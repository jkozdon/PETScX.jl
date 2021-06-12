using PETSc, MPI, Printf
MPI.Initialized() || MPI.Init()
PETSc.initialize()

# Set up the RHS for a cell centered scheme for PDE
#
#  Δu = -cos(π * x) * cos(π * y)
#
# with zeros Neumann boundary conditions
function rhs!(
    ksp::PETSc.AbstractKSP{PetscScalar},
    b_vec::PETSc.AbstractVec{PetscScalar},
) where {PetscScalar}
    dm = PETSc.KSPGetDM(ksp)
    comm = PETSc.PetscObjectGetComm(ksp)
    corners = PETSc.DMDAGetCorners(dm)
    global_size = PETSc.DMDAGetInfo(dm).global_size[1:2]

    # Grid spacing in each direction
    h = PetscScalar(1) ./ global_size

    # Build the RHS vector for the cell-centered grid. Since the b_vec is not a
    # julia vector but a PETSc vector we need to convert it before operating on it
    PETSc.map_unsafe_localarray!(b_vec; read = false) do b
        b = reshape(b, Int64(corners.size[1]), Int64(corners.size[2]))
        for (iy, y) in enumerate(corners.lower[2]:corners.upper[2])
            y = (2y - 1) * h[2] / 2
            for (ix, x) in enumerate(corners.lower[1]:corners.upper[1])
                x = (2x - 1) * h[1] / 2
                b[ix, iy] = -cos(π * x) * cos(π * y) * h[1] * h[2]
            end
        end
    end

    # Assemble the PETSc vector
    PETSc.assemblybegin(b_vec)
    PETSc.assemblyend(b_vec)

    # Hack to remove constant from the vector since we are using all Neumann
    # boundary conditions
    nullspace = PETSc.MatNullSpace{PetscScalar}(comm, PETSc.PETSC_TRUE)
    PETSc.MatNullSpaceRemove!(nullspace, b_vec)
    PETSc.destroy(nullspace)

    # zero is the PETSc sucess code
    return 0
end

function jacobian!(
    ksp::PETSc.AbstractKSP{PetscScalar},
    J::PETSc.AbstractMat{PetscScalar},
    jac::PETSc.AbstractMat{PetscScalar},
) where {PetscScalar}
    dm = PETSc.KSPGetDM(ksp)
    corners = PETSc.DMDAGetCorners(dm)
    PetscInt = eltype(corners.size)

    global_size = PETSc.DMDAGetInfo(dm).global_size[1:2]

    # Grid spacing in each direction
    h = PetscScalar(1) ./ global_size

    # XXX: Would Mvectors be better?
    Sten = PETSc.MatStencil{PetscInt}
    col = Vector{Sten}(undef, 5)
    row = Vector{Sten}(undef, 1)
    val = Vector{PetscScalar}(undef, 5)

    for j in corners.lower[2]:corners.upper[2]
        for i in corners.lower[1]:corners.upper[1]
            row[1] = Sten(i = i, j = j)
            num = 1
            fill!(val, 0)
            if i > 1
                val[num] = -h[2] / h[1]
                col[num] = Sten(i = i - 1, j = j)
                num += 1
            end
            if i < global_size[1]
                val[num] = -h[2] / h[1]
                col[num] = Sten(i = i + 1, j = j)
                num += 1
            end
            if j > 1
                val[num] = -h[1] / h[2]
                col[num] = Sten(i = i, j = j - 1)
                num += 1
            end
            if j < global_size[2]
                val[num] = -h[1] / h[2]
                col[num] = Sten(i = i, j = j + 1)
                num += 1
            end
            val[num] = -sum(val)
            col[num] = Sten(i = i, j = j)
            PETSc.MatSetValuesStencil!(
                jac,
                row,
                col,
                val,
                PETSc.INSERT_VALUES;
                num_cols = num,
            )
        end
    end

    PETSc.assemblybegin(jac)
    PETSc.assemblyend(jac)

    # Since we have all Neumann there is constant nullspace
    comm = PETSc.PetscObjectGetComm(ksp)
    nullspace = PETSc.MatNullSpace{PetscScalar}(comm, PETSc.PETSC_TRUE)
    PETSc.MatSetNullSpace!(J, nullspace)
    PETSc.destroy(nullspace)
    return 0
end

function main(PetscScalar; comm = MPI.COMM_WORLD, options...)
    boundary_type = (PETSc.DM_BOUNDARY_NONE, PETSc.DM_BOUNDARY_NONE)
    stencil_type = PETSc.DMDA_STENCIL_STAR
    global_size = (11, 11)
    procs = (PETSc.PETSC_DECIDE, PETSc.PETSC_DECIDE)
    dof_per_node = 1
    stencil_width = 1
    points_per_proc = (nothing, nothing)

    opts =
        isinteractive() ? (ksp_monitor = nothing, ksp_view = nothing) :
        PETSc.parse_options(ARGS)

    da = PETSc.DMDACreate2d(
        PetscScalar,
        comm,
        boundary_type...,
        stencil_type,
        global_size...,
        procs...,
        dof_per_node,
        stencil_width,
        points_per_proc...;
        opts...,
    )

    PETSc.DMSetUp!(da)

    ksp = PETSc.KSP(da; opts...)

    PETSc.KSPSetComputeRHS!(ksp, rhs!)
    PETSc.KSPSetComputeOperators!(ksp, jacobian!)
    PETSc.setfromoptions!(ksp)
    PETSc.solve!(ksp)
end

main(Float64)
# main(Float32)
