using Test
using MPI
using PETScX
using LinearAlgebra: norm

MPI.Initialized() || MPI.Init()

@testset "mpivec tests" begin
    comm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)

    # local first and last value
    n0 = sum(10 .+ (0:mpirank)) - 10
    n1 = sum(10 .+ (0:(mpirank + 1))) - 11

    # global last value
    ne = sum(10 .+ (0:mpisize)) - 11
    exact_length = ne + 1

    # exact norm
    exact_norm = norm(0:ne)

    for petsclib in PETScX.petsclibs
        PetscScalar = PETScX.scalartype(petsclib)
        PetscInt = PETScX.inttype(petsclib)

        for test_version in 1:3
            PETScX.Initialize(petsclib)
            if test_version == 1 # Test creation from array
                julia_x = PetscScalar.(n0:n1)
                petsc_x = PETScX.Vec(petsclib, comm, julia_x)
            elseif test_version == 2 # Test creation from local size
                petsc_x = PETScX.Vec(petsclib, comm, n1 - n0 + 1)

                vals = PetscScalar.(n0:n1)
                inds = PetscInt.(n0:n1)
                PETScX.setvalues!(petsc_x, inds, vals, PETScX.INSERT_VALUES)

            else # Test creation from global size
                petsc_x = PETScX.Vec(
                    petsclib,
                    comm,
                    PETScX.PETSC_DECIDE;
                    global_length = exact_length,
                )

                inds = PetscInt.(PETScX.ownershiprange(petsc_x))
                vals = PetscScalar.(inds)
                PETScX.setvalues!(petsc_x, inds, vals, PETScX.INSERT_VALUES)
            end

            @test length(petsc_x) == exact_length

            vec_norm = norm(petsc_x)
            @test exact_norm ≈ vec_norm

            # We cannot get a local form for this type
            @test_throws PETScX.PETScX_NoLocalForm PETScX.getlocalform(petsc_x)

            PETScX.destroy(petsc_x)

            PETScX.Finalize(petsclib)
        end
    end
end

@testset "mpivec with ghost tests" begin
    comm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)

    # local first and last value
    n0 = sum(10 .+ (0:mpirank)) - 10
    n1 = sum(10 .+ (0:(mpirank + 1))) - 11
    local_length = n1 - n0 + 1

    # global last value
    ne = sum(10 .+ (0:mpisize)) - 11
    exact_length = ne + 1

    # exact norm
    exact_norm = norm(0:ne)

    for petsclib in PETScX.petsclibs
        PetscScalar = PETScX.scalartype(petsclib)
        PetscInt = PETScX.inttype(petsclib)

        # two ghost to left and right
        ghost = PetscInt.([])
        if mpirank != 0
            ghost = PetscInt.([ghost..., n0 - 2, n0 - 1])
        end
        if mpirank != mpisize - 1
            ghost = PetscInt.([ghost..., n1 + 1, n1 + 2])
        end
        for with_array in (true, false)
            PETScX.Initialize(petsclib)

            # Test creation with array
            if with_array
                julia_x = zeros(PetscScalar, local_length + length(ghost))
                julia_x[1:local_length] = n0:n1
                petsc_x = PETScX.Vec(petsclib, comm, julia_x, ghost)
            else
                petsc_x = PETScX.Vec(petsclib, comm, local_length, ghost)

                inds = PetscInt.(PETScX.ownershiprange(petsc_x))
                vals = PetscScalar.(inds)
                PETScX.setvalues!(petsc_x, inds, vals, PETScX.INSERT_VALUES)
            end

            PETScX.assemblybegin!(petsc_x)
            PETScX.assemblyend!(petsc_x)

            @test length(petsc_x) == exact_length

            vec_norm = norm(petsc_x)
            @test exact_norm ≈ vec_norm

            l_x = PETScX.getlocalform(petsc_x)
            @test length(l_x) == local_length + length(ghost)

            if length(ghost) > 0
                vals = zeros(PetscScalar, length(ghost))
                inds = PetscInt.(local_length - 1 .+ (1:length(ghost)))
                PETScX.getvalues!(vals, l_x, inds)
                @test !(vals == ghost)

                PETScX.ghostupdatebegin!(petsc_x)
                PETScX.ghostupdateend!(petsc_x)

                vals = zeros(PetscScalar, length(ghost))
                inds = PetscInt.(local_length - 1 .+ (1:length(ghost)))
                PETScX.getvalues!(vals, l_x, inds)
                @test vals == ghost
            end

            PETScX.restorelocalform!(l_x)

            PETScX.withlocalform(petsc_x) do l_vec
                @test length(l_vec) == local_length + length(ghost)
            end

            PETScX.destroy(petsc_x)

            PETScX.Finalize(petsclib)
        end
    end
end
