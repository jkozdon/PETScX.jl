using Test
using PETSc, MPI
MPI.Initialized() || MPI.Init()
PETSc.initialize()

@testset "DMDACreate1D" begin
    comm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)
    for (_, ST, RT, IT) in PETSc.libtypes
        # Loop over all boundary types and try to use them
        for boundary_type in instances(PETSc.DMBoundaryType)
            @testset "$boundary_type" begin
                dof_per_node = 4
                stencil_width = 5

                # We test both setting and not setting the point distribution
                points_per_proc = [IT(10 + i) for i in 1:mpisize]
                proc_global_offsets = [IT(0), accumulate(+, points_per_proc)...]

                global_size = proc_global_offsets[end]

                # left and right ghost points
                gl =
                    boundary_type == PETSc.DM_BOUNDARY_NONE && mpirank == 0 ?
                    0 : stencil_width
                gr =
                    boundary_type == PETSc.DM_BOUNDARY_NONE &&
                    mpirank == mpisize - 1 ? 0 : stencil_width

                # Set the points
                da = PETSc.DMDACreate1d(
                    ST,
                    comm,
                    boundary_type,
                    global_size,
                    dof_per_node,
                    stencil_width,
                    points_per_proc,
                )
                PETSc.DMSetUp!(da)

                da_info = PETSc.DMDAGetInfo(da)
                # local_da_info = PETSc.DMDAGetLocalInfo(da)
                corners = PETSc.DMDAGetCorners(da)
                ghost_corners = PETSc.DMDAGetGhostCorners(da)

                @test da_info.dim == 1
                @test da_info.global_size == [global_size, 1, 1]
                @test da_info.procs_per_dim == [mpisize, 1, 1]
                @test da_info.boundary_type == [
                    boundary_type,
                    PETSc.DM_BOUNDARY_NONE,
                    PETSc.DM_BOUNDARY_NONE,
                ]
                @test da_info.stencil_type == PETSc.DMDA_STENCIL_BOX
                @test da_info.stencil_width == stencil_width
                @test corners.lower ==
                      [proc_global_offsets[mpirank + 1] + 1, 1, 1]
                @test corners.upper == [proc_global_offsets[mpirank + 2], 1, 1]
                @test corners.size == [points_per_proc[mpirank + 1], 1, 1]
                @test ghost_corners.lower ==
                      [proc_global_offsets[mpirank + 1] + 1 - gl, 1, 1]
                @test ghost_corners.upper ==
                      [proc_global_offsets[mpirank + 2] + gr, 1, 1]
                @test ghost_corners.size ==
                      [points_per_proc[mpirank + 1] + gl + gr, 1, 1]

                # Do not set the points and test option parsing
                da_refine = 2
                da = PETSc.DMDACreate1d(
                    ST,
                    comm,
                    boundary_type,
                    global_size,
                    dof_per_node,
                    stencil_width,
                    nothing;
                    da_refine = da_refine,
                )
                PETSc.DMSetUp!(da)

                da_info = PETSc.DMDAGetInfo(da)

                @test da_info.dim == 1
                if boundary_type == PETSc.DM_BOUNDARY_PERIODIC
                    @test da_info.global_size ==
                          [2^da_refine * global_size, 1, 1]
                else
                    @test da_info.global_size ==
                          [2^da_refine * (global_size - 1) + 1, 1, 1]
                end
                @test da_info.procs_per_dim == [mpisize, 1, 1]
                @test da_info.boundary_type == [
                    boundary_type,
                    PETSc.DM_BOUNDARY_NONE,
                    PETSc.DM_BOUNDARY_NONE,
                ]
                @test da_info.stencil_type == PETSc.DMDA_STENCIL_BOX
                @test da_info.stencil_width == stencil_width
                # In this case we cannot check the numbers locally
            end
        end
    end
end
nothing
