using Test
using MPI
using PETScX
using LinearAlgebra: norm

MPI.Initialized() || MPI.Init()

@testset "mpivec tests" begin
    comm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)

    # local fist and last value
    n0 = sum(10 .+ (0:mpirank))
    n1 = sum(10 .+ (0:(mpirank + 1))) - 1

    # global last value
    ne = sum(10 .+ (0:(mpisize))) - 1
    exact_length = ne - 9

    # exact norm
    exact_norm = norm(10:ne)

    for petsclib in PETScX.petsclibs
        PETScX.Initialize(petsclib)

        PetscScalar = PETScX.scalartype(petsclib)
        PetscInt = PETScX.inttype(petsclib)

        # Test creation from array
        julia_x = PetscScalar.(n0:n1)

        petsc_x = PETScX.Vec(petsclib, comm, julia_x)

        @test length(petsc_x) == exact_length

        vec_norm = norm(petsc_x)
        @test exact_norm ≈ vec_norm

        PETScX.destroy(petsc_x)

        # Test creation from local size
        petsc_x = PETScX.Vec(petsclib, comm, n1 - n0 + 1)

        vals = PetscScalar.(n0:n1)
        inds = PetscInt.((n0:n1) .- 10)
        PETScX.setvalues!(petsc_x, inds, vals, PETScX.INSERT_VALUES)

        @test length(petsc_x) == exact_length

        vec_norm = norm(petsc_x)
        @test exact_norm ≈ vec_norm

        PETScX.destroy(petsc_x)

        # Test creation from global size
        petsc_x = PETScX.Vec(
            petsclib,
            comm,
            PETScX.PETSC_DECIDE;
            global_length = exact_length,
        )

        inds = PetscInt.(PETScX.ownershiprange(petsc_x))
        vals = PetscScalar.(10 .+ inds)
        PETScX.setvalues!(petsc_x, inds, vals, PETScX.INSERT_VALUES)

        @test length(petsc_x) == exact_length

        vec_norm = norm(petsc_x)
        @test exact_norm ≈ vec_norm

        PETScX.destroy(petsc_x)

        PETScX.Finalize(petsclib)
    end
end
