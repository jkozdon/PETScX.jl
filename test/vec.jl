using Test
using PETScX
using LinearAlgebra: norm
using MPI: mpiexec

@testset "vec tests" begin
    for petsclib in PETScX.petsclibs
        PETScX.Initialize(petsclib)
        PetscScalar = PETScX.scalartype(petsclib)
        PetscInt = PETScX.inttype(petsclib)

        julia_x = [PetscScalar(i) for i in 1:10]
        petsc_x = PETScX.Vec(petsclib, julia_x)

        # Check the basics
        @test length(petsc_x) == length(julia_x)
        @test norm(petsc_x) == norm(julia_x)
        #XXX: For some reason non-l2 is broken...
        # @test norm(petsc_x, PETScX.NORM_INFINITY) == norm(julia_x, Inf)
        display(petsc_x)

        # Check the view
        _stdout = stdout
        (rd, wr) = redirect_stdout()
        @show petsc_x
        @test readline(rd) == "petsc_x = Vec Object: 1 MPI processes"
        @test readline(rd) == "  type: seq"
        for x in julia_x
            @test readline(rd) == "$(Int(x))."
        end
        redirect_stdout(_stdout)

        # Test reading and writing values
        for (i, x) in enumerate(julia_x)
            @test petsc_x[i] === x
            petsc_x[i] ^= 2
            @test petsc_x[i] === x^2
        end

        # Test writing many values
        i = PetscInt.([0, 2, 4])
        v = PetscScalar.([-2, 4, 10])
        PETScX.setvalues!(petsc_x, i, v, PETScX.INSERT_VALUES)
        @test julia_x[i .+ 1] == v
        PETScX.setvalues!(petsc_x, i, v, PETScX.ADD_VALUES)
        @test julia_x[i .+ 1] == 2v

        # Test reading many values
        vals = fill!(similar(v), 0)
        PETScX.getvalues!(vals, petsc_x, i)
        @test vals == 2v

        PETScX.Finalize(petsclib)
    end
end
