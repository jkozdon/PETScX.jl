using Test
using PETScX

@testset "lib tests" begin
    # This should error, not library Float64, Float64 is possible!
    @test_throws ErrorException PETScX.getpetsclib(Float64, Float64)

    PETScX.@for_libpetsc begin
        # Make sure we can find the library correctly
        @test $petsclib === PETScX.getpetsclib($PetscScalar, $PetscInt)

        # Check the type accessors
        @test $PetscScalar === PETScX.scalartype($petsclib)
        @test $PetscScalar === PETScX.scalartype($PetscLib)
        @test $PetscInt === PETScX.inttype($petsclib)
        @test $PetscInt === PETScX.inttype($PetscLib)
        @test $PetscReal === PETScX.realtype($petsclib)
        @test $PetscReal === PETScX.realtype($PetscLib)
    end
end
