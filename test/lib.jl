using Test
using PETScX

@testset "lib tests" begin
    PETScX.@for_libpetsc begin
        @test $petsclib === PETScX.getpetsclib($PetscScalar, $PetscInt)
    end
end
