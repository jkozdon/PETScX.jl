using Test
using PETScX

@testset "mat tests" begin
    for petsclib in PETScX.petsclibs
        PETScX.Initialize(petsclib)
        PetscScalar = PETScX.scalartype(petsclib)
        PetscInt = PETScX.inttype(petsclib)

        # Create a matrix
        m = 10
        n = m
        nz = 3
        A = PETScX.MatAIJ(petsclib, m, n, nz)

        # Check some dimension information
        row_rng = PETScX.ownershiprange(A)
        @test row_rng == 0:(m - 1)
        @test size(A) == (m, n)
        @test length(A) == m * n

        vals = PetscScalar.([1, -1, -1, 1])
        for r in row_rng
            rows = cols = PetscInt.([r - 1, r])
            PETScX.setvalues!(A, rows, cols, vals, PETScX.ADD_VALUES)
        end

        # Fix up the last entry
        rows = cols = PetscInt.([m - 1])
        PETScX.setvalues!(A, rows, cols, vals, PETScX.ADD_VALUES)

        # Do assembly
        PETScX.assemblybegin!(A)
        PETScX.assemblyend!(A)

        # Check the the values we inserted are correct
        vals = zeros(PetscScalar, 3)
        for r in row_rng
            rows = PetscInt.([r])
            if r == 0
                cols = PetscInt.([r, r + 1])
                PETScX.getvalues!(vals, A, rows, cols)
                @test PetscScalar.([2, -1]) == vals[1:2]
            elseif r == m - 1
                cols = PetscInt.([r - 1, r])
                PETScX.getvalues!(vals, A, rows, cols)
                @test PetscScalar.([-1, 2]) == vals[1:2]
            else
                cols = PetscInt.([r - 1, r, r + 1])
                PETScX.getvalues!(vals, A, rows, cols)
                @test PetscScalar.([-1, 2, -1]) == vals
            end
        end

        PETScX.destroy(A)

        PETScX.Finalize(petsclib)
    end
end
