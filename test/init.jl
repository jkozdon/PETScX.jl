using Test
using PETScX

@testset "init tests" begin
    for petsclib in PETScX.petsclibs
        # The first time through finalize should be false since we have never
        # initialized petsclib yet...
        initial_finalized_value = false

        for x in (petsclib, typeof(petsclib))
            # since we haven't called anything these should be false!
            @test !(PETScX.Initialized(x))
            @test PETScX.Finalized(x) == initial_finalized_value

            # The second time through  time through finalize should be true since
            # we have initialized petsclib yet...
            initial_finalized_value = true

            # Initialize PETSc
            PETScX.Initialize(x)

            # Check values again
            @test PETScX.Initialized(x)
            @test !(PETScX.Finalized(x))

            PETScX.Finalize(x)

            # Check values again
            @test !(PETScX.Initialized(x))
            @test PETScX.Finalized(x)
        end
    end
end
