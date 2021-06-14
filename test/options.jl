using Test
using PETScX

@testset "options tests" begin
    kw_opts = (
        ksp_monitor = nothing,
        ksp_view = nothing,
        da_grid_x = 100,
        da_grid_y = 100,
        pc_type = "mg",
        pc_mg_levels = 1,
        mg_levels_0_pc_type = "ilu",
    )
    for petsclib in PETScX.petsclibs
        PETScX.Initialize(petsclib)

        opts = PETScX.Options(petsclib; kw_opts...)

        # Check that all the keys got added
        for (key, val) in pairs(kw_opts)
            key = string(key)
            val = isnothing(val) ? "" : string(val)
            @test val == opts[key]
        end

        # Check that we throw when a bad key is asked for
        @test_throws KeyError opts["bad key"]

        # try to insert some new keys
        opts["new_opt"] = 1
        opts["nothing_opt"] = nothing
        @test "" == opts["nothing_opt"]
        @test "1" == opts["new_opt"]

        PETScX.Finalize(petsclib)
    end
end
