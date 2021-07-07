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

        # Check that viewer is working
        _stdout = stdout
        (rd, wr) = redirect_stdout()
        @show opts

        @test readline(rd) == "opts = #PETSc Option Table entries:"
        @test readline(rd) == "-da_grid_x 100"
        @test readline(rd) == "-da_grid_y 100"
        @test readline(rd) == "-ksp_monitor"
        @test readline(rd) == "-ksp_view"
        @test readline(rd) == "-mg_levels_0_pc_type ilu"
        @test readline(rd) == "-new_opt 1"
        @test readline(rd) == "-nothing_opt"
        @test readline(rd) == "-pc_mg_levels 1"
        @test readline(rd) == "-pc_type mg"
        @test readline(rd) == "#End of PETSc Option Table entries"
        @test readline(rd) == ""

        glo_opts = PETScX.GlobalOptions(petsclib)
        show(stdout, "text/plain", glo_opts)
        @test readline(rd) == "#No PETSc Option Table entries"

        # Try to set some options and check that they are set
        PETScX.with(opts) do
            show(stdout, "text/plain", glo_opts)
        end
        @test readline(rd) == "#PETSc Option Table entries:"
        @test readline(rd) == "-da_grid_x 100"
        @test readline(rd) == "-da_grid_y 100"
        @test readline(rd) == "-ksp_monitor"
        @test readline(rd) == "-ksp_view"
        @test readline(rd) == "-mg_levels_0_pc_type ilu"
        @test readline(rd) == "-new_opt 1"
        @test readline(rd) == "-nothing_opt"
        @test readline(rd) == "-pc_mg_levels 1"
        @test readline(rd) == "-pc_type mg"
        @test readline(rd) == "#End of PETSc Option Table entries"

        show(stdout, "text/plain", glo_opts)
        @test readline(rd) == "#No PETSc Option Table entries"

        redirect_stdout(_stdout)

        PETScX.Finalize(petsclib)
    end
end
