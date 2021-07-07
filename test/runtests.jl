using Test
using MPI: mpiexec

# Do the MPI tests first so we do not have mpi running inside MPI
@testset "mpivec tests" begin
    @test mpiexec() do cmd
        run(`$cmd -n 4 julia --project mpivec.jl`)
        true
    end
end

include("lib.jl")
include("init.jl")
include("options.jl")
include("vec.jl")
include("mat.jl")
