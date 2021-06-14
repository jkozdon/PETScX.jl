module PETScX

using MPI

MPI.Initialized() || MPI.Init()

using Libdl

include("const.jl")
include("startup.jl")
include("lib.jl")
include("init.jl")
include("options.jl")
# include("ref.jl")
# include("viewer.jl")
# include("vec.jl")
# include("mat.jl")
# include("matshell.jl")
# include("dm.jl")
# include("dmda.jl")
# include("ksp.jl")
# include("pc.jl")
# include("snes.jl")
# include("sys.jl")

end
