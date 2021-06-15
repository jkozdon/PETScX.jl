const CPetscOptions = Ptr{Cvoid}

"""
    AbstractOptions{PetscLib <: PetscLibType}

Abstract type of PETSc solver options.
"""
abstract type AbstractOptions{PetscLib <: PetscLibType} end

"""
    GlobalOptions{PetscLib <: PetscLibType}

The PETSc global options database.
"""
struct GlobalOptions{PetscLib} <: AbstractOptions{PetscLib} end
Base.cconvert(::Type{CPetscOptions}, obj::GlobalOptions) = C_NULL

"""
    Options{PetscLib <: PetscLibType}(kw -> arg, ...)
    Options(petsclib, kw -> arg, ...)

Create a PETSc options data structure for the `petsclib`.

For construction a set of keyword argment pairs should be given. If the option
has no value it should be set to `nothing`

# Examples
```julia-repl
julia> opt = PETScX.Options(
           petsclib,
           ksp_monitor = true,
           ksp_view = true,
           pc_type = "mg",
           pc_mg_levels = 1,
       )
#PETSc Option Table entries:
-ksp_monitor true
-ksp_view true
-pc_mg_levels 1
-pc_type mg
#End of PETSc Option Table entries


julia> opt["ksp_monitor"]
"true"

julia> opt["pc_type"]
"mg"

julia> opt["pc_type"] = "ilu"
"ilu"

julia> opt["pc_type"]
"ilu"

julia> opt["bad_key"]
ERROR: KeyError: key "bad_key" not found
```

Manual: [`PetscOptionsCreate`](https://petsc.org/release/docs/manualpages/Sys/PetscOptionsCreate.html)
"""
mutable struct Options{T} <: AbstractOptions{T}
    ptr::CPetscOptions
end
Base.cconvert(::Type{CPetscOptions}, obj::Options) = obj.ptr
Base.unsafe_convert(::Type{Ptr{CPetscOptions}}, obj::Options) =
    convert(Ptr{CPetscOptions}, pointer_from_objref(obj))

Options(petsclib; kwargs...) = Options(petsclib, kwargs...)
function Options(petsclib, ps::Pair...)
    opts = Options(petsclib)
    for (k, v) in ps
        opts[k] = v
    end
    return opts
end

@for_libpetsc begin
    function Options(::$UnionPetscLib)
        opts = Options{$PetscLib}(C_NULL)
        @assert Initialized($PetscLib)
        @chk ccall(
            (:PetscOptionsCreate, $petsc_library),
            PetscErrorCode,
            (Ptr{CPetscOptions},),
            opts,
        )
        finalizer(Finalize, opts)
        return opts
    end

    function Finalize(opts::Options{$PetscLib})
        Finalized($PetscLib) || @chk ccall(
            (:PetscOptionsDestroy, $petsc_library),
            PetscErrorCode,
            (Ptr{CPetscOptions},),
            opts,
        )
        return nothing
    end

    function Base.push!(::GlobalOptions{$PetscLib}, opts::Options{$PetscLib})
        @chk ccall(
            (:PetscOptionsPush, $petsc_library),
            PetscErrorCode,
            (CPetscOptions,),
            opts,
        )
        return nothing
    end

    function Base.pop!(::GlobalOptions{$PetscLib})
        @chk ccall((:PetscOptionsPop, $petsc_library), PetscErrorCode, ())
        return nothing
    end

    function Base.setindex!(opts::AbstractOptions{$PetscLib}, val, key)
        @chk ccall(
            (:PetscOptionsSetValue, $petsc_library),
            PetscErrorCode,
            (CPetscOptions, Cstring, Cstring),
            opts,
            string('-', key),
            isnothing(val) ? C_NULL : string(val),
        )
    end

    function Base.getindex(opts::AbstractOptions{$PetscLib}, key)
        val = Vector{UInt8}(undef, 256)
        set_ref = Ref{PetscBool}()
        @chk ccall(
            (:PetscOptionsGetString, $petsc_library),
            PetscErrorCode,
            (
                CPetscOptions,
                Cstring,
                Cstring,
                Ptr{UInt8},
                Csize_t,
                Ptr{PetscBool},
            ),
            opts,
            C_NULL,
            string('-', key),
            val,
            sizeof(val),
            set_ref,
        )
        val = GC.@preserve val unsafe_string(pointer(val))
        set_ref[] == PETSC_TRUE || throw(KeyError(key))
        return val
    end

    function view(
        opts::AbstractOptions{$PetscLib},
        viewer::AbstractViewer{$PetscLib} = ViewerStdout(
            $PetscLib,
            MPI.COMM_SELF,
        ),
    )
        @chk ccall(
            (:PetscOptionsView, $petsc_library),
            PetscErrorCode,
            (CPetscOptions, CPetscViewer),
            opts,
            viewer,
        )
        return nothing
    end

    GlobalOptions(::$UnionPetscLib) = GlobalOptions{$PetscLib}()
end

Base.show(io::IO, opts::AbstractOptions) = _show(io, opts)

"""
    with(f, opts::Options)

Call `f()` with the [`Options`](@ref) `opts` set temporarily (in addition to any
global options).
"""
function with(f, opts::Options{T}) where {T}
    global_opts = GlobalOptions{T}()
    push!(global_opts, opts)
    try
        f()
    finally
        pop!(global_opts)
    end
end
