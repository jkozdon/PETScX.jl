# AbstractVec
#   - VecGhost (TODO)
# for the MPI variants we won't be able to attach finalizers, as destroy needs
# to be called collectively.
using LinearAlgebra

const CVec = Ptr{Cvoid}

"""
    AbstractVec{PetscLib <: PetscLibType}

Abstract type of PETSc vectors.

Manual: [`Vec`](https://petsc.org/release/docs/manualpages/Vec/Vec.html)
"""
abstract type AbstractVec{T, PetscLib <: PetscLibType{T}} <: AbstractVector{T} end
Base.eltype(::Type{V}) where {V <: AbstractVec{T}} where {T} = T
Base.eltype(v::AbstractVec{T}) where {T} = T
Base.size(v::AbstractVec) = (length(v),)
Base.parent(v::AbstractVec) = v.array

# allows us to pass XXVec objects directly into CVec ccall signatures
Base.cconvert(::Type{CVec}, obj::AbstractVec) = obj.__ptr__

# allows us to pass XXVec objects directly into Ptr{CVec} ccall signatures
Base.unsafe_convert(::Type{Ptr{CVec}}, obj::AbstractVec) =
    convert(Ptr{CVec}, pointer_from_objref(obj))

mutable struct Vec{
    T,
    PetscLib <: PetscLibType{T},
    AT <: Union{Vector{T}, Nothing},
    CT <: Union{MPI.Comm, Nothing},
} <: AbstractVec{T, PetscLib}
    __ptr__::CVec
    __possible_array_ref__::AT
    __possible_comm_ref::CT
    function Vec{T, PetscLib}(;
        array::AT = nothing,
        comm::CT = nothing,
    ) where {PetscLib, T, AT, CT}
        new{T, PetscLib, AT, CT}(C_NULL, array, comm)
    end
end

@for_libpetsc begin
    function view(
        vec::AbstractVec{$PetscScalar, $PetscLib},
        viewer::AbstractViewer{$PetscLib} = ViewerStdout($PetscLib),
    )
        @chk ccall(
            (:VecView, $petsc_library),
            PetscErrorCode,
            (CVec, CPetscViewer),
            vec,
            viewer,
        )
        return nothing
    end
end
Base.show(io::IO, v::AbstractVec) = _show(io, v)

"""
    destroy(v::AbstractVec)

Destroy the PETSc vector `v`

Manual: [`VecDestroy`](https://petsc.org/release/docs/manualpages/Vec/VecDestroy.html)
"""
destroy(::AbstractVec)

@for_libpetsc begin
    function destroy(petsc_v::AbstractVec{$PetscScalar, $PetscLib})
        Finalized($PetscLib) || @chk ccall(
            (:VecDestroy, $petsc_library),
            PetscErrorCode,
            (Ptr{CVec},),
            petsc_v,
        )

        # Zero the pointer so that if VecDestroy gets called multiple times we
        # do not artificially decrease the internal petsc ref counter
        petsc_v.__ptr__ = C_NULL
        return nothing
    end
end

"""
    Vec(petsclib, v::Vector; blocksize = 1, local_length = length(v))

A standard, sequentially-stored serial PETSc vector, wrapping the Julia vector
`v`.

Manual: [`VecCreateSeqWithArray`](https://petsc.org/release/docs/manualpages/Vec/VecCreateSeqWithArray.html)

!!! warning
    This reuses the array `v` as storage, and so `v` should not be `resize!`-ed
    or otherwise have its length modified while the PETSc object exists.

!!! note
    Since the `Vec` is inherently serial `destroy` will be called by the garbage
    collector
"""
Vec(_, ::Vector)

@for_libpetsc begin
    function Vec(
        ::$UnionPetscLib,
        jl_v::Vector{$PetscScalar};
        blocksize = 1,
        local_length = length(jl_v),
    )
        @assert Initialized($PetscLib)
        petsc_v = Vec{$PetscScalar, $PetscLib}(array = jl_v)
        @chk ccall(
            (:VecCreateSeqWithArray, $petsc_library),
            PetscErrorCode,
            (MPI.MPI_Comm, $PetscInt, $PetscInt, Ptr{$PetscScalar}, Ptr{CVec}),
            MPI.COMM_SELF,
            blocksize,
            local_length,
            jl_v,
            petsc_v,
        )

        finalizer(destroy, petsc_v)
        return petsc_v
    end
end

"""
    Vec(
        petsclib,
        comm,
        v::Vector;
        blocksize = 1,
        local_length = length(v),
        global_length = PETSC_DETERMINE
    )

Create An MPI distributed vectors without ghost elements using the vector `v`
for storage.

Manual: [`VecCreateMPIWithArray`](https://petsc.org/release/docs/manualpages/Vec/VecCreateMPIWithArray.html)

!!! warning
    The array `v` is reused as the storage and should not be `resize!`-ed or
    otherwise have its length modified while the PETSc object exists.

!!! note
    The user is responsible for calling `destroy(vec)` on the `Vec` since
    this cannot be handled by the garbage collector do to the MPI nature of the
    object.
"""
Vec(_, ::MPI.Comm, ::Vector)

@for_libpetsc begin
    function Vec(
        ::$UnionPetscLib,
        comm::MPI.Comm,
        jl_v::Vector{$PetscScalar};
        blocksize = 1,
        global_length = PETSC_DETERMINE,
        local_length = length(jl_v),
    )
        global_length > 0 || (global_length = PETSC_DETERMINE)
        @assert Initialized($PetscLib)
        petsc_v = Vec{$PetscScalar, $PetscLib}(comm = comm, array = jl_v)
        @chk ccall(
            (:VecCreateMPIWithArray, $petsc_library),
            PetscErrorCode,
            (
                MPI.MPI_Comm,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                Ptr{$PetscScalar},
                Ptr{CVec},
            ),
            comm,
            blocksize,
            local_length,
            global_length,
            jl_v,
            petsc_v,
        )
        return petsc_v
    end
end

"""
    Vec(
        petsclib,
        comm,
        local_length::Integer;
        global_length::Integer = PETSC_DETERMINE,
    )

An MPI distributed vectors without ghost elements with `local_length` and
`global_length`.

If `global_length == PETSC_DETERMINE` then the global length is determined by
PETSc.

If `local_length == PETSC_DECIDE` then the local length on each MPI rank is
determined by PETSc.

Manual: [`VecCreateMPI`](https://petsc.org/release/docs/manualpages/Vec/VecCreateMPI.html)

!!! note
    The user is responsible for calling `destroy(vec)` on the `Vec` since
    this cannot be handled by the garbage collector do to the MPI nature of the
    object.
"""
Vec(_, ::MPI.Comm, ::Integer)

@for_libpetsc begin
    function Vec(
        ::$UnionPetscLib,
        comm::MPI.Comm,
        local_length;
        global_length = PETSC_DETERMINE,
    )
        @assert (global_length > 0) || (local_length > 0)

        @assert Initialized($PetscLib)

        petsc_v = Vec{$PetscScalar, $PetscLib}(comm = comm)
        @chk ccall(
            (:VecCreateMPI, $petsc_library),
            PetscErrorCode,
            (MPI.MPI_Comm, $PetscInt, $PetscInt, Ptr{CVec}),
            comm,
            local_length,
            global_length,
            petsc_v,
        )
        return petsc_v
    end
end

"""
    Vec(
        petsclib,
        comm,
        local_length::Integer,
        ghost::Vector{PetscInt};
        num_ghost::Integer = length(ghost),
        global_length::Integer = PETSC_DETERMINE,
    )


An MPI distributed vectors with ghost elements with `local_length` and
`global_length`. The global indices of the ghost element are determined by the
`ghost` vector.

If `global_length == PETSC_DETERMINE` then the global length is determined by
PETSc.

If `local_length == PETSC_DECIDE` then the local length on each MPI rank is
determined by PETSc.

Manual: [`VecCreateGhost`](https://petsc.org/release/docs/manualpages/Vec/VecCreateGhost.html)

!!! note
    The user is responsible for calling `destroy(vec)` on the `Vec` since
    this cannot be handled by the garbage collector do to the MPI nature of the
    object.
"""
Vec(_, ::MPI.Comm, ::Integer, ::Vector)

@for_libpetsc begin
    function Vec(
        ::$UnionPetscLib,
        comm::MPI.Comm,
        local_length,
        ghost::Vector{$PetscInt};
        num_ghost = length(ghost),
        global_length = PETSC_DETERMINE,
    )
        @assert (global_length > 0) || (local_length > 0)

        @assert Initialized($PetscLib)

        petsc_v = Vec{$PetscScalar, $PetscLib}(comm = comm)
        @chk ccall(
            (:VecCreateGhost, $petsc_library),
            PetscErrorCode,
            (
                MPI.MPI_Comm,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                Ptr{$PetscInt},
                Ptr{CVec},
            ),
            comm,
            local_length,
            global_length,
            num_ghost,
            ghost,
            petsc_v,
        )
        return petsc_v
    end
end

"""
    Vec(
        petsclib,
        comm,
        v::Vector{PetscScalar};
        ghost::Vector{PetscInt};
        local_length::Integer = length(v) - length(ghost),
        num_ghost::Integer = length(ghost),
        global_length::Integer = PETSC_DETERMINE,
    )


An MPI distributed vectors with ghost elements using the vector `v` for local
and ghost storage. The global indices of the ghost element are determined by the
`ghost` vector.

If `global_length == PETSC_DETERMINE` then the global length is determined by
PETSc.

Manual: [`VecCreateGhostWithArray`](https://petsc.org/release/docs/manualpages/Vec/VecCreateGhostWithArray.html)

!!! warning
    The array `v` is reused as the storage and should not be `resize!`-ed or
    otherwise have its length modified while the PETSc object exists.

!!! note
    The user is responsible for calling `destroy(vec)` on the `Vec` since
    this cannot be handled by the garbage collector do to the MPI nature of the
    object.
"""
Vec(_, ::MPI.Comm, ::Integer, ::Vector)

@for_libpetsc begin
    function Vec(
        ::$UnionPetscLib,
        comm::MPI.Comm,
        jl_v::Vector{$PetscScalar},
        ghost::Vector{$PetscInt};
        local_length = length(jl_v) - length(ghost),
        num_ghost = length(ghost),
        global_length = PETSC_DETERMINE,
    )
        @assert local_length + num_ghost <= length(jl_v)

        @assert Initialized($PetscLib)

        petsc_v = Vec{$PetscScalar, $PetscLib}(comm = comm, array = jl_v)
        @chk ccall(
            (:VecCreateGhostWithArray, $petsc_library),
            PetscErrorCode,
            (
                MPI.MPI_Comm,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                Ptr{$PetscInt},
                Ptr{$PetscScalar},
                Ptr{CVec},
            ),
            comm,
            local_length,
            global_length,
            num_ghost,
            ghost,
            jl_v,
            petsc_v,
        )
        return petsc_v
    end
end

"""
    ownershiprange(vec::AbstractVec)

The range of global indices owned by this MPI rank, assuming that the vectors
are laid out with the first `n1` elements on the first processor, next `n2`
elements on the second, etc. For certain parallel layouts this range may not be
well defined.

Manual: [`VecGetOwnershipRange`](https://petsc.org/release/docs/manualpages/Vec/VecGetOwnershipRange.html)

!!! note
    The range returned is inclusive (`idx_first:idx_last`)
"""
ownershiprange

@for_libpetsc begin
    function ownershiprange(vec::AbstractVec{$PetscScalar})
        r_lo = Ref{$PetscInt}()
        r_hi = Ref{$PetscInt}()
        @chk ccall(
            (:VecGetOwnershipRange, $petsc_library),
            PetscErrorCode,
            (CVec, Ptr{$PetscInt}, Ptr{$PetscInt}),
            vec,
            r_lo,
            r_hi,
        )
        return r_lo[]:(r_hi[] - $PetscInt(1))
    end
end

"""
    setvalues!(
        v::AbstractVec{PetscScalar, PetscLib},
        indices::Vector{PetscInt},
        vals::Array{PetscScalar},
        insertmode::InsertMode,
    )

Assign the values `vals` in 0-based global `indices` of `vec`. The `insertmode`
can be `INSERT_VALUES` or `ADD_VALUES`.

!!! warning
    This function uses 0-based indexing!

Manual: [`VecSetValues`](https://petsc.org/release/docs/manualpages/Vec/VecSetValues.html)
"""
setvalues!(::AbstractVec)

@for_libpetsc begin
    function setvalues!(
        v::AbstractVec{$PetscScalar, $PetscLib},
        idxs0::Vector{$PetscInt},
        vals::Array{$PetscScalar},
        insertmode::InsertMode,
    )
        @assert length(vals) >= length(idxs0)
        @chk ccall(
            (:VecSetValues, $petsc_library),
            PetscErrorCode,
            (CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}, InsertMode),
            v,
            length(idxs0),
            idxs0,
            vals,
            insertmode,
        )
        return nothing
    end
end

"""
    getvalues!(
        vals::Array{PetscScalar},
        v::AbstractVec{PetscScalar, PetscLib},
        indices::Vector{PetscInt},
    )

Get the 0-based global `indices` of `vec` into the preallocated array `vals`.

!!! warning
    This function uses 0-based indexing!

!!! note
    Can only access local values (not ghost values)

Manual: [`VecGetValues`](https://petsc.org/release/docs/manualpages/Vec/VecGetValues.html)
"""
getvalues(::AbstractVec)

@for_libpetsc begin
    function getvalues!(
        vals::Array{$PetscScalar},
        v::AbstractVec{$PetscScalar, $PetscLib},
        idxs0::Vector{$PetscInt},
    )
        @assert length(vals) >= length(idxs0)
        @chk ccall(
            (:VecGetValues, $petsc_library),
            PetscErrorCode,
            (CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}),
            v,
            length(idxs0),
            idxs0,
            vals,
        )
        return vals
    end
end

#=
"""
    setvalueslocal!(
        v::AbstractVec{PetscScalar, PetscLib},
        indices::Vector{PetscInt},
        vals::Array{PetscScalar},
        insertmode::InsertMode,
    )

Assign the values `vals` in 0-based local `indices` of `vec`. The `insertmode`
can be `INSERT_VALUES` or `ADD_VALUES`.

!!! warning
    This function uses 0-based indexing!

Manual: [`VecSetValuesLocal`]( https://petsc.org/release/docs/manualpages/Vec/VecSetValuesLocal.html)
"""
setvalueslocal!(::AbstractVec)

@for_libpetsc begin
    TODO: need more to allow this
    function setvalueslocal!(
        v::AbstractVec{$PetscScalar, $PetscLib},
        idxs0::Vector{$PetscInt},
        vals::Array{$PetscScalar},
        insertmode::InsertMode,
    )
        @assert length(vals) >= length(idxs0)
        @chk ccall(
            (:VecSetValuesLocal, $petsc_library),
            PetscErrorCode,
            (
                CVec,
                $PetscInt,
                Ptr{$PetscInt},
                Ptr{$PetscScalar},
                InsertMode,
            ),
            v,
            length(idxs0),
            idxs0,
            vals,
            insertmode,
        )
        return nothing
    end
end
=#

"""
    LocalVec(vec::AbstractVec)

Obtains the local ghosted representation of a [`Vec`](@ref).

!!! note
    When done with the object the user should call [`restorelocalform!`](@ref)

Manual: [`VecGhostGetLocalForm`](https://petsc.org/release/docs/manualpages/Vec/VecGhostGetLocalForm.html)
"""
LocalVec

"""
    restorelocalform!(local_vec::LocalVec)

Restore the `local_vec` to the associated global vector after a call to
[`getlocalform`](@ref).

Manual: [`VecGhostRestoreLocalForm`](https://petsc.org/release/docs/manualpages/Vec/VecGhostRestoreLocalForm.html)
"""
restorelocalform!

mutable struct LocalVec{
    T,
    PetscLib <: PetscLibType{T},
    GVec <: AbstractVec{T, PetscLib},
} <: AbstractVec{T, PetscLib}
    __ptr__::CVec
    __global_vec__::GVec
    function LocalVec{PetscLib}(gvec::GVec) where {PetscLib, GVec}
        PetscScalar = scalartype(PetscLib)
        new{PetscScalar, PetscLib, GVec}(C_NULL, gvec)
    end
end

@for_libpetsc begin
    function getlocalform(gvec::AbstractVec{$PetscScalar, $PetscLib})
        lvec = LocalVec{$PetscLib}(gvec)
        @chk ccall(
            (:VecGhostGetLocalForm, $petsc_library),
            PetscErrorCode,
            (CVec, Ptr{CVec}),
            gvec,
            lvec,
        )
        return lvec
    end

    function restorelocalform!(lvec::LocalVec{$PetscScalar, $PetscLib})
        @chk ccall(
            (:VecGhostRestoreLocalForm, $petsc_library),
            PetscErrorCode,
            (CVec, Ptr{CVec}),
            lvec.__global_vec__,
            lvec,
        )
        lvec.__ptr__ = C_NULL
        return lvec.__global_vec__
    end
end

# `LinearAlgebra` pirated functions for AbstractVec
# - norm
@for_libpetsc begin
    function LinearAlgebra.norm(
        v::AbstractVec{$PetscScalar, $PetscLib},
        normtype::NormType = NORM_2,
    )
        # For some reason on this currently works!
        @assert normtype == NORM_2
        r_val = Ref{$PetscReal}()
        @chk ccall(
            (:VecNorm, $petsc_library),
            PetscErrorCode,
            (CVec, NormType, Ptr{$PetscReal}),
            v,
            normtype,
            r_val,
        )
        return r_val[]
    end
end

# `Base` pirated functions for AbstractVec
# - length
# - setindex!
# - getindex
@for_libpetsc begin
    function Base.length(v::AbstractVec{$PetscScalar, $PetscLib})
        r_sz = Ref{$PetscInt}()
        @chk ccall(
            (:VecGetSize, $petsc_library),
            PetscErrorCode,
            (CVec, Ptr{$PetscInt}),
            v,
            r_sz,
        )
        return r_sz[]
    end

    function Base.setindex!(
        v::AbstractVec{$PetscScalar, $PetscLib},
        val,
        i::Integer,
    )
        @chk ccall(
            (:VecSetValues, $petsc_library),
            PetscErrorCode,
            (CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}, InsertMode),
            v,
            1,
            Ref{$PetscInt}(i - 1),
            Ref{$PetscScalar}(val),
            INSERT_VALUES,
        )

        return val
    end

    function Base.getindex(v::AbstractVec{$PetscScalar, $PetscLib}, i::Integer)
        vals = [$PetscScalar(0)]
        @chk ccall(
            (:VecGetValues, $petsc_library),
            PetscErrorCode,
            (CVec, $PetscInt, Ptr{$PetscInt}, Ptr{$PetscScalar}),
            v,
            1,
            Ref{$PetscInt}(i - 1),
            vals,
        )

        return vals[1]
    end
end

"""
    assemblybegin!(vec::AbstractVec)

Begin assembling `vec`

Manual: [`VecAssemblyBegin`](https://petsc.org/release/docs/manualpages/Vec/VecAssemblyBegin.html)
"""
assemblybegin!(::AbstractVec)

"""
    assemblyend!(vec::AbstractVec)

Finish assembling `vec`

Manual: [`VecAssemblyEnd`](https://petsc.org/release/docs/manualpages/Vec/VecAssemblyEnd.html)
"""
assemblyend!(::AbstractVec)

@for_libpetsc begin
    function assemblybegin!(vec::AbstractVec{$PetscScalar, $PetscLib})
        @chk ccall(
            (:VecAssemblyBegin, $petsc_library),
            PetscErrorCode,
            (CVec,),
            vec,
        )
        return nothing
    end

    function assemblyend!(vec::AbstractVec{$PetscScalar, $PetscLib})
        @chk ccall(
            (:VecAssemblyEnd, $petsc_library),
            PetscErrorCode,
            (CVec,),
            vec,
        )
        return nothing
    end
end

"""
    ghostupdatebegin!(
        vec::AbstractVec,
        insertmode = INSERT_VALUES,
        scattermode = SCATTER_FORWARD,
    )

Begins scattering `vec` to the local or global representations

Manual: [`VecGhostUpdateBegin`](https://petsc.org/release/docs/manualpages/Vec/VecGhostUpdateBegin.html)
"""
ghostupdatebegin!(::AbstractVec)

"""
    ghostupdateend!(
        vec::AbstractVec,
        insertmode = INSERT_VALUES,
        scattermode = SCATTER_FORWARD,
    )

Finishes scattering `vec` to the local or global representations

Manual: [`VecGhostUpdateEnd`](https://petsc.org/release/docs/manualpages/Vec/VecGhostUpdateEnd.html)
"""
ghostupdateend!(::AbstractVec)

@for_libpetsc begin
    function ghostupdatebegin!(
        vec::AbstractVec{$PetscScalar, $PetscLib},
        insertmode = INSERT_VALUES,
        scattermode = SCATTER_FORWARD,
    )
        @chk ccall(
            (:VecGhostUpdateBegin, $petsc_library),
            PetscErrorCode,
            (CVec, InsertMode, ScatterMode),
            vec,
            insertmode,
            scattermode,
        )
        return nothing
    end

    function ghostupdateend!(
        vec::AbstractVec{$PetscScalar, $PetscLib},
        insertmode = INSERT_VALUES,
        scattermode = SCATTER_FORWARD,
    )
        @chk ccall(
            (:VecGhostUpdateEnd, $petsc_library),
            PetscErrorCode,
            (CVec, InsertMode, ScatterMode),
            vec,
            insertmode,
            scattermode,
        )
        return nothing
    end
end

#=
    function unsafe_localarray(
        ::Type{$PetscScalar},
        cv::CVec;
        read::Bool = true,
        write::Bool = true,
    )
        r_pv = Ref{Ptr{$PetscScalar}}()
        if write
            if read
                @chk ccall(
                    (:VecGetArray, $petsc_library),
                    PetscErrorCode,
                    (CVec, Ptr{Ptr{$PetscScalar}}),
                    cv,
                    r_pv,
                )
            else
                @chk ccall(
                    (:VecGetArrayWrite, $petsc_library),
                    PetscErrorCode,
                    (CVec, Ptr{Ptr{$PetscScalar}}),
                    cv,
                    r_pv,
                )
            end
        else
            @chk ccall(
                (:VecGetArrayRead, $petsc_library),
                PetscErrorCode,
                (CVec, Ptr{Ptr{$PetscScalar}}),
                cv,
                r_pv,
            )
        end
        r_sz = Ref{$PetscInt}()
        @chk ccall(
            (:VecGetLocalSize, $petsc_library),
            PetscErrorCode,
            (CVec, Ptr{$PetscInt}),
            cv,
            r_sz,
        )
        v = unsafe_wrap(Array, r_pv[], r_sz[]; own = false)

        if write
            if read
                finalizer(v) do v
                    @chk ccall(
                        (:VecRestoreArray, $petsc_library),
                        PetscErrorCode,
                        (CVec, Ptr{Ptr{$PetscScalar}}),
                        cv,
                        Ref(pointer(v)),
                    )
                    return nothing
                end
            else
                finalizer(v) do v
                    @chk ccall(
                        (:VecRestoreArrayWrite, $petsc_library),
                        PetscErrorCode,
                        (CVec, Ptr{Ptr{$PetscScalar}}),
                        cv,
                        Ref(pointer(v)),
                    )
                    return nothing
                end
            end
        else
            finalizer(v) do v
                @chk ccall(
                    (:VecRestoreArrayRead, $petsc_library),
                    PetscErrorCode,
                    (CVec, Ptr{Ptr{$PetscScalar}}),
                    cv,
                    Ref(pointer(v)),
                )
                return nothing
            end
        end
        return v
    end

"""
    unsafe_localarray(PetscScalar, ptr:CVec; read=true, write=true)
    unsafe_localarray(ptr:AbstractVec; read=true, write=true)

Return an `Array{PetscScalar}` containing local portion of the PETSc data.

Use `read=false` if the array is write-only; `write=false` if read-only.

!!! note
    `Base.finalize` should be called on the `Array` before the data can be used.
"""
unsafe_localarray

unsafe_localarray(v::AbstractVec{T}; kwargs...) where {T} =
    unsafe_localarray(T, v.ptr; kwargs...)

"""
    map_unsafe_localarray!(f!, x::AbstractVec{T}; read=true, write=true)

Convert `x` to an `Array{T}` and apply the function `f!`.

Use `read=false` if the array is write-only; `write=false` if read-only.

# Examples
```julia-repl
julia> map_unsafe_localarray(x; write=true) do x
   @. x .*= 2
end

!!! note
    `Base.finalize` should is automatically called on the array.
"""
function map_unsafe_localarray!(f!, v::AbstractVec{T}; kwargs...) where {T}
    array = unsafe_localarray(T, v.ptr; kwargs...)
    f!(array)
    Base.finalize(array)
end
=#
