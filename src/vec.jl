# AbstractVec
#   - VecSeq: wrap
#   - VecMPI
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
Base.cconvert(::Type{CVec}, obj::AbstractVec) = obj.ptr

# allows us to pass XXVec objects directly into Ptr{CVec} ccall signatures
Base.unsafe_convert(::Type{Ptr{CVec}}, obj::AbstractVec) =
    convert(Ptr{CVec}, pointer_from_objref(obj))

"""
    VecSeq(petsclib, v::Vector)

A standard, sequentially-stored serial PETSc vector, wrapping the Julia vector
`v`.

Manual: [`VecCreateSeqWithArray`](https://petsc.org/release/docs/manualpages/Vec/VecCreateSeqWithArray.html)

!!! warning
    This reuses the array `v` as storage, and so `v` should not be `resize!`-ed
    or otherwise have its length modified while the PETSc object exists.
"""
mutable struct VecSeq{T, PetscLib <: PetscLibType{T}} <:
               AbstractVec{T, PetscLib}
    ptr::CVec
    array::Vector{T}
end
@for_libpetsc begin
    function VecSeq(::$UnionPetscLib, jl_v::Vector{$PetscScalar}; blocksize = 1)
        @assert Initialized($PetscLib)
        petsc_v = VecSeq{$PetscScalar, $PetscLib}(C_NULL, jl_v)
        @chk ccall(
            (:VecCreateSeqWithArray, $petsc_library),
            PetscErrorCode,
            (MPI.MPI_Comm, $PetscInt, $PetscInt, Ptr{$PetscScalar}, Ptr{CVec}),
            MPI.COMM_SELF,
            blocksize,
            length(jl_v),
            jl_v,
            petsc_v,
        )
        return petsc_v
    end
end

"""
    VecMPI(petsclib, comm, v::Vector)
    VecMPI(petsclib, comm, local_length::Int, global_length::Int = -1)

An MPI distributed vectors without ghost elements.

If `v` is given then this is taken to be the mpi rank local values of the array
and the local size is determined from `length(v)`.

If `local_length ≥ 0` then this is taken to be the local length of the vector.

If `local_length ≤ 0` then PETSc will decide on the partitioning based on
`global_length`.

Manual: [`VecCreateMPI`](https://petsc.org/release/docs/manualpages/Vec/VecCreateMPI.html)
Manual: [`VecCreateMPIWithArray`](https://petsc.org/release/docs/manualpages/Vec/VecCreateMPIWithArray.html)

!!! warning
    If `V` is specified, then the array `v` is reused as the storage and should
    not be `resize!`-ed or otherwise have its length modified while the PETSc
    object exists.

!!! note
    The user is responsible for calling `destroy(vec)` on the `VecMPI` since
    this cannot be handled by the garbage collector do to the MPI nature of the
    object.
"""
mutable struct VecMPI{T, PetscLib <: PetscLibType{T}} <:
               AbstractVec{T, PetscLib}
    ptr::CVec
    comm::MPI.Comm
    array::Union{Nothing, Vector{T}}
end
@for_libpetsc begin
    function VecMPI(
        ::$UnionPetscLib,
        comm,
        jl_v::Vector{$PetscScalar};
        blocksize = 1,
        global_length = -1,
    )
        global_length > 0 || (global_length = PETSC_DETERMINE)
        @assert Initialized($PetscLib)
        petsc_v = VecMPI{$PetscScalar, $PetscLib}(C_NULL, comm, jl_v)
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
            length(jl_v),
            global_length,
            jl_v,
            petsc_v,
        )
        return petsc_v
    end

    function VecMPI(::$UnionPetscLib, comm, local_length, global_length = -1)
        global_length > 0 || (global_length = PETSC_DETERMINE)
        local_length > 0 || (local_length = PETSC_DECIDE)

        @assert (global_length > 0) || (local_length > 0)

        @assert Initialized($PetscLib)

        petsc_v = VecMPI{$PetscScalar, $PetscLib}(C_NULL, comm, nothing)
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

"""
    getvalues!(
        vals::Array{PetscScalar},
        v::AbstractVec{PetscScalar, PetscLib},
        indices::Vector{PetscInt},
        insertmode::InsertMode,
    )

Get the 0-based global `indices` of `vec` into the preallocated array `vals`.

!!! warning
    This function uses 0-based indexing!

Manual: [`VecGetValues`](https://petsc.org/release/docs/manualpages/Vec/VecGetValues.html)
"""
getvalues(::AbstractVec)

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
=#

# Functions for AbstractVec types
@for_libpetsc begin
    function destroy(petsc_v::AbstractVec{$PetscScalar, $PetscLib})
        Finalized($PetscLib) || @chk ccall(
            (:VecDestroy, $petsc_library),
            PetscErrorCode,
            (Ptr{CVec},),
            petsc_v,
        )
        return nothing
    end

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

    #=
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
    =#

    function assemblybegin(V::AbstractVec{$PetscScalar})
        @chk ccall(
            (:VecAssemblyBegin, $petsc_library),
            PetscErrorCode,
            (CVec,),
            V,
        )
        return nothing
    end
    function assemblyend(V::AbstractVec{$PetscScalar})
        @chk ccall(
            (:VecAssemblyEnd, $petsc_library),
            PetscErrorCode,
            (CVec,),
            V,
        )
        return nothing
    end

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
    =#
end
Base.show(io::IO, v::AbstractVec) = _show(io, v)

#=
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

function Base.show(io::IO, ::MIME"text/plain", vec::AbstractVec)
    _show(io, vec)
end

VecSeq(X::Vector{T}; kwargs...) where {T} = VecSeq(MPI.COMM_SELF, X; kwargs...)
AbstractVec(X::AbstractVector) = VecSeq(X)

"""
    ownership_range(vec::AbstractVec)

The range of indices owned by this processor, assuming that the vectors are laid out with the first n1 elements on the first processor, next n2 elements on the second, etc. For certain parallel layouts this range may not be well defined.

Note: unlike the C function, the range returned is inclusive (`idx_first:idx_last`)

https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Vec/VecGetOwnershipRange.html
"""
ownershiprange
=#
