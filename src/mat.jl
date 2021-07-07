const CMat = Ptr{Cvoid}
const CMatType = Cstring

"""
    AbstractMat{PetscLib <: PetscLibType}

Abstract type of PETSc vectors.

Manual: [`Mat`](https://petsc.org/release/docs/manualpages/Mat/Mat.html)
"""
abstract type AbstractMat{T, PetscLib <: PetscLibType{T}} <: AbstractMatrix{T} end
Base.eltype(::Type{V}) where {V <: AbstractMat{T}} where {T} = T
Base.eltype(v::AbstractMat{T}) where {T} = T

# allows us to pass XXMat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMat}, obj::AbstractMat) = obj.__ptr__

# allows us to pass XXMat objects directly into Ptr{CMat} ccall signatures
Base.unsafe_convert(::Type{Ptr{CMat}}, obj::AbstractMat) =
    convert(Ptr{CMat}, pointer_from_objref(obj))

mutable struct MatAIJ{
    T,
    PetscLib <: PetscLibType{T},
    CT <: Union{MPI.Comm, Nothing},
} <: AbstractMat{T, PetscLib}
    __ptr__::CMat
    __possible_comm_ref::CT
    function MatAIJ{T, PetscLib}(; comm::CT = nothing) where {PetscLib, T, CT}
        new{T, PetscLib, CT}(C_NULL, comm)
    end
end

@for_libpetsc function view(
    mat::AbstractMat{$PetscScalar, $PetscLib},
    viewer::AbstractViewer{$PetscLib} = ViewerStdout($PetscLib),
)
    @chk ccall(
        (:MatView, $petsc_library),
        PetscErrorCode,
        (CMat, CPetscViewer),
        mat,
        viewer,
    )
    return nothing
end
Base.show(io::IO, mat::AbstractMat) = _show(io, mat)

"""
    destroy(A::AbstractMat)

Destroy the PETSc matrix `A`

Manual: [`MatDestroy`](https://petsc.org/release/docs/manualpages/Mat/MatDestroy.html)
"""
destroy(::AbstractMat)

@for_libpetsc function destroy(petsc_A::AbstractMat{$PetscScalar, $PetscLib})
    Finalized($PetscLib) || @chk ccall(
        (:MatDestroy, $petsc_library),
        PetscErrorCode,
        (Ptr{CMat},),
        petsc_A,
    )

    # Zero the pointer so that if MatDestroy gets called multiple times we
    # do not artificially decrease the internal petsc ref counter
    petsc_A.__ptr__ = C_NULL
    return nothing
end

"""
    MatAIJ(petsclib, m::Int, n::Int, nz::Union{Int, Vector{PetscInt}}

A sequentially-stored `m × n` serial PETSc sparse matrix in compressed row
storage with `nz` zeros per row; if `nz` is a vector then the `nz[i]` is the
number of non-zeros for row `i`.

Manual: [`MatCreateSeqAIJ`](https://petsc.org/release/docs/manualpages/Mat/MatCreateSeqAIJ.html)

!!! note
    Since the `MatAIJ` is inherently serial `destroy` will be called by the
    garbage collector
"""
MatAIJ(_, ::Int, ::Int, ::Union{Int, Vector{Int}})

@for_libpetsc function MatAIJ(
    ::$UnionPetscLib,
    num_rows::Int,
    num_cols::Int,
    nz_per_row::Union{Int, Vector{$PetscInt}},
)
    @assert Initialized($PetscLib)
    petsc_A = MatAIJ{$PetscScalar, $PetscLib}()

    if nz_per_row isa Int
        array_nz_per_row = C_NULL
    else
        array_nz_per_row = nz_per_row
        @assert length(array_nz_per_row) >= num_rows
        nz_per_row = 0
    end

    @chk ccall(
        (:MatCreateSeqAIJ, $petsc_library),
        PetscErrorCode,
        (
            MPI.MPI_Comm,
            $PetscInt,
            $PetscInt,
            $PetscInt,
            Ptr{$PetscInt},
            Ptr{CMat},
        ),
        MPI.COMM_SELF,
        num_rows,
        num_cols,
        nz_per_row,
        array_nz_per_row,
        petsc_A,
    )

    finalizer(destroy, petsc_A)
    return petsc_A
end

"""
    assemblybegin!(mat::AbstractMat)

Begin assembling `mat`

Manual: [`MatAssemblyBegin`](https://petsc.org/release/docs/manualpages/Mat/MatAssemblyBegin.html)
"""
assemblybegin!(::AbstractMat)

"""
    assemblyend!(mat::AbstractMat)

Finish assembling `mat`

Manual: [`MatAssemblyEnd`](https://petsc.org/release/docs/manualpages/Mat/MatAssemblyEnd.html)
"""
assemblyend!(::AbstractMat)

@for_libpetsc function assemblybegin!(
    mat::AbstractMat{$PetscScalar, $PetscLib},
    type = MAT_FINAL_ASSEMBLY,
)
    @assert Initialized($PetscLib)
    @chk ccall(
        (:MatAssemblyBegin, $petsc_library),
        PetscErrorCode,
        (CMat, MatAssemblyType),
        mat,
        type,
    )

    return mat
end

@for_libpetsc function assemblyend!(
    mat::AbstractMat{$PetscScalar, $PetscLib},
    type = MAT_FINAL_ASSEMBLY,
)
    @assert Initialized($PetscLib)
    @chk ccall(
        (:MatAssemblyEnd, $petsc_library),
        PetscErrorCode,
        (CMat, MatAssemblyType),
        mat,
        type,
    )

    return mat
end

@for_libpetsc function Base.size(mat::AbstractMat{$PetscScalar, $PetscLib})
    r_m = Ref{$PetscInt}()
    r_n = Ref{$PetscInt}()
    @chk ccall(
        (:MatGetSize, $petsc_library),
        PetscErrorCode,
        (CMat, Ptr{$PetscInt}, Ptr{$PetscInt}),
        mat,
        r_m,
        r_n,
    )
    return (r_m[], r_n[])
end

"""
    ownershiprange(mat::AbstractMat)

The range of global rows owned by this MPI rank, assuming that the matrix
is laid out with the first `n1` rows on the first processor, next `n2`
rows on the second, etc. For certain parallel layouts this range may not be
well defined.

Manual: [`MatGetOwnershipRange`](https://petsc.org/release/docs/manualpages/Mat/MatGetOwnershipRange.html)

!!! note
    The range returned is inclusive (`idx_first:idx_last`) with 0-based indexing
"""
ownershiprange(::AbstractMat)

@for_libpetsc function ownershiprange(vec::AbstractMat{$PetscScalar})
    r_lo = Ref{$PetscInt}()
    r_hi = Ref{$PetscInt}()
    @chk ccall(
        (:MatGetOwnershipRange, $petsc_library),
        PetscErrorCode,
        (CMat, Ptr{$PetscInt}, Ptr{$PetscInt}),
        vec,
        r_lo,
        r_hi,
    )
    return r_lo[]:(r_hi[] - $PetscInt(1))
end

"""
    setvalues!(
        mat::AbstractMat{PetscScalar, PetscLib},
        rows::Vector{PetscInt},
        cols::Vector{PetscInt},
        vals::Array{PetscScalar},
        insertmode::InsertMode,
    )

Assign the values `vals` in 0-based global `indices` of `vec`. The `insertmode`
can be `INSERT_VALUES` or `ADD_VALUES`.

!!! warning
    This function uses 0-based indexing!

Manual: [`MatSetValues`](https://petsc.org/release/docs/manualpages/Mat/MatSetValues.html)
"""
setvalues!(::AbstractMat)

@for_libpetsc function setvalues!(
    v::AbstractMat{$PetscScalar, $PetscLib},
    rows::Vector{$PetscInt},
    cols::Vector{$PetscInt},
    vals::Array{$PetscScalar},
    insertmode::InsertMode,
)
    @assert length(vals) >= length(rows) * length(cols)
    @chk ccall(
        (:MatSetValues, $petsc_library),
        PetscErrorCode,
        (
            CMat,
            $PetscInt,
            Ptr{$PetscInt},
            $PetscInt,
            Ptr{$PetscInt},
            Ptr{$PetscScalar},
            InsertMode,
        ),
        v,
        length(rows),
        rows,
        length(cols),
        cols,
        vals,
        insertmode,
    )
    return nothing
end

"""
    getvalues!(
        vals::Array{PetscScalar},
        mat::AbstractMat{PetscScalar, PetscLib},
        rows::Vector{PetscInt},
        cols::Vector{PetscInt},
    )

Get the 0-based global `rows` and `cols` of `mat` into the preallocated array
`vals`.

!!! warning
    This function uses 0-based indexing!

Manual: [`MatGetValues`](https://petsc.org/release/docs/manualpages/Mat/MatGetValues.html)
"""
getvalues!(::AbstractMat)

@for_libpetsc function getvalues!(
    vals::Array{$PetscScalar},
    v::AbstractMat{$PetscScalar, $PetscLib},
    rows::Vector{$PetscInt},
    cols::Vector{$PetscInt},
)
    @assert length(vals) >= length(rows) * length(cols)
    @chk ccall(
        (:MatGetValues, $petsc_library),
        PetscErrorCode,
        (
            CMat,
            $PetscInt,
            Ptr{$PetscInt},
            $PetscInt,
            Ptr{$PetscInt},
            Ptr{$PetscScalar},
        ),
        v,
        length(rows),
        rows,
        length(cols),
        cols,
        vals,
    )
    return vals
end
