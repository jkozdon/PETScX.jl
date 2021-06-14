const CMat = Ptr{Cvoid}
const CMatNullSpace = Ptr{Cvoid}

abstract type AbstractMat{T, PetscLib <: PetscLibType{T}} <: AbstractMatrix{T} end
scalartype(::AbstractMat{T}) where {T} = T

# allows us to pass XXMat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMat}, obj::AbstractMat) = obj.ptr
# allows us to pass XXMat objects directly into Ptr{CMat} ccall signatures
Base.unsafe_convert(::Type{Ptr{CMat}}, obj::AbstractMat) =
    convert(Ptr{CMat}, pointer_from_objref(obj))

Base.eltype(::Type{A}) where {A <: AbstractMat{T}} where {T} = T
Base.eltype(A::AbstractMat{T}) where {T} = T

"""
    MatSeqAIJ{T, PetscLib}

PETSc sparse array using AIJ format (also known as a compressed sparse row or
CSR format).

Memory allocation is handled by PETSc.
"""
mutable struct MatSeqAIJ{T, PetscLib <: PetscLibType{T}} <:
               AbstractMat{T, PetscLib}
    ptr::CMat
    _comm::MPI.Comm
end

"""
    Mat{T, PetscLib}

Container for an abstract PETSc matrix

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/Mat.html)
"""
struct Mat{T, PetscLib <: PetscLibType{T}} <: AbstractMat{T, PetscLib}
    ptr::CMat
end

"""
    MatSeqDense{T, PetscLib}

PETSc dense array. This wraps a Julia `Matrix{T}` object.
"""
mutable struct MatSeqDense{T, PetscLib} <: AbstractMat{T, PetscLib}
    ptr::CMat
    _comm::MPI.Comm
    array::Matrix{T}
end

"""
    MatStencil{PetscInt}

Equivalent to the `MatStencil` in PETSc

See [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatStencil.html)
"""
struct MatStencil{PetscInt}
    "third grid index"
    k::PetscInt
    "second grid index"
    j::PetscInt
    "first grid index"
    i::PetscInt
    "degree of freedom"
    c::PetscInt
    function MatStencil{PetscInt}(; i, j = 1, k = 1, c = 1) where {PetscInt}
        # convert to zero-based indexing
        new{PetscInt}(k - 1, j - 1, i - 1, c - 1)
    end
end
# Since julia uses 1-based indexing we need to convert on access
function Base.getproperty(
    obj::MatStencil{PetscInt},
    sym::Symbol,
) where {PetscInt}
    if sym in (:i, :j, :k, :c)
        return getfield(obj, sym) + PetscInt(1)
    else # fallback to getfield
        return getfield(obj, sym)
    end
end
# Since julia uses 1-based indexing we need to convert on show
function Base.show(io::IO, m::MatStencil{PetscInt}) where {PetscInt}
    print(io, typeof(m))
    print(io, "(i = ", m.i)
    print(io, ", j = ", m.j)
    print(io, ", k = ", m.k)
    print(io, ", c = ", m.c, ")")
end
function Base.show(
    io::IO,
    ::MIME"text/plain",
    m::MatStencil{PetscInt},
) where {PetscInt}
    print(io, "(i = ", m.i)
    print(io, ", j = ", m.j)
    print(io, ", k = ", m.k)
    print(io, ", c = ", m.c, ")")
end

"""
    MatNullSpace{T}

Object that removes a null space from a vector, i.e. orthogonalizes the vector
to a subspace;
see [MatNullSpace](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpace.html)
and [MatNullSpaceCreate](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpaceCreate.html)

!!! Note
    The caller is responsible for calling `destroy` on this object
"""
mutable struct MatNullSpace{T}
    ptr::CMatNullSpace
    _comm::MPI.Comm
end
# allows us to pass XXMat objects directly into CMat ccall signatures
Base.cconvert(::Type{CMatNullSpace}, obj::MatNullSpace) = obj.ptr
# allows us to pass XXMat objects directly into Ptr{CMat} ccall signatures
Base.unsafe_convert(::Type{Ptr{CMatNullSpace}}, obj::MatNullSpace) =
    convert(Ptr{CMatNullSpace}, pointer_from_objref(obj))

"""
    MatNullSpaceRemove!(nullspace, vec)

Removes all the components of a `nullspace` from `vec`

see [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatNullSpaceRemove.html)
"""
function MatNullSpaceRemove! end

"""
    MatSetNullSpace!(mat, nullspace)

Attach `nullspace` to `mat`

see [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetNullSpace.html)
"""
function MatSetNullSpace! end

"""
    MatSetValuesStencil!(mat::AbstractMat{PetscScalar},
        rows::Vector{MatStencil{PetscInt}}, 
        cols::Vector{MatStencil{PetscInt}}, 
        vals::Vector{PetscScalar},
        mode;
        num_col = length(col),
        num_row = length(row)
      )

Insert the `vals` specified by `rows` and `cols` stencil indices into the `mat`.
The optional arguments `num_col` and `num_row` allow the limiting of the
elements of the `rows` and `cols` vectors.

see [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatSetValuesStencil.html)
"""
function MatSetValuesStencil! end

@for_libpetsc begin
    function MatNullSpace{$PetscScalar}(
        comm::MPI.Comm,
        has_constant,
        n = 0,
        vecs = nothing,
    )
        initialize($PetscScalar)
        @assert n == 0 && isnothing(vecs)
        nullspace = MatNullSpace{$PetscScalar}(C_NULL, comm)
        @chk ccall(
            (:MatNullSpaceCreate, $libpetsc),
            PetscErrorCode,
            (MPI.MPI_Comm, PetscBool, $PetscInt, Ptr{CVec}, Ptr{CMatNullSpace}),
            comm,
            has_constant,
            n,
            C_NULL,
            nullspace,
        )
        return nullspace
    end

    function MatNullSpaceRemove!(
        nullspace::MatNullSpace{$PetscScalar},
        vec::AbstractVec{$PetscScalar},
    )
        @chk ccall(
            (:MatNullSpaceRemove, $libpetsc),
            PetscErrorCode,
            (CMatNullSpace, CVec),
            nullspace,
            vec,
        )
        return nothing
    end

    function MatSetNullSpace!(
        mat::Mat{$PetscScalar},
        nullspace::MatNullSpace{$PetscScalar},
    )
        @chk ccall(
            (:MatSetNullSpace, $libpetsc),
            PetscErrorCode,
            (CMat, CMatNullSpace),
            mat,
            nullspace,
        )
        return nothing
    end

    function destroy(nullspace::MatNullSpace{$PetscScalar})
        finalized($PetscScalar) || @chk ccall(
            (:MatNullSpaceDestroy, $libpetsc),
            PetscErrorCode,
            (Ptr{CMatNullSpace},),
            nullspace,
        )
        return nothing
    end

    function MatSeqAIJ(
        ::$UnionPetscLib,
        m::Integer,
        n::Integer,
        nnz::Vector{$PetscInt},
    )
        initialize($PetscScalar)
        comm = MPI.COMM_SELF
        mat = MatSeqAIJ{$PetscScalar, $PetscLib}(C_NULL, comm)
        @chk ccall(
            (:MatCreateSeqAIJ, $libpetsc),
            PetscErrorCode,
            (
                MPI.MPI_Comm,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                Ptr{$PetscInt},
                Ptr{CMat},
            ),
            comm,
            m,
            n,
            0,
            nnz,
            mat,
        )
        finalizer(destroy, mat)
        return mat
    end
    function MatSeqDense(::$UnionPetscLib, A::Matrix{$PetscScalar})
        initialize($PetscScalar)
        comm = MPI.COMM_SELF
        mat = MatSeqDense{$PetscScalar, $PetscLib}(C_NULL, comm, A)
        @chk ccall(
            (:MatCreateSeqDense, $libpetsc),
            PetscErrorCode,
            (MPI.MPI_Comm, $PetscInt, $PetscInt, Ptr{$PetscScalar}, Ptr{CMat}),
            comm,
            size(A, 1),
            size(A, 2),
            A,
            mat,
        )
        finalizer(destroy, mat)
        return mat
    end

    function MatSetValuesStencil!(
        mat::AbstractMat{$PetscScalar},
        rows::Vector{MatStencil{$PetscInt}},
        cols::Vector{MatStencil{$PetscInt}},
        vals::Vector{$PetscScalar},
        mode::InsertMode;
        num_rows = length(rows),
        num_cols = length(cols),
    )
        @assert length(vals) >= num_cols * num_rows
        @assert length(cols) >= num_cols
        @assert length(rows) >= num_rows
        @chk ccall(
            (:MatSetValuesStencil, $libpetsc),
            PetscErrorCode,
            (
                CMat,
                $PetscInt,
                Ptr{MatStencil{$PetscInt}},
                $PetscInt,
                Ptr{MatStencil{$PetscInt}},
                Ptr{$PetscScalar},
                InsertMode,
            ),
            mat,
            num_rows,
            rows,
            num_cols,
            cols,
            vals,
            mode,
        )
        return nothing
    end

    function destroy(M::AbstractMat{$PetscScalar})
        finalized($PetscScalar) || @chk ccall(
            (:MatDestroy, $libpetsc),
            PetscErrorCode,
            (Ptr{CMat},),
            M,
        )
        return nothing
    end

    function setvalues!(
        M::AbstractMat{$PetscScalar},
        row0idxs::Vector{$PetscInt},
        col0idxs::Vector{$PetscInt},
        rowvals::Array{$PetscScalar},
        insertmode::InsertMode,
    )
        @chk ccall(
            (:MatSetValues, $libpetsc),
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
            M,
            length(row0idxs),
            row0idxs,
            length(col0idxs),
            col0idxs,
            rowvals,
            insertmode,
        )
        return nothing
    end

    function Base.setindex!(
        M::AbstractMat{$PetscScalar},
        val,
        i::Integer,
        j::Integer,
    )
        @chk ccall(
            (:MatSetValues, $libpetsc),
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
            M,
            1,
            Ref{$PetscInt}(i - 1),
            1,
            Ref{$PetscInt}(j - 1),
            Ref{$PetscScalar}(val),
            INSERT_VALUES,
        )
        return val
    end

    function assemblybegin(
        M::AbstractMat{$PetscScalar},
        t::MatAssemblyType = MAT_FINAL_ASSEMBLY,
    )
        @chk ccall(
            (:MatAssemblyBegin, $libpetsc),
            PetscErrorCode,
            (CMat, MatAssemblyType),
            M,
            t,
        )
        return nothing
    end
    function assemblyend(
        M::AbstractMat{$PetscScalar},
        t::MatAssemblyType = MAT_FINAL_ASSEMBLY,
    )
        @chk ccall(
            (:MatAssemblyEnd, $libpetsc),
            PetscErrorCode,
            (CMat, MatAssemblyType),
            M,
            t,
        )
        return nothing
    end
    function view(
        mat::AbstractMat{$PetscScalar},
        viewer::Viewer{$PetscScalar} = ViewerStdout{$PetscScalar}(
            PetscObjectGetComm(mat),
        ),
    )
        @chk ccall(
            (:MatView, $libpetsc),
            PetscErrorCode,
            (CMat, CPetscViewer),
            mat,
            viewer,
        )
        return nothing
    end

    function Base.size(A::AbstractMat{$PetscScalar})
        m = Ref{$PetscInt}()
        n = Ref{$PetscInt}()
        @chk ccall(
            (:MatGetSize, $libpetsc),
            PetscErrorCode,
            (CMat, Ptr{$PetscInt}, Ptr{$PetscInt}),
            A,
            m,
            n,
        )
        return (m[], n[])
    end
    function Base.:(==)(
        A::AbstractMat{$PetscScalar},
        B::AbstractMat{$PetscScalar},
    )
        fr = Ref{PetscBool}()
        @chk ccall(
            (:MatEqual, $libpetsc),
            PetscErrorCode,
            (CMat, CMat, Ptr{PetscBool}),
            A,
            B,
            fr,
        )
        return fr[] == PETSC_TRUE
    end

    function LinearAlgebra.issymmetric(
        A::AbstractMat{$PetscScalar};
        tol = zero($PetscReal),
    )
        fr = Ref{PetscBool}()
        @chk ccall(
            (:MatIsSymmetric, $libpetsc),
            PetscErrorCode,
            (CMat, $PetscReal, Ptr{PetscBool}),
            A,
            tol,
            fr,
        )
        return fr[] == PETSC_TRUE
    end
    function LinearAlgebra.ishermitian(
        A::AbstractMat{$PetscScalar};
        tol = zero($PetscReal),
    )
        fr = Ref{PetscBool}()
        @chk ccall(
            (:MatIsHermitian, $libpetsc),
            PetscErrorCode,
            (CMat, $PetscReal, Ptr{PetscBool}),
            A,
            tol,
            fr,
        )
        return fr[] == PETSC_TRUE
    end
    function LinearAlgebra.norm(
        M::AbstractMat{$PetscScalar},
        normtype::NormType = NORM_FROBENIUS,
    )
        r_val = Ref{$PetscReal}()
        @chk ccall(
            (:MatNorm, $libpetsc),
            PetscErrorCode,
            (CMat, NormType, Ptr{$PetscReal}),
            M,
            normtype,
            r_val,
        )
        return r_val[]
    end

    function LinearAlgebra.mul!(
        y::AbstractVec{$PetscScalar},
        M::AbstractMat{$PetscScalar},
        x::AbstractVec{$PetscScalar},
    )
        @chk ccall(
            (:MatMult, $libpetsc),
            PetscErrorCode,
            (CMat, CVec, CVec),
            M,
            x,
            y,
        )
        return y
    end
    function LinearAlgebra.mul!(
        y::AbstractVec{$PetscScalar},
        M::Adjoint{T, A},
        x::AbstractVec{$PetscScalar},
    ) where {T, A <: AbstractMat{$PetscScalar}}
        @chk ccall(
            (:MatMultHermitianTranspose, $libpetsc),
            PetscErrorCode,
            (CMat, CVec, CVec),
            parent(M),
            x,
            y,
        )
        return y
    end
    function LinearAlgebra.mul!(
        y::AbstractVec{$PetscScalar},
        M::Transpose{T, A},
        x::AbstractVec{$PetscScalar},
    ) where {T, A <: AbstractMat{$PetscScalar}}
        @chk ccall(
            (:MatMultTranspose, $libpetsc),
            PetscErrorCode,
            (CMat, CVec, CVec),
            parent(M),
            x,
            y,
        )
        return y
    end
end

function assemble(M::AbstractMat, t::MatAssemblyType = MAT_FINAL_ASSEMBLY)
    assemblybegin(M, t)
    assemblyend(M, t)
end

function MatSeqAIJ(::PT, S::SparseMatrixCSC{ST}) where {PT, ST}
    PetscInt = inttype(PT)
    m, n = size(S)
    nnz = zeros(PetscInt, m)
    for r in S.rowval
        nnz[r] += 1
    end
    M = MatSeqAIJ(PT, m, n, nnz)
    for j in 1:n
        for ii in S.colptr[j]:(S.colptr[j + 1] - 1)
            i = S.rowval[ii]
            M[i, j] = S.nzval[ii]
        end
    end
    assemble(M)
    return M
end

function Base.copyto!(M::PETSc.MatSeqAIJ{T}, S::SparseMatrixCSC{T}) where {T}
    for j in 1:size(S, 2)
        for ii in S.colptr[j]:(S.colptr[j + 1] - 1)
            i = S.rowval[ii]
            M[i, j] = S.nzval[ii]
        end
    end
    assemble(M)
end

function Base.show(io::IO, ::MIME"text/plain", mat::AbstractMat)
    _show(io, mat)
end

const MatAT{T} = Union{
    AbstractMat{T},
    Transpose{T, <:AbstractMat{T}},
    Adjoint{T, <:AbstractMat{T}},
}

LinearAlgebra.mul!(
    y::AbstractVector{T},
    M::MatAT{T},
    x::AbstractVector{T},
) where {T} = parent(LinearAlgebra.mul!(AbstractVec(y), M, AbstractVec(x)))
