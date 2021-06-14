"""
    MatShell{T}(obj, m, n)

Create a `m×n` PETSc shell matrix object wrapping `obj`.

If `obj` is a `Function`, then the multiply action `obj(y,x)`; otherwise it calls `mul!(y, obj, x)`.
This can be changed by defining `PETSc._mul!`.

"""
mutable struct MatShell{T, PetscLib, A} <: AbstractMat{T, PetscLib}
    ptr::CMat
    comm::MPI.Comm
    obj::A
end

struct MatOp{T, Op} end

function _mul!(
    y,
    mat::MatShell{T, PetscLib, F},
    x,
) where {T, PetscLib, F <: Function}
    mat.obj(y, x)
end

function _mul!(y, mat::MatShell{T, PetscLib, F}, x) where {T, PetscLib, F}
    LinearAlgebra.mul!(y, mat.obj, x)
end

MatShell{T}(obj, m, n) where {T} = MatShell{T}(obj, MPI.COMM_SELF, m, n, m, n)

@for_libpetsc begin
    function MatShell{$PetscScalar}(
        obj::A,
        comm::MPI.Comm,
        m::$PetscInt,
        n::$PetscInt,
        M::$PetscInt,
        N::$PetscInt,
    ) where {A}
        mat = MatShell{$PetscScalar, $PetscLib, A}(C_NULL, comm, obj)
        # we use the MatShell object itsel
        ctx = pointer_from_objref(mat)
        @chk ccall(
            (:MatCreateShell, $libpetsc),
            PetscErrorCode,
            (
                MPI.MPI_Comm,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                $PetscInt,
                Ptr{Cvoid},
                Ptr{CMat},
            ),
            comm,
            m,
            n,
            M,
            N,
            ctx,
            mat,
        )

        mulptr = @cfunction(
            MatOp{$PetscScalar, MATOP_MULT}(),
            $PetscInt,
            (CMat, CVec, CVec)
        )
        @chk ccall(
            (:MatShellSetOperation, $libpetsc),
            PetscErrorCode,
            (CMat, MatOperation, Ptr{Cvoid}),
            mat,
            MATOP_MULT,
            mulptr,
        )
        return mat
    end

    function (::MatOp{$PetscScalar, MATOP_MULT})(
        M::CMat,
        cx::CVec,
        cy::CVec,
    )::$PetscInt
        r_ctx = Ref{Ptr{Cvoid}}()
        @chk ccall(
            (:MatShellGetContext, $libpetsc),
            PetscErrorCode,
            (CMat, Ptr{Ptr{Cvoid}}),
            M,
            r_ctx,
        )
        ptr = r_ctx[]
        mat = unsafe_pointer_to_objref(ptr)

        x = unsafe_localarray($PetscScalar, cx; write = false)
        y = unsafe_localarray($PetscScalar, cy; read = false)

        _mul!(y, mat, x)

        Base.finalize(y)
        Base.finalize(x)
        return $PetscInt(0)
    end
end
