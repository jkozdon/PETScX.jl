const CPetscObject = Ptr{Cvoid}

@for_libpetsc begin
    function PetscObjectGetComm(
        obj::Union{
            AbstractKSP{$PetscScalar},
            AbstractMat{$PetscScalar},
            AbstractVec{$PetscScalar},
            AbstractDM{$PetscScalar},
        },
    )
        comm = MPI.Comm()
        @chk ccall(
            (:PetscObjectGetComm, $libpetsc),
            PetscErrorCode,
            (CPetscObject, Ptr{MPI.MPI_Comm}),
            obj,
            comm,
        )
        return comm
    end
end
