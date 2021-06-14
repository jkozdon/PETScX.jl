"""
   Initialize(petsclib)

Initialize the `petsclib`.

Manual: [`PetscInitializeNoArguments`](https://petsc.org/release/docs/manualpages/Sys/PetscInitializeNoArguments.html)
"""
function Initialized end

"""
   Initialized(petsclib)

Check if `petsclib` is initialized

Manual: [`PetscInitialized`](https://petsc.org/release/docs/manualpages/Sys/PetscInitialized.html)
"""
function Initialize end

"""
   Finalize(petsclib)

Check if `petsclib` is initialized

Manual: [`PetscFinalize`](https://petsc.org/release/docs/manualpages/Sys/PetscFinalize.html)
"""
function Finalize end

"""
   Finalized(petsclib)

Check if `petsclib` is finalized

Manual: [`PetscFinalized`](https://petsc.org/release/docs/manualpages/Sys/PetscFinalized.html)
"""
function Finalized end

@for_libpetsc begin
    function Initialize(::$UnionPetscLib)
        if !Initialized($petsclib)
            MPI.Initialized() || MPI.Init()
            @chk ccall(
                (:PetscInitializeNoArguments, $petsc_library),
                PetscErrorCode,
                (),
            )

            # disable signal handler
            @chk ccall(
                (:PetscPopSignalHandler, $petsc_library),
                PetscErrorCode,
                (),
            )

            atexit(() -> Finalize($PetscLib))
        end
        return nothing
    end

    function Initialized(::$UnionPetscLib)
        r_flag = Ref{PetscBool}()
        @chk ccall(
            (:PetscInitialized, $petsc_library),
            PetscErrorCode,
            (Ptr{PetscBool},),
            r_flag,
        )
        return r_flag[] == PETSC_TRUE
    end

    function Finalized(::$UnionPetscLib)
        r_flag = Ref{PetscBool}()
        @chk ccall(
            (:PetscFinalized, $petsc_library),
            PetscErrorCode,
            (Ptr{PetscBool},),
            r_flag,
        )
        return r_flag[] == PETSC_TRUE
    end

    function Finalize(::$UnionPetscLib)
        if !Finalized($petsclib)
            @chk ccall((:PetscFinalize, $petsc_library), PetscErrorCode, ())
        end
        return nothing
    end
end
