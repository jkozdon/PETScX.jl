const CDM = Ptr{Cvoid}

abstract type AbstractDM{T} end
mutable struct _DM{T} <: AbstractDM{T}
    ptr::CDM
end
mutable struct DM{T} <: AbstractDM{T}
    ptr::CDM
    _comm::MPI.Comm
    opts::Options{T}
end
# allows us to pass XXDM objects directly into CDM ccall signatures
Base.cconvert(::Type{CDM}, obj::AbstractDM) = obj.ptr
# allows us to pass XXDM objects directly into Ptr{CDM} ccall signatures
function Base.unsafe_convert(::Type{Ptr{CDM}}, obj::AbstractDM)
    return convert(Ptr{CDM}, pointer_from_objref(obj))
end

"""
    DMSetUp!(da::DM)

see [PETSc manual](https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/DM/DMSetUp.html)
"""
function DMSetUp! end

@for_libpetsc begin
    function destroy(da::DM{$PetscScalar})
        finalized($PetscScalar) ||
            @chk ccall((:DMDestroy, $libpetsc), PetscErrorCode, (Ptr{CDM},), da)
        da.ptr = C_NULL
        return nothing
    end

    function DMSetUp!(da::DM{$PetscScalar})
        with(da.opts) do
            @chk ccall(
                (:DMSetFromOptions, $libpetsc),
                PetscErrorCode,
                (CDM,),
                da,
            )

            @chk ccall((:DMSetUp, $libpetsc), PetscErrorCode, (CDM,), da)
        end
    end
end
