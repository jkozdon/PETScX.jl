
"""
    PetscLibType{PetscScalar, PetscInt}(libpetsc)

A container for specific PETSc libraries.

All other containers for PETSc objects should be typed on this to ensure that
dispatch is correct.
"""
struct PetscLibType{PetscScalar, PetscInt, LibType}
    libpetsc::LibType
end
function PetscLibType{ST, IT}(libpetsc) where {ST, IT}
    LT = typeof(libpetsc)
    return PetscLibType{ST, IT, LT}(libpetsc)
end

"""
    scalartype(petsclib::PetscLibType)

return the scalar type for the associated `petsclib`
"""
scalartype(::PetscLibType{ST}) where {ST} = ST
scalartype(::Type{PetscLib}) where {PetscLib <: PetscLibType{ST}} where {ST} =
    ST

"""
    realtype(petsclib::PetscLibType)

return the real type for the associated `petsclib`
"""
realtype(::PetscLibType{ST}) where {ST} = real(ST)
realtype(::Type{PetscLib}) where {PetscLib <: PetscLibType{ST}} where {ST} =
    real(ST)

"""
    inttype(petsclib::PetscLibType)

return the int type for the associated `petsclib`
"""
inttype(::PetscLibType{ST, IT}) where {ST, IT} = IT
inttype(
    ::Type{PetscLib},
) where {PetscLib <: PetscLibType{ST, IT}} where {ST, IT} = IT

"""
    getpetsclib([PetscScalar = Float64, PetscInt = Int64])

return the [`PETScLibType`](@ref) for the associated parameters.
"""
function getpetsclib(PetscScalar = Float64, PetscInt = Int64)
    try
        _getpetsclib(PetscScalar, PetscInt)
    catch
        error(
            "No PETSc library loaded for types " *
            "(PetscScalar, PetscInt) = ($PetscScalar, $PetscInt)",
        )
    end
end

"""
    petsclibs

A `Tuple` of the available PETSc library objects; see [`PetscLibType`](@ref)
"""
const petsclibs = map(libs) do lib
    libhdl = dlopen(lib...)

    # initialize petsc
    PetscInitializeNoArguments_ptr =
        dlsym(libhdl, :PetscInitializeNoArguments)
    @chk ccall(PetscInitializeNoArguments_ptr, PetscErrorCode, ())

    PETSC_REAL = DataTypeFromString(libhdl, "Real")
    PETSC_SCALAR = DataTypeFromString(libhdl, "Scalar")
    PETSC_INT_SIZE = PetscDataTypeGetSize(libhdl, PETSC_INT)

    PetscReal =
        PETSC_REAL == PETSC_DOUBLE ? Cdouble :
        PETSC_REAL == PETSC_FLOAT ? Cfloat :
        error("PETSC_REAL = $PETSC_REAL not supported.")

    PetscScalar =
        PETSC_SCALAR == PETSC_REAL ? PetscReal :
        PETSC_SCALAR == PETSC_COMPLEX ? Complex{PetscReal} :
        error("PETSC_SCALAR = $PETSC_SCALAR not supported.")

    PetscInt =
        PETSC_INT_SIZE == 4 ? Int32 :
        PETSC_INT_SIZE == 8 ? Int64 :
        error("PETSC_INT_SIZE = $PETSC_INT_SIZE not supported.")

    # TODO: PetscBLASInt, PetscMPIInt ?
    return PetscLibType{PetscScalar, PetscInt}(lib[1])
end

"""
    @for_libpetsc

Macro for looping over the available petsc libraries.
"""
macro for_libpetsc(expr)
    quote
        for petsclib in petsclibs
            libpetsc = petsclib.libpetsc
            PetscScalar = scalartype(petsclib)
            PetscReal = realtype(petsclib)
            PetscInt = inttype(petsclib)
            PetscLib = typeof(petsclib)
            UnionPetscLib = Union{PetscLib, Type{PetscLib}}
            @eval esc($expr)
        end
    end
end

@for_libpetsc begin
    _getpetsclib(::Type{$PetscScalar}, ::Type{$PetscInt}) = $petsclib
end
