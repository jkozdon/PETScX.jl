using Libdl

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

function getlibs()
    libs = ()
    petsc_libs = ENV["JULIA_PETSC_LIBRARY"]

    flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL

    for petsc_lib in Base.parse_load_path(petsc_libs)
        libs = (libs..., (petsc_lib, flags))
    end
    return libs
end

# Find the PETSc libraries to use
const libs = @static if !haskey(ENV, "JULIA_PETSC_LIBRARY")
    using PETSc_jll
    ((PETSc_jll.libpetsc,),)
else
    getlibs()
end

function PetscInitialize(libhdl::Ptr{Cvoid})
    PetscInitializeNoArguments_ptr = dlsym(libhdl, :PetscInitializeNoArguments)
    @chk ccall(PetscInitializeNoArguments_ptr, PetscErrorCode, ())
end

function DataTypeFromString(libhdl::Ptr{Cvoid}, name::AbstractString)
    PetscDataTypeFromString_ptr = dlsym(libhdl, :PetscDataTypeFromString)
    dtype_ref = Ref{PetscDataType}()
    found_ref = Ref{PetscBool}()
    @chk ccall(
        PetscDataTypeFromString_ptr,
        PetscErrorCode,
        (Cstring, Ptr{PetscDataType}, Ptr{PetscBool}),
        name,
        dtype_ref,
        found_ref,
    )
    @assert found_ref[] == PETSC_TRUE
    return dtype_ref[]
end
function PetscDataTypeGetSize(libhdl::Ptr{Cvoid}, dtype::PetscDataType)
    PetscDataTypeGetSize_ptr = dlsym(libhdl, :PetscDataTypeGetSize)
    datasize_ref = Ref{Csize_t}()
    @chk ccall(
        PetscDataTypeGetSize_ptr,
        PetscErrorCode,
        (PetscDataType, Ptr{Csize_t}),
        dtype,
        datasize_ref,
    )
    return datasize_ref[]
end

const petsclibs = map(libs) do lib
    libhdl = dlopen(lib...)
    PetscInitialize(libhdl)
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

const scalar_types = map(x -> scalartype(x), petsclibs)
@assert length(scalar_types) == length(unique(scalar_types))

macro for_libpetsc(expr)
    quote
        for petsclib in petsclibs
            libpetsc = petsclib.libpetsc
            PetscScalar = scalartype(petsclib)
            PetscReal = realtype(petsclib)
            PetscInt = inttype(petsclib)
            PetscLib = typeof(petsclib)
            @eval esc($expr)
        end
    end
end

@for_libpetsc begin
    # TODO: Remove this after full change over to PetscLibType
    inttype(::Type{$PetscScalar}) = $PetscInt
end
