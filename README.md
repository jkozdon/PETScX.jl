Repo is being archived and everything is being merged back into [`PETSc.jl`](https://github.com/JuliaParallel/PETSc.jl)

# PETScX

[![Build Status](https://github.com/jkozdon/PETScX.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/jkozdon/PETScX.jl/actions/workflows/ci.yml?query=branch%3Amain)
[![codecov.io](http://codecov.io/github/jkozdon/PETScX.jl/coverage.svg?branch=main)](http://codecov.io/github/jkozdon/PETScX.jl?branch=main)
[![bors enabled](https://bors.tech/images/badge_small.svg)](https://app.bors.tech/repositories/34540)

This package provides a low level interface for
[PETSc](https://www.mcs.anl.gov/petsc/)


## Installation

This package can be added with the julia command:
```julia
]add https://github.com/jkozdon/PETScX.jl
```
The installation can be tested with
```julia
]test PETScX
```

## BinaryBuilder Version

The package requires the uses a pre-build binary of
[`PETSc_jll`](https://github.com/JuliaBinaryWrappers/PETSc_jll.jl) along with a
default installation of `MPI.jl`; use of system install MPI and PETSc is not
currently supported. Not that the distributed version of PETSc is using real,
`Float64` numbers; build details can be found
[here](https://github.com/JuliaPackaging/Yggdrasil/blob/master/P/PETSc/build_tarballs.jl)

## System Builds

If you want to use the package with custom builds of the PETSc library, this can
be done by specifying the environment variable `JULIA_PETSC_LIBRARY`. This is a
colon separated list of paths to custom builds of PETSc; the reason for using
multiple builds is to enable single, double, and complex numbers in the same
julia session. These should be built against the same version of MPI as used
with `MPI.jl`

## Historical Notes

This package is an experimental(?) form of
[`PETSc.jl`](https://github.com/JuliaParallel/PETSc.jl) which (currently) is
pretty stale.

The [`PETSc_jll`](https://github.com/JuliaBinaryWrappers/PETSc_jll.jl) where
originally created for [`GridapPETSc.jl`](https://github.com/gridap/GridapPETSc.jl)
which are PETSc wrappers for [`Gridap.jl`](https://github.com/gridap/Gridap.jl).
