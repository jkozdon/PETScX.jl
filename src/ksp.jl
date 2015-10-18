# provide some most commonly used options, leave rest as low level
# common options: Orthogonilization type, KSP type, PC type
# use PC context created as part of KSP

type KSP{T, MType}
  pksp::C.KSP{T}
  ppc::C.PC{T}
  own_pc::Bool  # is the PC owned by the ksp context
  A::Mat{T, MType}
end

function KSP{T, MType}(A::Mat{T, MType}, pc_mat::Mat{T, MType}=A)
  ksp_arr = Array(C.KSP{T}, 1)
  pc_arr = Array(C.PC{T}, 1)

  chk(C.KSPCreate(A.comm, ksp_arr))
  ksp = ksp_arr[1]
  C.KSPGetPC(ksp, pc_arr)
  pc = pc_arr[1]

  chk(C.KSPSetOperators(ksp, A.p, pc_mat.p))
  

  return KSP{T, MType}(ksp, pc, true, A)
end


function KSPDestroy(ksp::KSP)

  tmp = Array(PetscBool, 1)
  C.PetscFinalized(eltype(mat), tmp)
   
  if tmp[1] == 0  # if petsc has not been finalized yet
    if !ksp.own_pc
      C.PCDestroy([ksp.ppc])
    end

    C.KSPDestroy([ksp.pksp])
  end

   # if Petsc has been finalized, let the OS deallocate the memory
end

function KSPSetOptions{T, MType}(ksp::KSP{T, MType}, opts::Dict{ByteString, ByteString})
# sets some options in the options database, call KSPSetFromOptions, then reset the
# options database to the original state
# this prevents unexpected dynamic phenomena like setting an option for one KSP 
# contex and having it still be set for another
# the keys in the dictionary should have the prepended -
  # to get options we have to provide a string buffer of sufficient length
  # to be populated with the returned string
  # what is a 'sufficient length'? no way to know, so make it 256 characters
  # for now and enlarge if needed later

  len = Csize_t(256)
  string_buff = UTF8String(Array(Uint8, len))
  null_char = ' ' # need nul char because can't pass C_NULL
  null_str = UTF8String(" ")
  opts_orig = Dict{UTF8String, UTF8String}()
  isset = Ref{PetscBool}()
  for i in keys(opts)
    # record the original option value
    chk(C.PetscOptionsGetString(T, null_str, i, string_buff, len, isset))
    if isset[]  # if an option with the specified name was found
      str_len = findfirst(string_buff, null_char)
      opts_orig[i] = string_buff[1:str_len]
    else
      println(STDERR, "Warning: option $i not found by PETSc")
    end

    # now set the the option
    chk(C.PetscOptionsSetValues(i, opts[i]))
  end

  C.KSPSetFromOptions(ksp.pksp)
  

  # now reset the options
  for i in keys(opts_orig)
    chk(C.PetscOptionsSetValues(i, opts[i]))
  end

  return nothing
end



function settolerances{T}(ksp::KSP{T}; rtol=1e-8, abstol=1e-12, dtol=1e5, maxits=size(ksp.A, 1))

  C.KSPSetTolerances(ksp, rtol, abstol, dtol, maxits)
end

function solve{T}(ksp::KSP{T}, b::Vec{T}, x::Vec{T})
# perform the solve
# users should specify all the options they want to use
# before calling this function
# if solving multiple rhs with the same matrix A,
# the preconditioner is resued automatically
# if A changes, the preconditioner is recomputed

  C.KSPSetUp(ksp.pksp)   # this is called by KSPSolve if needed
                   # decreases logging accurace of setup operations
  chk(C.KSPSolve(ksp.pksp, b.p, x.p))

  reason_arr = Array(Cint, 1)
  chk(C.KSPGetConvergedReason(ksp.pksp, reason_arr))
  reason = reason_arr[1]



  if reason < 0
    println(STDERR, "Warning: KSP Solve did not converge")
  end

end

function solve{T}(ksp::KSP{T}, b::Vec{T})
# this only works for square systems
  x = similar(b)
  solve(ksp, b, x)
  return x
end




  
  
