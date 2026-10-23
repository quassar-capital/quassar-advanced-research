program test_quassar_euler
  use, intrinsic :: iso_fortran_env, only: real64
  use quassar_euler, only: conservative_state, physical_flux
  implicit none

  real(real64), parameter :: gamma = 1.4_real64
  real(real64), parameter :: tolerance = 1.0e-12_real64
  real(real64) :: u(3), f(3)

  u = conservative_state(1.0_real64, 2.0_real64, &
                         1.0_real64, gamma)

  if (any(abs(u - [1.0_real64, 2.0_real64, &
                   4.5_real64]) > tolerance)) then
    error stop "Conservative-state validation failed"
  end if

  f = physical_flux(u, gamma)

  if (any(abs(f - [2.0_real64, 5.0_real64, &
                   11.0_real64]) > tolerance)) then
    error stop "Physical-flux validation failed"
  end if

  print '(A)', "RESEARCH 03: FORTRAN EULER VALIDATION PASSED"
end program test_quassar_euler
