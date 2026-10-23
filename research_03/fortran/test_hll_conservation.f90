program test_quassar_hll_conservation
    use, intrinsic :: iso_fortran_env, only: real64
    use quassar_euler, only: conservative_state
    use quassar_hll, only: hll_flux
    implicit none

    real(real64), parameter :: gamma = 1.4_real64
    real(real64), parameter :: tolerance = 1.0e-12_real64
    real(real64), parameter :: dt = 0.001_real64
    real(real64), parameter :: dx = 0.01_real64

    real(real64) :: left(3), right(3), flux(3)
    real(real64) :: left_new(3), right_new(3)
    real(real64) :: initial_total(3), final_total(3)

    left = conservative_state(1.0_real64, 0.0_real64, &
                              1.0_real64, gamma)
    right = conservative_state(0.125_real64, 0.0_real64, &
                               0.1_real64, gamma)

    flux = hll_flux(left, right, gamma)

    initial_total = dx * (left + right)

    left_new = left - (dt / dx) * flux
    right_new = right + (dt / dx) * flux

    final_total = dx * (left_new + right_new)

    if (maxval(abs(final_total - initial_total)) > tolerance) then
        error stop "HLL interface conservation failed"
    end if

    print '(A)', "RESEARCH 03: FORTRAN HLL INTERFACE CONSERVATION PASSED"
end program test_quassar_hll_conservation
