program test_quassar_hll
    use, intrinsic :: iso_fortran_env, only: real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    use quassar_euler, only: conservative_state, physical_flux
    use quassar_hll, only: hll_flux
    implicit none

    real(real64), parameter :: gamma = 1.4_real64
    real(real64), parameter :: tolerance = 1.0e-12_real64
    real(real64) :: u(3), expected(3), actual(3)
    real(real64) :: left(3), right(3), interface_flux(3)

    ! Consistency: HLL(U,U) must equal the physical Euler flux F(U).
    u = conservative_state(1.0_real64, 0.5_real64, &
                           1.0_real64, gamma)

    expected = physical_flux(u, gamma)
    actual = hll_flux(u, u, gamma)

    if (.not. all(ieee_is_finite(actual))) then
        error stop "HLL consistency: nonfinite flux"
    end if

    if (maxval(abs(actual - expected)) > tolerance) then
        error stop "HLL consistency validation failed"
    end if

    ! A discontinuous Riemann interface must produce finite flux.
    left = conservative_state(1.0_real64, 0.0_real64, &
                              1.0_real64, gamma)
    right = conservative_state(0.125_real64, 0.0_real64, &
                               0.1_real64, gamma)

    interface_flux = hll_flux(left, right, gamma)

    if (.not. all(ieee_is_finite(interface_flux))) then
        error stop "HLL Riemann interface: nonfinite flux"
    end if

    print '(A)', "RESEARCH 03: FORTRAN HLL NUMERICAL VALIDATION PASSED"
end program test_quassar_hll
