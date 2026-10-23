module quassar_hll
  use, intrinsic :: iso_fortran_env, only: real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  use quassar_euler, only: physical_flux
  implicit none
  private

  public :: hll_flux

contains

  function hll_flux(left, right, gamma) result(flux)
    real(real64), intent(in) :: left(3), right(3), gamma
    real(real64) :: flux(3)
    real(real64) :: fl(3), fr(3)
    real(real64) :: rho_l, rho_r, vel_l, vel_r
    real(real64) :: p_l, p_r, a_l, a_r, s_l, s_r

    ! Validate both states through the physical flux routine.
    fl = physical_flux(left, gamma)
    fr = physical_flux(right, gamma)

    rho_l = left(1)
    rho_r = right(1)

    vel_l = left(2) / rho_l
    vel_r = right(2) / rho_r

    p_l = (gamma - 1.0_real64) * &
          (left(3) - 0.5_real64 * rho_l * vel_l**2)

    p_r = (gamma - 1.0_real64) * &
          (right(3) - 0.5_real64 * rho_r * vel_r**2)

    a_l = sqrt(gamma * p_l / rho_l)
    a_r = sqrt(gamma * p_r / rho_r)

    s_l = min(vel_l - a_l, vel_r - a_r)
    s_r = max(vel_l + a_l, vel_r + a_r)

    if (.not. ieee_is_finite(s_l) .or. &
        .not. ieee_is_finite(s_r)) then
      error stop "Nonfinite HLL wave speeds"
    end if

    if (s_l >= 0.0_real64) then
      flux = fl
    else if (s_r <= 0.0_real64) then
      flux = fr
    else
      flux = (s_r * fl - s_l * fr + &
              s_l * s_r * (right - left)) / (s_r - s_l)
    end if

  end function hll_flux

end module quassar_hll
