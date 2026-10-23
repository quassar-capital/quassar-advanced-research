module quassar_euler
  use, intrinsic :: iso_fortran_env, only: real64
  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
  implicit none
  private

  public :: conservative_state, physical_flux

contains

  pure function conservative_state(rho, velocity, pressure, gamma) result(u)
    real(real64), intent(in) :: rho, velocity, pressure, gamma
    real(real64) :: u(3)

    u(1) = rho
    u(2) = rho * velocity
    u(3) = pressure / (gamma - 1.0_real64) &
         + 0.5_real64 * rho * velocity**2
  end function conservative_state

  function physical_flux(u, gamma) result(f)
    real(real64), intent(in) :: u(3), gamma
    real(real64) :: f(3)
    real(real64) :: rho, velocity, pressure

    if (.not. all(ieee_is_finite(u))) error stop "Nonfinite state"
    if (.not. ieee_is_finite(gamma)) error stop "Nonfinite gamma"
    if (gamma <= 1.0_real64) error stop "Invalid gamma"

    rho = u(1)
    if (rho <= 0.0_real64) error stop "Nonpositive density"

    velocity = u(2) / rho
    pressure = (gamma - 1.0_real64) * &
               (u(3) - 0.5_real64 * rho * velocity**2)

    if (.not. ieee_is_finite(pressure)) error stop "Nonfinite pressure"
    if (pressure <= 0.0_real64) error stop "Nonpositive pressure"

    f(1) = u(2)
    f(2) = u(2) * velocity + pressure
    f(3) = velocity * (u(3) + pressure)
  end function physical_flux

end module quassar_euler
