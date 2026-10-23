# Research 03 — High-Order Computational Fluid Dynamics, Multiphysics Coupling & Uncertainty Quantification

## Scientific objective

Develop and independently verify conservative, high-order numerical methods for compressible flow, with a reproducible pathway to coupled physical transport and uncertainty quantification.

## Governing equations

The initial mathematical system is the one-dimensional compressible Euler system:

dU/dt + dF(U)/dx = 0

U = [rho, rho*u, E]

F(U) = [rho*u, rho*u^2 + p, u*(E+p)]

p = (gamma - 1)*(E - 0.5*rho*u^2)

where rho > 0, p > 0 and gamma > 1.

## Numerical research requirements

1. Conservative finite-volume formulation.
2. High-order spatial reconstruction with explicit treatment of discontinuities.
3. Consistent approximate Riemann flux.
4. Strong-stability-preserving time integration.
5. Positivity preservation for density and pressure.
6. Conservation diagnostics for mass, momentum and total energy.
7. Manufactured or analytic smooth-solution convergence studies.
8. Shock-tube benchmark verification.
9. Cross-language numerical consistency and performance measurements.
10. Explicit documentation of stability restrictions, limitations and reproducibility.

## Scientific extensions

After verification of the compressible-flow numerical foundation, investigate coupled thermal transport, uncertainty propagation, sensitivity analysis and additional high-order discretizations.

## Verification standard

Every claimed numerical order, conservation property, stability condition and performance improvement must be supported by reproducible experiments. Passing unit tests alone is insufficient.

## Preservation

Research 01, Research 02 and the approved repository navigation architecture remain unchanged.
