
# Research 04 — Near-Extremal Kerr: Certified Spectral Sensitivity, Structured Perturbations, and Non-Normal Dynamics

## Research Question

To what extent does the pseudospectral sensitivity of near-extremal Kerr black hole perturbations reflect physically admissible dynamical behavior, rather than numerical conditioning, discretization artifacts, or norm-dependent effects?

## Scientific Objective

This research investigates the reliability and physical interpretation of spectral sensitivity in near-extremal Kerr spacetime.

The central objective is to distinguish genuine operator-level sensitivity from artifacts introduced by numerical approximation, coordinate choices, boundary treatment, arithmetic precision, and norm selection.

The investigation combines non-self-adjoint spectral theory, structured perturbation analysis, numerical convergence, and transient dynamical response.

## Mathematical Framework

Let \(L(a)\) denote a closed, densely defined linear operator representing an appropriately formulated Kerr perturbation problem, with dimensionless rotation parameter \(a/M\).

The operator domain, boundary conditions, functional space, and norm must be specified before spectral or dynamical conclusions are drawn.

A numerical discretization produces a finite-dimensional approximation \(L_N(a)\), where \(N\) controls spatial resolution.

For a matrix \(A\), the epsilon-pseudospectrum is

\[
\sigma_\varepsilon(A)
=
\sigma(A)\cup
\left\{
z\in\mathbb C:
\|(zI-A)^{-1}\|>\varepsilon^{-1}
\right\}.
\]

The matrix norm and its relationship to the underlying physical energy norm must be explicit.

For a simple eigenvalue \(\lambda\), with right and left eigenvectors \(v\) and \(w\), respectively, the eigenvalue condition number under the Euclidean norm is

\[
\kappa(\lambda)
=
\frac{\|w\|_2\|v\|_2}{|w^*v|}.
\]

For a differentiable perturbation \(A(\eta)=A+\eta E\), first-order eigenvalue sensitivity is

\[
\left.\frac{d\lambda}{d\eta}\right|_{\eta=0}
=
\frac{w^*Ev}{w^*v},
\]

provided the eigenvalue remains simple and the perturbation is differentiable.

These finite-dimensional formulas are diagnostic tools, not substitutes for analysis of the continuous physical operator.

## Research Hypotheses

### H1 — Spectral robustness

Selected near-extremal Kerr resonances exhibit sensitivity patterns that remain distinguishable from discretization and floating-point artifacts under independent numerical refinement.

### H2 — Structured perturbations

Physically motivated, constrained perturbations can produce substantially different spectral responses from arbitrary matrix perturbations of comparable norm.

### H3 — Norm dependence

The quantitative extent of pseudospectral sensitivity depends on the chosen operator norm. Physically meaningful comparisons require a consistent functional framework and documented norm equivalences or limitations.

### H4 — Spectral versus dynamical behavior

Large resolvent norms or eigenvalue sensitivity do not, by themselves, establish exponentially growing physical perturbations. Any relationship with transient amplification must be assessed through an appropriate evolution formulation.

These hypotheses are testable propositions, not established findings.

## Quantitative Research Design

The investigation will assess:

1. Complex resonance trajectories as the Kerr rotation parameter approaches extremality.
2. Spectral conditioning and resolvent sensitivity.
3. Structured versus unstructured operator perturbations.
4. Resolution-dependent eigenvalue and pseudospectral convergence.
5. Arithmetic precision sensitivity and residual-based error diagnostics.
6. Norm-dependent differences in spectral interpretation.
7. Transient amplification when a mathematically consistent evolution generator is available.

Numerical parameter ranges and acceptance tolerances must be justified through analytical scaling, reference benchmarks, and observed convergence.

## Verification Strategy

Verification will distinguish:

- Algebraic residuals from physical eigenvalue error.
- Discrete matrix pseudospectra from continuous-operator pseudospectra.
- Floating-point sensitivity from genuine spectral sensitivity.
- Coordinate-dependent effects from invariant or physically interpretable conclusions.
- Spectral instability from dynamical instability.

Independent numerical formulations will be used where feasible. Agreement between implementations is necessary evidence of reproducibility but does not independently establish physical validity.

## Scientific Acceptance Criteria

A result may be characterized as numerically verified only when:

- Its mathematical and physical assumptions are explicit.
- Boundary conditions and operator domains are documented.
- Relevant numerical convergence has been demonstrated.
- Residuals and conditioning have been assessed.
- Precision sensitivity has been examined where material.
- Comparisons use consistent norms and parameter definitions.
- Claims are supported by reproducible numerical evidence.

Claims of physical instability, mathematical novelty, or new gravitational phenomena require additional evidence beyond numerical agreement.

## Research Boundaries

The study does not presume that pseudospectral sensitivity implies physical instability.

It does not claim a new solution to Einstein's equations, a proof of Kerr stability, or a discovery of previously unknown gravitational phenomena.

Any contribution beyond established literature must be identified and independently substantiated.

## Research Integrity

All numerical results must distinguish established theory, independently reproduced benchmarks, new computational observations, and scientific inference.

Reproducibility, methodological transparency, numerical reliability, and appropriately qualified conclusions govern the research.
