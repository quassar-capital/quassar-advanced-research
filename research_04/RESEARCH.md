
# Research 04 — Near-Extremal Kerr: Certified Spectral Sensitivity, Structured Perturbations, and Non-Normal Dynamics

## Research Question

To what extent does the pseudospectral sensitivity of near-extremal Kerr black hole perturbations reflect physically admissible dynamical behavior, rather than numerical conditioning, discretization artifacts, or norm-dependent effects?

## Scientific Scope

Research 04 establishes a reproducible computational framework for finite-dimensional non-normal spectral analysis, structured eigenvalue sensitivity, pseudospectral diagnostics, and separated Kerr perturbation equations.

The implementation includes scalar and gravitational Teukolsky differential coefficients, a gravitational angular Galerkin eigensolver, and local angular eigenpair continuation.

The computational results do not establish certified continuous-operator Kerr spectra, gravitational quasinormal frequencies, or physical dynamical instability.

## Mathematical Framework

The physical spectral problem requires a specified differential operator, domain, boundary conditions, functional space, and norm.

For a finite-dimensional matrix A, the Euclidean epsilon-pseudospectrum is

\[
\sigma_\varepsilon(A)=\sigma(A)\cup
\{z\in\mathbb C:\|(zI-A)^{-1}\|_2>\varepsilon^{-1}\}.
\]

For a simple eigenvalue lambda, with right and left eigenvectors v and w,

\[
\kappa(\lambda)=
\frac{\|w\|_2\|v\|_2}{|w^*v|}.
\]

For a differentiable perturbation A(eta) = A + eta E,

\[
\left.\frac{d\lambda}{d\eta}\right|_{\eta=0}
=\frac{w^*Ev}{w^*v}.
\]

These expressions characterize finite-dimensional spectral sensitivity under their stated assumptions. They do not independently establish physical properties of Kerr perturbations.

## Implemented Research Components

### 1. Spectral Sensitivity

Module: `spectral.py`

Implements simple-eigenvalue condition numbers, directional eigenvalue derivatives, left and right eigenvector residuals, and smallest-singular-value diagnostics.

Eigenvalue ordering is deterministic but is not a physical mode-tracking procedure. Numerical spectral separation is not a proof of eigenvalue simplicity.

### 2. Structured Perturbations

Module: `structured.py`

Computes the maximum first-order eigenvalue sensitivity over a researcher-specified complex-linear matrix perturbation subspace, using the Frobenius norm.

The implementation distinguishes structured and unstructured sensitivity. It does not establish that the supplied perturbations are physically admissible gravitational perturbations.

### 3. Pseudospectral Diagnostics

Module: `pseudospectrum.py`

Computes shifted-matrix singular values, resolvent norms, spectral distances, amplification ratios, complex-plane diagnostic grids, and epsilon-pseudospectral membership.

All quantities refer to finite-dimensional matrices under the Euclidean norm. Continuous-operator pseudospectral convergence is not certified.

### 4. Scalar Kerr Operator

Module: `kerr_operator.py`

Implements separated scalar Teukolsky coefficients and differential residuals on subextremal Kerr backgrounds.

Conventions:

- Geometric units G = c = 1.
- Fourier dependence exp(-i omega t + i m phi).
- Spin weight s = 0.
- Exterior radial domain r > r_+.

The module also computes horizon quantities and the scalar radial separation parameter.

It does not solve the physical scalar boundary-value problem.

### 5. Gravitational Teukolsky Operator

Module: `gravitational_teukolsky.py`

Implements separated spin-weight-minus-two radial and angular Teukolsky coefficients and differential residuals.

The radial equation is represented as

\[
\Delta R''-\Delta'R'+VR=0,
\]

where

\[
K=(r^2+a^2)\omega-am,
\]

\[
V=\frac{K^2+4i(r-M)K}{\Delta}
-8i\omega r-\lambda,
\]

and

\[
\lambda=A+a^2\omega^2-2am\omega.
\]

The angular separation eigenvalue A is supplied explicitly.

The module does not enforce ingoing horizon or outgoing infinity boundary conditions and therefore does not compute gravitational quasinormal frequencies.

### 6. Gravitational Angular Eigenvalues

Module: `angular_solver.py`

Implements a finite-dimensional Galerkin angular operator in a spin-weighted spherical-harmonic basis.

The solver returns angular eigenvalue estimates, truncation-comparison diagnostics, algebraic residuals, and reference-mode overlaps.

The angular spheroidicity parameter is c = a omega.

Finite-truncation differences are numerical convergence indicators, not certified spectral error bounds.

Direct selection by proximity to a spherical reference eigenvalue is a local identification rule, not a global continuation guarantee.

### 7. Angular Mode Continuation

Module: `mode_continuation.py`

Tracks gravitational angular eigenpairs along a supplied path in complex spheroidicity beginning at c = 0.

Implements:

- Eigenvector-overlap tracking.
- Overlap-margin diagnostics.
- Spectral-gap evaluation.
- Relative eigenpair residuals.
- Conservative path-displacement thresholds.
- Explicit rejection of ambiguous identification.

These diagnostics reduce silent mode-switching risk in the truncated angular problem.

They do not prove global analytic continuation or physical mode identity. Independent truncation refinement remains necessary.

## Research Hypotheses and Evidence

### H1 — Spectral Robustness

Hypothesis: Near-extremal Kerr resonances exhibit sensitivity distinguishable from numerical discretization and floating-point artifacts.

Evidence status: Not established.

The repository does not implement a boundary-conditioned gravitational resonance solver or demonstrate independently converged near-extremal quasinormal frequencies.

### H2 — Structured Perturbations

Hypothesis: Physically constrained perturbations may produce substantially different spectral responses from unrestricted perturbations.

Evidence status: Partially addressed mathematically.

Structured sensitivity is implemented for complex-linear matrix subspaces. Physically admissible gravitational perturbation classes have not been constructed or validated.

### H3 — Norm Dependence

Hypothesis: Pseudospectral sensitivity depends on the selected norm and requires a consistent physical functional framework.

Evidence status: Mathematical distinction documented; physical comparison not established.

Spectral diagnostics use Euclidean matrix norms. Structured perturbation amplitudes use the Frobenius norm.

A physical Kerr energy norm and corresponding operator-level norm analysis are not implemented.

### H4 — Spectral Versus Dynamical Behavior

Hypothesis: Large resolvent norms and eigenvalue condition numbers do not independently establish physical dynamical instability.

Evidence status: Methodological restriction documented; dynamical hypothesis not tested.

The repository does not implement a validated Kerr time-evolution generator or physical transient-amplification experiment.

## Verification and Reproducibility

The repository includes focused Python tests covering the seven implemented computational components.

The recorded scientific CI checkpoint passed 52 Python tests. The corresponding Rust and Julia scientific jobs and Research Navigation Integrity workflow also passed.

These checks verify tested software behavior. They do not certify the physical research hypotheses or continuous Kerr spectral properties.

To execute the Python scientific tests from the repository root:

    python -m pytest -q

For numerical reproducibility, record:

- Repository commit identifier.
- Kerr mass and rotation parameters.
- Spin weight and angular quantum numbers.
- Complex frequency and spheroidicity.
- Angular truncation settings.
- Continuation paths and acceptance thresholds.
- Floating-point environment and numerical library versions.
- Residuals, spectral gaps, and convergence diagnostics.

The implementation uses NumPy double-precision numerical linear algebra. Independent higher-precision physical resonance certification is outside the implemented scope.

## Scientific Interpretation

### Discrete and Continuous Spectra

A finite-dimensional pseudospectrum cannot automatically be identified with the pseudospectrum of a continuous differential operator.

Operator domains, discretization convergence, numerical conditioning, and norm consistency must be established before continuous-operator claims are justified.

### Angular and Radial Coupling

Angular separation eigenvalues alone do not determine gravitational quasinormal frequencies.

A physical resonance calculation requires coupling the angular and radial equations and enforcing the appropriate horizon and infinity boundary conditions.

### Conditioning and Residuals

Small algebraic eigenpair residuals do not necessarily imply small eigenvalue errors for non-normal operators.

Spectral separation, eigenvector conditioning, numerical precision, and independent refinement remain essential.

### Physical Admissibility

Arbitrary matrix perturbations are not equivalent to perturbations permitted by the gravitational field equations.

Physical perturbation classes require explicit mathematical construction and validation.

### Stability and Dynamics

Pseudospectral extent, large resolvent norms, and eigenvalue sensitivity do not independently establish exponential instability or physical transient growth.

Such claims require a mathematically consistent evolution problem and an appropriate physical norm.

## Verified Computational Outcomes

The implemented and tested framework provides:

1. Finite-dimensional spectral sensitivity diagnostics.
2. Complex-linear structured first-order perturbation sensitivity.
3. Finite-dimensional pseudospectral and resolvent diagnostics.
4. Scalar Kerr differential coefficients and residuals.
5. Gravitational Teukolsky differential coefficients and residuals.
6. Gravitational angular Galerkin eigenvalue calculations.
7. Conservative local angular eigenpair continuation.
8. Automated scientific software verification.

These are computational and methodological outcomes, not independently certified discoveries about the physical Kerr spectrum.

## Scientific Limitations

The implementation does not establish:

1. Certified near-extremal Kerr gravitational quasinormal frequencies.
2. Continuous-operator Kerr pseudospectral convergence.
3. Physically admissible gravitational perturbation sensitivity bounds.
4. A physical energy-norm pseudospectral comparison.
5. Kerr transient amplification or dynamical instability.
6. A new gravitational phenomenon or a proof of Kerr stability.

## Research Outcome

Research 04 delivers a reproducible computational foundation for investigating non-normal spectral sensitivity and Kerr-related angular perturbation problems.

The completed implementation provides explicit mathematical conventions, finite-dimensional numerical methods, diagnostics, local mode tracking, and automated software verification.

The central physical research question remains unresolved by the implemented calculations.

A subsequent physical investigation would require a separately scoped coupled angular-radial gravitational resonance solver, appropriate boundary conditions, independent numerical refinement, and norm-consistent dynamical analysis.

## Research Integrity

Established mathematical identities, implemented computational methods, tested software behavior, unresolved physical hypotheses, and scientific inferences are distinct categories of evidence.

Successful numerical software tests do not independently certify physical spectral stability, instability, or mathematical novelty.

Research conclusions must remain proportional to the available evidence.
