# Online Resource 1 — Derivations

Notation. Training features φᵢ ∈ ℝᵖ (MiniRocket output, standardised with the training
mean and standard deviation), n training series, K classes, one-vs-rest targets
Y ∈ {−1, +1}ⁿˣᴷ. Augmented inputs zᵢ = (1, φᵢ − φ̄) with unpenalised intercept.
Penalty matrix P = diag(0, λIₚ) and A = ZᵀZ + P (Z stacks the zᵢ). The ridge fit is
B = A⁻¹ZᵀY, fitted logits Ŷ = ZB = HY with hat matrix H = ZA⁻¹Zᵀ, and dᵢ = Hᵢᵢ.

## Proposition 1 (exact LOO logits)
Let B₋ᵢ be the fit without example i, with the same λ and the same standardisation. Then

  zᵢᵀB₋ᵢ = Yᵢ − (Yᵢ − Ŷᵢ)/(1 − dᵢ).

*Proof.* A₋ᵢ = A − zᵢzᵢᵀ. By Sherman–Morrison,
A₋ᵢ⁻¹ = A⁻¹ + A⁻¹zᵢzᵢᵀA⁻¹/(1 − dᵢ), so zᵢᵀA₋ᵢ⁻¹ = zᵢᵀA⁻¹ + dᵢzᵢᵀA⁻¹/(1 − dᵢ)
= zᵢᵀA⁻¹/(1 − dᵢ). Since B₋ᵢ = A₋ᵢ⁻¹(ZᵀY − zᵢYᵢ),
zᵢᵀB₋ᵢ = (zᵢᵀA⁻¹ZᵀY − dᵢYᵢ)/(1 − dᵢ) = (Ŷᵢ − dᵢYᵢ)/(1 − dᵢ) = Yᵢ − (Yᵢ − Ŷᵢ)/(1 − dᵢ). ∎
(This is the classical PRESS identity of Allen (1974); PreVal uses the same identity.)

## Proposition 2 (held-out leverage)
The leverage of training example i under the model trained without it is

  h₋ᵢ := zᵢᵀA₋ᵢ⁻¹zᵢ = dᵢ/(1 − dᵢ).

*Proof.* From the Sherman–Morrison expression above,
zᵢᵀA₋ᵢ⁻¹zᵢ = dᵢ + dᵢ²/(1 − dᵢ) = dᵢ/(1 − dᵢ). ∎

## Proposition 3 (closed-form quantities from one thin SVD)
Let Φc = USVᵀ be the thin SVD of the centred feature matrix (rank r ≤ min(n − 1, p)), and
D = diag(1/(s²ⱼ + λ)). For a test point with centred features ψ:

* dᵢ = 1/n + Σⱼ u²ᵢⱼ s²ⱼ/(s²ⱼ + λ);
* h(x) := zᵀA⁻¹z = 1/n + ‖D^{1/2}Vᵀψ‖² + (‖ψ‖² − ‖Vᵀψ‖²)/λ;
* k(x, i) := zᵀA⁻¹zᵢ = 1/n + (Vᵀψ)ᵀ D S uᵢ (the training points lie in span(V)).

*Proof.* Because Φc is centred, the intercept column is orthogonal to Φc and A is block
diagonal in the basis (1, Φc): the intercept block contributes 1/n. For the feature block,
(ΦcᵀΦc + λI)⁻¹ = V D Vᵀ + λ⁻¹(I − VVᵀ). ∎

All three cost O(nr) per training point and O(pr) per test point after the SVD.

## Proposition 4 (exact LOO model at a test point)
With residuals E = Y − Ŷ,

  z(x)ᵀB₋ᵢ = z(x)ᵀB − k(x, i) Eᵢ/(1 − dᵢ),    h₋ᵢ(x) = h(x) + k(x, i)²/(1 − dᵢ).

*Proof.* B₋ᵢ = A₋ᵢ⁻¹(ZᵀY − zᵢYᵢ) = B − A⁻¹zᵢYᵢ + A⁻¹zᵢ(Ŷᵢ − dᵢYᵢ)/(1 − dᵢ)
= B − A⁻¹zᵢ(Yᵢ − Ŷᵢ)/(1 − dᵢ) = B − A⁻¹zᵢEᵢ/(1 − dᵢ); multiplying by z(x)ᵀ gives the first
identity. The second is Sherman–Morrison applied to z(x)ᵀA₋ᵢ⁻¹z(x). ∎
(Both identities are checked numerically against explicit refits in `code/tests/test_core.py`.)

These give every quantity jackknife+ needs, s(x, y; f₋ᵢ) for all i, in O(nK) per test
point, with no refitting. The jackknife+ set is
C(x) = {y : #{i : sᵢ < s(x, y; f₋ᵢ)} < (1 − α)(n + 1)}, where sᵢ = s(xᵢ, yᵢ; f₋ᵢ).

## Interpretation of h(x)
Under the Bayesian linear model with prior β ~ N(0, (σ²/λ)I) and noise variance σ², the
posterior predictive variance of each logit at x is σ²·h(x) (with the intercept treated as
flat). The leverage-standardised logit z(x)/√h(x) is therefore the posterior mean divided
by its posterior standard deviation up to the common factor σ. In p ≫ n the out-of-span term
(‖ψ‖² − ‖Vᵀψ‖²)/λ dominates; exact held-out leverages h₋ᵢ include the same term for each
left-out example, so h(x)/h_ref ≈ 1 for in-distribution test inputs and grows for inputs
away from the training distribution.

## Caveats (stated in the paper)
* The identities are exact for fixed standardisation; the scaler and the MiniRocket biases
  are fitted on all n training series, so the LOO quantities ignore that dependence.
* λ and the temperature a are selected on the same LOO pairs, which makes the LOO NLL
  slightly optimistic and means the jackknife+ guarantee (≥ 1 − 2α, Barber et al. 2021)
  holds only approximately; coverage is therefore assessed empirically.
