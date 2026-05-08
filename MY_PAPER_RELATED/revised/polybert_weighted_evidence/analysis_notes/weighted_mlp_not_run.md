# Weighted MLP Not Run

The optional MLP weighted predictor was not run.

Reasons:
- The manuscript currently uses PolyBERT-Ridge as the surrogate prescreener.
- The requested CEJ evidence gap concerns whether interval sample weights improve high-conductivity screening behavior for the existing Ridge prescreener.
- Adding a weighted MLP would introduce a new model class with additional hyperparameters and early-stopping behavior, which would complicate rather than clarify the current manuscript claim.

Recommendation:
- Keep the weighted experiment as a Ridge-only sensitivity analysis.
- If a weighted MLP is later needed, it should be run as a separate supplementary experiment with fixed architecture, fixed seeds, weighted MSE, and the same folds.
