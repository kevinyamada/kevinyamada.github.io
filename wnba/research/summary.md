# WNBA spatial shooting study

Kevin Yamada · Updated November 13, 2026

The public page shows where shots occur, who scored the most field goal points in each square, and how well different models predict makes. This is an exploratory study of the 2025 WNBA season. It does not measure expected threat yet.

## Data and checks

The public wehoop / ESPN release was downloaded on November 13. The original desktop archive was unavailable during this update. Every saved game and team field goal total and every saved player region total matches the new release. This is strong aggregate agreement, but it is not proof that the files are identical row by row.

| Check | Result |
| :--- | ---: |
| Source play rows | 121,342 |
| Regular season games | 286 |
| Regular season attempts | 38,535 |
| Made field goals | 16,926 |
| Field goal points | 38,533 |
| Attempts shown on the half court | 38,441 |
| Longer attempts outside the view | 94 |
| Two point attempts | 24,679 |
| Three point attempts | 13,856 |
| Game and team box score comparisons | 624 |
| Duplicate game and play number keys | 0 |
| Missing shooter names in regular season shots | 0 |
| Current shot score sequence conflicts | 0 |

The 624 comparisons include regular season, playoff, and special games. The article excludes the All Star game and the Commissioner's Cup final. Free throws are excluded. A missed shooting foul that is not an official field goal attempt is also outside this target.

The new release includes `points_attempted`. Its two or three point labels match all 624 team box score totals for three point attempts and makes, and every made shot's point value. This resolves the missing attempt value problem in the earlier extraction. Sorting the public play feed by game and play number also gives zero conflicts between field goal points and the change in the scoreboard. The earlier count of 27 regular games with score inconsistencies must not be reused for this release. This check alone does not validate possessions or every event's timing.

Source hashes and download URLs are in [source_audit.json](results/source_audit.json). The main row grain is one recorded field goal attempt. All model prediction labels were joined back to the source using the game and play number key. Brier score and log loss were recalculated directly from probabilities.

## Coordinates and the sparse band

Coordinates are recorded in whole feet. Let `sign` be 1 for the home team and −1 for the away team. The transformation is `x = 41.75 − sign × coordinate_x`, `y = sign × coordinate_y`. The basket is at (0, 0), x runs toward half court, and y runs sideways. The display has y on its horizontal axis. The visible court is x from −5.25 to 41.75 feet and y from −25 to 25 feet.

The 2025 WNBA rule book gives an arc radius of 22 feet 1.75 inches, with the corner lines 3 feet inside each sideline. This gives lateral corner boundaries at ±22 feet on a 50 foot court. We use the geometric intersection of the arc and these segments. Line thickness and coordinate rounding prevent this geometry from identifying a player's exact foot position. Official attempt values are used for points calculations.

Signed distance is the shortest distance to either the curved arc or a corner segment. Negative means inside. Reflection, boundary, and known distance checks pass. No recorded attempt is removed to create an empty band.

There are 508 attempts in [−2, 0) feet and 4,380 in (0, 2] feet. The areas are 153.8 and 166.425 square feet, estimated on a 0.05 foot grid clipped to the half court. The density ratio is **7.968**, with a game bootstrap 95% interval of **7.273 to 8.830** from 1,000 resamples of 286 games. The 140 attempts recorded exactly on the line are excluded from this ratio. For the histogram, the bin convention places those points in the first outside bin. That convention does not classify them as official three point attempts.

Artificially perturbing coordinates by up to half a foot gives ratios around **6.26 to 7.14** across 100 scenarios. This is a sensitivity check, not inferred player movement or an uncertainty interval. The sparse band is robust, while its precise size depends on location resolution. The data supports fewer attempts just inside the line. It does not establish that individual players took a step backwards.

The frequency chart uses attempts per square foot and a log colour scale, with zero counts left blank. The raw make rate chart stripes cells with fewer than 20 attempts. A smaller square does not create more accurate coordinates. Player leaders are ranked by observed field goal points, with all joint leaders shown. This measures scoring volume in the selected area, not shooting skill. Team filters count only attempts recorded for that team.

## Two different evaluation questions

| Sample | Dates or scope | Half court attempts | Role |
| :--- | :--- | ---: | :--- |
| Training | Through August 9 | 26,686 | Fit all new models |
| Validation | August 10 to 26 | 6,129 | Select settings and model family |
| Later games | August 27 onward | 5,626 | Report the existing benchmark |
| Playoffs | 2025 playoffs | 3,311 | Additional transfer check |

Games do not cross the temporal splits. Models remain fitted on the original training sample when evaluated. We do not refit on validation and then compare against a different training size. Context boosting was selected by validation log loss. The later game and playoff samples have already been examined during development, including the earlier GP work. They are no longer untouched final tests.

For spatial validation, only the 26,686 early season attempts are used. Four fixed regions are defined by x = 16 feet and y = 0. Each region is held out once. Training locations whose nearest observed held out location is within 0, 2, or 4 feet are removed. The measured nearest separation exceeds the buffer in every fold. All rates, trends, variograms, and model fits are recomputed within each fold. The geometry of the held out locations is used to define the buffer, but their outcomes are not used for fitting.

The spatial settings are fixed for this diagnostic: IDW power 2, SPDE range 8 and sigma 0.7, and spatial boosting depth 3. This is not nested tuning of the final temporal winner. Games can occur on both sides of a spatial fold. It asks how well the model transfers to a missing area, not how well it predicts a future game. Four large regions are a first diagnostic; they do not establish performance across every possible partition. The buffers are a sensitivity study, not a claim that dependence ends at four feet.

## Model definitions

### Distance baseline and IDW

The distance baseline uses fixed bands at 0, 4, 8, 12, 16, 20, 22.1458, 24, 28, and 100 feet. Each observed band has a Beta(1, 1) prior and posterior mean (makes + 1) / (attempts + 2). An unobserved band falls back to the training sample's make rate.

IDW uses two foot cells and Jeffreys smoothed rates (makes + 0.5) / (attempts + 1). It averages the 16 nearest cell rates with weights proportional to inverse distance raised to a power. Powers 1, 2, and 3 are compared on validation. This interpolation does not provide a full Bayesian posterior and does not automatically account for different cell sample sizes in its distance weights.

### Ordinary and universal kriging

Cell logits are approximated by

$$z_j=\log\frac{k_j+0.5}{n_j-k_j+0.5},\qquad v_j\approx\frac1{k_j+0.5}+\frac1{n_j-k_j+0.5}.$$

Ordinary kriging uses a constant mean. Universal kriging adds distance and squared distance, scaled by 25 feet. After weighted trend fitting, residual semivariances are corrected by subtracting the approximate sampling noise. Only pairs of cells with at least 20 attempts each enter the variogram fit. Two foot lag bins extend to 32 feet and require more than 30 pairs. Negative corrected empirical values are retained because they can occur from sampling variability.

A constrained exponential covariance is fitted by weighted least squares. The fitted residual sill and length scale for universal kriging are 0.06753 and 2.62631 feet. This length scale is not the same convention as the SPDE practical range. Generalized least squares then estimates the trend using this covariance and the observation noise variances. Predictions transform the fitted logit mean back with the logistic function. They do not integrate transformed Gaussian uncertainty, and no kriging probability intervals are claimed. The Gaussian approximation to cell logits is a benchmark, not an exact binomial model. A stationary covariance may be a poor description across the rim and the perimeter.

### Bayesian GP from the earlier analysis

The retained model uses

$$k_{jp}\sim\operatorname{Binomial}(n_{jp},p_{jp}),\qquad
\operatorname{logit}(p_{jp})=\beta_0+f(s_j)+\alpha_p.$$

The intercept has a Normal(0, 1) prior. The spatial field uses a Matérn 5/2 covariance with a Hilbert space GP approximation. Each length scale has a Normal(8, 3) prior truncated to 3 through 20 feet. The field amplitude has a HalfNormal(1) prior. Player effects follow Normal(0, sigma_player), with sigma_player having a HalfNormal(0.5) prior. Sparse players are partially pooled toward the population mean. These effects are associations after the included adjustment, not causal shooting talent.

The displayed surface is the posterior make probability with the player effect set to zero. It is not the actual player mix at each location, nor an average over new player effects. Each square displays the nearest saved grid prediction to its centre, with a pointwise 95% credible interval for the probability. The interval is not a binary outcome prediction interval or a simultaneous band for the entire court.

The retained 24 by 24 basis fit used four chains with 2,000 tuning and 2,000 retained draws per chain. Maximum Rhat was 1.003902, minimum bulk ESS was 485.53, and there were zero divergences. A 16 versus 24 basis comparison changed evaluation probabilities by 0.00256 on average and 0.01566 at most. Prior predictive checks produced broad probabilities around a mean of 0.5. Such checks identify implausible assumptions; passing them does not prove calibration or model truth.

These outputs were retained from the earlier run, not rerun in this update. The original raw desktop file and event predictions were unavailable for a fresh paired comparison. Therefore the GP is shown separately from the new comparison table. Historical metrics, diagnostics, prior checks, and surface values are in `reference/`. `gp_model.py` provides the model code for a new run on the public source.

### Binomial SPDE

The new SPDE model has a binomial likelihood at each observed integer location and a Normal(0, 2) intercept prior. The Matérn field has smoothness nu = 1 in two dimensions. Piecewise linear basis functions on the triangulation interpolate the latent field to shot locations.

With lumped mass matrix C, stiffness matrix G, range rho, and field scale sigma,

$$\kappa=\sqrt{8}/\rho,\qquad \tau=(\sqrt{4\pi}\,\kappa\sigma)^{-1},$$

$$Q=\tau^2\left(\kappa^4 C+2\kappa^2G+GC^{-1}G\right).$$

The mesh extends beyond the court to reduce boundary effects. This does not prove the domain is large enough; a larger domain sensitivity check remains useful. The primary three foot mesh has 841 nodes. The precision matrix is sparse. A Newton solver finds the posterior mode, and the inverse Hessian provides a conditional Gaussian approximation. Fifteen point Gauss Hermite quadrature integrates the logistic probability under that approximation.

Ranges 4, 8, and 16 feet and sigma values 0.3, 0.7, and 1.5 were compared on validation. The selected setting is range 16 and sigma 0.3. These hyperparameters are fixed after selection. This is a **conditional Laplace approximation implemented in Python**, not an INLA fit, and it does not integrate hyperparameter uncertainty. The formula links the continuous SPDE prior to the finite element model; finite domain marginal variances need not exactly equal the infinite domain sigma squared.

The two and 1.5 foot meshes have 1,936 and 3,364 nodes. Refinement changed local predictions, even though validation scores were similar. We retain the prespecified three foot result in the primary table and do not claim complete mesh convergence. The page uses the earlier GP for its Bayesian probability map. Refinement metrics are in `mesh_refinement.csv`.

Numerical checks cover positive definite symmetric prior precision, interpolation weights summing to one, sparse posterior variance calculations matching a dense inverse on a synthetic example, finite bounded probabilities, and a small mode gradient. These are implementation checks, not proof of the Laplace approximation's statistical accuracy.

A fuller INLA analysis could assign PC priors through statements such as P(range < r0) = p_r and P(sigma > sigma0) = p_sigma, then integrate their uncertainty. The distance and scale choices should be elicited in feet and checked using simulated prior surfaces. We have not fitted this extension. The metre scale priors from an air quality exercise should not be copied onto a basketball court.

### Boosting and the ensemble

CatBoost compares depths 3 and 5, with 600 iterations, learning rate 0.035, L2 regularization 10, and seed 2026. The spatial version uses x, y, distance, and angle. The context version adds shooter, team, period, game clock, and home status. No made or missed descriptions, assist descriptions, or other outcome text are predictors. Both selected depth 3.

The temporal ensemble gives equal weight to the distance baseline, SPDE, and context boosting. The spatial diagnostic replaces context boosting with spatial boosting. We did not choose ensemble weights on the later game sample. This is a small comparison of plausible models, not an exhaustive search for the best possible algorithm.

## Results

Brier score is the average squared error in the predicted make probability. Log loss penalizes confident wrong predictions more heavily. Lower is better. Neither should be described as a percentage accuracy.

| Model | Validation Brier | Later games Brier | Playoff Brier |
| :--- | ---: | ---: | ---: |
| Context boosting | 0.23164 | 0.23286 | 0.23167 |
| Distance baseline | 0.23314 | 0.23378 | 0.23294 |
| Ensemble | 0.23178 | 0.23261 | 0.23176 |
| IDW | 0.23495 | 0.23436 | 0.23489 |
| Ordinary kriging | 0.23286 | 0.23315 | 0.23335 |
| SPDE | 0.23235 | 0.23313 | 0.23260 |
| Spatial boosting | 0.23212 | 0.23284 | 0.23311 |
| Universal kriging | 0.23287 | 0.23317 | 0.23328 |

The ensemble improves later game Brier from 0.233775 to 0.232610, a small relative reduction of about 0.50%. A paired game bootstrap with 2,000 resamples gives an ensemble minus baseline difference of −0.001165, with a 95% interval from −0.001960 to −0.000390. This is conditional on the inspected benchmark and the chosen models. It does not remove the effects of model search or turn the sample into a fresh final test. Context boosting wins validation and also has the lowest playoff Brier in this comparison.

| Model | Spatial Brier, 0 ft | Spatial Brier, 2 ft | Spatial Brier, 4 ft |
| :--- | ---: | ---: | ---: |
| Distance baseline | 0.23183 | 0.23272 | 0.24452 |
| IDW | 0.23449 | 0.24007 | 0.24595 |
| Ordinary kriging | 0.23432 | 0.24467 | 0.25006 |
| SPDE | 0.23279 | 0.23630 | 0.24816 |
| Spatial boosting | 0.23150 | 0.23322 | 0.24265 |
| Spatial ensemble | 0.23095 | 0.23305 | 0.24303 |
| Universal kriging | 0.23455 | 0.23821 | 0.24481 |

Spatial scores pool all held out shots across the four folds. No independence based confidence interval is inferred from just four fold scores. The full fold table includes sample counts and measured separation. Ordinary and universal kriging should be assessed by these results rather than preferred because they are course methods.

Calibration tables report predicted and observed make rates in fixed probability bins, as well as separately for two and three point attempts. Bins with few attempts should not be read as precise estimates. All model settings, temporal metrics, spatial fold metrics, bootstrap results, and numerical checks are published alongside the code.

## Expected points and expected threat

For an attempt with recorded value v in {2, 3}, a simple field goal points prediction is v times the predicted make probability. We now evaluate this using the verified source attempt value, rather than guessing a missed shot's value from coordinates. This is a plug in prediction conditional on an attempted shot and its recorded value. It is not the counterfactual value of choosing a two instead of a three.

For the ensemble in later games, predicted points average 0.99325 per attempt and observed points average 1.00729. RMSE is 1.15385 points, compared with 1.15602 for the distance baseline. These are small differences. The complete comparison is in `expected_points_metrics.csv`. The make models do not include attempt value explicitly; a separate model by attempt type is a reasonable next comparison, particularly near the line.

Free throws, offensive rebounds, continuation after misses, and future possession outcomes are excluded from this quantity. A location chart also gives P(location | recorded shot), not P(shot | the ball is at that location). The latter needs a denominator of shooting opportunities or possessions observed there.

Karun Singh's xT assigns value to states by combining an immediate scoring component with transitions into future states. A basketball version needs validated possessions, nonshot actions, turnovers, rebounds, fouls, and an explicit reward definition. The current play feed does not record every pass or movement. Even with consistent score updates, the last ten rows are not automatically the last ten on court actions. Sequence models and action credit assignment should wait for that audit.

## Connection to STA465

| Course idea | Role in this project |
| :--- | :--- |
| Spatial objects and coordinate systems | Court coordinates, units, bounds, and geometry checks; no geographic latitude or longitude projection is needed |
| EDA and spatial variation | Attempt density, make rates, separate team maps, and coordinate sensitivity |
| IDW | A simple local interpolation benchmark |
| Variograms | Residual dependence after removing a distance trend and approximate sampling noise |
| Ordinary and universal kriging | Constant versus distance dependent mean models on transformed cell rates |
| Spatial validation and buffers | Held out court regions with nearby training observations removed |
| Gaussian processes and priors | Smooth latent log odds, player pooling, prior predictive and sampling checks |
| SPDE and mesh | A finite element approximation to a Matérn prior with a binomial likelihood |
| Ensemble | Equal weight prediction averaging, checked against simpler baselines |
| Besag and BYM2 from HW2 | A possible areal model if the court is divided into adjacent cells; not fitted here |

The air quality model in HW2 used a continuous response and location based spatial structure. The disease mapping example used Poisson counts and an exposure. Field goal outcomes need a Bernoulli or binomial likelihood. Copying a Gaussian or Poisson response model without adapting the target would change the scientific question. A BYM2 extension could model binomial cell rates with a scaled structured component and an independent component, but mesh SPDE and BYM2 are different models, not interchangeable labels.

## Next research steps

1. Reserve an untouched season or future game block before further tuning. Use rolling temporal validation within the development seasons and report calibration alongside Brier and log loss.
2. Refit the GP on the verified public release, retain event predictions, and compare it with all other models on identical event keys. Test sensitivity to prior choices, mesh spacing, and the outer mesh boundary. An INLA SPDE fit can provide an independent implementation comparison.
3. Model make probability separately or hierarchically by recorded shot value, then extend expected points to possession rewards only after validating rebounds, fouls, free throws, and possession endings.
4. Confirm whether richer action or tracking data exists before attempting shot versus nonshot probabilities or the previous ten actions model. Record what each row represents and manually review a sample of possessions.

No Modal credits were used. The new CPU comparisons completed locally in the managed workspace. The desktop was offline, so its files and VS Code session were not changed. No claim of a maximum achievable accuracy is made.

## Reproduce the update

From this repository, use Python 3.12 in a virtual environment:

```bash
python -m pip install -r wnba/research/requirements.txt
python wnba/research/prepare_data.py
python wnba/research/spatial_summary.py
python wnba/research/verify_methods.py
python wnba/research/run_comparisons.py
python wnba/research/refine_mesh.py
python wnba/research/evaluate_predictions.py
python wnba/research/export_site_data.py
```

The preparation step downloads two public Parquet files if they are absent and verifies their hashes. If the release changes, stop and review the changed data rather than silently replacing the hash. The raw data and event predictions are excluded from version control. The website publishes only aggregated shot counts and saved model summaries.

The archived GP is optional and uses the separate dependencies in `requirements_gp.txt`. To fit it again, install those in a separate environment and run `python wnba/research/gp_model.py --players --basis 24 --draws 2000 --tune 2000`. Its outputs go into `results/gp/`. The website will continue to use the documented historical surface until the new run has been checked and explicitly promoted. Results may differ slightly between numerical libraries or systems.

## Sources

* [Public WNBA pipeline and data provenance](https://github.com/sportsdataverse/wehoop-wnba-data)
* [2025 WNBA official rule book](https://cdn.wnba.com/sites/4/2025/04/2025-WNBA-Official-Rule-Book.pdf), Rule 1 and the court diagram
* [Karun Singh, Introducing Expected Threat](https://karun.in/blog/expected-threat.html)
* [David Sumpter, Explaining Expected Threat](https://soccermatics.medium.com/explaining-expected-threat-cbc775d97935)
* [Christopher Fonnesbeck, Bayesian Spatial Modeling for Evaluating Hockey Goaltending Performance](https://www.pymc-labs.com/blog-posts/bayesian-spatial-modeling-for-evaluating-hockey-goaltending-performance)
* [Lindgren, Rue, and Lindström, 2011: the SPDE approach](https://doi.org/10.1111/j.1467-9868.2011.00777.x)
* [R INLA documentation for Matérn SPDE models and PC priors](https://www.r-inla.org/learnmore/docs/reference/inla.spde2.pcmatern.html)
* [Roberts et al., 2017: cross validation for structured data](https://florianhartig.github.io/publications/roberts2017cross/)

The page's interaction was inspired by these explanatory articles. Its code, visual design, and WNBA analysis were created for this project.
