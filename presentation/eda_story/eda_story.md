---
title: "Exploratory Data Story – Soil Moisture LSTM"
author: Bryan Nsoh
---

# soil moisture automation needs sensor context so here is the field we logged...

![sensor layout](images/sensor_layout_placeholder.png){width=85%}

- lorawan nodes sample volumetric water content at 6, 18, and 30 inch depths every hour
- the weather mast streams air temperature, humidity, wind, canopy temperature, and rainfall
- irrigation totals sync from the controller so agronomists see actuations alongside sensors

# before modeling we check coverage so you see the telemetry gaps...

![coverage](images/feature_missingness.png){width=95%}

- core soil moisture and weather channels exceed eighty percent availability for the season
- stress indices such as cwsi, swsi, and daily et stayed too sparse to trust in training
- categorical crop and growth stage notes were largely absent so we excluded them

# raw probes jitter so domain smoothing keeps irrigation pulses intact...

![cleaning](images/vwc_cleaning_highlight.png){width=100%}

- hourly volumetric water content is spiky because of sensor noise and extraction cycles
- daily means plus pchip interpolation honour the recharge curve between irrigations
- a savitzky–golay window of twenty one keeps irrigation peaks while suppressing jitter

# spike flags separate irrigation jumps from drift so alerts stay actionable...

![spike flags](images/vwc_spike_flags.png){width=100%}

- we flag +15% day-over-day jumps as irrigation or rainfall response candidates
- −15% drops reveal drainage events or sensor faults that need review
- these flags feed downstream labelling and dashboard alerts for the irrigation team

# we cut noisy channels and keep agronomy signals so the feature set stays focused...

![feature selection](images/feature_selection_matrix.png){width=100%}

- retained features cover soil moisture layers, temporal encoding, weather load, and irrigation context
- 42" probes showed persistent flatlines so we documented and removed them
- sparse crop metadata and redundant elevation columns were dropped to avoid leakage

# domain transforms teach the lstm to feel rates and actuator events...

- center each vwc depth per plot so the network learns deviations instead of static bias
- compute Δvwc_d(t) = vwc_d(t) − vwc_d(t−1) to encode drying velocity
- log-transform precipitation + 1 and add a binary irrigation flag to highlight rare events

# sequence framing and regularized lstm capture a week of history before forecasting four days...

- 168-hour input window aligns with irrigation planning cadence
- 96-hour horizon meets the farm requirement for scheduling and monitoring
- stacked five-layer lstm (512→256→128→128→64) uses dropout and l2 regularisation to curb overfit
- timeseriessplit was meant for leave-one-plot-out yet the bug reused the first plot folds

# plot 2003 shows surface tracking while deeper layers lag revealing irrigation sensitivity gaps...

![plot 2003](images/pred_vs_actual_plot2003.png){width=100%}

- 6" predictions follow the smoothed trend but under-react immediately after irrigation
- 18" and 30" layers respond too softly versus the ground-truth rebounds
- the model captured seasonal decline yet missed amplitude which is key for scheduling

# plot 2014 exposes positive bias because dry-down exemplars were scarce in training...

![plot 2014](images/pred_vs_actual_plot2014.png){width=100%}

- surface depth stays elevated even as actual moisture declines after events
- deeper probes fail to catch the sharp post-irrigation spike that agronomists expect
- highlights need for balanced dry and wet sequences from every plot in training

# plot 2015 lags forty eight hours proving the fold reuse bug and irrigation imbalance...

![plot 2015](images/pred_vs_actual_plot2015.png){width=100%}

- forecasts trail the observed rebound by roughly two days across depths
- training data lacked aggressive irrigation cycles represented in this plot
- fixing the fold logic ensures each site contributes sequences and stops this lag

# injection experiments fail because features never recompute and scaling shifts the inputs...

![injection](images/irrigation_injection_failure.png){width=100%}

- water gets added after scaling so engineered features like logs and cumulative sums stay frozen
- scaled precip_irrig exceeds its buffered range which pushes the network off manifold
- with little cross-plot diversity the lstm defaults to trend continuation instead of reacting

# lessons fuel next season so data prep stays but the modeling stack evolves...

- current pipeline delivers clean engineered datasets ready for retraining and sharing
- next iteration will recompute features before sensitivity tests and repair fold reuse
- documented findings become onboarding material for agronomists and the controls team

