# PRE-REGISTRATION — emergence-forecasting fusion analysis (FROZEN before any probe data was read)

*Analysis-only pre-registration over the two-gap grid's artifacts (the grid itself is governed by
PREREG_TWOGAP.md, unchanged). Committed while the grid's FIRST run is still training. Fuses the
mem-gen-delay forecasting methodology (composed anchors, calibrated intervals, manufactured-negative
false-alarm certification, blind gates, multiplicative anchor-fraction law) with the grid's
dilution x economics emergence testbed. Novelty lit-check completed 2026-07-17 (verdict: the
composition is unclaimed; must-cite list below; claims are scoped narrowly per that report).*

## Disclosure of everything seen at freeze time

Pilot runs (800 steps, m50, probe-free instrumentation era); the deleted 600-step dense-smoke run
(its probes.jsonl was touched once during instrumentation verification: at most one step-500 probe
line of a deleted smoke model); heartbeat lines (step / tok/s / memory ONLY) and the A_m50 header
from the running grid. NO grid eval values and NO grid probe values have been read. Monitoring rule
until the freeze sequence completes: status checks read heartbeat/DONE lines only; no eval lines of
m2 runs are displayed before step F3 below.

## Definitions (frozen now)

- **Event** (per run): first eval step (2,500 cadence) with mean(toggle-depth8, dial-depth8)
  accuracy >= 0.9 for two consecutive evals (absolute criterion; boxes reported secondarily). Runs
  never reaching it are non-eventing.
- **Anchor** (500-step cadence, composed per the paper-4 recipe): first step where
  probe_mean = (probe_toggle + probe_dial)/2 >= theta_p AND shallow_acc >= theta_b, each sustained
  two consecutive points. theta_p in {0.30,...,0.60 step 0.05}, theta_b in {0.55, 0.65, 0.75},
  selected ON CALIBRATION RUNS ONLY: maximize median lead subject to (i) fires before the event on
  every eventing calibration run, (ii) ZERO alarms on the m0 negatives (data-ablated: tracking
  cannot emerge). Alarm on a non-eventing run = false alarm.
- **Law**: c = t_anchor / t_event per eventing calibration run; frozen constants = median(c) and
  envelope [min(c), max(c)]. **Forecast** at anchor on a blind run: point t_hat = t_anchor/median(c),
  interval [t_anchor/max(c), t_anchor/min(c)].

## Frozen sequence (commit gates)

1. **THIS COMMIT**: definitions, split, selection procedure, predictions, kills.
2. Grid completes -> open ONLY calibration (m50: A,B,C,B' + m10: A,B,C) and negatives (m0: A,B,C);
   select thetas, compute median(c) + envelope; **FREEZE COMMIT** of the constants.
3. Only then open the blind cells (m2: A,B,C): score coverage, leads, false alarms (m2 non-eventing
   + m0 re-check), and the comparisons below. Verdicts commit. All steps reported regardless.

## Predictions

- **F1 (anchors lead):** the selected anchor fires before the event on >= 6/7 eventing calibration
  runs (denominator restricted to eventing runs if fewer event; see KF3).
- **F2 (a law exists):** calibration spread max(c)/min(c) <= 1.5.
- **F3 (blind gate):** every eventing m2 run's event falls inside its frozen interval; zero false
  alarms on m0 and on non-eventing m2 runs.
- **F4 (economics covariate, two-sided):** either c is stable across arms A/B/B'/C (the timing law
  is robust to write-gate economics) or it shifts systematically with pricing (gate economics moves
  the announced fraction) — both pre-registered as reportable findings; no directional bet.
- **F5 (cross-architecture universality):** the DeltaNet-class median(c) is compared against the
  mem-gen-delay frozen anchor fraction (0.843, blind envelope ~[0.735, 0.901]). Inside = first
  cross-architecture, cross-capability universality evidence for the fraction law; outside =
  architecture/capability-dependence. Either is the finding.
- **Baselines (both scored on the blind cells):** (a) best-case training-loss threshold rule
  (expected nowcast, per the paper-4 result); (b) additive fixed-offset alternative
  (Sivasankar-style t_event = t_anchor + k) calibrated identically — multiplicative vs additive
  decided by blind coverage.

## Kills

- **KF1:** no (theta_p, theta_b) satisfies the calibration constraints (pre-event firing + 0/3
  negative alarms) -> this probe family is inadequate as an anchor; the fusion closes as a
  pre-registered negative (echoing paper-4's label-free probe death).
- **KF2:** calibration spread > 2x -> no stable fraction law in this substrate; report.
- **KF3:** fewer than 4 eventing calibration runs -> the grid cannot support calibration; the
  fusion defers to the seed-replication / n_h=2 runs, disclosed (no thresholds fitted on m2, ever).

## FUSION AMENDMENT 1 (2026-07-17, mid-grid; triggered by a relayed external review)

**Exposure disclosure:** a review from another session, relayed by the user, exposed partial
mid-run CALIBRATION-cell values to the analyst (m50 baseline blended-pack accuracies "0.39-0.53,
non-monotone between evals 7.5k/10k"). No probe values, no depth-8 family values, and no m2 values
were exposed. The frozen definitions at commit 8de59c9 provably predate this exposure and are
UNCHANGED. Standing request recorded: no further grid values relayed into the analysis context
before the rule-freeze commit — m2 values especially.

**In response (all declared before any further data is seen):**
1. **Event definition unchanged.** Lowering the bar post-exposure would be contaminated. If arms
   fail to event, KF3 fires as frozen. Any relaxed-bar or shallower-depth event analysis that may
   later be run is HEREBY pre-labeled post-hoc exploratory and can never carry a pre-registered
   verdict.
2. **F5 dual-scoring (selection-bias repair):** the lead-maximized anchor biases fractions early
   relative to paper-4's fixed a-priori constant. F5 is now scored on BOTH (a) the optimized anchor
   and (b) a FIXED mid-grid anchor declared here: theta_p = 0.45, theta_b = 0.65 (grid midpoints,
   chosen without sight of any probe data). Universality claim language keys to (b).
3. **F2/F4 decomposition (pooling repair):** fractions are computed per cell; report the
   across-arms-within-mixture spread and the across-mixtures-within-arm spread separately; a pooled
   "law" claim requires BOTH within the F2 bound. The seed-replication rule extends to fusion
   F-verdicts: any positive F2/F3/F5 verdict requires >= 2 calibration cells replicated at seed 1
   before the claim is made.
4. **Cadence correction (disclosed):** the actual eval cadence is 2,500 steps (launcher), not the
   2,000 in PREREG_TWOGAP's frozen text; TWOGAP Amendment 1's "same evals" was inaccurate as to
   cadence. Infrastructure-only; the fusion doc's definitions already said 2,500.
5. **Heartbeat-memory note:** hb lines report resting allocator memory, not peak; the
   memory-fraction guard (loud OOM) is the actual spill protection. No mid-grid code change (would
   make instrumentation inconsistent across runs).

## Must-cite scoping (from the 2026-07-17 lit-check)

2606.12966 (FSD precursor; additive, seed-averaged, transformer grokking — the nearest work);
2505.17863 (emergence-time power laws in data properties); 2603.29805 (spectral early-warning
indicator, uncalibrated); 2605.20441 (retrospective timing regimes incl. Mamba); 2604.08510
(implicit curriculum; post-hoc); 2511.16893 (seed-invariant induction prediction); 2411.16035,
2405.10938, 2310.03262 (cross-scale emergence prediction); 2306.13253 (grokking prediction);
plus the mem-gen-delay series and PREREG_TWOGAP's literature. Claims are the COMPOSITION only:
per-seed calibrated forecasts x recurrent substrate x dilution+economics covariates x the
fraction-law universality test, with manufactured-negative false-alarm certification.
