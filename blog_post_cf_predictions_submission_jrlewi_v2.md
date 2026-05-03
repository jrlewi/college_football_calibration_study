# Are ESPN's College Football Win Probabilities Any Good?

*A calibration study across 14,100 games and 2.4 million plays.*

*By: John Lewis*

---

For roughly 30% of college football games on ESPN, the live win probability tracker isn't doing real-time probability estimation. It's doing something much simpler: returning a number close to the historical home team win rate and calling it a day.

That's not a bug, exactly. It's a tell — a window into how the model was built and what it was optimized for. And it's one of several things I found after scraping ESPN's play-by-play data for every available Division I game I could get: roughly 14,100 games across the 2015–2025 seasons, yielding 2.4 million individual play-level predictions.

The headline is that ESPN's model is well-calibrated. But the more interesting story is in the places where it isn't — and what those failures reveal about what's happening under the hood. Similar calibration audits have been done for ESPN's NBA win probability model ([Inpredictable, 2018](https://www.inpredictable.com/2018/01/judging-win-probability-models.html); [Lopez & Matthews, 2020](https://arxiv.org/pdf/2010.00781)), but to my knowledge, no one has done this systematically for college football.

The code and data pipeline are in the [GitHub repo](https://github.com/jrlewi/college_football_calibration_study). This post is about what I found.

---

## What Does "Good" Mean?

A probability forecast is **calibrated** if the stated probabilities match observed frequencies. If ESPN says a team has a 70% chance of winning, that team should win about 70% of the time — across all plays where that probability was given.

Three metrics do most of the work here.

**Brier Score** measures average squared error between predicted probability *p* and actual outcome *y* ∈ {0,1}:

$$BS = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$

A score of 0 is perfect; 0.25 is what you'd get from always predicting 50%. Lower is better.

**Expected Calibration Error (ECE)** groups predictions into bins by probability and measures the weighted average gap between mean predicted probability and actual win rate in each bin:

$$ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |p_b - \bar{y}_b|$$

An ECE of 0 means perfectly calibrated; 0.05 means the model is off by about 5 percentage points on average.

**Observed-to-Expected Ratio (OTE)** is a simpler aggregate check: the ratio of actual wins to the sum of predicted win probabilities. Above 1.0 means the model is *underpredicting* wins; below 1.0 means overpredicting.

$$OTE = \frac{\sum y_i}{\sum p_i}$$

These three measure related but distinct things. Brier rewards confidence: predicting 0.9 on a team that wins is much better than predicting 0.6 on the same winner. ECE and OTE both capture systematic bias — OTE tells you the *direction*, ECE measures the average *magnitude* across the probability range.

> **A note on sample size:** Estimating win probability from play-by-play data is statistically tricky. [Brill, Yurko, and Wyner (2025)](https://arxiv.org/pdf/2406.16171) showed via simulation that because every play within a game shares the same win/loss outcome, the dependence structure substantially inflates estimator bias and variance — reducing effective sample size well below the raw play count. With 14,100 games and 2.4 million plays, the effective sample size is closer to the game count than the play count. The aggregate findings here are robust to this; the minute-by-minute and overtime analyses should be interpreted with that caveat in mind.

---

<!-- CODE SNIPPET 1: Place after the metrics definitions above -->
<!-- Suggested location: immediately below the metrics section, under a "Computing the Metrics" subheader -->
<!--
### Computing the Metrics

Here's how each metric is computed from a dataframe of play-level predictions:

```python
import numpy as np
import pandas as pd

def brier_score(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def expected_calibration_error(y_true, y_pred, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_pred[mask].mean()
        ece += mask.mean() * abs(bin_conf - bin_acc)
    return ece

def ote(y_true, y_pred):
    return y_true.sum() / y_pred.sum()
```
-->

---

## A Quick Look at the Data

The ESPN play-by-play API returns one row per play, including a home team win probability estimate at that moment. Each game starts with a pre-game probability (before the first snap) and updates throughout.

[*Figure: Example win probability trace for a single game — Ohio State 2025. Probability updates roughly play-by-play from kickoff through final.*]

Some games look like a slow walk from one endpoint to certainty; others are chaotic back-and-forth. Across the full dataset, the distribution of all in-game win probabilities is roughly U-shaped — lots of plays near 0 and 1, which is expected because every game trace ends at one of those endpoints.

The pre-game distribution is where things get interesting.

[*Figure: Histogram of all (left) and pre-game (right) win probabilities.*]

There's a sharp spike near 0.60 — which is exactly where we're going next.

---

<!-- CODE SNIPPET 2: Place after the data section above, under a "Pulling the Data" subheader -->
<!--
### Pulling the Data

The scraper fetches game-level play-by-play from ESPN's API and stores results locally:

```python
from cfb_calibration.scrapers import ESPNPlayByPlayScraper

scraper = ESPNPlayByPlayScraper()

# Fetch all available Division I games for a season
plays = scraper.fetch_season(year=2024)

# plays is a DataFrame with columns:
# game_id, play_id, home_team, away_team, minute, period,
# home_win_prob, home_score, away_score, home_win (actual outcome)
print(plays.shape)         # (N_plays, ...)
print(plays["home_win_prob"].describe())
```

Full scraping instructions and API details are in the [README](https://github.com/jrlewi/college_football_calibration_study).
-->

---

## The Spike at 60%: ESPN Isn't Always Estimating

Out of 14,100 games, **4,034 — about 29%** — had a pre-game probability between 0.55 and 0.65. That's a lot of games landing in the same narrow window. Home teams win college football games at roughly 61% overall, and that spike is sitting right on top of that base rate.

The pattern is almost entirely explained by *which* teams are playing:

| Group | Near 60% | All Others | % Near 60% |
|:---:|:---:|:---:|:---:|
| G5 or FCS | 3,600 | 5,948 | 37.7% |
| Power Five | 434 | 4,118 | 9.5% |

Drilling into FCS specifically — ESPN's group 81 — the concentration is even sharper:

| Subdivision | Near 60% | % Near 60% |
|:---:|:---:|:---:|
| FCS (group 81) | 3,115 | 49.4% |
| FBS (group 80) | 919 | 11.8% |

Nearly half of all FCS pre-game probabilities cluster near 60%. Attendance tells the same story: games near 60% have a median attendance of 7,179; other games median 22,348. Smaller programs generate fewer national games, fewer historical data points, and less signal for ESPN's model.

The interpretation is straightforward: when the model doesn't have enough information to give a meaningful spread, it falls back toward the historical base rate. It's not broken behavior. It's a prior — and a sensible one. But it's also a significant departure from what "real-time win probability" implies.

---

<!-- CODE SNIPPET 3: Place in the spike section above, showing how to identify the 60% cluster -->
<!--
```python
# Identify pre-game plays (minute 0, before first snap)
pregame = plays[plays["minute"] == 0].copy()

# Flag games near the 60% base rate
pregame["near_60"] = pregame["home_win_prob"].between(0.55, 0.65)

# Break down by subdivision
summary = (
    pregame.groupby("subdivision")["near_60"]
    .agg(["sum", "count"])
    .assign(pct_near_60=lambda df: df["sum"] / df["count"])
    .rename(columns={"sum": "near_60", "count": "total"})
)
print(summary)
```
-->

---

## Calibration During Regulation

### The Aggregate View

Setting aside the pre-game spike, how well does the model perform across all 2.4 million plays?

| Metric | Value |
|:---:|:---:|
| Brier Score | 0.119 |
| ECE | 0.006 |
| Pre-game OTE | 1.002 |

An ECE of 0.006 means the model is off by less than a percentage point on average. An OTE of 1.002 means home teams won almost exactly as often as predicted. For context, a model that simply predicts the base rate (~61%) every play would score around 0.24 on Brier — so the gap between 0.119 and that naive baseline is the real signal.

That's a well-calibrated model.

### Does Calibration Hold Through the Game?

**OTE by Minute**

[*Figure: OTE by minute of game (regulation, minutes 1–60). Horizontal reference line at 1.0.*]

Through the first half, OTE hovers close to 1.0. In the second half it creeps upward, reaching around 1.018 by Q4. ESPN is consistently *underpredicting* home team wins by a small margin late in games. The effect is small (~1.5–2%), but it appears systematic.

**Brier Score and ECE by Minute**

[*Figure: Brier score and ECE by minute of regulation.*]

Brier drops steadily as games progress — from ~0.18 in minute 1 to ~0.05 near the final whistle. This is expected: early predictions stay closer to 0.5 because outcomes are uncertain; late predictions move toward 0 or 1 as games resolve. A well-calibrated model will score lower as certainty increases.

ECE tells a different story. It's low (0.001–0.010) through most of the first half, then picks up slightly in the second half (0.012–0.016 in individual minutes). This aligns with the OTE pattern: a modest but real directional bias emerges late, with the model underpredicting home team wins in the final stretch.

### Reliability Diagrams by Quarter

The reliability diagram plots predicted probability on the x-axis against actual win rate on the y-axis. The diagonal is perfect — when the model says 70%, the win rate should be 70%. Points above the diagonal are underconfident; below is overconfident.

[*Figure: 2×3 grid of reliability diagrams by quarter (Q1–Q4, Overtime).*]

Q1 through Q3 all sit close to the diagonal, with a slight break noticeable starting in Q3. Q4 shows a cleaner pattern: when the model gives the home team a lower probability (20–40%), the actual win rate is slightly higher. The model is underpredicting home team win probability late in games — the same signal as before, now visible in the calibration curve.

---

<!-- CODE SNIPPET 4: Place in the reliability diagram section, showing how to generate one -->
<!--
```python
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def plot_reliability_diagram(y_true, y_pred, ax, title="", n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_accs, bin_sizes = [], [], []

    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() < 10:
            continue
        bin_centers.append(y_pred[mask].mean())
        bin_accs.append(y_true[mask].mean())
        bin_sizes.append(mask.sum())

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.scatter(bin_centers, bin_accs, s=[s / 500 for s in bin_sizes],
               alpha=0.8, label="Observed")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title(title)
    ax.legend(fontsize=8)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for ax, (quarter, group) in zip(axes.flat, plays.groupby("period")):
    plot_reliability_diagram(
        group["home_win"].values,
        group["home_win_prob"].values,
        ax=ax,
        title=f"Q{quarter}" if quarter <= 4 else "OT"
    )
plt.tight_layout()
```
-->

---

## When It Breaks: Overtime

Overtime is where the model falls apart.

| Metric | Regulation | Overtime |
|:---:|:---:|:---:|
| Brier Score | 0.119 | 0.225 |
| ECE | 0.006 | 0.081 |

The Brier score nearly doubles. ECE jumps from 0.6% to 8.1% — a 13× increase. The reliability diagram for overtime looks like chaos compared to the clean regulation curves above.

The pattern in the overtime diagram is also directional: if ESPN predicts a low home team win probability (say, 0.20), the actual win rate is closer to 0.35–0.40. When it predicts high (0.80), the actual rate is somewhat lower (~0.75). The model is miscalibrated in both directions.

Three things explain this:

1. **College football overtime is a fundamentally different format** — alternating possessions from the 25-yard line, with forced 2-point conversions after several exchanges. A model trained on regulation play has no native representation of this structure.
2. **The rules have changed over the years**, which means historical overtime data comes from structurally different formats, making any learned model partially wrong by construction.
3. **Overtime is rare** — only 535 of 14,100 games went to overtime. There simply isn't enough data to learn from, and even what exists is heterogeneous.

This isn't an indictment of ESPN's model. It's a hard problem with sparse, structurally inconsistent data. But it's worth knowing that the number on screen during overtime has less than a tenth of the precision it has during regulation.

---

## Bottom Line

ESPN's win probability model is well-calibrated during regulation. The headline numbers (ECE ~0.006, Brier ~0.119 across 2.4 million plays, pre-game OTE of 1.002) are genuinely good.

The more interesting story is what the model reveals about itself in the places where it struggles:

- **~29% of games** have pre-game probabilities anchored near the historical home win rate — not because the model is broken, but because it lacks information to do better for smaller programs.
- **A modest but systematic bias** emerges in Q3 and Q4: home teams win slightly more often than predicted late in games (~1.5–2%).
- **Overtime is a different sport** as far as the model is concerned — ECE jumps 13× and the reliability diagram degrades substantially.

None of these are failures in the traditional sense. Each one is a tell — a consequence of how the model was built, what it was trained on, and where it runs out of signal. The 60% spike is the clearest example: what looks like a sophisticated real-time model is, for a large fraction of games, doing something much simpler under the hood.

---

## References

Brill, R. S., Yurko, R., & Wyner, A. J. (2025). Exploring the difficulty of estimating win probability: a simulation study. *arXiv preprint arXiv:2406.16171*. [https://arxiv.org/abs/2406.16171](https://arxiv.org/abs/2406.16171)

Lopez, M. J. & Matthews, G. J. (2020). Evaluating real-time probabilistic forecasts with application to National Basketball Association outcome prediction. *arXiv preprint arXiv:2010.00781*. [https://arxiv.org/abs/2010.00781](https://arxiv.org/abs/2010.00781)

Stern, H. (2018). Judging win probability models. *Inpredictable*. [https://www.inpredictable.com/2018/01/judging-win-probability-models.html](https://www.inpredictable.com/2018/01/judging-win-probability-models.html)

---

*Data: ESPN unofficial play-by-play API, 2015–2025. FBS + FCS Division I games. Code to reproduce all analyses is available at [github.com/jrlewi/college\_football\_calibration\_study](https://github.com/jrlewi/college_football_calibration_study).*
