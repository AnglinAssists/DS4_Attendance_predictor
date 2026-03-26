#!/usr/bin/env python3
"""
CSD 4 (East Harlem) K-4 Attendance Analysis & Blizzard Impact Prediction

I'm analyzing NYC DOE daily attendance data for Community School District 4,
grades K-4, from October 2025 through February 2026. The goal is to predict
how badly the Blizzard of 2026 hit attendance on February 24th.

The dataset covers 84 school days across 18 elementary schools (1,512 records total).
I'm using enrollment numbers from NYSED 2023-24 (3,196 K-4 students) and weather
data from NOAA's Central Park station.

I train three models (OLS regression, Random Forest, Gradient Boosting), then
combine them into a weighted ensemble to get the final prediction.
"""

# I need pandas for data manipulation, numpy for math, sklearn for the ML models,
# and matplotlib/seaborn for the charts.
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # I use 'Agg' so matplotlib doesn't try to open a GUI window
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings('ignore')

# I'm setting up output and styling here so I can change it in one place later.
# OUTPUT_DIR is where my CSV lives and where I'll save my figures.
import os
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# seaborn's whitegrid gives me clean gridlines without clutter.
# dpi=150 makes the saved images crisp enough for a presentation.
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# I set a random seed so every time I run this, I get the same results.
# This is important for reproducibility — if someone else runs my code,
# they should see the exact same numbers I did.
np.random.seed(42)

# My color palette. I keep these in a dictionary so I'm not hardcoding
# hex values all over the place.
COLORS = {
    'primary': '#1a5276',
    'secondary': '#2e86c1',
    'accent': '#e74c3c',
    'warm': '#f39c12',
    'green': '#27ae60',
    'purple': '#8e44ad',
    'gray': '#7f8c8d',
}

print(__doc__)


# SECTION 1: LOADING THE DATA
# I'm reading in my CSV file. It has one row per school per day, with
# attendance counts and weather data already merged in.
# parse_dates tells pandas to treat the "date" column as actual dates
# instead of plain text strings. This lets me do date math later.
print("SECTION 1: Loading Dataset")
print("-" * 50)

df = pd.read_csv(f"{OUTPUT_DIR}/csd4_k4_attendance.csv", parse_dates=["date"])
print(f"  Records: {len(df):,}")
print(f"  Schools: {df['dbn'].nunique()}")
print(f"  Dates: {df['date'].nunique()} school days")
print(f"  Period: {df['date'].min().date()} to {df['date'].max().date()}")

# I need to collapse from school-level rows to one row per day.
# .agg() lets me sum up enrollment/present/absent across all 18 schools,
# while grabbing the first weather value (since weather is the same for
# every school on the same day — they're all in East Harlem).
daily = df.groupby("date").agg(
    enrolled=("k4_enrolled", "sum"),
    present=("present", "sum"),
    absent=("absent", "sum"),
    temp_avg=("temp_avg_f", "first"),
    temp_high=("temp_high_f", "first"),
    temp_low=("temp_low_f", "first"),
    snow=("snow_in", "first"),
    precip=("precip_in", "first"),
    wind_chill=("wind_chill_f", "first"),
    wind_speed=("wind_speed_mph", "first"),
    day_of_week=("day_of_week", "first"),
    month=("month", "first"),
    is_return=("is_return_from_break", "first"),
).reset_index()

# The attendance rate is just: how many showed up / how many are enrolled.
# I store it as a decimal (like 0.88) rather than a percentage (88%).
# I'll format it as a percentage when I print it out.
daily["rate"] = daily["present"] / daily["enrolled"]


# SECTION 2: DESCRIPTIVE STATISTICS
# Before I build any models, I want to understand the shape of my data.
# What's the average attendance? How much does it vary? Are there patterns
# by month or day of the week?
print("\n\nSECTION 2: Descriptive Statistics")
print("=" * 60)

stats = {
    'Mean': f"{daily['rate'].mean():.1%}",
    'Median': f"{daily['rate'].median():.1%}",
    'Std Dev': f"{daily['rate'].std():.3f}",
    'Min': f"{daily['rate'].min():.1%} ({daily.loc[daily['rate'].idxmin(), 'date'].strftime('%b %d')})",
    'Max': f"{daily['rate'].max():.1%} ({daily.loc[daily['rate'].idxmax(), 'date'].strftime('%b %d')})",
    'IQR': f"[{daily['rate'].quantile(0.25):.1%}, {daily['rate'].quantile(0.75):.1%}]",
}
for k, v in stats.items():
    print(f"  {k:12s}: {v}")

# I break it down by month to see the seasonal trend.
# In NYC, attendance usually dips in winter — cold weather, flu season, holidays.
print("\n  By Month:")
for m in ["October", "November", "December", "January", "February"]:
    md = daily[daily["month"] == m]
    if len(md):
        print(f"    {m:12s}: {md['rate'].mean():.1%} (n={len(md)})")

# Day-of-week patterns: Mondays and Fridays tend to have lower attendance.
# This is a well-documented pattern in education research.
print("\n  By Day of Week:")
for d in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
    dd = daily[daily["day_of_week"] == d]
    if len(dd):
        print(f"    {d:12s}: {dd['rate'].mean():.1%} (n={len(dd)})")

# The correlation matrix shows me how strongly each weather variable
# relates to attendance. A positive number means "when this goes up,
# attendance goes up too." A negative number means the opposite.
# I expect temperature to be positive (warmer = more kids show up)
# and snow/precip to be negative (bad weather = fewer kids).
print("\n  Correlation Matrix (Weather vs Attendance):")
corr_vars = ['rate', 'temp_avg', 'wind_chill', 'snow', 'precip']
print(daily[corr_vars].corr().round(3).to_string())


# SECTION 3: FEATURE ENGINEERING
# This is where I create the input variables ("features") that my models
# will learn from. I start with the raw weather data, then I create
# additional columns that capture patterns the models might miss otherwise.
print("\n\nSECTION 3: Feature Engineering")
print("-" * 50)

# One-hot encoding turns categorical text into numbers the model can use.
# For example, "Monday" becomes a column that's 1 on Mondays, 0 otherwise.
# drop_first=True avoids a math problem called "multicollinearity" —
# basically, if I know it's not Tue/Wed/Thu/Fri, it must be Monday,
# so I don't need a separate Monday column from the dummies.
dow_dummies = pd.get_dummies(daily['day_of_week'], prefix='dow', drop_first=True)
month_dummies = pd.get_dummies(daily['month'], prefix='month', drop_first=True)

# temp_range captures how much the temperature swung in a day.
# Big swings can mean unstable weather that keeps families home.
daily['temp_range'] = daily['temp_high'] - daily['temp_low']

# Binary flags — these are yes/no columns (1 or 0).
# I'm flagging extreme conditions because the relationship between
# weather and attendance isn't linear. Going from 50F to 40F matters
# less than going from 30F to 20F.
daily['is_freezing'] = (daily['temp_avg'] < 32).astype(int)
daily['is_extreme_cold'] = (daily['wind_chill'] < 20).astype(int)
daily['heavy_snow'] = (daily['snow'] > 3).astype(int)
daily['any_precip'] = (daily['precip'] > 0).astype(int)

# is_return flags the first day back after a break (like winter recess).
# Those days tend to have weird attendance — some families extend vacations.
daily['is_return'] = daily['is_return'].astype(int)

# Monday and Friday flags, because those days have their own attendance patterns.
daily['is_monday'] = (daily['day_of_week'] == 'Monday').astype(int)
daily['is_friday'] = (daily['day_of_week'] == 'Friday').astype(int)

# days_since_start is a simple trend variable. It captures any gradual
# drift in attendance over the school year (like a slow decline as
# winter drags on).
daily['days_since_start'] = (daily['date'] - daily['date'].min()).dt.days

# rate_lag1 is yesterday's attendance rate. Today's attendance is often
# similar to yesterday's — this is called "autocorrelation." If attendance
# was low yesterday, it'll probably be low-ish today too.
# I fill the first day (which has no "yesterday") with the overall average.
daily['rate_lag1'] = daily['rate'].shift(1).fillna(daily['rate'].mean())

# I bundle all my features into one list and one dataframe.
feature_cols = [
    'temp_avg', 'wind_chill', 'snow', 'precip', 'wind_speed',
    'temp_range', 'is_freezing', 'is_extreme_cold', 'heavy_snow',
    'any_precip', 'is_return', 'is_monday', 'is_friday',
    'days_since_start', 'rate_lag1',
]
features = pd.concat([daily[feature_cols], dow_dummies, month_dummies], axis=1)
feature_names = list(features.columns)
print(f"  Total features: {len(feature_names)}")
print(f"  Weather: temp_avg, wind_chill, snow, precip, wind_speed, temp_range")
print(f"  Binary:  is_freezing, is_extreme_cold, heavy_snow, any_precip, is_return")
print(f"  Calendar: day-of-week dummies, month dummies, is_monday, is_friday")
print(f"  Lag:     rate_lag1 (prior day attendance)")

# X is my feature matrix (the inputs), y is my target (attendance rate).
# I split off Feb 24 as my test set — that's the day I'm trying to predict.
# Everything else is training data the models learn from.
X = features.values
y = daily['rate'].values
feb24_mask = daily['date'] == '2026-02-24'
X_train, y_train = X[~feb24_mask], y[~feb24_mask]
X_test, y_test = X[feb24_mask], y[feb24_mask]


# SECTION 4: MODEL TRAINING & EVALUATION
# I'm training three different models and comparing them. Each one
# approaches the problem differently, which is why combining them
# (ensembling) usually beats any single model.
print("\n\nSECTION 4: Model Training & Evaluation")
print("=" * 60)

# TimeSeriesSplit is like regular cross-validation, but it respects
# the order of time. It never trains on future data to predict the past,
# which would be cheating. I use 5 splits.
tscv = TimeSeriesSplit(n_splits=5)
results = {}

# MODEL 1: OLS (Ordinary Least Squares) Linear Regression
# This is the simplest model — it draws a straight line (well, a hyperplane
# in many dimensions) through the data. It's fast, interpretable, and gives
# me a baseline to compare the fancier models against.
lr = LinearRegression()
lr.fit(X_train, y_train)
cv_lr = cross_val_score(lr, X_train, y_train, cv=tscv, scoring='r2')
results['OLS'] = {
    'model': lr,
    'pred': lr.predict(X_test)[0],
    'cv_r2': cv_lr.mean(),
    'train_r2': r2_score(y_train, lr.predict(X_train)),
}

# MODEL 2: Random Forest
# This model builds 200 decision trees, each trained on a random subset
# of the data. Then it averages their predictions. The randomness helps
# prevent overfitting (memorizing the training data instead of learning
# real patterns). max_depth=8 keeps individual trees from getting too
# complex. min_samples_leaf=3 means every leaf node needs at least 3
# data points, which also fights overfitting.
rf = RandomForestRegressor(
    n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
)
rf.fit(X_train, y_train)
cv_rf = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='r2')
results['RF'] = {
    'model': rf,
    'pred': rf.predict(X_test)[0],
    'cv_r2': cv_rf.mean(),
    'train_r2': r2_score(y_train, rf.predict(X_train)),
}

# MODEL 3: Gradient Boosting
# This model also uses many trees, but instead of building them
# independently (like Random Forest), it builds them sequentially.
# Each new tree focuses on correcting the mistakes of the previous ones.
# learning_rate=0.05 means each tree only makes a small correction,
# which helps avoid overshooting. This is usually the most accurate
# of my three models, but it's also the most prone to overfitting.
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    min_samples_leaf=3, random_state=42
)
gb.fit(X_train, y_train)
cv_gb = cross_val_score(gb, X_train, y_train, cv=tscv, scoring='r2')
results['GB'] = {
    'model': gb,
    'pred': gb.predict(X_test)[0],
    'cv_r2': cv_gb.mean(),
    'train_r2': r2_score(y_train, gb.predict(X_train)),
}

# I print each model's performance so I can compare them.
# Train R² is how well it fits the data it learned from.
# CV R² is how well it predicts data it hasn't seen (more trustworthy).
for name, r in results.items():
    print(f"  {name:5s} | Train R²: {r['train_r2']:.4f} | CV R²: {r['cv_r2']:.4f} | Feb 24: {r['pred']:.1%}")

# ENSEMBLE: I combine all three models with weighted averaging.
# I give more weight to Gradient Boosting (0.45) because it usually
# performs best, then Random Forest (0.35), then OLS (0.20).
# The idea is that combining diverse models smooths out each one's
# individual weaknesses.
ensemble = (
    0.20 * results['OLS']['pred']
    + 0.35 * results['RF']['pred']
    + 0.45 * results['GB']['pred']
)

# BOOTSTRAP CONFIDENCE INTERVAL
# I want to know not just my prediction, but how uncertain I am about it.
# Bootstrap works by resampling my training data 500 times (picking random
# rows with replacement), training a new model each time, and seeing how
# much the predictions vary. The middle 95% of those predictions gives me
# my confidence interval.
boot_preds = []
for _ in range(500):
    idx = np.random.choice(len(X_train), len(X_train), replace=True)
    gb_b = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05
    )
    gb_b.fit(X_train[idx], y_train[idx])
    boot_preds.append(gb_b.predict(X_test)[0])

ci_low, ci_high = np.percentile(boot_preds, 2.5), np.percentile(boot_preds, 97.5)
enrolled = daily.loc[feb24_mask, 'enrolled'].iloc[0]


# SECTION 5: FINAL PREDICTION
# This is the payoff — my best guess at how many kids showed up on
# February 24, 2026, the day after the Blizzard of 2026.
print("\n\n" + "=" * 60)
print("FINAL PREDICTION: February 24, 2026")
print("=" * 60)
print(f"""
  Context: Blizzard of 2026 aftermath
  - 22" snow on ground, wind chill 18F
  - MTA running with delays, buses at reduced capacity
  - Schools reopened after Monday snow day
  - CSD 4: 86% economically disadvantaged, 82% Black/Hispanic

  ENSEMBLE PREDICTION:  {ensemble:.1%}
  95% CI: [{ci_low:.1%}, {ci_high:.1%}]

  Expected present:  {int(enrolled * ensemble):,} of {enrolled:,} students
  Expected absent:   {enrolled - int(enrolled * ensemble):,} students
  Normal baseline:   88% ({int(enrolled * 0.88):,} students)
  Attendance deficit: ~{int(enrolled * 0.88) - int(enrolled * ensemble)} additional absences
""")


# SECTION 6: HYPOTHESIS TESTING
# This is the part where I put the model on trial. It's not enough to
# make a prediction — I need to formally test whether the blizzard
# actually caused a statistically significant drop in attendance, and
# whether my model's prediction was accurate enough to be useful.
#
# I'm framing this around a business question:
# "Does severe winter weather cause a meaningful drop in school attendance
#  in CSD 4, and can we predict the magnitude of that drop?"
#
# To answer that, I need two hypothesis tests:
#
# TEST 1 (Did the blizzard actually hurt attendance?):
#   H0 (null): Feb 24 attendance came from the same distribution as normal days.
#              In other words, the blizzard had no real effect.
#   H1 (alternative): Feb 24 attendance was significantly lower than normal.
#
# TEST 2 (Did my model predict it accurately?):
#   H0 (null): The model's prediction is not significantly different from
#              the actual attendance. (This is the one I WANT to keep.)
#   H1 (alternative): The model's prediction is significantly off from reality.

print("\n\n" + "=" * 60)
print("SECTION 6: HYPOTHESIS TESTING")
print("=" * 60)

# First, I grab the actual Feb 24 attendance rate from the data.
actual_rate = daily.loc[feb24_mask, 'rate'].iloc[0]
actual_present = daily.loc[feb24_mask, 'present'].iloc[0]
actual_absent = daily.loc[feb24_mask, 'absent'].iloc[0]
baseline_rate = daily['rate'].mean()

print(f"\n  Actual Feb 24 attendance: {actual_rate:.1%} ({actual_present:,} of {enrolled:,})")
print(f"  Ensemble prediction:     {ensemble:.1%} ({int(enrolled * ensemble):,})")
print(f"  Season baseline (mean):  {baseline_rate:.1%}")

# TEST 1: Did the blizzard significantly reduce attendance?
# I use a one-sample t-test. I'm asking: "Is the Feb 24 rate significantly
# below the distribution of all other days?"
# But since I only have one observation for Feb 24 (one district-wide rate),
# I'll use the school-level data instead — 18 schools give me 18 observations
# for that day, which gives the t-test something to work with.
print("\n  " + "-" * 50)
print("  TEST 1: Did the blizzard reduce attendance?")
print("  " + "-" * 50)
print("  H0: Feb 24 attendance = normal attendance (no blizzard effect)")
print("  H1: Feb 24 attendance < normal attendance (blizzard hurt attendance)")

# I pull the per-school attendance rates for Feb 24.
feb24_school_rates = df.loc[df['date'] == '2026-02-24', 'attendance_rate'].values

# And the per-school mean rates across all other days (each school's typical rate).
non_feb24 = df[df['date'] != '2026-02-24']
school_means = non_feb24.groupby('dbn')['attendance_rate'].mean().values

# One-sample t-test: are the Feb 24 school rates significantly below
# the schools' normal rates? I use a paired approach since each school
# has its own baseline.
t_stat_1, p_value_1_two = scipy_stats.ttest_rel(feb24_school_rates, school_means)
# I want a one-tailed test (is Feb 24 LOWER?), so I halve the p-value
# and only count it if the t-statistic is negative (meaning Feb 24 is lower).
p_value_1 = p_value_1_two / 2 if t_stat_1 < 0 else 1 - p_value_1_two / 2

print(f"\n  Paired t-test (18 schools, Feb 24 vs their season average):")
print(f"    t-statistic: {t_stat_1:.4f}")
print(f"    p-value (one-tailed): {p_value_1:.6f}")

alpha = 0.05
if p_value_1 < alpha:
    print(f"    Result: REJECT H0 (p < {alpha})")
    print(f"    Conclusion: The blizzard caused a statistically significant")
    print(f"    drop in attendance. This wasn't random variation.")
else:
    print(f"    Result: FAIL TO REJECT H0 (p >= {alpha})")
    print(f"    Conclusion: Cannot confirm the blizzard caused the drop.")

# I also calculate the effect size using Cohen's d.
# This tells me not just whether the effect is statistically significant,
# but how BIG it is. A small p-value with a tiny effect size isn't very
# interesting in practice.
diff = feb24_school_rates - school_means
cohens_d = diff.mean() / diff.std()
print(f"\n    Effect size (Cohen's d): {cohens_d:.2f}")
if abs(cohens_d) >= 0.8:
    print(f"    Interpretation: LARGE effect (|d| >= 0.8)")
elif abs(cohens_d) >= 0.5:
    print(f"    Interpretation: MEDIUM effect (|d| >= 0.5)")
else:
    print(f"    Interpretation: SMALL effect (|d| < 0.5)")

# I also run a Wilcoxon signed-rank test as a non-parametric alternative.
# This doesn't assume the data is normally distributed, which is a safer
# assumption with only 18 schools. If both tests agree, I'm more confident.
w_stat, p_value_w_two = scipy_stats.wilcoxon(feb24_school_rates, school_means, alternative='less')
print(f"\n    Wilcoxon signed-rank (non-parametric confirmation):")
print(f"    W-statistic: {w_stat:.1f}, p-value: {p_value_w_two:.6f}")
if p_value_w_two < alpha:
    print(f"    Confirms: Significant drop (p < {alpha})")
else:
    print(f"    Does not confirm significant drop.")


# TEST 2: How accurate was my model?
# Here I'm testing whether the model's prediction was close enough to
# the actual value to be operationally useful.
print("\n  " + "-" * 50)
print("  TEST 2: Model prediction accuracy")
print("  " + "-" * 50)
print("  H0: Predicted attendance = Actual attendance (model is accurate)")
print("  H1: Predicted attendance != Actual attendance (model is off)")

# First, the simple error metrics.
prediction_error = ensemble - actual_rate
abs_error = abs(prediction_error)
pct_error = abs_error / actual_rate
student_error = abs(int(enrolled * ensemble) - actual_present)

print(f"\n  Point Estimate Accuracy:")
print(f"    Predicted:  {ensemble:.1%}")
print(f"    Actual:     {actual_rate:.1%}")
print(f"    Error:      {prediction_error:+.1%} ({prediction_error * 100:+.1f} percentage points)")
print(f"    Abs Error:  {abs_error:.1%} ({abs_error * 100:.1f} pp)")
print(f"    MAPE:       {pct_error:.1%}")
print(f"    Student-level error: {student_error} students")

# Did the actual rate fall within my 95% confidence interval?
# This is a key test — if my CI captured the true value, my uncertainty
# estimate was well-calibrated.
in_ci = ci_low <= actual_rate <= ci_high
print(f"\n  Confidence Interval Check:")
print(f"    95% CI:     [{ci_low:.1%}, {ci_high:.1%}]")
print(f"    Actual:     {actual_rate:.1%}")
print(f"    Within CI:  {'YES' if in_ci else 'NO'}")
if in_ci:
    print(f"    The actual value falls within my predicted range.")
    print(f"    My uncertainty estimate was well-calibrated.")
else:
    print(f"    The actual value fell outside my predicted range.")
    print(f"    My model underestimated the uncertainty.")

# I use the bootstrap distribution to formally test whether the actual
# value is consistent with my model's predictions. I calculate a z-score:
# how many standard deviations is the actual value from the center of
# my bootstrap distribution?
boot_mean = np.mean(boot_preds)
boot_std = np.std(boot_preds)
z_score = (actual_rate - boot_mean) / boot_std
p_value_2 = 2 * (1 - scipy_stats.norm.cdf(abs(z_score)))

print(f"\n  Bootstrap Distribution Test:")
print(f"    Bootstrap mean: {boot_mean:.1%}")
print(f"    Bootstrap std:  {boot_std:.3f}")
print(f"    Z-score:        {z_score:.2f}")
print(f"    p-value:        {p_value_2:.4f}")

if p_value_2 >= alpha:
    print(f"    Result: FAIL TO REJECT H0 (p >= {alpha})")
    print(f"    Conclusion: The model's prediction is NOT significantly")
    print(f"    different from actual. The model performed well.")
else:
    print(f"    Result: REJECT H0 (p < {alpha})")
    print(f"    Conclusion: The model's prediction IS significantly")
    print(f"    different from actual. The model needs improvement.")

# Per-model accuracy breakdown — I want to see which model got closest.
print(f"\n  Per-Model Accuracy:")
print(f"    {'Model':8s} | {'Predicted':>10s} | {'Error':>10s} | {'Abs Error':>10s}")
print(f"    {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10}")
for name, r in results.items():
    pred = r['pred']
    err = pred - actual_rate
    print(f"    {name:8s} | {pred:>9.1%} | {err:>+9.1%} | {abs(err):>9.1%}")
print(f"    {'Ensemble':8s} | {ensemble:>9.1%} | {ensemble - actual_rate:>+9.1%} | {abs(ensemble - actual_rate):>9.1%}")

# Final scorecard — I summarize the hypothesis testing results in one place.
print(f"\n  " + "=" * 50)
print(f"  HYPOTHESIS TESTING SCORECARD")
print(f"  " + "=" * 50)
print(f"  Business Question: Does severe weather cause a meaningful,")
print(f"  predictable drop in CSD 4 attendance?")
print(f"")
print(f"  Test 1 — Blizzard effect is real?      {'YES' if p_value_1 < alpha else 'NO':>5s}  (p={p_value_1:.4f})")
print(f"  Test 1 — Effect size is meaningful?     {'YES' if abs(cohens_d) >= 0.5 else 'NO':>5s}  (d={cohens_d:.2f})")
print(f"  Test 2 — Prediction within CI?          {'YES' if in_ci else 'NO':>5s}  ([{ci_low:.1%}, {ci_high:.1%}])")
print(f"  Test 2 — Prediction statistically close? {'YES' if p_value_2 >= alpha else 'NO':>5s}  (p={p_value_2:.4f})")
print(f"  Prediction error:                       {abs_error * 100:.1f} pp ({student_error} students)")
print(f"")

# I determine the overall verdict based on all the tests together.
blizzard_confirmed = p_value_1 < alpha and abs(cohens_d) >= 0.5
model_accurate = in_ci and p_value_2 >= alpha

if blizzard_confirmed and model_accurate:
    verdict = "STRONG SUPPORT"
    explanation = (
        "The blizzard caused a statistically significant and practically\n"
        "  meaningful drop in attendance, and the model predicted it within\n"
        "  its confidence interval. This model is operationally useful for\n"
        "  planning around severe weather events."
    )
elif blizzard_confirmed and not model_accurate:
    verdict = "PARTIAL SUPPORT"
    explanation = (
        "The blizzard effect is real and significant, but the model's\n"
        "  point estimate was off by more than expected. The model captures\n"
        "  the direction and general magnitude, but needs refinement for\n"
        "  precise headcount planning."
    )
elif not blizzard_confirmed:
    verdict = "WEAK SUPPORT"
    explanation = (
        "The statistical tests did not confirm a significant blizzard\n"
        "  effect, which undermines the premise of the prediction model."
    )

print(f"  VERDICT: {verdict}")
print(f"  {explanation}")
print()


# FIGURE 7: Hypothesis Test Visualization
# I want a clear visual that shows the prediction vs actual, with the
# confidence interval and the baseline, so anyone can see at a glance
# whether the model worked.
fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Bootstrap distribution with actual value marked
ax7a.hist(np.array(boot_preds) * 100, bins=30, color=COLORS['secondary'],
          alpha=0.6, edgecolor='white', label='Bootstrap predictions')
ax7a.axvline(x=actual_rate * 100, color=COLORS['accent'], linewidth=2.5,
             linestyle='-', label=f'Actual: {actual_rate:.1%}')
ax7a.axvline(x=ensemble * 100, color=COLORS['primary'], linewidth=2.5,
             linestyle='--', label=f'Ensemble: {ensemble:.1%}')
ax7a.axvline(x=baseline_rate * 100, color=COLORS['green'], linewidth=2,
             linestyle=':', label=f'Baseline: {baseline_rate:.1%}')

# I shade the 95% CI region
ax7a.axvspan(ci_low * 100, ci_high * 100, alpha=0.15, color=COLORS['warm'],
             label=f'95% CI [{ci_low:.1%}, {ci_high:.1%}]')

ax7a.set_xlabel('Attendance Rate (%)')
ax7a.set_ylabel('Frequency')
ax7a.set_title('Bootstrap Distribution vs Actual Attendance')
ax7a.legend(fontsize=8, loc='upper left')

# Right panel: School-level paired comparison (Feb 24 vs season average)
# I show each school as a dot pair connected by a line.
# This makes the paired t-test result visually obvious.
school_names_short = [s[:15] for s in df.loc[df['date'] == '2026-02-24', 'school_name'].values]
x_pos = np.arange(len(school_names_short))

ax7b.scatter(x_pos, school_means * 100, color=COLORS['green'], s=50,
             zorder=3, label='Season Average', marker='o')
ax7b.scatter(x_pos, feb24_school_rates * 100, color=COLORS['accent'], s=50,
             zorder=3, label='Feb 24 Actual', marker='s')

# I draw lines connecting each school's average to its Feb 24 rate.
# Every line should point downward if the blizzard hurt attendance.
for i in range(len(x_pos)):
    ax7b.plot([x_pos[i], x_pos[i]],
              [school_means[i] * 100, feb24_school_rates[i] * 100],
              color=COLORS['gray'], linewidth=1, alpha=0.6)

ax7b.set_xticks(x_pos)
ax7b.set_xticklabels(school_names_short, rotation=70, ha='right', fontsize=7)
ax7b.set_ylabel('Attendance Rate (%)')
ax7b.set_title(f'School-Level: Season Avg vs Feb 24 (p={p_value_1:.4f})')
ax7b.legend(fontsize=9)

plt.tight_layout()
fig7.savefig(f"{OUTPUT_DIR}/fig7_hypothesis_tests.png", bbox_inches='tight')
plt.close(fig7)
print("  Saved fig7_hypothesis_tests.png")


# SECTION 7: VISUALIZATIONS
# I generate 6 figures that tell the story of this analysis visually.
# Each one answers a different question about the data.

# FIGURE 1: Time Series of Daily Attendance
# This shows attendance over the entire school year, day by day.
# I'm looking for trends, seasonal patterns, and any obvious outliers.
# The blizzard day should stick out like a sore thumb.
fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(daily['date'], daily['rate'] * 100, color=COLORS['primary'],
         linewidth=1.5, alpha=0.8, label='Daily Rate')
ax1.axhline(y=daily['rate'].mean() * 100, color=COLORS['warm'],
            linestyle='--', alpha=0.7, label=f"Mean ({daily['rate'].mean():.1%})")

# I highlight Feb 24 with a red dot so it's easy to spot.
if feb24_mask.any():
    feb24_rate = daily.loc[feb24_mask, 'rate'].iloc[0] * 100
    feb24_date = daily.loc[feb24_mask, 'date'].iloc[0]
    ax1.scatter([feb24_date], [feb24_rate], color=COLORS['accent'],
                s=100, zorder=5, label=f"Feb 24 ({feb24_rate:.1f}%)")

ax1.set_xlabel("Date")
ax1.set_ylabel("Attendance Rate (%)")
ax1.set_title("CSD 4 K-4 Daily Attendance Rate (Oct 2025 - Feb 2026)")
ax1.legend(loc='lower left')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.xticks(rotation=45)
plt.tight_layout()
fig1.savefig(f"{OUTPUT_DIR}/fig1_timeseries.png")
plt.close(fig1)
print("  Saved fig1_timeseries.png")

# FIGURE 2: Weather vs Attendance Scatter Plots
# These four panels show how each weather variable relates to attendance.
# I'm looking for clear trends — does attendance drop as snow increases?
# Does it rise with temperature? The scatter pattern tells me whether
# these are useful features for my models.
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
weather_pairs = [
    ('temp_avg', 'Avg Temperature (F)', COLORS['secondary']),
    ('wind_chill', 'Wind Chill (F)', COLORS['purple']),
    ('snow', 'Snowfall (inches)', COLORS['accent']),
    ('precip', 'Precipitation (inches)', COLORS['green']),
]
for ax, (col, label, color) in zip(axes2.flat, weather_pairs):
    ax.scatter(daily[col], daily['rate'] * 100, alpha=0.6, color=color, s=40)
    # I add a trend line (numpy polyfit degree 1 = straight line)
    # to make the relationship easier to see.
    z = np.polyfit(daily[col], daily['rate'] * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(daily[col].min(), daily[col].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.7)
    corr = daily[col].corr(daily['rate'])
    ax.set_xlabel(label)
    ax.set_ylabel("Attendance Rate (%)")
    ax.set_title(f"{label} vs Attendance (r={corr:.3f})")

plt.suptitle("Weather Variables vs Attendance Rate", fontsize=14, y=1.02)
plt.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/fig2_correlations.png", bbox_inches='tight')
plt.close(fig2)
print("  Saved fig2_correlations.png")

# FIGURE 3: Correlation Heatmap
# This is a matrix that shows how every variable relates to every other
# variable. Dark blue means strong positive correlation, dark red means
# strong negative. I use this to spot which features are most useful
# and which ones are just duplicating each other (multicollinearity).
fig3, ax3 = plt.subplots(figsize=(10, 8))
corr_cols = ['rate', 'temp_avg', 'temp_high', 'temp_low', 'wind_chill',
             'snow', 'precip', 'wind_speed', 'temp_range']
corr_matrix = daily[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax3)
ax3.set_title("Correlation Heatmap: Attendance & Weather Variables")
plt.tight_layout()
fig3.savefig(f"{OUTPUT_DIR}/fig3_heatmap.png")
plt.close(fig3)
print("  Saved fig3_heatmap.png")

# FIGURE 4: Attendance Distributions by Month and Day of Week
# Box plots show me the spread of attendance for each group.
# The box covers the middle 50% of values, the line inside is the median,
# and the whiskers show the full range (minus outliers).
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))

month_order = ["October", "November", "December", "January", "February"]
daily['month_cat'] = pd.Categorical(daily['month'], categories=month_order, ordered=True)
sns.boxplot(data=daily, x='month_cat', y='rate', ax=ax4a,
            palette='coolwarm', order=month_order)
ax4a.set_xlabel("Month")
ax4a.set_ylabel("Attendance Rate")
ax4a.set_title("Attendance Distribution by Month")

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
sns.boxplot(data=daily, x='day_of_week', y='rate', ax=ax4b,
            palette='Set2', order=dow_order)
ax4b.set_xlabel("Day of Week")
ax4b.set_ylabel("Attendance Rate")
ax4b.set_title("Attendance Distribution by Day of Week")

plt.tight_layout()
fig4.savefig(f"{OUTPUT_DIR}/fig4_distributions.png")
plt.close(fig4)
print("  Saved fig4_distributions.png")

# FIGURE 5: Model Comparison
# I show two things side by side:
# Left panel — how each model's predictions compare to the actual values
#   over the training period. Closer to the diagonal line = better.
# Right panel — feature importance from the Gradient Boosting model.
#   This tells me which inputs the model relied on most.
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 5))

# Left: predicted vs actual for each model
for name, color in [('OLS', COLORS['secondary']), ('RF', COLORS['green']), ('GB', COLORS['purple'])]:
    model = results[name]['model']
    y_pred_train = model.predict(X_train)
    ax5a.scatter(y_train * 100, y_pred_train * 100, alpha=0.4, s=30,
                 color=color, label=f"{name} (R²={results[name]['cv_r2']:.3f})")

# I draw a diagonal reference line — if predictions were perfect,
# all dots would sit on this line.
ax5a.plot([80, 95], [80, 95], 'k--', alpha=0.5, label='Perfect prediction')
ax5a.set_xlabel("Actual Attendance (%)")
ax5a.set_ylabel("Predicted Attendance (%)")
ax5a.set_title("Model Predictions vs Actual (Training Data)")
ax5a.legend(fontsize=9)

# Right: feature importance from Gradient Boosting.
# Higher importance = the model used this feature more to make decisions.
importances = gb.feature_importances_
sorted_idx = np.argsort(importances)[-15:]  # top 15 features
ax5b.barh(range(len(sorted_idx)), importances[sorted_idx], color=COLORS['primary'])
ax5b.set_yticks(range(len(sorted_idx)))
ax5b.set_yticklabels([feature_names[i] for i in sorted_idx])
ax5b.set_xlabel("Feature Importance")
ax5b.set_title("Top 15 Features (Gradient Boosting)")

plt.tight_layout()
fig5.savefig(f"{OUTPUT_DIR}/fig5_models.png")
plt.close(fig5)
print("  Saved fig5_models.png")

# FIGURE 6: February 24 Prediction Summary
# This is a visual summary card for the final prediction.
# I show the ensemble prediction, individual model predictions,
# and the confidence interval on a number line so it's easy to
# see how much uncertainty there is.
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.set_xlim(65, 100)
ax6.set_ylim(0, 10)
ax6.axis('off')

# Title
ax6.text(82.5, 9.2, "February 24, 2026 — Attendance Prediction",
         ha='center', fontsize=16, fontweight='bold', color=COLORS['primary'])

# I draw a horizontal number line showing the prediction range.
ax6.axhline(y=5, xmin=0.05, xmax=0.95, color='gray', linewidth=2, alpha=0.3)

# Confidence interval as a shaded band
ax6.axvspan(ci_low * 100, ci_high * 100, ymin=0.35, ymax=0.65,
            alpha=0.2, color=COLORS['secondary'], label='95% CI')

# Individual model predictions as dots
for name, marker, yoff in [('OLS', 's', 5.8), ('RF', '^', 5.8), ('GB', 'D', 5.8)]:
    pred = results[name]['pred'] * 100
    ax6.plot(pred, yoff, marker=marker, markersize=10, color=COLORS['gray'],
             label=f"{name}: {pred:.1f}%")

# Ensemble prediction as a big red dot
ax6.plot(ensemble * 100, 5, 'o', markersize=15, color=COLORS['accent'],
         zorder=5, label=f"Ensemble: {ensemble:.1f}%")

# Baseline marker
ax6.axvline(x=88, ymin=0.25, ymax=0.75, color=COLORS['green'],
            linestyle='--', linewidth=2, alpha=0.7)
ax6.text(88, 3.5, f"Baseline\n88%", ha='center', fontsize=10,
         color=COLORS['green'], fontweight='bold')

# Annotation
ax6.text(82.5, 1.5,
         f"Ensemble: {ensemble:.1%}  |  95% CI: [{ci_low:.1%}, {ci_high:.1%}]  |  "
         f"~{int(enrolled * 0.88) - int(enrolled * ensemble)} extra absences",
         ha='center', fontsize=11, color=COLORS['primary'],
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax6.legend(loc='upper left', fontsize=9)
plt.tight_layout()
fig6.savefig(f"{OUTPUT_DIR}/fig6_feb24.png")
plt.close(fig6)
print("  Saved fig6_feb24.png")

# SECTION 7: RECOMMENDATIONS
# This is where I connect the numbers to real decisions. A prediction
# is only useful if someone can act on it. I'm thinking about this
# from the perspective of someone running a school program in CSD 4 —
# what would I want to know, and what would I do differently?
print("\n\n" + "=" * 60)
print("RECOMMENDATIONS")
print("=" * 60)
print("""
  Based on this analysis, here's what I'd recommend for school programs
  operating in CSD 4 during severe weather events:

  1. PRE-POSITION RESOURCES THE DAY BEFORE REOPENING
     If weather models show a major storm, assume 70-80% attendance on
     the return day and staff accordingly. Don't wait until morning
     headcounts to adjust — by then it's too late to reassign tutors
     or cancel sessions.

  2. PRIORITIZE OUTREACH FOR CHRONICALLY ABSENT STUDENTS
     The students most likely to miss post-storm days are the ones
     already at risk of chronic absenteeism. A targeted text or call
     the evening before reopening can move the needle. In my experience
     at Reading Partners, families respond when they know someone is
     expecting their child.

  3. DELAY ASSESSMENTS BY AT LEAST ONE DAY
     Testing on a 75% attendance day means 25% of students need makeups,
     which creates scheduling chaos for weeks. Push assessments back one
     day and use the return day for review and re-engagement instead.

  4. COORDINATE WITH MEAL PROGRAMS
     If only 2,400 kids show up instead of 2,800, that's 400 meals
     that either go to waste or 400 kids who don't get fed at home
     because the family expected school lunch. Meal programs need the
     same forecasts that academic programs do.

  5. USE THIS MODEL PROACTIVELY
     Feed in the weather forecast for the week ahead and flag any day
     where predicted attendance drops below 80%. That gives programs
     2-3 days of lead time to adjust staffing, outreach, and logistics.

  The bottom line: weather impacts on attendance aren't random — they're
  predictable. And if they're predictable, programs can plan for them
  instead of reacting to them.
""")

print("Script complete. All 6 figures saved to output directory.")
