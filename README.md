# CSD 4 Attendance Predictor — Blizzard Impact Analysis

## The Problem

On February 22, 2026, the Blizzard of 2026 dumped 22 inches of snow on New York City. Schools closed Monday the 23rd. When they reopened Tuesday the 24th, the question for anyone managing school programs in East Harlem was: **how many kids are actually going to show up?**

That question matters. Tutoring programs like Reading Partners need to know how many students to expect so they can staff sessions, adjust schedules, and plan outreach to families. Meal programs need headcounts. Administrators need to decide whether to hold assessments or push them back.

I built this project to answer that question with data instead of guesswork.

## What This Project Does

I analyzed 84 school days of attendance data across all 18 elementary schools in Community School District 4 (East Harlem), covering October 2025 through February 2026. I combined that with daily weather data from NOAA's Central Park station, then trained three machine learning models to predict attendance based on weather, calendar patterns, and recent trends.

**The prediction: ~75% attendance on February 24th** — about 2,397 of 3,196 students, roughly 415 fewer than a normal day. The 95% confidence interval runs from 72% to 82%.

## Why This Matters

CSD 4 serves a community where 86% of students are economically disadvantaged and 82% are Black or Hispanic. These families are disproportionately affected by weather disruptions — fewer have cars, backup childcare is harder to arrange, and bus routes through East Harlem are among the last to get plowed. A 13-point attendance drop isn't just a number — it's hundreds of kids missing instruction, meals, and support services.

## How This Started

This project grew directly out of my work at Reading Partners. I manage tutoring operations at schools across CSD 4 — coordinating 42 tutor-student pairs, scheduling sessions, and tracking attendance in Salesforce. I also built a Google Apps Script tool that automates session logging for the entire center.

When the Blizzard of 2026 hit, I was standing outside one of my schools on the morning of the 24th, wondering how many kids would actually show up. We had tutors scheduled, sessions planned, and no idea whether we'd be running at full capacity or half-empty. That morning I thought: I have the data, I have the skills — why am I guessing?

So I built this. It started as a question from my real job and turned into a full data science project that answers it.

## Live Dashboard

The project includes an interactive Streamlit dashboard where you can input a weather forecast and get a predicted attendance rate in real time. No code required — just slide the temperature and snowfall sliders and watch the prediction update.

```bash
streamlit run app.py
```

## What's In This Repo

| File | What It Is |
|------|-----------|
| `app.py` | Interactive Streamlit dashboard — input weather, get attendance predictions |
| `main.py` | The full analysis — data loading, statistics, feature engineering, model training, prediction, and 7 visualizations |
| `predictor.py` | Core prediction module — shared by the dashboard, tests, and main script |
| `csd4_k4_attendance.csv` | The dataset: 1,512 records (84 days x 18 schools) with attendance counts and weather data |
| `executive_summary.py` | Generates a polished PDF summary with key findings and figures |
| `analysis_notebook.ipynb` | Jupyter Notebook version with narrative + code + output together |
| `GLOSSARY.md` | Plain-language definitions of every technical term, formula, and concept used |
| `blog_post.md` | LinkedIn-style article explaining the project for a non-technical audience |
| `tests/` | Unit tests for the prediction pipeline (run with `pytest tests/ -v`) |
| `requirements.txt` | Python dependencies needed to run the project |
| `fig1-fig7` | Visualizations (time series, correlations, heatmap, distributions, models, prediction, hypothesis tests) |
| `executive_summary.pdf` | The generated summary (after running `executive_summary.py`) |

## How to Run It

```bash
# 1. Clone the repo
git clone https://github.com/anglinassists/DS4_Attendance_predictor.git
cd DS4_Attendance_predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the main analysis (generates all figures + hypothesis tests)
python main.py

# 4. Launch the interactive dashboard
streamlit run app.py

# 5. Generate the executive summary PDF
python executive_summary.py

# 6. Run the test suite
pytest tests/ -v

# 7. (Optional) Open the notebook
jupyter notebook analysis_notebook.ipynb
```

## The Models

I trained three models and combined them into a weighted ensemble:

- **OLS Linear Regression** — the simplest model, draws a straight line through the data. Fast and interpretable, gives me a baseline.
- **Random Forest** — builds 200 decision trees on random subsets of the data and averages them. Good at capturing non-linear patterns.
- **Gradient Boosting** — builds trees sequentially, each one correcting the mistakes of the last. Usually the most accurate, gets the highest ensemble weight.
- **Ensemble** — a weighted blend (20% OLS, 35% RF, 45% GB) that smooths out each model's weaknesses.

## Key Findings

- **Baseline attendance** in CSD 4 is about 88% on a normal day
- **Temperature is the strongest weather predictor** (r = 0.84 correlation with attendance)
- **Snowfall has a strong negative relationship** (r = -0.51) — more snow, fewer kids
- **Mondays and Fridays** consistently have lower attendance than mid-week days
- **January is the worst month** for attendance (77.8% average), even before the blizzard
- **The blizzard prediction (75%)** represents a 13-point drop from baseline — ~415 extra absences

## Hypothesis Testing: Prediction vs Reality

I didn't just make a prediction — I tested it against what actually happened.

**Test 1: Did the blizzard actually reduce attendance?**
- H0 (null): Feb 24 attendance was normal — the blizzard had no effect
- H1 (alternative): Feb 24 attendance was significantly lower than normal
- Result: **REJECT H0** — paired t-test across 18 schools (t = -12.83, p < 0.0001), confirmed by Wilcoxon signed-rank test (p < 0.0001). Cohen's d = -3.11 (large effect). The blizzard absolutely crushed attendance.

**Test 2: Was the model's prediction accurate?**
- H0 (null): The prediction matches reality (model is accurate)
- H1 (alternative): The prediction is significantly off
- Result: **FAIL TO REJECT H0** — actual attendance (79.3%) falls within the 95% CI [72.2%, 82.1%]. Bootstrap z-test (z = 0.93, p = 0.35) confirms the prediction is not statistically different from actual.

**Scorecard:**

| Test | Result | Evidence |
|------|--------|----------|
| Blizzard effect is real? | YES | Paired t-test p < 0.0001 |
| Effect size is meaningful? | YES | Cohen's d = -3.11 (large) |
| Prediction within 95% CI? | YES | Actual 79.3% in [72.2%, 82.1%] |
| Model statistically accurate? | YES | Bootstrap z-test p = 0.35 |

**Verdict: STRONG SUPPORT** — severe weather causes a meaningful, predictable drop in CSD 4 attendance, and this model is operationally useful for planning around it.

## Recommendations

Based on this analysis, here's what I'd recommend for school programs operating in CSD 4 during severe weather events:

1. **Pre-position resources the day before reopening.** If weather models show a major storm, assume 70-80% attendance on the return day and staff accordingly.
2. **Prioritize family outreach for chronically absent students.** The students most likely to miss post-storm days are the ones already at risk. A targeted text/call the evening before can move the needle.
3. **Delay assessments by at least one day after reopening.** Testing on a 75% attendance day means 25% of students need makeups, which creates scheduling chaos.
4. **Coordinate with meal programs.** If only 2,400 kids show up instead of 2,800, that's 400 meals that either go to waste or 400 kids who don't get fed at home because the family expected school lunch.
5. **Use this model proactively.** Feed in the weather forecast for the week ahead and flag any day where predicted attendance drops below 80%. That gives programs 2-3 days of lead time to adjust.

## Data Sources

- **Attendance data:** NYC DOE daily attendance records for CSD 4 elementary schools (K-4)
- **Enrollment:** NYSED 2023-24 enrollment data (3,196 K-4 students across 18 schools)
- **Weather:** NOAA Central Park daily observations (temperature, precipitation, snowfall, wind)
- **Research basis:** McCormack (Harvard/RFF), Goodman (NBER), NYC Open Data

## About Me

I'm Mark Anglin — a Senior Literacy Intervention AmeriCorps Member at Reading Partners, serving elementary schools in CSD 4 (East Harlem). I built this project because I wanted to answer a real question that came up in my actual work: when weather shuts the city down, how do we plan for the kids who still show up?

At Reading Partners, I manage a center serving 60+ students with 42 tutor-student pairs across multiple sites. I built a Google Apps Script automation tool that my supervisor called "extremely innovative" — this project is the same instinct applied to a bigger problem. I see operational friction, and I build tools to fix it.

I'm currently pursuing my B.S. in Human Resources Management at SUNY Empire State and hold a Certificate in Data Science from Bloom Institute of Technology. I'm looking for roles where I can combine data skills with the kind of people-centered program work I do every day — whether that's in education technology, program management, data analysis, or community operations.

- [LinkedIn](https://linkedin.com/in/markanglin)
- [GitHub](https://github.com/anglinassists)
- mark.anglin24@gmail.com
