# Glossary of Terms

A plain-language guide to every technical term, formula, and concept used in this project. Written so anyone — regardless of background — can follow what's happening.

---

## Data & Statistics Terms

**Dataset / Data**
A collection of information organized in rows and columns, like a spreadsheet. Each row is one observation (in this project, one school on one day). Each column is a piece of information about that observation (like temperature or attendance count).

**CSV (Comma-Separated Values)**
A simple file format for storing tabular data. It's basically a spreadsheet saved as plain text, where commas separate the columns. You can open it in Excel, Google Sheets, or any text editor.

**Record**
One row in the dataset. In this project, one record = one school on one day. So "PS 72 on October 3rd" is one record.

**Aggregation**
Combining multiple records into a summary. For example, I take 18 school records for a single day and sum them up to get one district-wide daily number. That's aggregation.

**Mean (Average)**
Add up all the values and divide by how many there are. If attendance rates for 5 days are 85%, 87%, 90%, 88%, 86%, the mean is (85+87+90+88+86)/5 = 87.2%.

**Median**
The middle value when you sort all values from lowest to highest. Unlike the mean, the median isn't thrown off by extreme values. If one day had 50% attendance due to a blizzard, the median would barely move, but the mean would drop noticeably.

**Standard Deviation (Std Dev)**
A measure of how spread out the values are. A small standard deviation means most days are close to the average. A large one means there's a lot of day-to-day variation. Think of it as "how predictable is attendance?"

**IQR (Interquartile Range)**
The range covering the middle 50% of values. If the IQR is [86%, 90%], that means half of all school days had attendance between 86% and 90%. It's a way of describing "typical" that ignores extreme outliers.

**Correlation**
A number between -1 and +1 that measures how two things move together. +1 means they move in perfect lockstep (temperature goes up, attendance goes up). -1 means they move in opposite directions (snowfall goes up, attendance goes down). 0 means no relationship at all.

**Outlier**
A data point that's far away from the rest. If attendance is usually 85-90% but one day it's 65%, that day is an outlier. It might be a real event (like a blizzard) or a data error.

---

## Machine Learning Terms

**Model**
A mathematical formula that learns patterns from data and uses those patterns to make predictions. I give it historical attendance data (with weather, day of week, etc.) and it figures out the relationship between those inputs and attendance rates.

**Feature**
An input variable that the model uses to make predictions. Temperature, snowfall, day of the week, and "is it a Monday?" are all features. Think of features as the clues the model uses to guess attendance.

**Target (or Label)**
The thing I'm trying to predict. In this project, the target is the attendance rate.

**Training Data**
The historical data the model learns from. I give it 83 days of attendance data (everything except Feb 24) so it can learn the patterns.

**Test Data**
Data the model has never seen, used to check how well it predicts. I hold out Feb 24 as my test — the model makes its prediction, and I compare it to what actually happened.

**Overfitting**
When a model memorizes the training data too well, including the noise and randomness, instead of learning the real patterns. It performs great on training data but poorly on new data. It's like a student who memorizes test answers but can't solve new problems.

**Cross-Validation (CV)**
A technique for testing a model without wasting data. I split the training data into 5 chunks, train on 4, test on 1, and rotate which chunk is the test set. The average score across all 5 rounds gives me a more trustworthy estimate of how good the model really is.

**TimeSeriesSplit**
A special version of cross-validation for time-ordered data. It always trains on earlier data and tests on later data, never the reverse. This prevents "peeking into the future," which would make results artificially good.

**R-squared (R²)**
A score between 0 and 1 that tells me how much of the variation in attendance my model explains. R²=0.80 means the model explains 80% of the ups and downs. Higher is better. R²=1.0 would be a perfect prediction (basically impossible in the real world).

**One-Hot Encoding**
A way to turn text categories into numbers. "Monday" becomes a column that's 1 on Mondays and 0 every other day. "Tuesday" gets its own column, and so on. Models need numbers, not words, so this is a necessary translation step.

**Multicollinearity**
When two or more features are so closely related that they're basically saying the same thing. For example, if I include both "is it Monday?" and all the day-of-week dummy columns, the model gets confused because the information is redundant. I drop one category (drop_first=True) to avoid this.

**Feature Importance**
A score showing how much each feature contributed to the model's predictions. High importance means the model relied heavily on that feature. Low importance means it mostly ignored it. This helps me understand what's actually driving attendance.

**Lag Variable**
A feature that uses a previous time period's value. "rate_lag1" is yesterday's attendance rate used as today's input. The idea is that today's attendance is partly explained by what happened yesterday.

---

## The Three Models

**OLS (Ordinary Least Squares) Linear Regression**
The simplest model. It draws a straight line (or flat surface, in multiple dimensions) through the data that minimizes the total distance between the line and all the data points. It's fast and easy to interpret, but it assumes the relationship is linear — meaning it can't capture curved or complex patterns.

**Random Forest**
A model that builds many decision trees (200 in this project) and averages their predictions. Each tree is trained on a random sample of the data and a random subset of features, so they each see the problem slightly differently. Averaging them out cancels individual errors. Think of it as asking 200 different experts and going with the average opinion.

**Gradient Boosting**
Similar to Random Forest in that it uses many trees, but instead of building them independently, it builds them one at a time. Each new tree specifically targets the mistakes of all the previous trees combined. It's like a student who reviews their wrong answers after each practice test and studies those topics harder.

**Ensemble**
A combination of multiple models. Instead of trusting one model's opinion, I blend all three. I weight Gradient Boosting highest (45%) because it's usually the most accurate, then Random Forest (35%), then OLS (20%). This almost always performs better than any single model alone.

---

## Confidence & Uncertainty

**Confidence Interval (CI)**
A range of values that likely contains the true answer. A 95% CI means: if I repeated this analysis 100 times with slightly different data each time, the true value would fall within this range about 95 times. A wider CI means more uncertainty.

**Bootstrap**
A technique for estimating uncertainty. I take my training data and randomly resample it (pick rows with replacement — so some rows get picked twice, others get skipped) to create a new "fake" dataset. I train a model on each resampled dataset and see how much the predictions vary. Doing this 500 times gives me a good sense of how uncertain my prediction is.

---

## Weather Terms

**Wind Chill**
What the temperature "feels like" when you factor in wind. A 30F day with strong winds might feel like 18F because the wind strips heat from your body faster. This matters for attendance because families decide whether to send kids to school based on how cold it feels, not just the thermometer reading.

**Precipitation**
Any water falling from the sky — rain, sleet, snow, hail. Measured in inches. In this project, precipitation and snowfall are tracked separately because snow has a much bigger impact on transportation and attendance than rain.

---

## Project-Specific Terms

**CSD 4 (Community School District 4)**
A geographic grouping of public schools in East Harlem, Manhattan. NYC is divided into 32 community school districts for administrative purposes. CSD 4 covers roughly 96th to 142nd Street on the east side.

**DBN (District-Borough Number)**
A unique ID code for each NYC public school. For example, "04M012" means District 4, Manhattan borough, school #12. It's like a Social Security number but for schools.

**K-4**
Kindergarten through 4th grade. These are the youngest students in elementary school, typically ages 5-10.

**Attendance Rate**
The percentage of enrolled students who showed up on a given day. If 3,196 students are enrolled and 2,812 showed up, the attendance rate is 2,812 / 3,196 = 88%.

**Baseline**
The "normal" or expected value before any special event. CSD 4's baseline attendance is about 88%. I compare my blizzard prediction against this baseline to measure the impact.

**Attendance Deficit**
The difference between how many kids normally show up and how many actually showed up. If 2,812 usually attend but only 2,397 did, the deficit is 415 students — 415 extra absences beyond what's normal.

**Economically Disadvantaged**
A classification used by the NYC DOE meaning the student's family meets certain low-income thresholds (typically qualifying for free or reduced-price lunch). In CSD 4, 86% of students are classified this way, which matters because low-income families are more affected by weather disruptions (less reliable transportation, fewer backup childcare options).

---

## Hypothesis Testing Terms

**Hypothesis**
A statement you can test with data. In science and data analysis, you always start with two competing hypotheses and use statistics to decide which one the data supports.

**Null Hypothesis (H0)**
The "nothing is happening" hypothesis. It assumes there's no real effect — any difference you see is just random chance. For example: "The blizzard had no effect on attendance; the low turnout was just a normal bad day." You try to disprove this one.

**Alternative Hypothesis (H1)**
The "something IS happening" hypothesis — what you actually think is true. For example: "The blizzard caused attendance to drop significantly below normal." If the data gives you strong enough evidence, you reject H0 in favor of H1.

**p-value**
The probability of seeing your data (or something more extreme) IF the null hypothesis were true. A small p-value (below 0.05) means: "It would be very unlikely to see results this extreme by pure chance, so the null hypothesis is probably wrong." It does NOT tell you how big the effect is — just whether it's real.

**Alpha (Significance Level)**
The threshold you set before testing. The standard is 0.05 (5%). If the p-value is below alpha, you reject the null hypothesis. Think of it as your tolerance for being wrong — a 5% chance you're calling something "real" when it's actually just noise.

**t-test**
A statistical test that checks whether two groups are meaningfully different. A paired t-test (what I use here) compares matched pairs — each school's blizzard-day rate vs its own season average. The "t-statistic" measures how far apart the two groups are, scaled by how much variation there is.

**Wilcoxon Signed-Rank Test**
A backup test that does the same job as the paired t-test but doesn't assume the data follows a bell curve (normal distribution). With only 18 schools, I can't be sure the data is normally distributed, so I run both tests. If both agree, I'm more confident in the result.

**Cohen's d (Effect Size)**
A measure of HOW BIG the difference is, not just whether it's statistically significant. A tiny effect can be "statistically significant" with enough data, but that doesn't mean it matters in practice. Cohen's d thresholds: small (0.2), medium (0.5), large (0.8). In this project, d = -3.11 is an enormous effect — the blizzard didn't just nudge attendance, it hammered it.

**Z-score**
How many standard deviations a value is from the mean. A z-score of 0 means the value is exactly at the average. A z-score of 2 means it's 2 standard deviations above average (pretty unusual). I use this to test whether the actual attendance is consistent with my model's bootstrap predictions.

**MAPE (Mean Absolute Percentage Error)**
A way to express prediction error as a percentage of the actual value. If I predicted 75% and the actual was 79.3%, the MAPE is |75 - 79.3| / 79.3 = 5.5%. Lower is better. Under 10% is generally considered a good prediction for this kind of problem.

**Type I Error (False Positive)**
Concluding there's an effect when there isn't one. Like a fire alarm going off when there's no fire. The alpha level (0.05) controls how often this happens.

**Type II Error (False Negative)**
Missing a real effect — concluding "nothing happened" when something actually did. Like a fire alarm NOT going off during a real fire. This is why I run multiple tests and check effect sizes, not just p-values.

---

## Python / Code Terms

**pandas (pd)**
A Python library for working with tabular data. Think of it as Excel inside Python. A "DataFrame" is pandas' version of a spreadsheet.

**numpy (np)**
A Python library for math and number-crunching. It handles arrays (lists of numbers) very efficiently and provides functions for statistics, linear algebra, and random number generation.

**scikit-learn (sklearn)**
A Python library for machine learning. It provides ready-to-use implementations of models like Linear Regression, Random Forest, and Gradient Boosting, plus tools for evaluating how well they work.

**matplotlib / seaborn**
Python libraries for creating charts and graphs. matplotlib is the foundational library (handles the low-level drawing), and seaborn sits on top of it to make statistical plots look nicer with less code.

**Random Seed**
A starting number for the random number generator. Setting it to a specific value (like 42) means the "random" numbers come out the same every time I run the code. This makes my results reproducible — someone else running the same code will get the same answer.

---

## Formulas Used

**Attendance Rate**
```
attendance_rate = present_students / enrolled_students
```
Example: 2,812 present / 3,196 enrolled = 0.88 (or 88%)

**Mean (Average)**
```
mean = sum of all values / number of values
```

**R-squared**
```
R² = 1 - (sum of squared errors / total variance)
```
In plain English: how much better is my model than just guessing the average every time? R²=0 means my model is no better than guessing the average. R²=1 means perfect prediction.

**Ensemble Prediction**
```
ensemble = (0.20 x OLS_prediction) + (0.35 x RF_prediction) + (0.45 x GB_prediction)
```
A weighted average where better-performing models get more influence.

**95% Confidence Interval (Bootstrap)**
```
Lower bound = 2.5th percentile of 500 bootstrap predictions
Upper bound = 97.5th percentile of 500 bootstrap predictions
```
I cut off the bottom 2.5% and top 2.5% of predictions, and the remaining range is my 95% CI.
