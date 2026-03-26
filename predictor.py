"""
I pulled the core prediction logic into this module so the dashboard,
tests, and main.py can all share the same functions. This avoids
duplicating code across files.

The main pieces:
- load_data(): reads the CSV and aggregates to daily level
- engineer_features(): creates all the input features from raw data
- train_models(): trains OLS, RF, GB and returns the fitted models
- predict_attendance(): takes weather inputs and returns a prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# I store the path to the CSV relative to this file so it works
# regardless of where the script is run from.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "csd4_k4_attendance.csv")


def load_data(csv_path=None):
    """
    I load the CSV and aggregate from school-level rows to one row per day.
    Returns both the raw school-level dataframe and the daily aggregated one.
    """
    if csv_path is None:
        csv_path = CSV_PATH

    df = pd.read_csv(csv_path, parse_dates=["date"])

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

    daily["rate"] = daily["present"] / daily["enrolled"]

    return df, daily


def engineer_features(daily):
    """
    I create all the input features from the daily dataframe.
    Returns the feature matrix X, target array y, feature column names,
    and the modified daily dataframe.

    I make a copy so I don't mutate the original — this matters for
    testing and for the dashboard where I call this multiple times.
    """
    daily = daily.copy()

    # One-hot encoding for day of week and month.
    dow_dummies = pd.get_dummies(daily['day_of_week'], prefix='dow', drop_first=True)
    month_dummies = pd.get_dummies(daily['month'], prefix='month', drop_first=True)

    # Engineered weather features.
    daily['temp_range'] = daily['temp_high'] - daily['temp_low']
    daily['is_freezing'] = (daily['temp_avg'] < 32).astype(int)
    daily['is_extreme_cold'] = (daily['wind_chill'] < 20).astype(int)
    daily['heavy_snow'] = (daily['snow'] > 3).astype(int)
    daily['any_precip'] = (daily['precip'] > 0).astype(int)
    daily['is_return'] = daily['is_return'].astype(int)
    daily['is_monday'] = (daily['day_of_week'] == 'Monday').astype(int)
    daily['is_friday'] = (daily['day_of_week'] == 'Friday').astype(int)
    daily['days_since_start'] = (daily['date'] - daily['date'].min()).dt.days
    daily['rate_lag1'] = daily['rate'].shift(1).fillna(daily['rate'].mean())

    feature_cols = [
        'temp_avg', 'wind_chill', 'snow', 'precip', 'wind_speed',
        'temp_range', 'is_freezing', 'is_extreme_cold', 'heavy_snow',
        'any_precip', 'is_return', 'is_monday', 'is_friday',
        'days_since_start', 'rate_lag1',
    ]
    features = pd.concat([daily[feature_cols], dow_dummies, month_dummies], axis=1)
    feature_names = list(features.columns)

    X = features.values
    y = daily['rate'].values

    return X, y, feature_names, daily


def train_models(X_train, y_train, random_state=42):
    """
    I train the three models and return them in a dictionary.
    Each entry has the fitted model object.
    """
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    gb = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_samples_leaf=3, random_state=random_state
    )
    gb.fit(X_train, y_train)

    return {'OLS': lr, 'RF': rf, 'GB': gb}


def ensemble_predict(models, X, weights=None):
    """
    I combine all three model predictions using weighted averaging.
    Default weights: 20% OLS, 35% RF, 45% GB.
    """
    if weights is None:
        weights = {'OLS': 0.20, 'RF': 0.35, 'GB': 0.45}

    prediction = sum(
        weights[name] * model.predict(X)[0]
        for name, model in models.items()
    )
    return prediction


def build_scenario_features(temp_avg, wind_chill, snow, precip, wind_speed,
                            day_of_week, month, is_return_from_break,
                            daily_ref):
    """
    I build a single-row feature vector for a hypothetical scenario.
    This is what the dashboard uses — the user enters weather conditions
    and I construct the same features the model was trained on.

    daily_ref is the training daily dataframe, which I need for:
    - the lag value (I use the last known attendance rate)
    - the days_since_start reference point
    - ensuring the dummy columns match the training data
    """
    temp_high = temp_avg + 5  # I approximate if not given exact values
    temp_low = temp_avg - 5
    temp_range = temp_high - temp_low

    # I use the most recent attendance rate as the lag value.
    rate_lag1 = daily_ref['rate'].iloc[-1]

    # days_since_start: I assume we're roughly at the end of the dataset.
    days_since_start = daily_ref['days_since_start'].max() if 'days_since_start' in daily_ref.columns else 145

    row = {
        'temp_avg': temp_avg,
        'wind_chill': wind_chill,
        'snow': snow,
        'precip': precip,
        'wind_speed': wind_speed,
        'temp_range': temp_range,
        'is_freezing': int(temp_avg < 32),
        'is_extreme_cold': int(wind_chill < 20),
        'heavy_snow': int(snow > 3),
        'any_precip': int(precip > 0),
        'is_return': int(is_return_from_break),
        'is_monday': int(day_of_week == 'Monday'),
        'is_friday': int(day_of_week == 'Friday'),
        'days_since_start': days_since_start,
        'rate_lag1': rate_lag1,
    }

    # I need to match the exact dummy columns from training.
    # pandas drop_first=True drops the first alphabetically.
    # For days: Friday is dropped (alphabetically first).
    # For months: December is dropped (alphabetically first).
    dow_kept = ['Monday', 'Thursday', 'Tuesday', 'Wednesday']
    for d in dow_kept:
        row[f'dow_{d}'] = int(day_of_week == d)

    month_kept = ['February', 'January', 'November', 'October']
    for m in month_kept:
        row[f'month_{m}'] = int(month == m)

    return row


def get_trained_pipeline(csv_path=None):
    """
    I load data, engineer features, train models, and return everything
    the dashboard needs in one call. This avoids the dashboard having
    to orchestrate the steps itself.
    """
    df, daily = load_data(csv_path)
    X, y, feature_names, daily_eng = engineer_features(daily)

    # I train on all data (the dashboard is for forecasting, not backtesting).
    models = train_models(X, y)

    return {
        'df': df,
        'daily': daily,
        'daily_eng': daily_eng,
        'X': X,
        'y': y,
        'feature_names': feature_names,
        'models': models,
        'enrolled': daily['enrolled'].iloc[-1],
    }
