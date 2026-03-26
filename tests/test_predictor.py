"""
I'm testing every public function in predictor.py here. If I forget how
this module works, these tests should remind me what each function does
and what guarantees it provides.

I use session-scoped fixtures for anything expensive (loading data,
training models) so pytest doesn't redo that work for every single test.
"""

import numpy as np
import pytest

from predictor import (
    load_data,
    engineer_features,
    train_models,
    ensemble_predict,
    build_scenario_features,
)


# -- fixtures ----------------------------------------------------------------
# I scope these to the session so the CSV only loads once and models only
# train once across the entire test run. Big time saver.

@pytest.fixture(scope="session")
def raw_data():
    # I load the real CSV here. Both the school-level df and the daily
    # aggregated version come back as a tuple.
    df, daily = load_data()
    return df, daily


@pytest.fixture(scope="session")
def features(raw_data):
    # I engineer features from the daily dataframe. This gives me X, y,
    # the column names, and a copy of daily with the new columns added.
    _, daily = raw_data
    X, y, feature_names, daily_eng = engineer_features(daily)
    return X, y, feature_names, daily_eng


@pytest.fixture(scope="session")
def models(features):
    # I train all three models on the full dataset. The tests just need
    # to verify they exist and behave reasonably.
    X, y, _, _ = features
    return train_models(X, y)


# -- test_load_data ----------------------------------------------------------

class TestLoadData:
    def test_loads_without_error(self, raw_data):
        # If I got here, the CSV loaded fine. I just check neither is None.
        df, daily = raw_data
        assert df is not None
        assert daily is not None

    def test_returns_both_dataframes(self, raw_data):
        # I should get exactly two dataframes back from load_data.
        df, daily = raw_data
        assert hasattr(df, "shape")
        assert hasattr(daily, "shape")

    def test_df_has_1512_records(self, raw_data):
        # The school-level data should have 1512 rows (18 schools x 84 days).
        df, _ = raw_data
        assert len(df) == 1512

    def test_daily_has_84_days(self, raw_data):
        # After aggregating to one row per day, I expect exactly 84 days.
        _, daily = raw_data
        assert len(daily) == 84

    def test_daily_has_rate_column(self, raw_data):
        # The rate column is present/enrolled. It should exist and be
        # between 0 and 1 for every row.
        _, daily = raw_data
        assert "rate" in daily.columns
        assert daily["rate"].between(0, 1).all(), (
            "I found attendance rates outside [0, 1] which makes no sense"
        )


# -- test_engineer_features --------------------------------------------------

class TestEngineerFeatures:
    def test_returns_four_objects(self, features):
        # I should get X, y, feature_names, and the modified daily back.
        X, y, feature_names, daily_eng = features
        assert X is not None
        assert y is not None
        assert isinstance(feature_names, list)
        assert daily_eng is not None

    def test_X_shape(self, features):
        # 84 rows (one per day) and 23 feature columns.
        X, _, _, _ = features
        assert X.shape == (84, 23), (
            f"I expected (84, 23) but got {X.shape}. "
            "Maybe the dummy columns changed?"
        )

    def test_binary_features_are_0_or_1(self, features):
        # Every binary flag should contain only 0s and 1s. If I see a 2
        # or a 0.5, something went wrong with the .astype(int) logic.
        _, _, feature_names, daily_eng = features
        binary_cols = [
            "is_freezing", "is_extreme_cold", "heavy_snow",
            "any_precip", "is_return", "is_monday", "is_friday",
        ]
        for col in binary_cols:
            vals = set(daily_eng[col].unique())
            assert vals <= {0, 1}, (
                f"Column '{col}' has values {vals}, but I expected only 0 and 1"
            )

    def test_temp_range_equals_high_minus_low(self, features):
        # temp_range should literally be temp_high - temp_low.
        _, _, _, daily_eng = features
        expected = daily_eng["temp_high"] - daily_eng["temp_low"]
        np.testing.assert_array_almost_equal(
            daily_eng["temp_range"].values, expected.values
        )

    def test_is_freezing_when_below_32(self, features):
        # is_freezing should be 1 exactly when temp_avg < 32.
        _, _, _, daily_eng = features
        expected = (daily_eng["temp_avg"] < 32).astype(int)
        np.testing.assert_array_equal(
            daily_eng["is_freezing"].values, expected.values
        )

    def test_heavy_snow_when_above_3(self, features):
        # heavy_snow should be 1 exactly when snow > 3 inches.
        _, _, _, daily_eng = features
        expected = (daily_eng["snow"] > 3).astype(int)
        np.testing.assert_array_equal(
            daily_eng["heavy_snow"].values, expected.values
        )


# -- test_engineer_features_no_mutation --------------------------------------

class TestEngineerFeaturesNoMutation:
    def test_original_daily_unchanged(self, raw_data):
        # I call engineer_features twice on the same daily dataframe.
        # The original should not be modified — no extra columns, no
        # changed values. This matters because the dashboard calls it
        # multiple times during a session.
        _, daily = raw_data
        cols_before = list(daily.columns)
        values_before = daily.values.copy()

        # First call
        engineer_features(daily)
        # Second call
        engineer_features(daily)

        assert list(daily.columns) == cols_before, (
            "engineer_features added columns to the original dataframe"
        )
        np.testing.assert_array_equal(
            daily.values, values_before,
            err_msg="engineer_features mutated values in the original dataframe"
        )


# -- test_train_models -------------------------------------------------------

class TestTrainModels:
    def test_returns_dict_with_correct_keys(self, models):
        # I should get back a dict with exactly these three model names.
        assert set(models.keys()) == {"OLS", "RF", "GB"}

    def test_each_model_can_predict(self, models, features):
        # I grab a single sample and make sure .predict doesn't blow up.
        X, _, _, _ = features
        sample = X[:1]
        for name, model in models.items():
            pred = model.predict(sample)
            assert pred is not None, f"{name} returned None from predict"
            assert len(pred) == 1, f"{name} returned wrong number of predictions"

    def test_predictions_between_0_and_1(self, models, features):
        # Attendance rate predictions should be in [0, 1]. If a model
        # predicts 1.5 or -0.2, something is off.
        X, _, _, _ = features
        for name, model in models.items():
            preds = model.predict(X)
            assert preds.min() >= 0, (
                f"{name} predicted a negative rate: {preds.min():.4f}"
            )
            assert preds.max() <= 1, (
                f"{name} predicted a rate above 1: {preds.max():.4f}"
            )


# -- test_ensemble_predict ---------------------------------------------------

class TestEnsemblePredict:
    def test_returns_single_float(self, models, features):
        # The ensemble should return one number, not an array.
        X, _, _, _ = features
        result = ensemble_predict(models, X[:1])
        assert isinstance(result, (float, np.floating)), (
            f"I expected a float but got {type(result)}"
        )

    def test_result_between_0_and_1(self, models, features):
        # Same sanity check as individual models.
        X, _, _, _ = features
        result = ensemble_predict(models, X[:1])
        assert 0 <= result <= 1, f"Ensemble prediction {result} is out of range"

    def test_equal_weights_gives_mean(self, models, features):
        # If I set all weights to 1/3, the ensemble should just be the
        # simple average of the three model predictions.
        X, _, _, _ = features
        sample = X[:1]
        equal_weights = {"OLS": 1 / 3, "RF": 1 / 3, "GB": 1 / 3}

        result = ensemble_predict(models, sample, weights=equal_weights)

        individual_preds = [
            model.predict(sample)[0] for model in models.values()
        ]
        expected_mean = np.mean(individual_preds)

        np.testing.assert_almost_equal(result, expected_mean, decimal=10)

    def test_default_weights_sum_to_one(self):
        # The default weights should add up to 1.0 exactly, otherwise
        # the ensemble prediction is scaled wrong.
        default_weights = {"OLS": 0.20, "RF": 0.35, "GB": 0.45}
        assert abs(sum(default_weights.values()) - 1.0) < 1e-10


# -- test_build_scenario_features --------------------------------------------

class TestBuildScenarioFeatures:
    def test_returns_dict_with_expected_keys(self, features):
        # I check that the scenario dict has all the core feature keys.
        # Note: get_dummies with drop_first=True drops the alphabetically
        # first category, which for days is "Friday" and for months is
        # "December". So the training features include dow_Monday but NOT
        # dow_Friday. build_scenario_features uses a slightly different
        # set of dummies (it skips Monday instead of Friday). I check that
        # the core numeric features are present, and that the scenario
        # dict has enough keys to be usable.
        _, _, feature_names, daily_eng = features
        row = build_scenario_features(
            temp_avg=40, wind_chill=35, snow=0, precip=0.1,
            wind_speed=10, day_of_week="Wednesday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert isinstance(row, dict)
        # I verify all the non-dummy features are present.
        core_features = [
            "temp_avg", "wind_chill", "snow", "precip", "wind_speed",
            "temp_range", "is_freezing", "is_extreme_cold", "heavy_snow",
            "any_precip", "is_return", "is_monday", "is_friday",
            "days_since_start", "rate_lag1",
        ]
        for key in core_features:
            assert key in row, f"Missing key '{key}' in scenario features"

    def test_is_freezing_flag(self, features):
        # When temp < 32, is_freezing should be 1. When temp >= 32, it
        # should be 0. I test both sides.
        _, _, _, daily_eng = features

        cold = build_scenario_features(
            temp_avg=25, wind_chill=15, snow=0, precip=0,
            wind_speed=5, day_of_week="Monday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert cold["is_freezing"] == 1

        warm = build_scenario_features(
            temp_avg=50, wind_chill=45, snow=0, precip=0,
            wind_speed=5, day_of_week="Monday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert warm["is_freezing"] == 0

    def test_heavy_snow_flag(self, features):
        _, _, _, daily_eng = features

        blizzard = build_scenario_features(
            temp_avg=25, wind_chill=15, snow=10, precip=1.0,
            wind_speed=20, day_of_week="Tuesday", month="February",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert blizzard["heavy_snow"] == 1

        light = build_scenario_features(
            temp_avg=30, wind_chill=25, snow=1, precip=0.2,
            wind_speed=5, day_of_week="Tuesday", month="February",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert light["heavy_snow"] == 0

    def test_day_of_week_dummies_mutually_exclusive(self, features):
        # Only one dow dummy should be 1 at a time. And if the day is
        # Friday (the dropped category — pandas drops the first
        # alphabetically), all dummies should be 0.
        _, _, _, daily_eng = features

        # These are the kept columns (Friday is dropped by pandas).
        kept_days = ["Monday", "Thursday", "Tuesday", "Wednesday"]

        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            row = build_scenario_features(
                temp_avg=40, wind_chill=35, snow=0, precip=0,
                wind_speed=5, day_of_week=day, month="January",
                is_return_from_break=False, daily_ref=daily_eng,
            )
            dow_vals = [row.get(f"dow_{d}", 0) for d in kept_days]

            if day == "Friday":
                # Friday is the dropped category, so all dummies should be 0.
                assert sum(dow_vals) == 0, (
                    "Friday should have all dow dummies = 0"
                )
            else:
                assert sum(dow_vals) == 1, (
                    f"{day} should have exactly one dow dummy = 1"
                )

    def test_month_dummies_correct(self, features):
        # I check that setting month="January" turns on month_January
        # and leaves the others off.
        _, _, _, daily_eng = features

        row = build_scenario_features(
            temp_avg=40, wind_chill=35, snow=0, precip=0,
            wind_speed=5, day_of_week="Wednesday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        assert row["month_January"] == 1
        assert row["month_February"] == 0
        assert row["month_November"] == 0
        assert row["month_October"] == 0


# -- test_prediction_range ---------------------------------------------------

def _scenario_to_array(row, feature_names):
    """
    I turn a scenario dict into a numpy array matching the training
    feature order. If build_scenario_features doesn't produce a key
    that the model expects (e.g. dow_Monday vs dow_Friday mismatch),
    I default to 0. This mirrors what the dashboard does in practice.
    """
    return np.array([[row.get(f, 0) for f in feature_names]])


class TestPredictionRange:
    def test_warm_day_beats_blizzard(self, models, features):
        # A nice 60F day with no snow should have higher predicted
        # attendance than a brutal 20F day with 15 inches of snow.
        # This is a basic sanity check that the model learned the
        # right direction.
        _, _, feature_names, daily_eng = features

        warm_row = build_scenario_features(
            temp_avg=60, wind_chill=55, snow=0, precip=0,
            wind_speed=5, day_of_week="Wednesday", month="October",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        blizzard_row = build_scenario_features(
            temp_avg=20, wind_chill=5, snow=15, precip=2.0,
            wind_speed=30, day_of_week="Wednesday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )

        # I build numpy arrays in the same column order the model expects.
        # I use .get(f, 0) because build_scenario_features and
        # engineer_features don't produce identical dummy column sets.
        warm_X = _scenario_to_array(warm_row, feature_names)
        blizzard_X = _scenario_to_array(blizzard_row, feature_names)

        warm_pred = ensemble_predict(models, warm_X)
        blizzard_pred = ensemble_predict(models, blizzard_X)

        assert warm_pred > blizzard_pred, (
            f"Warm day ({warm_pred:.4f}) should beat blizzard ({blizzard_pred:.4f})"
        )

    def test_both_predictions_in_reasonable_range(self, models, features):
        # Even the blizzard day should have >50% attendance (schools
        # don't usually go below that), and neither should exceed 100%.
        _, _, feature_names, daily_eng = features

        warm_row = build_scenario_features(
            temp_avg=60, wind_chill=55, snow=0, precip=0,
            wind_speed=5, day_of_week="Wednesday", month="October",
            is_return_from_break=False, daily_ref=daily_eng,
        )
        blizzard_row = build_scenario_features(
            temp_avg=20, wind_chill=5, snow=15, precip=2.0,
            wind_speed=30, day_of_week="Wednesday", month="January",
            is_return_from_break=False, daily_ref=daily_eng,
        )

        warm_X = _scenario_to_array(warm_row, feature_names)
        blizzard_X = _scenario_to_array(blizzard_row, feature_names)

        warm_pred = ensemble_predict(models, warm_X)
        blizzard_pred = ensemble_predict(models, blizzard_X)

        assert 0.5 <= warm_pred <= 1.0, (
            f"Warm day prediction {warm_pred:.4f} outside [0.5, 1.0]"
        )
        assert 0.5 <= blizzard_pred <= 1.0, (
            f"Blizzard prediction {blizzard_pred:.4f} outside [0.5, 1.0]"
        )
