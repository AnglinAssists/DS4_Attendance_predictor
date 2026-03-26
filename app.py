"""
I built this Streamlit dashboard so that anyone at Reading Partners NYC
(or CSD 4 school leaders) can plug in a weather forecast and instantly
see the predicted attendance rate for K-4 schools in East Harlem.

To run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd

# I import the three functions I need from predictor.py, which lives in
# the same directory. get_trained_pipeline gives me everything at once,
# build_scenario_features turns user inputs into a feature row, and
# ensemble_predict combines all three model predictions.
from predictor import get_trained_pipeline, build_scenario_features, ensemble_predict

# I set the page config first because Streamlit requires it before any
# other st.* calls. Wide layout gives more room for charts.
st.set_page_config(
    page_title="CSD 4 Attendance Predictor",
    page_icon="🏫",
    layout="wide",
)

# I define my color palette here so I can reference them throughout
# the app without magic strings everywhere.
PRIMARY = "#1a5276"
ACCENT = "#e74c3c"
GREEN = "#27ae60"
YELLOW = "#f39c12"
ORANGE = "#e67e22"
RED = "#e74c3c"

# I inject custom CSS to make the metric cards and overall look cleaner.
# Streamlit's default styling is fine but I want the metrics to pop more.
st.markdown(f"""
<style>
    .main-title {{
        color: {PRIMARY};
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0;
    }}
    .subtitle {{
        color: #5d6d7e;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border-left: 5px solid {PRIMARY};
    }}
    .metric-value {{
        font-size: 2.4rem;
        font-weight: 700;
        color: {PRIMARY};
    }}
    .metric-label {{
        font-size: 0.9rem;
        color: #5d6d7e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .warning-bar {{
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
    }}
    .model-table {{
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """
    I cache this with st.cache_resource so the models only train once,
    no matter how many times the user tweaks the sliders. Training
    three models on ~150 rows is fast, but there's no reason to redo it
    on every interaction.
    """
    return get_trained_pipeline()


# I load the pipeline right away. On first run this trains the models;
# after that Streamlit serves the cached version.
pipeline = load_pipeline()

# I pull out the pieces I need from the pipeline dict so the code below
# is easier to read.
models = pipeline['models']
feature_names = pipeline['feature_names']
daily_eng = pipeline['daily_eng']
daily = pipeline['daily']
enrolled = pipeline['enrolled']

# --- Sidebar: Weather Inputs ---
# I put all the inputs in the sidebar so the main area stays clean
# and focused on results. The user can adjust these and the prediction
# updates instantly because Streamlit reruns the script on every change.
st.sidebar.markdown(f"<h2 style='color:{PRIMARY}'>Weather Conditions</h2>",
                    unsafe_allow_html=True)

# I chose slider ranges based on realistic NYC winter weather.
# Default values represent a typical cold but not extreme day.
temp_avg = st.sidebar.slider("Temperature (avg F)", 0, 70, 35)
wind_chill = st.sidebar.slider("Wind Chill (F)", -10, 70, 30)
snowfall = st.sidebar.slider("Snowfall (inches)", 0.0, 25.0, 0.0, step=0.5)
precip = st.sidebar.slider("Precipitation (inches)", 0.0, 2.0, 0.0, step=0.05)
wind_speed = st.sidebar.slider("Wind Speed (mph)", 0, 40, 8)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<h2 style='color:{PRIMARY}'>Schedule</h2>",
                    unsafe_allow_html=True)

# I use selectboxes for day and month because these are categorical,
# not continuous. The order matches a normal school week / school year.
day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
)

month = st.sidebar.selectbox(
    "Month",
    ["October", "November", "December", "January", "February"]
)

is_return = st.sidebar.checkbox("Returning from break?")

# --- About Section in Sidebar ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:0.85rem; color:#5d6d7e;'>
<strong>About</strong><br>
Built by <strong>Mark Anglin</strong> at
<strong>Reading Partners NYC</strong>.<br><br>
This tool analyzes CSD 4 (East Harlem) K-4 attendance data
alongside weather conditions to predict daily attendance rates.<br><br>
<a href="https://github.com/anglinassists" target="_blank">
GitHub: anglinassists</a>
</div>
""", unsafe_allow_html=True)

# --- Build Prediction ---
# I use build_scenario_features to turn the user's inputs into the same
# feature dictionary the models expect. Then I convert it to a numpy
# array in the exact column order the models were trained on. If a
# feature name isn't in the scenario dict (shouldn't happen, but just
# in case), I default to 0.
scenario = build_scenario_features(
    temp_avg=temp_avg,
    wind_chill=wind_chill,
    snow=snowfall,
    precip=precip,
    wind_speed=wind_speed,
    day_of_week=day_of_week,
    month=month,
    is_return_from_break=is_return,
    daily_ref=daily_eng,
)

X_scenario = np.array([[scenario.get(f, 0) for f in feature_names]])

# I get the ensemble prediction (weighted average of OLS, RF, GB).
predicted_rate = ensemble_predict(models, X_scenario)

# I also get each individual model's prediction so the user can see
# how they compare. This builds trust — if they're all close together,
# the ensemble is more credible.
individual_preds = {}
for name, model in models.items():
    individual_preds[name] = model.predict(X_scenario)[0]
individual_preds['Ensemble'] = predicted_rate

# I calculate headcounts from the rate. These are more concrete and
# useful for school leaders than a percentage alone.
expected_present = int(round(predicted_rate * enrolled))
expected_absent = int(enrolled - expected_present)

# --- Main Area ---
st.markdown('<p class="main-title">CSD 4 Attendance Predictor</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">'
    'Predict daily attendance for CSD 4 (East Harlem) K-4 schools '
    'based on weather conditions and schedule factors. '
    'Adjust the inputs in the sidebar to see how attendance changes.'
    '</p>',
    unsafe_allow_html=True,
)

# --- Metric Cards ---
# I use three columns so the big numbers sit side by side. This is the
# first thing the user's eye hits, which is what I want.
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Attendance Rate</div>
        <div class="metric-value">{predicted_rate * 100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Expected Present (of {enrolled:,})</div>
        <div class="metric-value">{expected_present:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {ACCENT};">
        <div class="metric-label">Expected Absent</div>
        <div class="metric-value" style="color: {ACCENT};">{expected_absent:,}</div>
    </div>
    """, unsafe_allow_html=True)

# --- Warning Bar ---
# I color-code this so the user instantly knows if the predicted rate
# is concerning. The thresholds come from NYC DOE chronic absenteeism
# benchmarks (loosely).
if predicted_rate >= 0.85:
    bar_color = GREEN
    bar_text = "Attendance looks healthy — above 85%"
elif predicted_rate >= 0.80:
    bar_color = YELLOW
    bar_text = "Caution — attendance may dip below 85%"
elif predicted_rate >= 0.75:
    bar_color = ORANGE
    bar_text = "Warning — attendance predicted between 75-80%"
else:
    bar_color = RED
    bar_text = "Alert — attendance predicted below 75%"

st.markdown(f"""
<div class="warning-bar" style="background-color: {bar_color}22; color: {bar_color}; border: 2px solid {bar_color};">
    {bar_text}
</div>
""", unsafe_allow_html=True)

# --- Individual Model Predictions ---
# I show this so people can see whether the models agree. Big
# disagreements between OLS and the tree models might mean the
# scenario is outside training data range.
st.markdown(f"<h3 style='color:{PRIMARY}; margin-top:1.5rem;'>Model Breakdown</h3>",
            unsafe_allow_html=True)

model_cols = st.columns(4)
model_names = ['OLS', 'RF', 'GB', 'Ensemble']
model_labels = ['OLS (Linear)', 'Random Forest', 'Gradient Boosting', 'Ensemble (Weighted)']

for i, (mname, mlabel) in enumerate(zip(model_names, model_labels)):
    pred_val = individual_preds[mname]
    with model_cols[i]:
        # I bold the ensemble column so it stands out as the "official" one.
        weight = "700" if mname == "Ensemble" else "400"
        border_color = PRIMARY if mname == "Ensemble" else "#dee2e6"
        st.markdown(f"""
        <div style="background:#f8f9fa; border-radius:8px; padding:0.8rem;
                    text-align:center; border:2px solid {border_color};">
            <div style="font-size:0.8rem; color:#5d6d7e;">{mlabel}</div>
            <div style="font-size:1.6rem; font-weight:{weight}; color:{PRIMARY};">
                {pred_val * 100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- Historical Time Series Chart ---
# I show the actual historical attendance data so the user can see where
# their prediction falls relative to real days. The horizontal line
# makes it easy to compare.
st.markdown(f"<h3 style='color:{PRIMARY}; margin-top:2rem;'>Historical Context</h3>",
            unsafe_allow_html=True)
st.markdown(
    '<p style="color:#5d6d7e; font-size:0.9rem;">'
    'The chart below shows actual daily attendance rates from the training data. '
    'The dashed line shows where your current prediction falls.'
    '</p>',
    unsafe_allow_html=True,
)

# I build the chart data using the daily dataframe from the pipeline.
# I need the date and rate columns, plus I add the prediction as a
# constant column so I can plot it as a horizontal reference line.
chart_df = daily[['date', 'rate']].copy()
chart_df = chart_df.rename(columns={'rate': 'Actual Rate'})
chart_df['Predicted Rate'] = predicted_rate
chart_df = chart_df.set_index('date')

# I use st.line_chart for simplicity, but I want more control over
# styling, so I use Streamlit's built-in Altair integration instead.
import altair as alt

# I build two layers: the actual time series as a solid line, and the
# prediction as a dashed horizontal rule.
base = alt.Chart(chart_df.reset_index()).encode(
    x=alt.X('date:T', title='Date')
)

actual_line = base.mark_line(
    color=PRIMARY,
    strokeWidth=2
).encode(
    y=alt.Y('Actual Rate:Q', title='Attendance Rate',
            scale=alt.Scale(domain=[
                max(0.60, min(chart_df['Actual Rate'].min(), predicted_rate) - 0.02),
                min(1.0, max(chart_df['Actual Rate'].max(), predicted_rate) + 0.02)
            ])),
    tooltip=[
        alt.Tooltip('date:T', title='Date'),
        alt.Tooltip('Actual Rate:Q', title='Rate', format='.1%'),
    ]
)

pred_rule = alt.Chart(pd.DataFrame({'y': [predicted_rate]})).mark_rule(
    color=ACCENT,
    strokeDash=[6, 4],
    strokeWidth=2,
).encode(
    y='y:Q'
)

# I add a text label on the prediction line so the user knows what it is.
pred_label = alt.Chart(pd.DataFrame({
    'y': [predicted_rate],
    'label': [f'Prediction: {predicted_rate * 100:.1f}%']
})).mark_text(
    align='left',
    dx=5,
    dy=-10,
    color=ACCENT,
    fontWeight='bold',
    fontSize=12,
).encode(
    y='y:Q',
    text='label:N',
)

chart = (actual_line + pred_rule + pred_label).properties(
    height=380
).configure_axis(
    gridColor='#eee',
    labelFontSize=11,
    titleFontSize=12,
)

st.altair_chart(chart, use_container_width=True)
