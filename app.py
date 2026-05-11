import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from sklearn.linear_model import LinearRegression

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LifeLens",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #333;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

DATA_PATH = "data/life_data.csv"

df = pd.read_csv(DATA_PATH)

# Convert date column safely
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce"
)

# Remove invalid dates if any
df = df.dropna(subset=["date"])

# =========================================================
# LIFE SCORE CALCULATION
# =========================================================

def calculate_life_score(row):

    score = 0

    # Sleep score
    score += min(row["sleep_hours"] * 10, 100) * 0.25

    # Productivity score
    score += row["productivity"] * 10 * 0.30

    # Exercise score
    score += min(row["exercise_minutes"], 60) / 60 * 100 * 0.15

    # Water intake score
    score += min(row["water_intake"], 3) / 3 * 100 * 0.10

    # Stress reduction score
    score += (10 - row["stress_level"]) * 10 * 0.20

    return round(score, 1)

df["life_score"] = df.apply(
    calculate_life_score,
    axis=1
)

# =========================================================
# BURNOUT RISK CALCULATION
# =========================================================

def calculate_burnout_risk(row):

    risk_score = 0

    if row["sleep_hours"] < 6:
        risk_score += 3

    if row["stress_level"] > 7:
        risk_score += 3

    if row["screen_time"] > 8:
        risk_score += 2

    if row["productivity"] < 4:
        risk_score += 2

    if risk_score >= 7:
        return "High"

    elif risk_score >= 4:
        return "Medium"

    else:
        return "Low"

df["burnout_risk"] = df.apply(
    calculate_burnout_risk,
    axis=1
)

# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

features = df[[
    "sleep_hours",
    "study_hours",
    "screen_time",
    "exercise_minutes",
    "water_intake",
    "stress_level"
]]

target = df["productivity"]

model = LinearRegression()

model.fit(features, target)

df["predicted_productivity"] = model.predict(features)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 LifeLens")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Add Daily Entry"]
)

# =========================================================
# DASHBOARD PAGE
# =========================================================

if page == "Dashboard":

    st.title("🚀 LifeLens")
    st.subheader("AI-Powered Personal Life Analytics")

    # =====================================================
    # SIDEBAR STATS
    # =====================================================

    st.sidebar.write("## 📈 Quick Stats")

    st.sidebar.info(
        f"🌟 Avg Life Score: {round(df['life_score'].mean(),1)}"
    )

    st.sidebar.info(
        f"⚡ Avg Productivity: {round(df['productivity'].mean(),1)}"
    )

    st.sidebar.info(
        f"😴 Avg Sleep: {round(df['sleep_hours'].mean(),1)} hrs"
    )

    # =====================================================
    # KPI CARDS
    # =====================================================

    latest_score = round(
        df.iloc[-1]["life_score"],
        1
    )

    latest_burnout = df.iloc[-1]["burnout_risk"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌟 Life Score",
        latest_score
    )

    col2.metric(
        "⚡ Productivity",
        round(df["productivity"].mean(),1)
    )

    col3.metric(
        "😴 Avg Sleep",
        f"{round(df['sleep_hours'].mean(),1)} hrs"
    )

    col4.metric(
        "🔥 Burnout",
        latest_burnout
    )

    st.divider()

    # =====================================================
    # LIFE SCORE TREND
    # =====================================================

    st.write("# 🌟 Life Score Trend")

    life_chart = px.line(
        df,
        x="date",
        y="life_score",
        markers=True
    )

    st.plotly_chart(
        life_chart,
        use_container_width=True
    )

    # =====================================================
    # PRODUCTIVITY TREND
    # =====================================================

    st.write("# ⚡ Productivity Trend")

    productivity_chart = px.line(
        df,
        x="date",
        y="productivity",
        markers=True
    )

    st.plotly_chart(
        productivity_chart,
        use_container_width=True
    )

    # =====================================================
    # PREDICTED PRODUCTIVITY
    # =====================================================

    st.write("# 🤖 Predicted Productivity")

    prediction_chart = px.line(
        df,
        x="date",
        y=["productivity", "predicted_productivity"],
        markers=True,
        title="Actual vs Predicted Productivity"
    )

    st.plotly_chart(
        prediction_chart,
        use_container_width=True
    )

    # =====================================================
    # SCREEN TIME IMPACT
    # =====================================================

    st.write("# 📱 Screen Time Impact")

    scatter_chart = px.scatter(
        df,
        x="screen_time",
        y="productivity",
        size="stress_level",
        color="mood",
        hover_data=["sleep_hours"]
    )

    st.plotly_chart(
        scatter_chart,
        use_container_width=True
    )

    # =====================================================
    # MOOD ANALYSIS
    # =====================================================

    st.write("# 😊 Mood Analysis")

    mood_chart = px.histogram(
        df,
        x="mood",
        color="mood"
    )

    st.plotly_chart(
        mood_chart,
        use_container_width=True
    )

    # =====================================================
    # BURNOUT ANALYSIS
    # =====================================================

    st.write("# 🔥 Burnout Analysis")

    burnout_chart = px.histogram(
        df,
        x="burnout_risk",
        color="burnout_risk"
    )

    st.plotly_chart(
        burnout_chart,
        use_container_width=True
    )

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    st.write("# 🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(
        include=["number"]
    )

    correlation = numeric_df.corr()

    heatmap = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto"
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    # =====================================================
    # AI RECOMMENDATIONS
    # =====================================================

    st.write("# 🧠 AI Recommendations")

    avg_sleep = df["sleep_hours"].mean()

    if avg_sleep < 6:

        st.warning(
            "😴 Increase sleep duration for better productivity."
        )

    avg_screen = df["screen_time"].mean()

    if avg_screen > 7:

        st.warning(
            "📱 Reduce screen time to lower stress."
        )

    avg_water = df["water_intake"].mean()

    if avg_water < 2:

        st.warning(
            "💧 Drink more water daily."
        )

    best_days = df[df["life_score"] > 80]

    if not best_days.empty:

        best_sleep = round(
            best_days["sleep_hours"].mean(),
            1
        )

        st.success(
            f"🌟 Your best days happen around {best_sleep} hours of sleep."
        )

    # =====================================================
    # WEEKLY SUMMARY
    # =====================================================

    st.write("# 📅 Weekly Summary")

    st.info(f"""
    🌟 Average Life Score: {round(df['life_score'].mean(),1)}

    ⚡ Average Productivity: {round(df['productivity'].mean(),1)}

    😴 Average Sleep: {round(df['sleep_hours'].mean(),1)} hrs

    🔥 Burnout Risk: {df.iloc[-1]['burnout_risk']}
    """)

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.write("# 📥 Download Report")

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Life Report",
        data=csv,
        file_name="lifelens_report.csv",
        mime="text/csv"
    )

    # =====================================================
    # DATA TABLE
    # =====================================================

    st.write("# 📋 Life Records")

    st.dataframe(
        df,
        use_container_width=True
    )

# =========================================================
# ADD ENTRY PAGE
# =========================================================

elif page == "Add Daily Entry":

    st.title("📝 Add Daily Entry")

    with st.form("life_form"):

        # =====================================================
        # DATE
        # =====================================================

        entry_date = st.date_input(
            "Date",
            date.today()
        )

        # =====================================================
        # SLEEP
        # =====================================================

        sleep_hours = st.slider(
            "😴 Sleep Hours",
            0.0, 12.0, 7.0
        )

        # =====================================================
        # MOOD
        # =====================================================

        mood = st.selectbox(
            "😊 Mood",
            ["Happy", "Focused", "Neutral", "Tired", "Stressed"]
        )

        # =====================================================
        # STUDY HOURS
        # =====================================================

        study_hours = st.slider(
            "📚 Study Hours",
            0.0, 15.0, 2.0
        )

        # =====================================================
        # SCREEN TIME
        # =====================================================

        screen_time = st.slider(
            "📱 Screen Time",
            0.0, 15.0, 5.0
        )

        # =====================================================
        # MANUAL EXPENSE ENTRY
        # =====================================================

        st.write("### 💰 Daily Expenses")

        expense_text = st.text_area(
            "Enter expenses manually",
            placeholder="Food 120, Travel 50, Snacks 40"
        )

        expenses = 0

        if expense_text:

            items = expense_text.split(",")

            for item in items:

                parts = item.strip().split()

                if len(parts) >= 2:

                    try:
                        amount = float(parts[-1])
                        expenses += amount

                    except:
                        pass

        st.info(f"Total Expenses: ₹{expenses}")

        # =====================================================
        # EXERCISE
        # =====================================================

        exercise_minutes = st.slider(
            "🏃 Exercise Minutes",
            0, 180, 20
        )

        # =====================================================
        # WATER INTAKE
        # =====================================================

        water_intake = st.slider(
            "💧 Water Intake (Liters)",
            0.0, 5.0, 2.0
        )

        # =====================================================
        # PRODUCTIVITY
        # =====================================================

        productivity = st.slider(
            "⚡ Productivity Score",
            1, 10, 5
        )

        # =====================================================
        # STRESS LEVEL
        # =====================================================

        stress_level = st.slider(
            "😓 Stress Level",
            1, 10, 5
        )

        # =====================================================
        # SUBMIT BUTTON
        # =====================================================

        submitted = st.form_submit_button(
            "Save Entry"
        )

        # =====================================================
        # SAVE DATA
        # =====================================================

        if submitted:

            new_data = {
                "date": entry_date.strftime("%Y-%m-%d"),
                "sleep_hours": sleep_hours,
                "mood": mood,
                "study_hours": study_hours,
                "screen_time": screen_time,
                "expenses": expenses,
                "exercise_minutes": exercise_minutes,
                "water_intake": water_intake,
                "productivity": productivity,
                "stress_level": stress_level
            }

            new_df = pd.DataFrame([new_data])

            df = pd.concat(
                [df, new_df],
                ignore_index=True
            )

            df.to_csv(
                DATA_PATH,
                index=False
            )

            st.success(
                "✅ Entry Saved Successfully!"
            )