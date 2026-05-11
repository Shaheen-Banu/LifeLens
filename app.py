import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import date
from sklearn.linear_model import LinearRegression

# =========================================================
# LOAD LOGO
# =========================================================

logo = Image.open("assets/lifelens_logo.png")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="LifeLens",
    page_icon=logo,
    layout="wide"
)

# =========================================================
# PREMIUM UI STYLING
# =========================================================

st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #111827;
    border-right: 1px solid #374151;
}

/* Titles */
h1, h2, h3 {
    color: white;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: 0.3s ease;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-5px);
    border: 1px solid #00ADB5;
}

/* Metric Value */
div[data-testid="stMetricValue"] {
    font-size: 30px;
    font-weight: 700;
    color: #00E5FF;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00ADB5, #007CF0);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 1.5rem;
    font-weight: 600;
}

/* Download Button */
.stDownloadButton>button {
    background: linear-gradient(90deg, #10B981, #059669);
    color: white;
    border-radius: 12px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

DATA_PATH = "data/life_data.csv"

df = pd.read_csv(DATA_PATH)

# Convert dates safely
df["date"] = pd.to_datetime(
    df["date"],
    format="mixed",
    errors="coerce"
)

# Remove invalid rows
df = df.dropna(subset=["date"])

# =========================================================
# LIFE SCORE
# =========================================================

def calculate_life_score(row):

    score = 0

    score += min(row["sleep_hours"] * 10, 100) * 0.25
    score += row["productivity"] * 10 * 0.30
    score += min(row["exercise_minutes"], 60) / 60 * 100 * 0.15
    score += min(row["water_intake"], 3) / 3 * 100 * 0.10
    score += (10 - row["stress_level"]) * 10 * 0.20

    return round(score, 1)

df["life_score"] = df.apply(
    calculate_life_score,
    axis=1
)

# =========================================================
# BURNOUT RISK
# =========================================================

def calculate_burnout_risk(row):

    risk = 0

    if row["sleep_hours"] < 6:
        risk += 3

    if row["stress_level"] > 7:
        risk += 3

    if row["screen_time"] > 8:
        risk += 2

    if row["productivity"] < 4:
        risk += 2

    if risk >= 7:
        return "High"

    elif risk >= 4:
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

st.sidebar.image(logo, width=180)
st.sidebar.markdown("## LifeLens")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Add Daily Entry"]
)

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":


    # HERO SECTION
    st.title("LifeLens")

    st.markdown("""
    <div style='
    background: linear-gradient(90deg,#00ADB5,#007CF0);
    padding: 1.2rem;
    border-radius: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    '>
    <h2 style='color:white;margin:0;'>
    AI-Powered Personal Life Analytics
    </h2>
    <p style='color:white;font-size:18px;'>
    Track habits, predict burnout, and optimize productivity.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # KPI CARDS
    # =====================================================

    latest_score = round(df.iloc[-1]["life_score"], 1)
    latest_burnout = df.iloc[-1]["burnout_risk"]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌟 Life Score",
        latest_score
    )

    col2.metric(
        "⚡ Avg Productivity",
        round(df["productivity"].mean(), 1)
    )

    col3.metric(
        "😴 Avg Sleep",
        f"{round(df['sleep_hours'].mean(),1)} hrs"
    )

    col4.metric(
        "🔥 Burnout Risk",
        latest_burnout
    )

    st.divider()

    # =====================================================
    # LIFE SCORE TREND
    # =====================================================

    st.subheader("🌟 Life Score Trend")

    life_chart = px.line(
        df,
        x="date",
        y="life_score",
        markers=True,
        template="plotly_dark"
    )

    life_chart.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(
        life_chart,
        use_container_width=True
    )

    # =====================================================
    # PRODUCTIVITY TREND
    # =====================================================

    st.subheader("⚡ Productivity Trend")

    productivity_chart = px.area(
        df,
        x="date",
        y="productivity",
        template="plotly_dark"
    )

    productivity_chart.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(
        productivity_chart,
        use_container_width=True
    )

    # =====================================================
    # PRODUCTIVITY PREDICTION
    # =====================================================

    st.subheader("🤖 Predicted Productivity")

    prediction_chart = px.line(
        df,
        x="date",
        y=["productivity", "predicted_productivity"],
        markers=True,
        template="plotly_dark"
    )

    prediction_chart.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(
        prediction_chart,
        use_container_width=True
    )

    # =====================================================
    # SCREEN TIME IMPACT
    # =====================================================

    st.subheader("📱 Screen Time vs Productivity")

    scatter_chart = px.scatter(
        df,
        x="screen_time",
        y="productivity",
        size="stress_level",
        color="life_score",
        hover_data=["sleep_hours"],
        template="plotly_dark"
    )

    scatter_chart.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=500
    )

    st.plotly_chart(
        scatter_chart,
        use_container_width=True
    )

    # =====================================================
    # MOOD ANALYSIS
    # =====================================================

    st.subheader("😊 Mood Distribution")

    mood_chart = px.histogram(
        df,
        x="mood",
        color="mood",
        template="plotly_dark"
    )

    mood_chart.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white")
    )

    st.plotly_chart(
        mood_chart,
        use_container_width=True
    )

    # =====================================================
    # CORRELATION HEATMAP
    # =====================================================

    st.subheader("🔥 Correlation Heatmap")

    numeric_df = df.select_dtypes(include=["number"])

    correlation = numeric_df.corr()

    heatmap = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        template="plotly_dark"
    )

    heatmap.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="white"),
        height=700
    )

    st.plotly_chart(
        heatmap,
        use_container_width=True
    )

    # =====================================================
    # AI RECOMMENDATIONS
    # =====================================================

    st.subheader("🧠 AI Recommendations")

    if df["sleep_hours"].mean() < 6:
        st.warning("😴 Increase sleep duration for better productivity.")

    if df["screen_time"].mean() > 7:
        st.warning("📱 Reduce screen time to lower stress.")

    if df["water_intake"].mean() < 2:
        st.warning("💧 Drink more water daily.")

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
    # DOWNLOAD REPORT
    # =====================================================

    st.subheader("📥 Download Report")

    csv = df.to_csv(index=False)

    st.download_button(
        label="📥 Download Life Report",
        data=csv,
        file_name="lifelens_report.csv",
        mime="text/csv"
    )

    # =====================================================
    # LIFE RECORDS
    # =====================================================

    st.subheader("📋 Life Records")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================================
    # DELETE RECORDS
    # =====================================================

    st.subheader("🗑 Delete Records")

    record_index = st.number_input(
        "Enter Row Number to Delete",
        min_value=0,
        max_value=len(df)-1,
        step=1
    )

    if st.button("Delete Selected Record"):

        df = df.drop(index=record_index)

        df = df.reset_index(drop=True)

        df.to_csv(
            DATA_PATH,
            index=False
        )

        st.success("✅ Record Deleted Successfully")

        st.rerun()

# =========================================================
# ADD DAILY ENTRY
# =========================================================

elif page == "Add Daily Entry":

    st.title("📝 Add Daily Entry")

    with st.form("life_form"):

        entry_date = st.date_input(
            "Date",
            date.today()
        )

        sleep_hours = st.slider(
            "😴 Sleep Hours",
            0.0, 12.0, 7.0
        )

        mood = st.selectbox(
            "😊 Mood",
            ["Happy", "Focused", "Neutral", "Tired", "Stressed"]
        )

        study_hours = st.slider(
            "📚 Study Hours",
            0.0, 15.0, 2.0
        )

        screen_time = st.slider(
            "📱 Screen Time",
            0.0, 15.0, 5.0
        )

        st.subheader("💰 Daily Expenses")

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

        exercise_minutes = st.slider(
            "🏃 Exercise Minutes",
            0, 180, 20
        )

        water_intake = st.slider(
            "💧 Water Intake",
            0.0, 5.0, 2.0
        )

        productivity = st.slider(
            "⚡ Productivity Score",
            1, 10, 5
        )

        stress_level = st.slider(
            "😓 Stress Level",
            1, 10, 5
        )

        submitted = st.form_submit_button(
            "Save Entry"
        )

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

            st.success("✅ Entry Saved Successfully!")

            st.rerun()
