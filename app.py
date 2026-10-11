from pathlib import Path
import pickle

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="LaunchLens AI | Student Placement Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "dataset.csv"

FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 8% 0%, #202c55 0%, #111827 42%, #0b1220 100%);
        color: #eef2ff;
    }
    .block-container { max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    .hero {
        padding: 1.7rem 1.9rem;
        border-radius: 24px;
        background: linear-gradient(115deg, rgba(99,102,241,.25), rgba(6,182,212,.12));
        border: 1px solid rgba(165,180,252,.28);
        box-shadow: 0 18px 55px rgba(0,0,0,.18);
        margin-bottom: 1.2rem;
    }
    .hero-kicker { color: #a5b4fc; text-transform: uppercase; letter-spacing: .16em; font-size: .76rem; font-weight: 800; }
    .hero h1 { color: #fff; font-size: clamp(2rem, 4vw, 3.1rem); margin: .25rem 0 .35rem 0; }
    .hero p { color: #cbd5e1; font-size: 1.02rem; margin: 0; }
    .section-card {
        padding: 1.15rem 1.25rem;
        border: 1px solid rgba(148,163,184,.22);
        border-radius: 18px;
        background: rgba(255,255,255,.035);
        margin-bottom: .7rem;
    }
    .result-card {
        padding: 1.7rem 1.3rem;
        border-radius: 22px;
        text-align: center;
        border: 1px solid rgba(255,255,255,.18);
        box-shadow: 0 15px 40px rgba(0,0,0,.22);
        margin-top: .6rem;
    }
    .result-positive { background: linear-gradient(135deg, rgba(5,150,105,.28), rgba(16,185,129,.08)); border-color: rgba(52,211,153,.48); }
    .result-prep { background: linear-gradient(135deg, rgba(217,119,6,.25), rgba(245,158,11,.07)); border-color: rgba(251,191,36,.45); }
    .result-icon { font-size: 2.5rem; line-height: 1.25; }
    .result-title { color: #fff; font-size: 1.7rem; font-weight: 850; margin: .55rem 0 .4rem 0; }
    .result-copy { color: #dbeafe; font-size: .98rem; }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.045); border: 1px solid rgba(148,163,184,.2); padding: 1rem; border-radius: 16px; }
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; min-height: 3.1rem; border: 0; border-radius: 13px;
        color: white; font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #0891b2);
    }
    div[data-testid="stTabs"] button { font-weight: 700; }
    .footnote { color: #94a3b8; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI-powered career insights</div>
      <h1>🎯 LaunchLens AI</h1>
      <p>Explore a student's placement outlook using academic, skill and experience indicators.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.exists():
    st.error("The trained model is missing. Open the VS Code terminal and run: python train_model.py")
    st.stop()

try:
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
except Exception as exc:
    st.error("Could not load model.pkl. Rebuild it with `python train_model.py` using the requirements in this project.")
    st.caption(f"Technical detail: {exc}")
    st.stop()

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip()
else:
    data = pd.DataFrame()

predict_tab, dashboard_tab, about_tab = st.tabs(
    ["🎯 Placement Predictor", "📊 Dataset Overview", "ℹ️ About the Project"]
)

with predict_tab:
    st.markdown("### Student profile")
    st.caption("Enter the details below, then select Predict Outlook. No confidence percentage is displayed.")

    with st.form("student_prediction_form"):
        left, right = st.columns(2, gap="large")
        with left:
            iq = st.slider("🧠 IQ score", min_value=50, max_value=150, value=100)
            prev_sem = st.slider("📚 Previous semester result", min_value=0.0, max_value=10.0, value=6.5, step=0.01)
            cgpa = st.slider("🎓 CGPA", min_value=0.0, max_value=10.0, value=6.5, step=0.01)
            academic = st.slider("📖 Academic performance", min_value=0, max_value=10, value=7)
        with right:
            internship = st.selectbox("💼 Internship experience", options=["No", "Yes"])
            extracurricular = st.slider("🏆 Extra-curricular score", min_value=0, max_value=10, value=5)
            communication = st.slider("🗣 Communication skills", min_value=0, max_value=10, value=5)
            projects = st.slider("💻 Projects completed", min_value=0, max_value=10, value=2)
        submitted = st.form_submit_button("✨ Predict Placement Outlook")

    if submitted:
        student = pd.DataFrame(
            [{
                "IQ": iq,
                "Prev_Sem_Result": prev_sem,
                "CGPA": cgpa,
                "Academic_Performance": academic,
                "Internship_Experience": 1 if internship == "Yes" else 0,
                "Extra_Curricular_Score": extracurricular,
                "Communication_Skills": communication,
                "Projects_Completed": projects,
            }],
            columns=FEATURES,
        )
        try:
            prediction = int(model.predict(student)[0])
            st.markdown("### Your result")
            if prediction == 1:
                st.markdown(
                    """
                    <div class="result-card result-positive">
                      <div class="result-icon">🌟</div>
                      <div class="result-title">Positive Placement Outlook</div>
                      <div class="result-copy">The model predicts a positive placement outcome for this profile.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info("Keep strengthening interview skills, practical projects and job-specific preparation.")
            else:
                st.markdown(
                    """
                    <div class="result-card result-prep">
                      <div class="result-icon">🚀</div>
                      <div class="result-title">More Preparation Recommended</div>
                      <div class="result-copy">The model predicts that this profile may need further preparation to improve its placement outlook.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info("Consider improving technical skills, building projects, practising aptitude and preparing for interviews.")
            st.markdown(
                '<p class="footnote">This is a machine-learning estimate based on the dataset, not a hiring decision or a guarantee of employment.</p>',
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error("Prediction could not be completed. Please retrain the model and try again.")
            st.caption(f"Technical detail: {exc}")

with dashboard_tab:
    st.markdown("### Dataset at a glance")
    if data.empty:
        st.warning("dataset.csv was not found beside app.py. The predictor can still run, but this overview is unavailable.")
    elif "Placement" not in data.columns:
        st.warning("The dataset must contain a column named Placement.")
    else:
        outcomes = data["Placement"].astype(str).str.strip().str.title()
        total = len(data)
        positive = int((outcomes == "Yes").sum())
        other = total - positive
        col1, col2, col3 = st.columns(3)
        col1.metric("Student records", f"{total:,}")
        col2.metric("Positive outcomes in dataset", f"{positive:,}")
        col3.metric("Other outcomes in dataset", f"{other:,}")
        chart_left, chart_right = st.columns(2, gap="large")
        with chart_left:
            st.markdown("**Recorded placement outcomes**")
            st.bar_chart(outcomes.value_counts())
        with chart_right:
            if "CGPA" in data.columns:
                st.markdown("**Average CGPA by recorded outcome**")
                chart_data = data.assign(_outcome=outcomes).groupby("_outcome")["CGPA"].mean()
                st.bar_chart(chart_data)
        with st.expander("Preview first 20 dataset rows"):
            st.dataframe(data.head(20), use_container_width=True)
        st.caption("College_ID is kept for reference and is not used as a model input.")

with about_tab:
    st.markdown("### How LaunchLens AI works")
    st.markdown(
        """
        1. Reads the student placement dataset.
        2. Cleans the target and internship fields.
        3. Trains a Random Forest classification model.
        4. Evaluates it with a held-out test set and cross-validation.
        5. Uses the saved model to predict a placement outlook from the form.

        **Important:** the model learns patterns in the supplied dataset. It cannot know all factors that affect real hiring outcomes and should not be used to make actual hiring decisions.
        """
    )

st.markdown("---")
st.markdown('<p class="footnote">LaunchLens AI · Student project for educational and demonstration purposes</p>', unsafe_allow_html=True)
