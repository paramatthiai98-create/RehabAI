import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="RehabAI", layout="wide")

st.title("🏥 RehabAI – Real-time Recovery Intelligence System")
st.caption("AI ที่ช่วยให้การฟื้นฟูร่างกายเร็วขึ้น ปลอดภัยขึ้น และถูกต้องขึ้น แบบ real-time")

if "running" not in st.session_state:
    st.session_state.running = False
if "correct_count" not in st.session_state:
    st.session_state.correct_count = 0
if "incorrect_count" not in st.session_state:
    st.session_state.incorrect_count = 0
if "scores" not in st.session_state:
    st.session_state.scores = []
if "angles" not in st.session_state:
    st.session_state.angles = []
if "rep_stage" not in st.session_state:
    st.session_state.rep_stage = "down"

st.sidebar.header("👤 Patient Profile")
patient_name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.slider("Age", 20, 80, 45)
condition = st.sidebar.selectbox(
    "Condition",
    ["Shoulder Rehab", "Knee Rehab", "Elbow Rehab"]
)
session_goal = st.sidebar.slider("Target Reps", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛 Demo Control")
angle = st.sidebar.slider("Simulated Angle", 30, 120, 70)

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("▶ Start Session", use_container_width=True):
        st.session_state.running = True
with c2:
    if st.button("⏹ Stop Session", use_container_width=True):
        st.session_state.running = False

if st.sidebar.button("🔄 Reset Session", use_container_width=True):
    st.session_state.running = False
    st.session_state.correct_count = 0
    st.session_state.incorrect_count = 0
    st.session_state.scores = []
    st.session_state.angles = []
    st.session_state.rep_stage = "down"

def get_risk_and_recommendation(angle_value):
    if angle_value < 55:
        return "Incorrect ❌", 45, "HIGH 🔴", "Raise your arm higher / reduce compensation"
    elif 55 <= angle_value < 75:
        return "Almost Correct ⚠️", 70, "MEDIUM 🟠", "Try to improve range of motion"
    else:
        return "Correct ✅", 92, "LOW 🟢", "Good posture and movement"

def count_rep(angle_value):
    if angle_value < 55:
        st.session_state.rep_stage = "down"
    if angle_value > 75 and st.session_state.rep_stage == "down":
        st.session_state.rep_stage = "up"
        return True
    return False

left, center, right = st.columns([1.1, 2.2, 1.2])

with left:
    st.subheader("👤 Patient Info")
    st.write(f"**Name:** {patient_name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Condition:** {condition}")
    total_reps = st.session_state.correct_count + st.session_state.incorrect_count
    st.progress(min(total_reps / session_goal, 1.0))
    st.subheader("📊 Session Summary")
    st.metric("Correct Reps", st.session_state.correct_count)
    st.metric("Incorrect Reps", st.session_state.incorrect_count)
    avg_score = int(np.mean(st.session_state.scores)) if st.session_state.scores else 0
    max_angle = int(np.max(st.session_state.angles)) if st.session_state.angles else 0
    st.metric("Average Score", avg_score)
    st.metric("Max Angle", f"{max_angle}°")

with center:
    st.subheader("📹 Live Monitoring")
    frame_placeholder = st.empty()
    chart_placeholder = st.empty()

with right:
    st.subheader("🧠 Live Clinical Metrics")
    status_box = st.empty()
    score_box = st.empty()
    risk_box = st.empty()
    angle_box = st.empty()
    rec_box = st.empty()

if st.session_state.running:
    status, score, risk, recommendation = get_risk_and_recommendation(angle)

    rep_done = count_rep(angle)
    if rep_done:
        if "Correct" in status:
            st.session_state.correct_count += 1
        else:
            st.session_state.incorrect_count += 1

    st.session_state.scores.append(score)
    st.session_state.angles.append(angle)

    frame_placeholder.markdown(
        f"""
        <div style="
            height:420px;
            border-radius:18px;
            background:#111827;
            color:white;
            display:flex;
            flex-direction:column;
            justify-content:center;
            align-items:center;
            text-align:center;">
            <h2>RehabAI Demo Mode</h2>
            <h1>{angle}°</h1>
            <p>Simulated Joint Angle</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    df = pd.DataFrame({"Recovery Score": st.session_state.scores[-20:]})
    chart_placeholder.line_chart(df, height=220)

    status_box.metric("Posture Status", status)
    score_box.metric("Recovery Score", score)
    risk_box.metric("Risk Level", risk)
    angle_box.metric("Joint Angle", f"{angle}°")

    if "Correct" in status:
        rec_box.success(f"✅ {recommendation}")
    elif "Almost" in status:
        rec_box.warning(f"⚠️ {recommendation}")
    else:
        rec_box.error(f"🚨 {recommendation}")
else:
    frame_placeholder.info("กด ▶ Start Session เพื่อเริ่ม Demo")
    chart_placeholder.empty()
    status_box.metric("Posture Status", "Waiting")
    score_box.metric("Recovery Score", 0)
    risk_box.metric("Risk Level", "-")
    angle_box.metric("Joint Angle", "0°")
    rec_box.info("ระบบพร้อมใช้งาน")
