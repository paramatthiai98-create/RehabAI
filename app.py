import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="RehabAI – Real-time Recovery Intelligence System",
    layout="wide"
)

st.title("🏥 RehabAI – Real-time Recovery Intelligence System")
st.caption("AI ที่ช่วยให้การฟื้นฟูร่างกาย ‘เร็วขึ้น ปลอดภัยขึ้น และถูกต้องขึ้น’ แบบ real-time")

# -----------------------------
# Session State
# -----------------------------
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

if "last_status" not in st.session_state:
    st.session_state.last_status = "Waiting"

if "rep_stage" not in st.session_state:
    st.session_state.rep_stage = "down"

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("👤 Patient Profile")
patient_name = st.sidebar.text_input("Name", "John Doe")
age = st.sidebar.slider("Age", 20, 80, 45)
condition = st.sidebar.selectbox(
    "Condition",
    ["Shoulder Rehab", "Knee Rehab", "Elbow Rehab"]
)

mode = st.sidebar.radio("Mode", ["Demo (No Camera)", "Camera"])
session_goal = st.sidebar.slider("Target Reps", 5, 30, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🎛 Demo Control")

simulated_angle = st.sidebar.slider(
    "Simulated Angle",
    min_value=30,
    max_value=120,
    value=70,
    step=1,
    disabled=(mode != "Demo (No Camera)")
)

col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    if st.button("▶ Start Session", use_container_width=True):
        st.session_state.running = True

with col_btn2:
    if st.button("⏹ Stop Session", use_container_width=True):
        st.session_state.running = False

if st.sidebar.button("🔄 Reset Session", use_container_width=True):
    st.session_state.running = False
    st.session_state.correct_count = 0
    st.session_state.incorrect_count = 0
    st.session_state.scores = []
    st.session_state.angles = []
    st.session_state.last_status = "Waiting"
    st.session_state.rep_stage = "down"

# -----------------------------
# Mediapipe Setup
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle
    return angle

def get_risk_and_recommendation(angle):
    """
    ตัวอย่าง logic เบื้องต้นสำหรับ Shoulder Rehab
    """
    if angle < 55:
        return "Incorrect ❌", 45, "HIGH 🔴", "Raise your arm higher / reduce compensation"
    elif 55 <= angle < 75:
        return "Almost Correct ⚠️", 70, "MEDIUM 🟠", "Try to improve range of motion"
    else:
        return "Correct ✅", 92, "LOW 🟢", "Good posture and movement"

def count_rep(angle):
    """
    นับ rep แบบง่าย
    - angle ต่ำ = down
    - angle สูง = up
    """
    if angle < 55:
        st.session_state.rep_stage = "down"

    if angle > 75 and st.session_state.rep_stage == "down":
        st.session_state.rep_stage = "up"
        return True

    return False

def get_progress_value(current_reps, target_reps):
    if target_reps <= 0:
        return 0.0
    return min(current_reps / target_reps, 1.0)

# -----------------------------
# Layout
# -----------------------------
col1, col2, col3 = st.columns([1.15, 2.2, 1.2])

# LEFT PANEL
with col1:
    st.subheader("👤 Patient Info")
    st.write(f"**Name:** {patient_name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Condition:** {condition}")
    st.write(f"**Mode:** {mode}")

    total_reps = st.session_state.correct_count + st.session_state.incorrect_count
    progress_value = get_progress_value(total_reps, session_goal)

    st.write("**Session Progress**")
    st.progress(progress_value)

    st.subheader("📊 Session Summary")
    st.metric("Correct Reps", st.session_state.correct_count)
    st.metric("Incorrect Reps", st.session_state.incorrect_count)

    avg_score = int(np.mean(st.session_state.scores)) if st.session_state.scores else 0
    max_angle = int(np.max(st.session_state.angles)) if st.session_state.angles else 0

    st.metric("Average Score", avg_score)
    st.metric("Max Angle", f"{max_angle}°")

    if st.session_state.scores:
        st.line_chart(st.session_state.scores, height=180)

# CENTER PANEL
with col2:
    st.subheader("📹 Live Monitoring")
    frame_window = st.empty()

# RIGHT PANEL
with col3:
    st.subheader("🧠 Live Clinical Metrics")
    status_box = st.empty()
    score_box = st.empty()
    risk_box = st.empty()
    angle_box = st.empty()
    rec_box = st.empty()

# -----------------------------
# Main Logic
# -----------------------------
if st.session_state.running:

    # -------------------------
    # DEMO MODE
    # -------------------------
    if mode == "Demo (No Camera)":
        angle = simulated_angle
        status, score, risk, recommendation = get_risk_and_recommendation(angle)

        rep_done = count_rep(angle)
        if rep_done:
            if "Correct" in status:
                st.session_state.correct_count += 1
            else:
                st.session_state.incorrect_count += 1

        st.session_state.scores.append(score)
        st.session_state.angles.append(angle)
        st.session_state.last_status = status

        # ทำภาพ placeholder สำหรับ demo
        demo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        demo_frame[:] = (25, 25, 25)

        cv2.putText(
            demo_frame,
            "RehabAI Demo Mode",
            (170, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            demo_frame,
            f"Angle: {angle} deg",
            (210, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            demo_frame,
            f"Status: {status}",
            (170, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if "Correct" in status else (0, 165, 255) if "Almost" in status else (0, 0, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            demo_frame,
            f"Risk: {risk}",
            (210, 320),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        frame_window.image(
            cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB),
            channels="RGB",
            use_container_width=True
        )

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

    # -------------------------
    # CAMERA MODE
    # -------------------------
    else:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            frame_window.error("ไม่พบกล้อง หรือไม่สามารถเปิดกล้องได้")
            status_box.metric("Posture Status", "Camera Error")
            score_box.metric("Recovery Score", 0)
            risk_box.metric("Risk Level", "UNKNOWN")
            angle_box.metric("Joint Angle", "0°")
            rec_box.error("🚨 กรุณาเช็กการเชื่อมต่อกล้อง หรือเปลี่ยนเป็น Demo Mode")
            st.session_state.running = False
        else:
            ret, frame = cap.read()

            if not ret:
                frame_window.error("ไม่สามารถอ่านภาพจากกล้องได้")
                st.session_state.running = False
            else:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                with mp_pose.Pose(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                ) as pose:
                    results = pose.process(rgb)

                status = "Detecting..."
                score = 0
                risk = "UNKNOWN"
                recommendation = "Please stand in camera view"
                angle = 0

                if results.pose_landmarks:
                    lm = results.pose_landmarks.landmark

                    # ใช้แขนซ้ายเป็นตัวอย่าง
                    shoulder = [lm[11].x, lm[11].y]
                    elbow = [lm[13].x, lm[13].y]
                    wrist = [lm[15].x, lm[15].y]

                    angle = calculate_angle(shoulder, elbow, wrist)
                    status, score, risk, recommendation = get_risk_and_recommendation(angle)

                    rep_done = count_rep(angle)
                    if rep_done:
                        if "Correct" in status:
                            st.session_state.correct_count += 1
                        else:
                            st.session_state.incorrect_count += 1

                    st.session_state.scores.append(score)
                    st.session_state.angles.append(angle)
                    st.session_state.last_status = status

                    mp_draw.draw_landmarks(
                        rgb,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS
                    )

                    h, w, _ = rgb.shape
                    elbow_px = tuple(np.multiply(elbow, [w, h]).astype(int))
                    cv2.putText(
                        rgb,
                        f"{int(angle)} deg",
                        elbow_px,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    status = "Body Not Detected ⚠️"
                    score = 0
                    risk = "MEDIUM 🟠"
                    recommendation = "Move into frame and ensure proper lighting"

                frame_window.image(rgb, channels="RGB", use_container_width=True)

                status_box.metric("Posture Status", status)
                score_box.metric("Recovery Score", score)
                risk_box.metric("Risk Level", risk)
                angle_box.metric("Joint Angle", f"{int(angle)}°")

                if "Correct" in status:
                    rec_box.success(f"✅ {recommendation}")
                elif "Almost" in status:
                    rec_box.warning(f"⚠️ {recommendation}")
                else:
                    rec_box.error(f"🚨 {recommendation}")

            cap.release()

else:
    frame_window.info("กดปุ่ม ▶ Start Session เพื่อเริ่มการทำงาน")

    status_box.metric("Posture Status", "Waiting")
    score_box.metric("Recovery Score", 0)
    risk_box.metric("Risk Level", "LOW 🟢")
    angle_box.metric("Joint Angle", "0°")
    rec_box.info("ระบบพร้อมใช้งาน กรุณาเลือกโหมดและกด Start Session")

# -----------------------------
# Footer Notes
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **แนวทางต่อยอดโปรเจกต์**
    - เพิ่มกราฟ real-time ของ score และ angle
    - เพิ่ม Pain Score (0–10)
    - เพิ่มหลายท่ากายภาพ เช่น Shoulder Flexion / Knee Extension
    - เพิ่ม AI Summary หลังจบ session
    - บันทึกผลลงฐานข้อมูลหรือ CSV
    """
)
