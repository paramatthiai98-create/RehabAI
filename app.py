"""
RehabAI v2.4 – Real-time Recovery Intelligence System
Camera Mode Upgrade: Full MediaPipe pose detection for Shoulder Flexion.
All existing Demo Mode logic, UI, KPI cards, charts, and AI insights preserved.
"""

import math
import time
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RehabAI · Recovery Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
EXERCISE_CONFIG = {
    "Shoulder Flexion": {
        "description": "Raise arm forward from resting position",
        "description_th": "ยกแขนไปข้างหน้าจากท่าพัก",
        "target_angle": 90,
        "correct_min": 80,
        "almost_min": 60,
        "joint_label": "Shoulder",
        "joint_label_th": "ไหล่",
        # Camera Mode: hip → shoulder → elbow (as required)
        "mediapipe_joints": ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
        "icon": "💪",
    },
    "Elbow Flexion": {
        "description": "Bend elbow bringing hand toward shoulder",
        "description_th": "งอข้อศอกดึงมือเข้าหาไหล่",
        "target_angle": 90,
        "correct_min": 75,
        "almost_min": 55,
        "joint_label": "Elbow",
        "joint_label_th": "ข้อศอก",
        "mediapipe_joints": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
        "icon": "🦾",
    },
    "Knee Extension": {
        "description": "Straighten knee from seated position",
        "description_th": "เหยียดเข่าจากท่านั่ง",
        "target_angle": 150,
        "correct_min": 140,
        "almost_min": 120,
        "joint_label": "Knee",
        "joint_label_th": "เข่า",
        "mediapipe_joints": ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
        "icon": "🦵",
    },
    "Hip Abduction": {
        "description": "Move leg outward away from body",
        "description_th": "แกว่งขาออกด้านข้างลำตัว",
        "target_angle": 30,
        "correct_min": 25,
        "almost_min": 15,
        "joint_label": "Hip",
        "joint_label_th": "สะโพก",
        "mediapipe_joints": ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
        "icon": "🏃",
    },
}

STATUS_COLORS = {
    "Correct":   {"hex": "#16a34a", "bg": "#f0fdf4", "border": "#22c55e", "text": "#14532d"},
    "Almost":    {"hex": "#ea580c", "bg": "#fff7ed", "border": "#f97316", "text": "#7c2d12"},
    "Incorrect": {"hex": "#dc2626", "bg": "#fff1f2", "border": "#f43f5e", "text": "#881337"},
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  (unchanged from v2.3)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Sora:wght@400;600;700;800&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] { font-family: 'Sora', sans-serif; background: #eef2f7; }
[data-testid="stSidebar"] { background: #06101f; border-right: 1px solid #0f2040; }
[data-testid="stSidebar"] * { color: #c9d8ee !important; }
[data-testid="stSidebar"] .stButton button { width: 100%; border-radius: 10px; font-weight: 700; letter-spacing: .04em; }
#MainMenu, footer { visibility: hidden; }
.top-bar {
    background: linear-gradient(120deg, #1347b0 0%, #0284c7 100%);
    border-radius: 18px; padding: 18px 28px; color: white; margin-bottom: 20px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 4px 24px rgba(19,71,176,.22);
}
.top-bar-title { font-size: 22px; font-weight: 800; letter-spacing: -.02em; }
.top-bar-sub   { font-size: 13px; opacity: .75; margin-top: 3px; }
.top-bar-badge { background: rgba(255,255,255,.18); border-radius: 20px; padding: 5px 16px; font-size: 12px; font-weight: 700; letter-spacing: .04em; }
.top-bar-time  { font-family: 'IBM Plex Mono', monospace; font-size: 12px; opacity: .6; margin-top: 5px; }
.kpi-card {
    background: white; border-radius: 16px; padding: 18px 20px; text-align: center;
    box-shadow: 0 2px 14px rgba(6,16,31,.08); border: 1px solid #dde5f0;
    height: 115px; display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.kpi-label { font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: #6b7a99; margin-bottom: 6px; }
.kpi-value { font-size: 34px; font-weight: 800; line-height: 1; }
.kpi-sub   { font-size: 11px; color: #94a3b8; margin-top: 4px; }
.c-green { color: #15803d; } .c-orange { color: #c2570a; }
.c-red   { color: #b91c1c; } .c-blue   { color: #1d4ed8; }
.section-card {
    background: white; border-radius: 16px; padding: 22px 24px;
    box-shadow: 0 2px 10px rgba(6,16,31,.06); border: 1px solid #dde5f0; margin-bottom: 16px;
}
.section-label {
    font-size: 10px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
    color: #1d4ed8; border-bottom: 2px solid #bfdbfe; padding-bottom: 8px; margin-bottom: 16px;
}
.rec-box { border-radius: 12px; padding: 16px 20px; font-size: 14px; line-height: 1.7; font-weight: 500; border-left: 5px solid; }
.rec-green  { background:#f0fdf4; border-color:#22c55e; color:#14532d; }
.rec-orange { background:#fff7ed; border-color:#f97316; color:#7c2d12; }
.rec-red    { background:#fff1f2; border-color:#f43f5e; color:#881337; }
.insight-item { display:flex; justify-content:space-between; align-items:center; padding:10px 0; border-bottom:1px solid #f1f5f9; font-size:14px; }
.insight-key   { color:#64748b; font-weight:500; }
.insight-value { font-weight:800; font-size:15px; }
.sess-tile { background:#f8fafc; border-radius:12px; padding:14px; text-align:center; border:1px solid #e2e8f0; }
.sess-num  { font-size:28px; font-weight:800; }
.sess-lbl  { font-size:10px; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-top:3px; font-weight:600; }
.exercise-badge { display:inline-flex; align-items:center; gap:8px; background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:8px 14px; font-size:13px; font-weight:700; color:#1d4ed8; margin-bottom:14px; }
.exercise-desc  { font-size:12px; color:#64748b; margin-bottom:16px; }
.angle-display  { font-family:'IBM Plex Mono',monospace; font-size:72px; font-weight:700; line-height:1; letter-spacing:-.04em; }
.angle-target-label { font-size:11px; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:.08em; margin-top:6px; }
.skeleton-wrap { display:flex; align-items:center; justify-content:center; background:#f8fafc; border-radius:12px; min-height:260px; border:1.5px dashed #cbd5e1; }
.stProgress > div > div { border-radius:8px !important; }
[data-testid="stSlider"] label { color:#c9d8ee !important; }
.mono { font-family:'IBM Plex Mono',monospace; }

/* ── Camera Mode specific ── */
.cam-hud {
    background: rgba(6,16,31,0.82); border-radius: 14px; padding: 16px 20px;
    color: white; margin-bottom: 12px; display: flex;
    align-items: center; justify-content: space-between;
    backdrop-filter: blur(6px);
}
.cam-hud-angle { font-family:'IBM Plex Mono',monospace; font-size: 42px; font-weight: 700; line-height: 1; }
.cam-hud-label { font-size: 11px; opacity: .6; text-transform: uppercase; letter-spacing: .1em; margin-bottom: 4px; }
.cam-hud-status { font-size: 15px; font-weight: 700; }
.cam-warning {
    background: #fff7ed; border: 1.5px solid #fed7aa; border-radius: 12px;
    padding: 14px 18px; font-size: 13px; color: #9a3412; text-align: center;
    margin: 8px 0;
}
.cam-info-bar {
    background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px;
    padding: 10px 16px; font-size: 12px; color: #0369a1; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}

/* ── Areas for Improvement ── */
.aoi-section {
    background: white; border-radius: 18px; padding: 28px 32px;
    box-shadow: 0 2px 14px rgba(6,16,31,.07); border: 1px solid #dde5f0; margin-top: 8px;
}
.aoi-header    { font-size: 16px; font-weight: 800; color: #0f172a; margin-bottom: 6px; }
.aoi-subheader { font-size: 13px; color: #64748b; margin-bottom: 24px; line-height: 1.6; }
.aoi-grid      { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media (max-width: 900px) { .aoi-grid { grid-template-columns: 1fr; } }
.aoi-card      { border-radius: 14px; padding: 20px 22px; border: 1px solid; }
.aoi-card-icon { font-size: 22px; margin-bottom: 10px; display: block; }
.aoi-card-title{ font-size: 14px; font-weight: 800; margin-bottom: 6px; }
.aoi-card-body { font-size: 13px; line-height: 1.65; }
.aoi-tag { display:inline-block; font-size:10px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:10px; }
.aoi-now    { background:#eff6ff; border-color:#bfdbfe; }
.aoi-now    .aoi-card-title { color:#1d4ed8; } .aoi-now    .aoi-tag { background:#dbeafe; color:#1e40af; } .aoi-now    .aoi-card-body { color:#1e3a5f; }
.aoi-med    { background:#fdf4ff; border-color:#e9d5ff; }
.aoi-med    .aoi-card-title { color:#7e22ce; } .aoi-med    .aoi-tag { background:#f3e8ff; color:#6b21a8; } .aoi-med    .aoi-card-body { color:#3b0764; }
.aoi-later  { background:#f0fdf4; border-color:#bbf7d0; }
.aoi-later  .aoi-card-title { color:#15803d; } .aoi-later  .aoi-tag { background:#dcfce7; color:#166534; } .aoi-later  .aoi-card-body { color:#14532d; }
.aoi-vision { background:#fff7ed; border-color:#fed7aa; }
.aoi-vision .aoi-card-title { color:#c2570a; } .aoi-vision .aoi-tag { background:#ffedd5; color:#9a3412; } .aoi-vision .aoi-card-body { color:#431407; }
</style>
""", unsafe_allow_html=True)

# Sidebar toggle fix
st.markdown("""
<style>
[data-testid="collapsedControl"] { display:flex !important; visibility:visible !important; opacity:1 !important; z-index:99999 !important; }
header { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = {
    "running": False,
    "score_history": [],
    "angle_history": [],
    "correct_reps": 0,
    "incorrect_reps": 0,
    "last_stage": "idle",
    # Camera Mode state
    "cam_angle": 0.0,
    "cam_pose_detected": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v if not isinstance(v, list) else []

# ─────────────────────────────────────────────────────────────────────────────
# ── NEW ── CAMERA ANGLE CALCULATION HELPER
# calculate_angle(a, b, c) — reusable for any 3-point joint angle
# Takes 3 numpy arrays [x, y], returns angle at point b in degrees (0–180)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle at joint b formed by vectors b→a and b→c.
    Returns angle in degrees, clamped to [0, 180].
    Args:
        a: numpy array [x, y] — first point (e.g. hip)
        b: numpy array [x, y] — vertex point (e.g. shoulder)
        c: numpy array [x, y] — third point  (e.g. elbow)
    Returns:
        float: angle in degrees at point b
    """
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    angle  = math.degrees(math.acos(np.clip(cosine, -1.0, 1.0)))
    return min(angle, 180.0)


# ─────────────────────────────────────────────────────────────────────────────
# EXISTING HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def classify_angle(angle: float, exercise: str, lang: str = "EN") -> dict:
    cfg  = EXERCISE_CONFIG[exercise]
    jl   = cfg["joint_label_th"] if lang == "TH" else cfg["joint_label"]
    tgt  = cfg["target_angle"]
    diff = int(tgt - angle)

    if angle >= cfg["correct_min"]:
        rec = (f"ยอดเยี่ยม! มุม{jl}ของคุณอยู่ที่ {angle:.0f}° อยู่ในช่วงเป้าหมาย ({tgt}°) แล้ว คงท่านี้ไว้"
               if lang == "TH" else
               f"Excellent! Your {jl} angle is {angle:.0f}° — within the target zone ({tgt}°). Hold this position.")
        return dict(status="Correct",   risk="LOW",    score_delta=+3, icon="✅", rec_cls="rec-green",  rec=rec)

    if angle >= cfg["almost_min"]:
        rec = (f"ใกล้แล้ว! มุม{jl}ปัจจุบัน {angle:.0f}° ขยับเพิ่มอีกประมาณ {diff}° ให้ถึง {tgt}°"
               if lang == "TH" else
               f"Almost there! {jl} angle is {angle:.0f}° — extend ~{diff}° more to reach the target of {tgt}°.")
        return dict(status="Almost",    risk="MEDIUM", score_delta=+1, icon="⚠️", rec_cls="rec-orange", rec=rec)

    rec = (f"มุม{jl}แคบเกินไป ({angle:.0f}°) กรุณาขยับเพิ่มอีกประมาณ {diff}° ให้ถึง {tgt}° ค่อยๆ ทำช้าๆ"
           if lang == "TH" else
           f"{jl} angle too low ({angle:.0f}°). Increase by ~{diff}° to reach {tgt}°. Slow down and correct gradually.")
    return dict(status="Incorrect", risk="HIGH",   score_delta=-2, icon="❌", rec_cls="rec-red",    rec=rec)


def update_session_history(posture: dict, angle: float):
    """Update score/angle history and rep counting — shared by Demo and Camera modes."""
    new_score = update_score(posture["score_delta"])
    st.session_state.score_history.append(new_score)
    st.session_state.angle_history.append(angle)
    stage = posture["status"].lower()
    if st.session_state.last_stage != stage:
        if stage == "correct" and st.session_state.last_stage in ("almost", "incorrect", "idle"):
            st.session_state.correct_reps += 1
        elif stage == "incorrect":
            st.session_state.incorrect_reps += 1
        st.session_state.last_stage = stage
    if len(st.session_state.score_history) > 200:
        st.session_state.score_history = st.session_state.score_history[-200:]
        st.session_state.angle_history = st.session_state.angle_history[-200:]
    return new_score


def update_score(delta: int) -> int:
    hist = st.session_state.score_history
    prev = hist[-1] if hist else 50
    return max(0, min(100, prev + delta))


def score_color(s: int) -> str:
    if s >= 70: return "#15803d"
    if s >= 40: return "#c2570a"
    return "#b91c1c"


def kpi_color_class(status: str) -> str:
    return {"Correct": "c-green", "Almost": "c-orange", "Incorrect": "c-red"}.get(status, "c-blue")


def risk_color_class(risk: str) -> str:
    return {"LOW": "c-green", "MEDIUM": "c-orange", "HIGH": "c-red"}.get(risk, "c-blue")


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_score_chart(history):
    data = history[-30:]; x = list(range(len(data)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data, mode="lines",
        line=dict(color="#3b82f6", width=2.5, shape="spline"),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.08)"))
    fig.update_layout(height=140, margin=dict(l=8,r=8,t=8,b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0,100], gridcolor="#f1f5f9",
                   tickfont=dict(size=10, color="#94a3b8"), zeroline=False),
        showlegend=False)
    return fig


def build_angle_chart(history, exercise):
    data = history[-30:]; x = list(range(len(data)))
    target = EXERCISE_CONFIG[exercise]["correct_min"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=data, mode="lines+markers",
        line=dict(color="#8b5cf6", width=2, shape="spline"),
        marker=dict(size=4, color="#8b5cf6")))
    fig.add_hline(y=target, line=dict(color="#22c55e", dash="dot", width=1.5),
                  annotation_text="Target", annotation_position="right",
                  annotation_font=dict(size=10, color="#22c55e"))
    fig.update_layout(height=120, margin=dict(l=8,r=8,t=8,b=8),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(gridcolor="#f1f5f9",
                   tickfont=dict(size=10, color="#94a3b8"), zeroline=False),
        showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DEMO SKELETON SVG  (rendered via components.html — preserves SMIL animation)
# ─────────────────────────────────────────────────────────────────────────────
def body_svg_html(angle: float, posture: dict, exercise: str, running: bool) -> str:
    status = posture["status"]
    c  = STATUS_COLORS[status]["hex"]
    bs = "#94a3b8"
    cfg = EXERCISE_CONFIG[exercise]

    head  = (120, 42)
    sp_t  = (120, 68);  sp_b  = (120, 188)
    sh_l  = (62, 90);   sh_r  = (178, 90)
    hip_l = (88, 188);  hip_r = (152, 188)
    le    = (50, 146);  lw    = (46, 192)
    lk    = (80, 255);  la    = (78, 314)
    re    = [190, 146]; rw    = [194, 192]
    rk    = [160, 255]; ra    = [162, 314]
    r_hip_for_leg = list(hip_r)
    active_joint  = None
    r2d = math.pi / 180

    if exercise == "Shoulder Flexion":
        rad = (270 + angle) * r2d
        re  = [int(sh_r[0] + 52*math.cos(rad)), int(sh_r[1] + 52*math.sin(rad))]
        rw  = [int(re[0]   + 42*math.cos(rad)), int(re[1]   + 42*math.sin(rad))]
        active_joint = list(sh_r)
    elif exercise == "Elbow Flexion":
        re = [sh_r[0]+6, sh_r[1]+56]
        rw = [int(re[0]+46*math.sin(angle*r2d)), int(re[1]-46*math.cos(angle*r2d))]
        active_joint = list(re)
    elif exercise == "Knee Extension":
        sh = [hip_r[0], hip_r[1]+8]
        r_hip_for_leg = sh
        rk = [sh[0]+60, sh[1]+2]
        sr = (angle-90)*r2d
        ra = [int(rk[0]+58*math.cos(sr)), int(rk[1]+58*math.sin(sr))]
        active_joint = list(rk)
    elif exercise == "Hip Abduction":
        off = int(min(angle*0.85, 52))
        rk  = [hip_r[0]+off//2, hip_r[1]+62]
        ra  = [hip_r[0]+off,    hip_r[1]+122]
        active_joint = list(hip_r)

    target_angle = cfg["target_angle"]
    gre, grw, grk, gra = re[:], rw[:], rk[:], ra[:]
    if exercise == "Shoulder Flexion":
        rad_t = (270+target_angle)*r2d
        gre   = [int(sh_r[0]+52*math.cos(rad_t)), int(sh_r[1]+52*math.sin(rad_t))]
        grw   = [int(gre[0]+42*math.cos(rad_t)),  int(gre[1]+42*math.sin(rad_t))]
    elif exercise == "Elbow Flexion":
        gre = [sh_r[0]+6, sh_r[1]+56]
        grw = [int(gre[0]+46*math.sin(target_angle*r2d)), int(gre[1]-46*math.cos(target_angle*r2d))]
    elif exercise == "Knee Extension":
        sr_t = (target_angle-90)*r2d
        gra  = [int(rk[0]+58*math.cos(sr_t)), int(rk[1]+58*math.sin(sr_t))]
    elif exercise == "Hip Abduction":
        off_t = int(min(target_angle*0.85, 52))
        grk   = [hip_r[0]+off_t//2, hip_r[1]+62]
        gra   = [hip_r[0]+off_t,    hip_r[1]+122]

    arm_c = c  if exercise in ("Shoulder Flexion","Elbow Flexion") else bs
    leg_c = c  if exercise in ("Knee Extension","Hip Abduction")   else bs
    arm_w = 3.5 if arm_c != bs else 2.5
    leg_w = 3.5 if leg_c != bs else 2.5

    def seg(a, b, col=bs, w=2.5):
        return f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{col}" stroke-width="{w}" stroke-linecap="round"/>'

    def dot(p, r=5, f="white", s=bs, sw=2.2):
        return f'<circle cx="{p[0]}" cy="{p[1]}" r="{r}" fill="{f}" stroke="{s}" stroke-width="{sw}"/>'

    def pulse_ring(p):
        if running:
            anim_op = '<animate attributeName="opacity" values="0.22;0.04;0.22" dur="1.4s" repeatCount="indefinite"/>'
            anim_r  = '<animate attributeName="r" values="13;21;13" dur="1.4s" repeatCount="indefinite"/>'
        else:
            anim_op = anim_r = ""
        return (f'<circle cx="{p[0]}" cy="{p[1]}" r="13" fill="{c}" opacity="0.20">{anim_op}{anim_r}</circle>'
                f'<circle cx="{p[0]}" cy="{p[1]}" r="8" fill="{c}" stroke="white" stroke-width="2.5"/>')

    def angle_badge(p, val):
        ax, ay = p
        bw = 52; rx = max(4, min(ax-bw-6, 240-bw-4)); tx = rx+bw//2
        return (f'<rect x="{rx}" y="{ay-13}" width="{bw}" height="22" rx="7" fill="{c}" opacity="0.14"/>'
                f'<text x="{tx}" y="{ay+4}" text-anchor="middle" font-size="13" fill="{c}" font-weight="700" font-family="monospace">{val:.0f}°</text>')

    pc    = STATUS_COLORS[status]
    icons = {"Correct": "✓ Correct", "Almost": "~ Almost", "Incorrect": "✕ Incorrect"}
    g_alpha = "0.22"
    els = []

    if exercise in ("Shoulder Flexion","Elbow Flexion"):
        els.append(f'<line x1="{sh_r[0]}" y1="{sh_r[1]}" x2="{gre[0]}" y2="{gre[1]}" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="6 4" opacity="{g_alpha}" stroke-linecap="round"/>')
        els.append(f'<line x1="{gre[0]}" y1="{gre[1]}" x2="{grw[0]}" y2="{grw[1]}" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="6 4" opacity="{g_alpha}" stroke-linecap="round"/>')
    elif exercise == "Knee Extension":
        els.append(f'<line x1="{rk[0]}" y1="{rk[1]}" x2="{gra[0]}" y2="{gra[1]}" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="6 4" opacity="{g_alpha}" stroke-linecap="round"/>')
    elif exercise == "Hip Abduction":
        els.append(f'<line x1="{hip_r[0]}" y1="{hip_r[1]}" x2="{grk[0]}" y2="{grk[1]}" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="6 4" opacity="{g_alpha}" stroke-linecap="round"/>')
        els.append(f'<line x1="{grk[0]}" y1="{grk[1]}" x2="{gra[0]}" y2="{gra[1]}" stroke="#22c55e" stroke-width="2.5" stroke-dasharray="6 4" opacity="{g_alpha}" stroke-linecap="round"/>')

    els += [
        seg(sp_t,sp_b,bs,3), seg(sh_l,sh_r,bs,3), seg(hip_l,hip_r,bs,3),
        seg(sh_l,le,bs,2.5), seg(le,lw,bs,2.5),
        seg(hip_l,lk,bs,2.5), seg(lk,la,bs,2.5),
        seg(sh_r,re,arm_c,arm_w), seg(re,rw,arm_c,arm_w),
        seg(r_hip_for_leg,rk,leg_c,leg_w), seg(rk,ra,leg_c,leg_w),
    ]
    for p, r in [(sh_l,6),(sh_r,6),(hip_l,6),(hip_r,6),(le,5),(lw,4),(re,5),(rw,4),(lk,5),(la,4),(rk,5),(ra,4)]:
        els.append(dot(p, r))
    if active_joint:
        els.append(pulse_ring(active_joint))

    target_badge = (
        f'<rect x="158" y="8" width="74" height="20" rx="10" fill="#22c55e" opacity="0.12"/>'
        f'<text x="195" y="22" text-anchor="middle" font-size="10" fill="#15803d" font-weight="700" font-family="monospace">target {target_angle}°</text>')

    status_pill = (
        f'<rect x="65" y="350" width="110" height="24" rx="12" fill="{pc["hex"]}" opacity="0.15"/>'
        f'<text x="120" y="367" text-anchor="middle" font-size="11" fill="{pc["hex"]}" font-weight="700" font-family="sans-serif">{icons[status]}</text>')
    ex_label = f'<text x="120" y="390" text-anchor="middle" font-size="10" fill="#94a3b8" font-weight="600" font-family="sans-serif" letter-spacing="1.5">{exercise.upper()}</text>'
    body_html = "\n".join(els)
    a_badge   = angle_badge(active_joint, angle) if active_joint else ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden;}}svg{{display:block;margin:0 auto;}}</style>
</head><body>
<svg viewBox="0 0 240 400" width="220" height="367" style="max-height:380px;display:block;margin:0 auto;">
  <defs>
    <linearGradient id="sk-bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/><stop offset="100%" stop-color="#eef2f7"/>
    </linearGradient>
  </defs>
  <rect width="240" height="400" rx="14" fill="url(#sk-bg)"/>
  <rect width="240" height="400" rx="14" fill="none" stroke="#dde5f0" stroke-width="1.5"/>
  <circle cx="{head[0]}" cy="{head[1]}" r="26" fill="white" stroke="{bs}" stroke-width="2.5"/>
  <circle cx="113" cy="39" r="2.5" fill="{bs}"/>
  <circle cx="127" cy="39" r="2.5" fill="{bs}"/>
  <path d="M114 49 Q120 54 126 49" stroke="{bs}" stroke-width="1.5" fill="none" stroke-linecap="round"/>
  {body_html}
  {a_badge}
  {target_badge}
  {status_pill}
  {ex_label}
</svg></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# ── NEW ── CAMERA MODE: MediaPipe pose processing
# Extracts landmark coordinates, calls calculate_angle(), draws skeleton overlay
# Currently supports: Shoulder Flexion (hip → shoulder → elbow)
# Architecture: called once per rerun when running + Camera mode
# ─────────────────────────────────────────────────────────────────────────────
def process_camera_frame(frame, pose_model, mp_pose, mp_draw, mp_styles,
                          exercise: str, lang: str):
    """
    Process one camera frame:
      1. Convert BGR → RGB
      2. Run MediaPipe Pose
      3. Extract the 3 landmarks for the selected exercise
      4. Call calculate_angle() to get the joint angle
      5. Draw MediaPipe skeleton + HUD overlay on frame
      6. Return (annotated_frame, detected_angle, pose_detected)

    Shoulder Flexion landmark order: hip → shoulder → elbow (a→b→c)
    """
    import cv2

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    angle_out    = 0.0
    pose_detected = False

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # ── Map MediaPipe landmark names to indices ──
        LANDMARK_IDX = {
            "RIGHT_HIP":      mp_pose.PoseLandmark.RIGHT_HIP.value,
            "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            "RIGHT_ELBOW":    mp_pose.PoseLandmark.RIGHT_ELBOW.value,
            "RIGHT_WRIST":    mp_pose.PoseLandmark.RIGHT_WRIST.value,
            "RIGHT_KNEE":     mp_pose.PoseLandmark.RIGHT_KNEE.value,
            "RIGHT_ANKLE":    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        }

        j_names = EXERCISE_CONFIG[exercise]["mediapipe_joints"]

        # ── Extract pixel coordinates for the 3 joints ──
        def get_coord(name):
            idx = LANDMARK_IDX[name]
            return np.array([lm[idx].x * w, lm[idx].y * h])

        pt_a = get_coord(j_names[0])  # e.g. RIGHT_HIP
        pt_b = get_coord(j_names[1])  # e.g. RIGHT_SHOULDER  (vertex)
        pt_c = get_coord(j_names[2])  # e.g. RIGHT_ELBOW

        # ── Calculate angle using the shared helper ──
        angle_out     = calculate_angle(pt_a, pt_b, pt_c)
        pose_detected = True

        # ── Draw full MediaPipe skeleton ──
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_styles.get_default_pose_landmarks_style(),
        )

        # ── Draw angle arc at vertex joint (pt_b) ──
        vx, vy = int(pt_b[0]), int(pt_b[1])
        posture = classify_angle(angle_out, exercise, lang)
        status  = posture["status"]
        col_bgr = {
            "Correct":   (22,  163, 74),
            "Almost":    (234, 88,  12),
            "Incorrect": (220, 38,  38),
        }.get(status, (59, 130, 246))

        # Vertex circle highlight
        cv2.circle(frame, (vx, vy), 14, col_bgr, -1)
        cv2.circle(frame, (vx, vy), 14, (255, 255, 255), 2)

        # Angle text next to joint
        cv2.putText(frame, f"{angle_out:.0f}°",
                    (vx + 18, vy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, col_bgr, 2, cv2.LINE_AA)

        # ── HUD box (top-left) ──
        cv2.rectangle(frame, (8, 8), (320, 72), (255, 255, 255), -1)
        cv2.rectangle(frame, (8, 8), (320, 72), col_bgr, 2)
        cv2.putText(frame, f"{exercise}  {angle_out:.1f}°",
                    (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{status}  |  Target: {EXERCISE_CONFIG[exercise]['target_angle']}°",
                    (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col_bgr, 2, cv2.LINE_AA)

    else:
        # ── No pose detected: show warning overlay ──
        cv2.rectangle(frame, (8, 8), (300, 50), (255, 247, 237), -1)
        cv2.rectangle(frame, (8, 8), (300, 50), (234, 88, 12), 2)
        cv2.putText(frame, "No body detected — adjust position",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (154, 52, 18), 2, cv2.LINE_AA)

    return frame, angle_out, pose_detected


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  (extended with Camera Mode note)
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 RehabAI v2.4")
    st.markdown("---")

    st.markdown("**Patient Profile**")
    patient_name = st.text_input("Name", "Sarah Mitchell",
                                 label_visibility="collapsed", placeholder="Patient name")
    col_a, col_b = st.columns(2)
    with col_a:
        age = st.number_input("Age", 10, 100, 34, label_visibility="collapsed")
    with col_b:
        st.markdown(f"<div style='padding-top:8px;font-size:13px;'>Age: {age}</div>",
                    unsafe_allow_html=True)
    condition = st.selectbox("Condition",
        ["ACL Recovery","Shoulder Rehab","Hip Replacement","Stroke Recovery","Custom"])
    st.markdown("---")

    st.markdown("**Exercise Type**")
    exercise = st.selectbox("Exercise", list(EXERCISE_CONFIG.keys()),
        format_func=lambda x: f"{EXERCISE_CONFIG[x]['icon']} {x}",
        label_visibility="collapsed")
    st.markdown("---")

    st.markdown("**Monitoring Mode**")
    mode = st.radio("", ["Demo","Camera"], horizontal=True, label_visibility="collapsed")

    # ── Camera Mode note ──
    if mode == "Camera":
        cam_note_color = "#16a34a" if exercise == "Shoulder Flexion" else "#c2570a"
        cam_note_text  = ("✅ Full support" if exercise == "Shoulder Flexion"
                          else "⚠️ Prototype: Shoulder Flexion works best")
        st.markdown(f"""
        <div style="background:#0f2040;border:1px solid #1e3a5f;border-radius:8px;
                    padding:10px 12px;font-size:11px;margin-top:6px;color:{cam_note_color};">
            📷 Camera Mode<br>
            <span style="color:#94a3b8;">{cam_note_text}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Language / ภาษา**")
    lang = st.radio("", ["EN","TH"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")

    if mode == "Demo":
        st.markdown("**Simulate Angle**")
        cfg = EXERCISE_CONFIG[exercise]
        demo_angle = st.slider("Joint Angle (°)", 0, 170, cfg["almost_min"],
                               label_visibility="collapsed")

    target_reps = st.number_input("Target Reps", 5, 50, 15)
    st.markdown("---")

    b1, b2 = st.columns(2)
    with b1:
        start = st.button("▶ Start", type="primary", use_container_width=True)
    with b2:
        stop  = st.button("⏹ Stop",  use_container_width=True)
    reset = st.button("↺ Reset Session", use_container_width=True)

    if start: st.session_state.running = True
    if stop:  st.session_state.running = False
    if reset:
        for k, v in DEFAULTS.items():
            st.session_state[k] = v if not isinstance(v, list) else []
        st.rerun()

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:11px;color:#475569;text-align:center;line-height:1.9;">
        RehabAI v2.4 · {mode} Mode<br>
        <span style="font-family:'IBM Plex Mono',monospace;">
        {datetime.now().strftime('%H:%M · %b %d, %Y')}</span>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DETERMINE ANGLE & POSTURE
# Demo Mode:   angle = slider value
# Camera Mode: angle = st.session_state.cam_angle (updated inside camera block)
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Demo":
    angle = float(demo_angle)
else:
    # Use last known camera angle until the camera block updates it this rerun
    angle = st.session_state.cam_angle

posture       = classify_angle(angle, exercise, lang)
current_score = st.session_state.score_history[-1] if st.session_state.score_history else 50

# Demo Mode history update
if st.session_state.running and mode == "Demo":
    current_score = update_session_history(posture, angle)

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
mode_dot  = "🟢" if st.session_state.running else "⚪"
ex_cfg    = EXERCISE_CONFIG[exercise]
lang_flag = "🇹🇭" if lang == "TH" else "🇬🇧"
st.markdown(f"""
<div class="top-bar">
  <div>
    <div class="top-bar-title">RehabAI · Recovery Intelligence System</div>
    <div class="top-bar-sub">{patient_name} · {condition} · {age} yrs
      &nbsp;·&nbsp; {ex_cfg['icon']} {exercise} &nbsp;·&nbsp; {lang_flag} {lang}</div>
  </div>
  <div style="text-align:right;">
    <div class="top-bar-badge">{mode_dot} {mode} Mode</div>
    <div class="top-bar-time">{datetime.now().strftime('%H:%M:%S')}</div>
  </div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI BAR  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

def kpi_card(label, value, sub, cls):
    return f"""<div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {cls}">{value}</div>
      <div class="kpi-sub">{sub}</div></div>"""

with k1:
    st.markdown(kpi_card("Posture Status",
        f'{posture["icon"]} {posture["status"]}', "Real-time detection",
        kpi_color_class(posture["status"])), unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Recovery Score</div>
      <div class="kpi-value mono" style="color:{score_color(current_score)};">{current_score}</div>
      <div class="kpi-sub">out of 100</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Risk Level", posture["risk"], "Injury probability",
        risk_color_class(posture["risk"])), unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Joint Angle</div>
      <div class="kpi-value mono c-blue">{angle:.1f}°</div>
      <div class="kpi-sub">{ex_cfg['joint_label']} · target {ex_cfg['target_angle']}°</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT ROW
# ─────────────────────────────────────────────────────────────────────────────
main_col, right_col = st.columns([1.55, 1], gap="medium")

# ╔═══════════════════════════════════════════╗
# ║  CENTER PANEL                             ║
# ╚═══════════════════════════════════════════╝
with main_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    # ─────────────────────────── DEMO MODE (unchanged) ───────────────────────
    if mode == "Demo":
        st.markdown('<div class="section-label">📡 Simulated Monitoring Panel</div>',
                    unsafe_allow_html=True)
        desc = ex_cfg["description_th"] if lang == "TH" else ex_cfg["description"]
        jl   = ex_cfg["joint_label_th"] if lang == "TH" else ex_cfg["joint_label"]
        st.markdown(f"""
        <div class="exercise-badge">{ex_cfg['icon']} {exercise}
          <span style="font-weight:400;color:#3b82f6;font-size:12px;">· {jl} joint</span>
        </div>
        <div class="exercise-desc">{desc}</div>
        """, unsafe_allow_html=True)

        angle_col, skel_col = st.columns([1, 1.2])
        with angle_col:
            c_hex      = STATUS_COLORS[posture["status"]]["hex"]
            c_bg       = STATUS_COLORS[posture["status"]]["bg"]
            c_txt      = STATUS_COLORS[posture["status"]]["text"]
            target_pct = min(angle / ex_cfg["target_angle"], 1.0)
            st.markdown(f"""
            <div style="text-align:center;padding:16px 0 8px 0;">
              <div style="font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                          color:#6b7a99;margin-bottom:10px;">{jl} Angle</div>
              <div class="angle-display" style="color:{c_hex};">{angle:.0f}°</div>
              <div class="angle-target-label">Target: {ex_cfg['target_angle']}°</div>
              <div style="margin:14px auto 0;max-width:160px;">
                <div style="background:#e2e8f0;border-radius:6px;height:8px;overflow:hidden;">
                  <div style="width:{int(target_pct*100)}%;background:{c_hex};height:100%;border-radius:6px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#94a3b8;margin-top:3px;">
                  <span>0°</span><span>{ex_cfg['target_angle']}°</span>
                </div>
              </div>
              <div style="margin-top:16px;">
                <span style="background:{c_bg};color:{c_txt};border-radius:20px;padding:5px 18px;
                             font-size:14px;font-weight:700;border:1px solid {c_hex}33;">
                  {posture['icon']} {posture['status']}
                </span>
              </div>
            </div>""", unsafe_allow_html=True)

        with skel_col:
            # Skeleton SVG via iframe — preserves SMIL animation
            html_str = body_svg_html(angle, posture, exercise, st.session_state.running)
            components.html(html_str, height=385, scrolling=False)

    # ─────────────────────────── CAMERA MODE (new) ───────────────────────────
    else:
        st.markdown('<div class="section-label">📷 Live Camera Feed · MediaPipe Pose</div>',
                    unsafe_allow_html=True)

        # ── Check if MediaPipe + OpenCV are available ──
        try:
            import cv2
            import mediapipe as mp
            cam_libs_ok = True
        except ImportError:
            cam_libs_ok = False

        if not cam_libs_ok:
            # ── Fallback: libraries missing ──
            st.markdown("""
            <div style="background:#fff7ed;border:1.5px solid #fed7aa;border-radius:14px;
                        padding:32px;text-align:center;">
              <div style="font-size:40px;">⚠️</div>
              <div style="font-size:17px;font-weight:700;color:#9a3412;margin-top:12px;">
                Camera libraries not installed</div>
              <div style="color:#c2570a;font-size:14px;margin-top:10px;line-height:1.7;">
                Install required packages:<br>
                <code style="background:#fff3e0;padding:4px 10px;border-radius:6px;">
                  pip install mediapipe opencv-python-headless</code><br><br>
                Or switch to <strong>Demo Mode</strong> in the sidebar.
              </div>
            </div>""", unsafe_allow_html=True)

        elif not st.session_state.running:
            # ── Waiting to start ──
            st.markdown(f"""
            <div style="background:#f0f9ff;border:1.5px solid #bae6fd;border-radius:14px;
                        padding:32px;text-align:center;">
              <div style="font-size:40px;">📷</div>
              <div style="font-size:16px;font-weight:700;color:#0369a1;margin-top:12px;">
                Camera Mode Ready</div>
              <div style="color:#0284c7;font-size:13px;margin-top:10px;line-height:1.7;">
                Exercise: <strong>{ex_cfg['icon']} {exercise}</strong><br>
                Joint tracked: <strong>hip → shoulder → elbow</strong><br><br>
                Press <strong>▶ Start</strong> in the sidebar to activate webcam.
              </div>
            </div>""", unsafe_allow_html=True)

        else:
            # ══════════════════════════════════════════════════════════════════
            # ── LIVE CAMERA BLOCK ──
            # Opens webcam, runs MediaPipe, calls calculate_angle(),
            # updates session state, feeds real angle into dashboard logic
            # ══════════════════════════════════════════════════════════════════

            mp_pose   = mp.solutions.pose
            mp_draw   = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles

            # Info bar: shows which joints are being tracked
            jnames = EXERCISE_CONFIG[exercise]["mediapipe_joints"]
            st.markdown(f"""
            <div class="cam-info-bar">
              ℹ️ Tracking: <strong>{jnames[0]}</strong> → <strong>{jnames[1]}</strong>
              → <strong>{jnames[2]}</strong> &nbsp;|&nbsp;
              Vertex joint: <strong>{jnames[1]}</strong>
            </div>""", unsafe_allow_html=True)

            frame_slot = st.empty()
            cap        = cv2.VideoCapture(0)

            if not cap.isOpened():
                # ── Fallback: no webcam found ──
                st.markdown("""
                <div style="background:#fff1f2;border:1.5px solid #fecdd3;border-radius:14px;
                            padding:32px;text-align:center;">
                  <div style="font-size:40px;">🚫</div>
                  <div style="font-size:17px;font-weight:700;color:#9f1239;margin-top:12px;">
                    Camera not available</div>
                  <div style="color:#be123c;font-size:14px;margin-top:10px;">
                    No webcam detected on this device.<br>
                    Please switch to <strong>Demo Mode</strong> in the sidebar.
                  </div>
                </div>""", unsafe_allow_html=True)
                st.session_state.running = False

            else:
                with mp_pose.Pose(
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=0.55,
                    min_tracking_confidence=0.55,
                ) as pose_model:

                    ret, frame = cap.read()

                    if not ret:
                        st.markdown("""
                        <div class="cam-warning">
                          ⚠️ Could not read camera frame. Check webcam connection.
                        </div>""", unsafe_allow_html=True)
                    else:
                        # ── Process frame: pose detection + angle calculation ──
                        annotated_frame, cam_angle, pose_detected = process_camera_frame(
                            frame, pose_model, mp_pose, mp_draw, mp_styles, exercise, lang
                        )

                        # ── Show live video feed ──
                        frame_slot.image(
                            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            channels="RGB",
                            use_column_width=True,
                        )

                        if pose_detected:
                            # ── Feed real camera angle into dashboard logic ──
                            st.session_state.cam_angle       = cam_angle
                            st.session_state.cam_pose_detected = True

                            cam_posture   = classify_angle(cam_angle, exercise, lang)
                            current_score = update_session_history(cam_posture, cam_angle)

                            # Update locals so KPI cards and right panel reflect camera data
                            angle   = cam_angle
                            posture = cam_posture
                        else:
                            # No pose detected: keep last known angle, show warning
                            st.session_state.cam_pose_detected = False
                            st.markdown("""
                            <div class="cam-warning">
                              👤 No body detected — move into frame and face the camera.
                            </div>""", unsafe_allow_html=True)

                cap.release()

    st.markdown("</div>", unsafe_allow_html=True)

    # Angle history chart (shared by both modes)
    if len(st.session_state.angle_history) > 1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">📐 Joint Angle History</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(build_angle_chart(st.session_state.angle_history, exercise),
                        use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════╗
# ║  RIGHT PANEL — AI Insights (unchanged)    ║
# ╚═══════════════════════════════════════════╝
with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🤖 AI Insights</div>', unsafe_allow_html=True)

    risk_hex   = {"LOW":"#15803d","MEDIUM":"#c2570a","HIGH":"#b91c1c"}.get(posture["risk"],"#1d4ed8")
    status_hex = STATUS_COLORS.get(posture["status"],{}).get("hex","#1d4ed8")

    # Camera Mode: show pose detection status
    if mode == "Camera":
        dot_col  = "#16a34a" if st.session_state.cam_pose_detected else "#dc2626"
        dot_lbl  = "Pose detected" if st.session_state.cam_pose_detected else "No pose"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 0 12px 0;
                    border-bottom:1px solid #f1f5f9;margin-bottom:4px;">
          <div style="width:10px;height:10px;border-radius:50%;background:{dot_col};flex-shrink:0;"></div>
          <span style="font-size:12px;color:{dot_col};font-weight:700;">{dot_lbl}</span>
          <span style="font-size:11px;color:#94a3b8;margin-left:auto;">Live · MediaPipe</span>
        </div>""", unsafe_allow_html=True)

    rows = [
        ("Posture Status",
         f'<span style="color:{status_hex};font-weight:800;">{posture["icon"]} {posture["status"]}</span>'),
        ("Recovery Score",
         f'<span class="mono" style="color:{score_color(current_score)};font-weight:800;">{current_score} / 100</span>'),
        ("Risk Level",
         f'<span style="color:{risk_hex};font-weight:800;">{posture["risk"]}</span>'),
        ("Joint Angle",
         f'<span class="mono" style="color:#1d4ed8;font-weight:800;">{angle:.1f}°</span>'),
        ("Exercise",
         f'<span style="color:#6b21a8;font-weight:700;">{ex_cfg["icon"]} {exercise}</span>'),
        ("Target Angle",
         f'<span class="mono" style="color:#0369a1;font-weight:700;">{ex_cfg["target_angle"]}°</span>'),
    ]
    for key, val in rows:
        st.markdown(f'<div class="insight-item"><span class="insight-key">{key}</span>'
                    f'<span class="insight-value">{val}</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    lbl_rec = "คำแนะนำ AI" if lang == "TH" else "AI Recommendation"
    st.markdown(f"""
    <div style="font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
                color:#6b7a99;margin-bottom:8px;">{lbl_rec}</div>
    <div class="rec-box {posture['rec_cls']}">🧠 {posture['rec']}</div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recovery score trend
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📈 Recovery Score Trend</div>', unsafe_allow_html=True)
    if len(st.session_state.score_history) > 1:
        st.plotly_chart(build_score_chart(st.session_state.score_history),
                        use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""<div style="height:80px;display:flex;align-items:center;
            justify-content:center;color:#94a3b8;font-size:13px;">
            Press ▶ Start to begin recording…</div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Session summary
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📋 Session Summary</div>', unsafe_allow_html=True)

    total_reps   = st.session_state.correct_reps + st.session_state.incorrect_reps
    progress_pct = min(total_reps / max(target_reps, 1), 1.0)
    accuracy_pct = int(st.session_state.correct_reps / max(total_reps, 1) * 100) if total_reps > 0 else 0

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(f'<div class="sess-tile"><div class="sess-num" style="color:#15803d;">{st.session_state.correct_reps}</div><div class="sess-lbl">Correct</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="sess-tile"><div class="sess-num" style="color:#b91c1c;">{st.session_state.incorrect_reps}</div><div class="sess-lbl">Incorrect</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="sess-tile"><div class="sess-num" style="color:#1d4ed8;">{target_reps}</div><div class="sess-lbl">Target</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.progress(progress_pct)
    acc_color = "#15803d" if accuracy_pct>=70 else "#c2570a" if accuracy_pct>=40 else "#b91c1c"
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-top:4px;">
      <span>0</span>
      <span style="font-weight:700;color:#1d4ed8;">{total_reps} / {target_reps} reps</span>
      <span>{target_reps}</span>
    </div>
    <div style="margin-top:12px;padding:10px 14px;background:#f8fafc;
                border-radius:10px;border:1px solid #e2e8f0;">
      <div style="font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;
                  letter-spacing:.07em;margin-bottom:4px;">Session Accuracy</div>
      <div style="font-size:22px;font-weight:800;color:{acc_color};">{accuracy_pct}%</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AREAS FOR IMPROVEMENT  (unchanged from v2.3)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="aoi-section">
  <div class="aoi-header">🔬 Areas for Improvement — Roadmap to a Decision Support System</div>
  <div class="aoi-subheader">
    เพื่อยกระดับ RehabAI ให้เป็น Decision Support System ที่ช่วยผู้ป่วยได้อย่างแท้จริง
    &nbsp;|&nbsp; To evolve RehabAI into a fully actionable clinical decision support tool.
  </div>
  <div class="aoi-grid">
    <div class="aoi-card aoi-now">
      <span class="aoi-card-icon">📷</span>
      <div class="aoi-tag">ทำแล้ว v2.4 · Done</div>
      <div class="aoi-card-title">Camera Mode — MediaPipe Pose Integration</div>
      <div class="aoi-card-body">
        เพิ่ม Camera Mode ที่ใช้ MediaPipe Pose ตรวจจับท่าทางจากกล้อง Live แล้วคำนวณองศาข้อต่อ
        จริงส่งเข้า Dashboard โดยตรง — Shoulder Flexion (hip→shoulder→elbow) พร้อมใช้งานแล้ว<br><br>
        Camera Mode now uses MediaPipe to detect real shoulder angle live.
        The <code>calculate_angle(a, b, c)</code> helper is reusable for any exercise.
        Real angle feeds directly into posture classification, scoring, and AI recommendation.
      </div>
    </div>
    <div class="aoi-card aoi-now">
      <span class="aoi-card-icon">🎯</span>
      <div class="aoi-tag">ทำแล้ว v2.3 · Done</div>
      <div class="aoi-card-title">Actionable AI Recommendations</div>
      <div class="aoi-card-body">
        คำแนะนำระบุองศาปัจจุบัน + ต้องขยับเพิ่มอีกกี่องศา เช่น
        <em>"มุมเข่าแคบเกินไป (56°) กรุณายืดเพิ่ม 94° ให้ถึง 150°"</em><br><br>
        Angle-specific, joint-specific recommendations — no more vague "correct your posture".
      </div>
    </div>
    <div class="aoi-card aoi-now">
      <span class="aoi-card-icon">🇹🇭</span>
      <div class="aoi-tag">ทำแล้ว v2.3 · Done</div>
      <div class="aoi-card-title">Thai Language Support</div>
      <div class="aoi-card-body">
        เลือกภาษา EN / TH ใน Sidebar — คำแนะนำ AI, ชื่อข้อต่อ, คำอธิบายท่าเป็นภาษาไทยทั้งหมด
        เหมาะสำหรับผู้ป่วยสูงอายุในประเทศไทย<br><br>
        Toggle EN/TH in sidebar. All AI recommendations and labels switch to Thai.
      </div>
    </div>
    <div class="aoi-card aoi-med">
      <span class="aoi-card-icon">🔊</span>
      <div class="aoi-tag">แนะนำ · Next Step</div>
      <div class="aoi-card-title">Voice Feedback / เสียงแจ้งเตือน (TTS)</div>
      <div class="aoi-card-body">
        เพิ่ม Text-to-Speech ภาษาไทย (Web Speech API หรือ gTTS) เพื่ออ่านคำแนะนำออกเสียง
        ผู้ป่วยสูงอายุไม่ต้องอ่านหน้าจอขณะออกกำลังกาย<br><br>
        Add Thai TTS to read recommendations aloud. Critical for elderly users
        who cannot read a screen while exercising.
      </div>
    </div>
    <div class="aoi-card aoi-med">
      <span class="aoi-card-icon">📐</span>
      <div class="aoi-tag">แนะนำ · Next Step</div>
      <div class="aoi-card-title">Expand Camera Mode to All 4 Exercises</div>
      <div class="aoi-card-body">
        ขณะนี้ Camera Mode รองรับ Shoulder Flexion เต็มรูปแบบ ขั้นต่อไปคือขยาย
        <code>calculate_angle()</code> ไปยัง Elbow Flexion, Knee Extension, Hip Abduction
        โดยใช้ landmark triplets ที่กำหนดไว้ใน EXERCISE_CONFIG แล้ว<br><br>
        The architecture already supports all exercises via EXERCISE_CONFIG mediapipe_joints.
        Just validate landmark visibility and test per exercise.
      </div>
    </div>
    <div class="aoi-card aoi-later">
      <span class="aoi-card-icon">📷</span>
      <div class="aoi-tag">Camera Enhancement</div>
      <div class="aoi-card-title">Full-width Camera + Floating HUD</div>
      <div class="aoi-card-body">
        ขยาย camera feed ให้เต็ม center panel พร้อม HUD แสดงองศา+status ลอยทับวิดีโอ
        แทนการใช้ box ด้านบน ทำให้ผู้ป่วยมองเห็นตัวเองชัดเจนขึ้น<br><br>
        Expand video to full-width with a floating angle/status HUD overlay
        so patients can clearly see themselves while exercising.
      </div>
    </div>
    <div class="aoi-card aoi-vision">
      <span class="aoi-card-icon">🏥</span>
      <div class="aoi-tag">Vision · ระยะยาว</div>
      <div class="aoi-card-title">Physiotherapist Dashboard + Patient History</div>
      <div class="aoi-card-body">
        หน้า Dashboard สำหรับนักกายภาพ — ดู ROM ของผู้ป่วยหลายคน, ตั้ง target เฉพาะบุคคล,
        ส่ง PDF report อัตโนมัติ<br><br>
        A PT dashboard showing all patients' ROM trends, individual target configuration,
        and auto-generated PDF progress reports.
      </div>
    </div>
    <div class="aoi-card aoi-vision">
      <span class="aoi-card-icon">🤖</span>
      <div class="aoi-tag">Vision · ระยะยาว</div>
      <div class="aoi-card-title">LLM-powered Adaptive Coaching</div>
      <div class="aoi-card-body">
        ผสาน LLM เพื่อปรับแผนการออกกำลังกายอัตโนมัติตาม ROM trend และ pattern ของข้อผิดพลาด
        แนะนำ corrective exercises เฉพาะบุคคล<br><br>
        Integrate an LLM to adaptively adjust the exercise plan day-by-day,
        flag recurring posture errors, and suggest correctives.
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH
# Demo mode:   0.8s (smooth slider animation)
# Camera mode: 0.05s (near real-time frame rate)
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.running and mode == "Demo":
    time.sleep(0.8)
    st.rerun()
elif st.session_state.running and mode == "Camera":
    time.sleep(0.05)
    st.rerun()
