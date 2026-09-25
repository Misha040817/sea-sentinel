import os, io, cv2, tempfile, base64, sqlite3, json, subprocess, shutil, time, html
from datetime import datetime
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

st.set_page_config(page_title='Sea Sentinel', page_icon='🚢', layout='wide', initial_sidebar_state='expanded')
load_dotenv()

# ---------------- PROJECT CONSTANTS ----------------
WORKSPACE = 'misha-r'
WORKFLOW_ID = 'custom-workflow-2'
VESSEL_CLASSES = ['cargo_ship','container_ship','fishing_boat','military_vessel','passenger_ferry','speedboat','tanker','yacht']
DATASET_COUNTS = {
    'Cargo Ship':117, 'Container Ship':126, 'Fishing Boat':100, 'Military Vessel':208,
    'Passenger Ferry':111, 'Speedboat':78, 'Tanker':107, 'Yacht':151
}
DATASET = {'Total Images':2241, 'Training':1569, 'Validation':403, 'Testing':269, 'Classes':8}
MODEL = {'mAP@50':87.8, 'Precision':81.5, 'Recall':83.6, 'F1 Score':82.5}
CLASS_MAP50 = {'Cargo Ship':97,'Container Ship':98,'Fishing Boat':54,'Military Vessel':92,'Passenger Ferry':94,'Speedboat':84,'Tanker':86,'Yacht':98}

# ---------------- STYLE ----------------
st.markdown('''
<style>
:root{--navy:#071827;--panel:#0c2233;--cyan:#25c7e8;--muted:#C3D8E2;--line:rgba(104,166,195,.20)}
.stApp{background:radial-gradient(circle at 80% 5%,rgba(27,145,190,.10),transparent 25%),#06111b;color:#F4FAFD}
.block-container{max-width:1500px;padding-top:1.25rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#071827,#081e2e 58%,#06131e);border-right:1px solid var(--line)}
[data-testid="stSidebar"] .block-container{padding-top:1.1rem}
/* Sidebar contrast / navigation visibility */
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label{color:#F2FAFD!important;opacity:1!important}
[data-testid="stSidebar"] .stRadio label,[data-testid="stSidebar"] .stRadio label p,[data-testid="stSidebar"] div[role="radiogroup"] label,[data-testid="stSidebar"] div[role="radiogroup"] label p{color:#eef8fc!important;font-weight:600!important;opacity:1!important}
[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]{accent-color:#25c7e8!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{color:#C8DEE8!important;opacity:1!important}
[data-testid="stSidebar"] .eyebrow{color:#34d5f3!important}
[data-testid="stSidebar"] .brand h2{color:#ffffff!important}
[data-testid="stSidebar"] .brand p{color:#C8DEE8!important}
#MainMenu,footer{visibility:hidden} header{background:transparent!important}
.brand{padding:10px 3px 20px;border-bottom:1px solid var(--line);margin-bottom:16px}.brand h2{margin:0;color:#fff;font-size:1.35rem}.brand p{margin:4px 0 0;color:#C8DEE8;font-size:.76rem}
.eyebrow{font-size:.68rem;letter-spacing:1.4px;font-weight:800;color:#34c8e8;margin:18px 0 7px}
.hero{position:relative;overflow:hidden;min-height:245px;padding:34px 36px;border-radius:24px;border:1px solid rgba(72,188,222,.25);margin-bottom:18px;background:linear-gradient(90deg,rgba(3,18,29,.96) 0%,rgba(4,25,39,.88) 46%,rgba(4,25,39,.38) 100%),radial-gradient(circle at 82% 40%,rgba(37,199,232,.24),transparent 24%),linear-gradient(135deg,#0a3249,#0b2131);box-shadow:0 18px 55px rgba(0,0,0,.28)}
.hero:before{content:'';position:absolute;right:-60px;top:-105px;width:560px;height:560px;border-radius:50%;border:1px solid rgba(70,214,239,.18);box-shadow:0 0 0 55px rgba(70,214,239,.035),0 0 0 115px rgba(70,214,239,.025)}
.hero:after{content:'';position:absolute;right:30px;bottom:-54px;width:500px;height:190px;transform:skewX(-18deg) rotate(-4deg);background:linear-gradient(155deg,rgba(24,151,186,.26),rgba(5,37,55,.15));border-top:1px solid rgba(86,219,240,.22);box-shadow:-110px -34px 0 rgba(12,89,116,.12),-230px -60px 0 rgba(12,89,116,.08)}
.hero-content{position:relative;z-index:3;max-width:760px}.hero-kicker{font-size:.72rem;letter-spacing:2px;font-weight:900;color:#55d9f1;margin-bottom:10px}.hero h1{font-size:2.55rem;letter-spacing:-.8px;margin:0 0 7px;color:#fff}.hero p{color:#D7EAF2;margin:0;max-width:680px;font-size:.95rem;line-height:1.55}.hero-actions{margin-top:19px;display:flex;gap:9px;flex-wrap:wrap}.hero-chip{display:inline-block;padding:7px 10px;border-radius:999px;background:rgba(4,20,31,.55);border:1px solid rgba(93,205,231,.22);color:#EAF8FC;font-size:.72rem}.online{position:absolute;z-index:4;right:24px;top:22px;display:inline-block;padding:7px 12px;border-radius:999px;background:rgba(8,35,37,.68);border:1px solid rgba(34,197,94,.35);color:#7ee2a4;font-size:.75rem;font-weight:800;backdrop-filter:blur(8px)}
.ship-art{position:absolute;z-index:2;right:68px;bottom:39px;width:285px;height:72px;opacity:.86}.ship-hull{position:absolute;right:0;bottom:0;width:250px;height:24px;background:linear-gradient(90deg,#148eb1,#44d5ed);clip-path:polygon(0 0,100% 0,88% 100%,12% 100%)}.ship-deck{position:absolute;right:62px;bottom:24px;width:130px;height:22px;background:rgba(212,247,252,.85);clip-path:polygon(8% 0,88% 0,100% 100%,0 100%)}.ship-cabin{position:absolute;right:92px;bottom:46px;width:56px;height:18px;background:#e9fbff}.ship-stack{position:absolute;right:115px;bottom:64px;width:9px;height:12px;background:#25c7e8}.waterline{position:absolute;z-index:2;right:28px;bottom:23px;width:360px;height:1px;background:linear-gradient(90deg,transparent,#4ddaf2,transparent);box-shadow:0 8px 0 rgba(77,218,242,.22),0 16px 0 rgba(77,218,242,.10)}
@media(max-width:900px){.hero{min-height:230px;padding:28px 24px}.hero h1{font-size:2rem}.ship-art{opacity:.28;right:10px}.online{position:relative;right:auto;top:auto;margin-bottom:14px}.hero-content{max-width:100%}}
.kpi{background:rgba(12,34,51,.88);border:1px solid var(--line);border-radius:16px;padding:17px 18px;min-height:108px;box-shadow:0 10px 28px rgba(0,0,0,.15)}.kpi .name{color:#C7DBE5;font-size:.70rem;text-transform:uppercase;letter-spacing:.8px;font-weight:800}.kpi .value{font-size:1.75rem;font-weight:850;margin-top:7px}.kpi .note{color:#B8CED9;font-size:.72rem;margin-top:4px}
.section{font-size:1.05rem;font-weight:800;margin:12px 0}.panel{background:rgba(12,34,51,.75);border:1px solid var(--line);border-radius:17px;padding:18px;margin-bottom:16px}.soft{color:#C3D8E2;font-size:.86rem}.pill{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(37,199,232,.10);border:1px solid rgba(37,199,232,.25);color:#6edcf2;font-size:.72rem;margin:3px}
[data-testid="stFileUploader"]{background:rgba(10,30,45,.65);border:1px dashed rgba(82,175,210,.35);border-radius:16px;padding:8px}
.stButton>button,.stDownloadButton>button{border-radius:10px!important;font-weight:750!important}
hr{border-color:var(--line)!important}
/* High-contrast text across the entire dashboard */
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6{color:#F4FAFD!important}
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p{color:#C8DEE8!important;opacity:1!important}
.stApp [data-testid="stMarkdownContainer"] p{color:#F2FAFD!important}
.stApp .soft{color:#C8DEE8!important}
.stApp .kpi .name{color:#b7ccd7!important}.stApp .kpi .note{color:#C4DAE4!important}
[data-testid="stWidgetLabel"] p{color:#F4FAFD!important;font-weight:600!important}
[data-baseweb="select"] *{color:#F4FAFD!important}
[data-testid="stFileUploader"] small{color:#C8DEE8!important}
[data-testid="stDataFrame"]{color:#F4FAFD!important}

/* SEA SENTINEL ACCESSIBILITY / CONTRAST OVERRIDE */
html, body, [class*="css"] { color:#F4FAFD !important; }
.stApp { color:#F4FAFD !important; }
.stApp p, .stApp li, .stApp span:not([data-baseweb="tag"] span), .stApp label {
  color:#EAF6FB !important; opacity:1 !important; -webkit-text-fill-color:#EAF6FB !important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6 {color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important}
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stSlider"] label, [data-testid="stSlider"] p, [data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p {color:#EAF6FB!important;opacity:1!important;-webkit-text-fill-color:#EAF6FB!important}
[data-testid="stFileUploaderDropzone"] {background:#F5F7FA!important}
[data-testid="stFileUploaderDropzone"] p,[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] small {color:#34495E!important;-webkit-text-fill-color:#34495E!important}
[data-testid="stFileUploaderDropzone"] button {color:#173042!important;-webkit-text-fill-color:#173042!important}
[data-baseweb="select"] > div {background:#F5F7FA!important;color:#173042!important}
[data-baseweb="select"] input,[data-baseweb="select"] svg {color:#173042!important;-webkit-text-fill-color:#173042!important}
[data-baseweb="select"] span:not([data-baseweb="tag"] span) {color:#173042!important;-webkit-text-fill-color:#173042!important}
[data-baseweb="tag"] {color:#FFFFFF!important}
[data-baseweb="tag"] span {color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important}
[data-testid="stSidebar"] * {opacity:1!important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div[role="radiogroup"] label,[data-testid="stSidebar"] div[role="radiogroup"] label p {
 color:#F2FAFD!important;-webkit-text-fill-color:#F2FAFD!important;opacity:1!important;
}
[data-testid="stSidebar"] .eyebrow,[data-testid="stSidebar"] .eyebrow * {color:#35D8F4!important;-webkit-text-fill-color:#35D8F4!important}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
 color:#C8DEE8!important;-webkit-text-fill-color:#C8DEE8!important;opacity:1!important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover p {color:#55DDF5!important;-webkit-text-fill-color:#55DDF5!important}
[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {accent-color:#25C7E8!important}
.stAlert p,.stAlert span {color:inherit!important;-webkit-text-fill-color:currentColor!important}
[data-testid="stDataFrame"] * {opacity:1!important}
.soft{color:#C3D8E2!important;-webkit-text-fill-color:#C3D8E2!important}
.kpi .name{color:#C7DBE5!important;-webkit-text-fill-color:#C7DBE5!important}.kpi .note{color:#B8CED9!important;-webkit-text-fill-color:#B8CED9!important}

/* === FINAL ACCESSIBILITY / CONTRAST OVERRIDE === */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span:not([data-baseweb="tag"] span),
.stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stCaptionContainer"] p {
    color:#F4FAFD !important;
    opacity:1 !important;
}
.stApp [data-testid="stCaptionContainer"] p,
.stApp small, .stApp .soft, .stApp .kpi .note, .stApp .kpi .name {
    color:#C8DEE8 !important;
    opacity:1 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color:#F4FAFD !important;
    -webkit-text-fill-color:#F4FAFD !important;
    opacity:1 !important;
}
[data-testid="stSidebar"] .eyebrow,
[data-testid="stSidebar"] .eyebrow * {
    color:#35D8F4 !important;
    -webkit-text-fill-color:#35D8F4 !important;
}
/* Light controls need dark ink */
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-baseweb="select"] svg,
[data-baseweb="select"] > div span:not([data-baseweb="tag"] span),
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    color:#173042 !important;
    -webkit-text-fill-color:#173042 !important;
    opacity:1 !important;
}
[data-baseweb="tag"] span {
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
}
/* Slider and radio labels */
[data-testid="stSlider"] p, [data-testid="stSlider"] label,
[data-testid="stRadio"] p, [data-testid="stCheckbox"] p {
    color:#F4FAFD !important;
    -webkit-text-fill-color:#F4FAFD !important;
    opacity:1 !important;
}

/* ===== COMPETITION VISUAL POLISH ===== */
.kpi,.panel{border:2px solid rgba(34,211,238,.38)!important;box-shadow:0 12px 34px rgba(0,0,0,.25),inset 0 0 22px rgba(34,211,238,.025)!important}
.kpi:hover,.panel:hover{border-color:rgba(34,211,238,.72)!important;box-shadow:0 14px 38px rgba(0,0,0,.30),0 0 20px rgba(34,211,238,.08)!important}
.section{color:#FFFFFF!important;border-left:5px solid #22D3EE;padding-left:11px;line-height:1.15}
[data-testid="stPlotlyChart"]{background:linear-gradient(145deg,rgba(10,34,51,.96),rgba(6,23,36,.96));border:2px solid rgba(34,211,238,.42);border-radius:18px;padding:8px 10px 2px;box-shadow:0 12px 30px rgba(0,0,0,.24);overflow:hidden}
[data-testid="stDataFrame"]{border:2px solid rgba(34,211,238,.46)!important;border-radius:14px!important;overflow:hidden!important;box-shadow:0 10px 28px rgba(0,0,0,.22)!important;background:#0A2132!important}
[data-testid="stDataFrame"] [role="columnheader"]{background:#10354B!important;color:#FFFFFF!important;font-weight:800!important;border-bottom:2px solid #22D3EE!important}
[data-testid="stDataFrame"] [role="gridcell"]{color:#EAF6FB!important;border-bottom:1px solid rgba(120,188,214,.18)!important}
[data-testid="stFileUploader"]{border:2px dashed rgba(34,211,238,.62)!important;box-shadow:inset 0 0 20px rgba(34,211,238,.035)!important}
.stButton>button,.stDownloadButton>button{border:2px solid rgba(34,211,238,.65)!important;background:linear-gradient(135deg,#0E4057,#0A2D42)!important;color:#FFFFFF!important;box-shadow:0 8px 22px rgba(0,0,0,.22)!important;min-height:42px}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#67E8F9!important;background:linear-gradient(135deg,#12617B,#0B3C55)!important;box-shadow:0 0 18px rgba(34,211,238,.18)!important}
[data-testid="stExpander"]{border:2px solid rgba(100,171,201,.36)!important;border-radius:13px!important;background:rgba(12,34,51,.72)!important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:#F4FAFD!important;-webkit-text-fill-color:#F4FAFD!important;font-weight:700!important}
/* Ensure chart-adjacent markdown titles never fall back to dark grey */
.stApp [data-testid="stMarkdownContainer"] h1,.stApp [data-testid="stMarkdownContainer"] h2,.stApp [data-testid="stMarkdownContainer"] h3,.stApp [data-testid="stMarkdownContainer"] h4{color:#FFFFFF!important;-webkit-text-fill-color:#FFFFFF!important}
/* Stronger input outlines */
[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{border:2px solid rgba(83,142,171,.45)!important}
/* Raw prediction output: force Sea Sentinel dark theme and readable JSON */
[data-testid="stCodeBlock"]{background:#071827!important;border:2px solid rgba(34,211,238,.46)!important;border-radius:12px!important;overflow:hidden!important}
[data-testid="stCodeBlock"] pre,[data-testid="stCodeBlock"] code,[data-testid="stCodeBlock"] span{background:#071827!important;color:#EAF6FB!important;-webkit-text-fill-color:#EAF6FB!important;opacity:1!important}

</style>''', unsafe_allow_html=True)

st.markdown('''
<style>
/* FINAL: make expander header/label clearly visible */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
    opacity:1 !important;
    font-weight:800 !important;
}
[data-testid="stExpander"] summary {
    background:#173247 !important;
}
</style>
''', unsafe_allow_html=True)


st.markdown('''
<style>
/* === RAW PREDICTION OUTPUT: HIGH-CONTRAST LIGHT VIEWER === */
[data-testid="stJson"],
[data-testid="stJson"] > div,
[data-testid="stJson"] pre,
[data-testid="stJson"] code,
[data-testid="stJson"] span,
[data-testid="stJson"] div,
[data-testid="stJson"] button,
[data-testid="stJson"] svg,
[data-testid="stCode"],
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stCode"] span,
[data-testid="stCode"] div,
[data-testid="stCodeBlock"],
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code,
[data-testid="stCodeBlock"] span {
    color:#071827 !important;
    -webkit-text-fill-color:#071827 !important;
    opacity:1 !important;
}
[data-testid="stJson"] pre,
[data-testid="stJson"] code,
[data-testid="stCode"] pre,
[data-testid="stCode"] code,
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code {
    background:#F7FAFC !important;
}
</style>
''', unsafe_allow_html=True)


# ---------------- STATE ----------------
for key, default in {
    'image_predictions':[], 'image_obj':None, 'image_name':None, 'history':[],
    'video_stats':None, 'video_output':None, 'video_original':None, 'video_original_name':None, 'video_last_original':None, 'video_last_annotated':None, 'video_last_result':None, 'video_last_predictions':[], 'live_last_predictions':[], 'resilience_result':None
}.items():
    if key not in st.session_state: st.session_state[key] = default

# ---------------- HELPERS ----------------
def pretty(name): return str(name).replace('_',' ').title()
def kpi(name,value,note=''):
    st.markdown(f'<div class="kpi"><div class="name">{name}</div><div class="value">{value}</div><div class="note">{note}</div></div>',unsafe_allow_html=True)
def title(text, subtitle=''):
    st.markdown(f'''
    <div class="hero">
      <span class="online">● SYSTEM ONLINE</span>
      <div class="hero-content">
        <div class="hero-kicker">MARITIME VISION INTELLIGENCE</div>
        <h1>{text}</h1>
        <p>{subtitle}</p>
        <div class="hero-actions">
          <span class="hero-chip">⚓ YOLO26 Nano</span>
          <span class="hero-chip">◉ 8 Vessel Classes</span>
          <span class="hero-chip">⌁ Roboflow Workflow</span>
        </div>
      </div>
      <div class="ship-art"><div class="ship-hull"></div><div class="ship-deck"></div><div class="ship-cabin"></div><div class="ship-stack"></div></div>
      <div class="waterline"></div>
    </div>''', unsafe_allow_html=True)
def transparent(fig, height=340):
    # Competition-ready chart styling: high contrast, thick traces and vivid maritime colours.
    palette = ['#22D3EE','#3B82F6','#8B5CF6','#14B8A6','#F59E0B','#F43F5E','#84CC16','#EC4899']
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(8,27,42,0.92)',
        plot_bgcolor='rgba(8,27,42,0.92)',
        font=dict(color='#F4FAFD', family='Arial', size=13),
        title=dict(font=dict(color='#FFFFFF', size=18, family='Arial Black'), x=0.02, xanchor='left'),
        margin=dict(l=42,r=28,t=65,b=42),
        colorway=palette,
        legend=dict(
            bgcolor='rgba(7,24,39,0.70)', bordercolor='rgba(34,211,238,0.45)', borderwidth=1,
            font=dict(color='#EAF6FB', size=12), title_font=dict(color='#FFFFFF', size=12)
        ),
        hoverlabel=dict(bgcolor='#0B2234', bordercolor='#22D3EE', font=dict(color='#FFFFFF', size=13))
    )
    fig.update_xaxes(
        gridcolor='rgba(121,184,211,0.16)', zerolinecolor='rgba(121,184,211,0.22)',
        linecolor='rgba(34,211,238,0.48)', tickfont=dict(color='#D8ECF5'),
        title_font=dict(color='#F4FAFD', size=14), showline=True, linewidth=1.4
    )
    fig.update_yaxes(
        gridcolor='rgba(121,184,211,0.16)', zerolinecolor='rgba(121,184,211,0.22)',
        linecolor='rgba(34,211,238,0.48)', tickfont=dict(color='#D8ECF5'),
        title_font=dict(color='#F4FAFD', size=14), showline=True, linewidth=1.4
    )
    # Thicker line/scatter traces and outlined bars/pie slices where supported.
    fig.update_traces(selector=dict(type='scatter'), line=dict(width=4), marker=dict(size=9, line=dict(width=1.5, color='#EAFBFF')))
    fig.update_traces(selector=dict(type='bar'), marker_line_color='rgba(232,251,255,0.75)', marker_line_width=1.5, textfont=dict(color='#FFFFFF', size=13))
    fig.update_traces(selector=dict(type='pie'), marker=dict(line=dict(color='#071827', width=3)), textfont=dict(color='#FFFFFF', size=13))
    return fig
def get_client():
    key = os.getenv('ROBOFLOW_API_KEY')
    if not key:
        try:
            key = st.secrets.get('ROBOFLOW_API_KEY')
        except Exception:
            key = None
    if not key:
        st.error('ROBOFLOW_API_KEY is missing. Add it to .env locally or Streamlit Secrets when deployed.')
        st.stop()
    return InferenceHTTPClient(api_url='https://serverless.roboflow.com', api_key=key)
def box_iou(a, b):
    """IoU for Roboflow center-format boxes (x, y, width, height)."""
    def corners(p):
        x=float(p.get('x',0) or 0); y=float(p.get('y',0) or 0)
        w=float(p.get('width',0) or 0); h=float(p.get('height',0) or 0)
        return x-w/2, y-h/2, x+w/2, y+h/2
    ax1,ay1,ax2,ay2=corners(a); bx1,by1,bx2,by2=corners(b)
    iw=max(0.0,min(ax2,bx2)-max(ax1,bx1)); ih=max(0.0,min(ay2,by2)-max(ay1,by1))
    inter=iw*ih; union=max(0.0,(ax2-ax1)*(ay2-ay1))+max(0.0,(bx2-bx1)*(by2-by1))-inter
    return inter/union if union>0 else 0.0

def apply_nms(predictions, iou_threshold=0.30):
    """Class-aware NMS so the displayed IoU threshold is a real post-processing control."""
    ordered=sorted(predictions or [],key=lambda p:float(p.get('confidence',0) or 0),reverse=True)
    kept=[]
    for pred in ordered:
        cls=str(pred.get('class',''))
        if all(str(k.get('class',''))!=cls or box_iou(pred,k)<=iou_threshold for k in kept):
            kept.append(pred)
    return kept

def run_workflow(image, classes, confidence, iou_threshold, text_scale, box_thickness, box_palette, label_palette):
    result = get_client().run_workflow(workspace_name=WORKSPACE,workflow_id=WORKFLOW_ID,images={'image':image},parameters={
        'class_filter':classes,'text_scale':text_scale,'text_color':'Black','confidence':confidence,'text_thickness':1,
        'text_position':'CENTER','bounding_box_thickness':box_thickness,'bounding_box_color_palette':box_palette,'label_color_palette':label_palette})
    predictions=result[0].get('predictions',{}).get('predictions',[])
    return apply_nms(predictions,iou_threshold)

def prediction_detail_df(predictions):
    rows=[]
    for i,p in enumerate(predictions or [],1):
        rows.append({'#':i,'Class':pretty(p.get('class','unknown')),'Confidence (%)':round(float(p.get('confidence',0) or 0)*100,1),
                     'X':round(float(p.get('x',0) or 0),1),'Y':round(float(p.get('y',0) or 0),1),
                     'Width':round(float(p.get('width',0) or 0),1),'Height':round(float(p.get('height',0) or 0),1)})
    return pd.DataFrame(rows)

def show_detection_results(predictions, heading='Detection Results', image_page=False, clean_table=False):
    st.markdown(f'<div class="section">{heading}</div>',unsafe_allow_html=True)
    df=prediction_detail_df(predictions)
    if df.empty:
        st.info('No detections above the selected confidence threshold.')
    else:
        if image_page or clean_table:
            # Clean HTML table avoids Streamlit dataframe bottom/focus underline.
            # dataframe cyan bottom/focus line cannot appear.
            display_df=df.copy()
            display_df['Confidence (%)']=display_df['Confidence (%)'].map(lambda v: f'{v:.1f}')
            for col in ['X','Y','Width','Height']:
                display_df[col]=display_df[col].map(lambda v: f'{v:g}' if isinstance(v,(int,float,np.integer,np.floating)) else v)
            table_html=display_df.to_html(index=False,escape=True,classes='image-results-clean-table')
            image_table_html = '''<style>
.image-results-table-wrap{width:100%;overflow:hidden;border:1px solid #B8CBD5;border-radius:10px;background:#FFFFFF;margin:0;padding:0;box-shadow:none}
table.image-results-clean-table{width:100%;border-collapse:collapse;border-spacing:0;margin:0!important;background:#FFFFFF;color:#243746!important;font-size:14px}
table.image-results-clean-table thead th{background:#F4F6F8!important;color:#6B7280!important;-webkit-text-fill-color:#6B7280!important;text-align:left;font-weight:500;padding:11px 10px;border-right:1px solid #E1E5E9;border-bottom:1px solid #DDE3E7}
table.image-results-clean-table tbody td{background:#FFFFFF!important;color:#374151!important;-webkit-text-fill-color:#374151!important;padding:11px 10px;border-right:1px solid #E5E7EB;border-bottom:0!important;vertical-align:middle}
table.image-results-clean-table th:last-child,table.image-results-clean-table td:last-child{border-right:0}
table.image-results-clean-table tbody tr:last-child td{border-bottom:0!important}
table.image-results-clean-table th:first-child,table.image-results-clean-table td:first-child{text-align:right;width:9%}
table.image-results-clean-table td:nth-child(n+3){text-align:right}
</style><div class="image-results-table-wrap">''' + table_html + '''</div>'''
            st.html(image_table_html)
        else:
            st.dataframe(df,width='stretch',hide_index=True)
        with st.expander('Raw Prediction Output'):
            readable_json(predictions)
def draw_detections(image,predictions,thickness=3):
    out=image.copy(); draw=ImageDraw.Draw(out)
    try: font=ImageFont.truetype('arial.ttf',20)
    except: font=ImageFont.load_default()
    for p in predictions:
        x,y,w,h=[p.get(k,0) for k in ('x','y','width','height')]; conf=p.get('confidence',0); cls=pretty(p.get('class','unknown'))
        x1,y1,x2,y2=int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)
        draw.rectangle([x1,y1,x2,y2],outline='#28d0ef',width=max(1,int(thickness)))
        label=f'{cls} {conf*100:.1f}%'; box=draw.textbbox((0,0),label,font=font); tw,th=box[2]-box[0],box[3]-box[1]; ly=max(0,y1-th-10)
        draw.rounded_rectangle([x1,ly,x1+tw+12,ly+th+8],radius=4,fill='#071827'); draw.text((x1+6,ly+4),label,fill='white',font=font)
    return out
def png_bytes(img):
    b=io.BytesIO(); img.save(b,format='PNG'); return b.getvalue()

def draw_video_detections(image, predictions, thickness=6):
    """Competition-readable annotations for Video Detection only."""
    out=image.copy(); draw=ImageDraw.Draw(out)
    # Scale label text with video resolution while keeping it readable on large displays.
    font_size=max(22, min(28, int(min(out.size) * 0.040)))
    font=None
    for font_name in ('arialbd.ttf','Arial Bold.ttf','arial.ttf'):
        try:
            font=ImageFont.truetype(font_name,font_size); break
        except Exception:
            pass
    if font is None: font=ImageFont.load_default()
    line_width=max(3, min(4, int(thickness)))
    for p in predictions:
        x,y,w,h=[p.get(k,0) for k in ('x','y','width','height')]
        conf=float(p.get('confidence',0) or 0); cls=pretty(p.get('class','unknown')).upper()
        x1,y1,x2,y2=int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2)
        draw.rectangle([x1,y1,x2,y2],outline='#22D3EE',width=line_width)
        label=f'{cls}  {conf*100:.1f}%'
        box=draw.textbbox((0,0),label,font=font); tw,th=box[2]-box[0],box[3]-box[1]
        pad_x=max(10,font_size//3); pad_y=max(6,font_size//5)
        label_h=th+2*pad_y
        ly=y1-label_h if y1>=label_h else y1
        label_w=tw+2*pad_x
        lx=max(0, min(x1, out.size[0]-label_w))
        draw.rounded_rectangle([lx,ly,lx+label_w,ly+label_h],radius=5,fill='#061827',outline='#22D3EE',width=2)
        draw.text((lx+pad_x,ly+pad_y),label,fill='#FFFFFF',font=font)
    return out


def readable_json(data):
    """Render JSON vertically with guaranteed line breaks and indentation."""
    raw = json.dumps(data, indent=4, default=str)
    safe = html.escape(raw)
    safe = safe.replace(" ", "&nbsp;").replace("\\n", "<br>")

    st.markdown(
        f"""<div style="
            background:#071827;
            border:2px solid #22D3EE;
            border-radius:12px;
            padding:18px 20px;
            box-shadow:0 0 16px rgba(34,211,238,.08);
            overflow-x:auto;
            color:#FFFFFF !important;
            -webkit-text-fill-color:#FFFFFF !important;
            font-family:Consolas,'Courier New',monospace;
            font-size:14px;
            line-height:1.65;
            font-weight:600;
        ">{safe}</div>""",
        unsafe_allow_html=True
    )


# ---------------- MODEL RESILIENCE HELPERS ----------------
def apply_visual_degradation(image, degradation, severity):
    """Create a controlled degraded copy of a PIL image for resilience testing."""
    rgb = np.array(image.convert('RGB'))
    level = max(1, min(int(severity), 5))

    if degradation == 'Gaussian Noise':
        sigma = [8, 16, 25, 35, 48][level - 1]
        noise = np.random.default_rng(42).normal(0, sigma, rgb.shape)
        degraded = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif degradation == 'Blur':
        kernel = [3, 5, 7, 11, 15][level - 1]
        degraded = cv2.GaussianBlur(rgb, (kernel, kernel), 0)

    elif degradation == 'Low Light':
        factor = [0.80, 0.65, 0.50, 0.35, 0.22][level - 1]
        degraded = np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    elif degradation == 'Fog / Haze':
        alpha = [0.12, 0.22, 0.32, 0.45, 0.58][level - 1]
        haze = np.full_like(rgb, 225)
        degraded = cv2.addWeighted(rgb, 1.0 - alpha, haze, alpha, 0)
        blur_kernel = [1, 3, 3, 5, 7][level - 1]
        if blur_kernel > 1:
            degraded = cv2.GaussianBlur(degraded, (blur_kernel, blur_kernel), 0)
    else:
        degraded = rgb.copy()

    return Image.fromarray(degraded)

def summarize_predictions(predictions):
    """Return compact detection statistics without treating detections as unique vessels."""
    predictions = predictions or []
    confidences = [float(p.get('confidence', 0) or 0) for p in predictions]
    top = max(predictions, key=lambda p: float(p.get('confidence', 0) or 0), default=None)
    return {
        'detections': len(predictions),
        'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
        'max_confidence': float(max(confidences)) if confidences else 0.0,
        'top_class': pretty(top.get('class', 'None')) if top else 'No Detection',
        'top_confidence': float(top.get('confidence', 0) or 0) if top else 0.0,
    }

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sea_sentinel_history.db')

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_type TEXT NOT NULL,
            filename TEXT,
            detections INTEGER DEFAULT 0,
            avg_confidence REAL DEFAULT 0,
            max_confidence REAL DEFAULT 0,
            vessel_types INTEGER DEFAULT 0,
            threshold REAL DEFAULT 0,
            class_counts TEXT DEFAULT '{}',
            details TEXT DEFAULT '[]'
        )""")
        conn.commit()

def save_history(source_type, filename, predictions=None, threshold=0.0, detections=None, confidences=None, class_counts=None, details=None):
    predictions = predictions or []
    if confidences is None:
        confidences = [float(p.get('confidence', 0) or 0) for p in predictions]
    if class_counts is None:
        class_counts = Counter(pretty(p.get('class','unknown')) for p in predictions)
    if detections is None:
        detections = len(predictions)
    if details is None:
        details = predictions
    avg = float(np.mean(confidences)) if confidences else 0.0
    mx = float(max(confidences)) if confidences else 0.0
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""INSERT INTO detection_history
            (timestamp, source_type, filename, detections, avg_confidence, max_confidence, vessel_types, threshold, class_counts, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(timespec='seconds'), source_type, filename, int(detections), avg, mx, len(class_counts), float(threshold), json.dumps(dict(class_counts)), json.dumps(details, default=str)))
        conn.commit()

def load_history(limit=500):
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM detection_history ORDER BY id DESC LIMIT ?", conn, params=(limit,))

def clear_history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('DELETE FROM detection_history'); conn.commit()

def history_preview(limit=8, heading='Recent Detection History'):
    hist = load_history(limit)
    st.markdown(f'<div class="section">{heading}</div>', unsafe_allow_html=True)
    if hist.empty:
        st.info('No saved detections yet. Run Image Detection or Video Detection; each completed analysis will be saved automatically.')
        return
    view = hist[['timestamp','source_type','filename','detections','avg_confidence','vessel_types']].copy()
    view['timestamp'] = pd.to_datetime(view['timestamp']).dt.strftime('%d %b %Y, %H:%M:%S')
    view['avg_confidence'] = (view['avg_confidence'] * 100).round(1).astype(str) + '%'
    view.columns = ['Date / Time','Source','File','Detections','Avg Confidence','Vessel Types']
    st.dataframe(view, width='stretch', hide_index=True)
    st.caption(f'Persistent local history database: {os.path.basename(DB_PATH)} • {len(hist)} recent record(s) shown')

init_db()

def detection_controls(prefix='img'):
    classes=st.multiselect('Vessel classes',VESSEL_CLASSES,default=VESSEL_CLASSES,format_func=pretty,key=f'{prefix}_classes')
    confidence=st.slider('Confidence threshold',0.10,1.00,0.40,0.05,key=f'{prefix}_conf')
    iou=st.slider('IoU threshold',0.10,0.90,0.30,0.05,key=f'{prefix}_iou',help='Class-aware non-maximum suppression threshold for overlapping detections. Lower values suppress overlapping boxes more aggressively.')
    box=st.slider('Bounding box thickness',1,6,3,key=f'{prefix}_box')
    text=st.slider('Text scale',0.5,2.0,0.7,0.1,key=f'{prefix}_text')
    label=st.selectbox('Label color palette',['Matplotlib Pastel1','Matplotlib Set1','Matplotlib Tab10'],key=f'{prefix}_label')
    palette=st.selectbox('Bounding box palette',['ROBOFLOW','Matplotlib Cividis'],key=f'{prefix}_palette')
    return classes,confidence,iou,box,text,label,palette

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="brand"><h2>🌊 SEA SENTINEL</h2><p>Maritime Vision Intelligence</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">NAVIGATION</div>',unsafe_allow_html=True)
    page=st.radio('Page',['Dashboard','Image Detection','Video Detection','Live Detection','Detection Analytics','Dataset Analytics','Model Performance','Methodology','About Sea Sentinel'],label_visibility='collapsed')
    st.markdown('<div class="eyebrow">SYSTEM</div>',unsafe_allow_html=True)
    st.caption('Model · YOLO26 Nano')
    st.caption('Workflow · Roboflow')
    st.caption('Classes · 8 vessel classes')

# ---------------- DASHBOARD ----------------
if page=='Dashboard':
    title('SEA SENTINEL','Maritime Vessel Object Detection & Visual Analytics')
    st.markdown('AI-powered computer vision for maritime vessel detection, dataset exploration and model-performance reporting.')
    cols=st.columns(4)
    for c,(n,v,note) in zip(cols,[('Total Images','2,241','Maritime dataset'),('Training','1,569','70.0% of dataset'),('Validation','403','18.0% of dataset'),('Testing','269','12.0% of dataset')]):
        with c:kpi(n,v,note)
    st.write('')
    left,right=st.columns([1.25,1],gap='large')
    with left:
        st.markdown('<div class="section">Dataset Class Distribution</div>',unsafe_allow_html=True)
        df=pd.DataFrame({'Vessel':DATASET_COUNTS.keys(),'Images':DATASET_COUNTS.values()})
        fig=px.bar(df,x='Vessel',y='Images',text='Images'); fig.update_traces(textposition='outside'); st.plotly_chart(transparent(fig),width='stretch',config={'displayModeBar':False})
    with right:
        st.markdown('<div class="section">Model Performance</div>',unsafe_allow_html=True)
        mc=st.columns(2)
        for i,(n,v) in enumerate(MODEL.items()):
            with mc[i%2]:kpi(n,f'{v:.1f}%','Validation result')
        st.markdown('<div class="panel"><b>Model information</b><br><br><span class="soft">Model</span> &nbsp; YOLO26 Nano<br><span class="soft">Platform</span> &nbsp; Roboflow Workflows<br><span class="soft">Input</span> &nbsp; Maritime imagery<br><span class="soft">Classes</span> &nbsp; 8 vessel classes</div>',unsafe_allow_html=True)
    history_preview(8)

# ---------------- IMAGE ----------------
elif page=='Image Detection':
    st.markdown('''<style>
    /* Image Detection only: make image toolbar / zoom-fullscreen controls visible */
    [data-testid="stImage"] [data-testid="stElementToolbar"] button,
    [data-testid="stImage"] [data-testid="stElementToolbar"] button *,
    [data-testid="stImage"] [data-testid="stElementToolbar"] svg{
        color:#071827!important;
        stroke:#071827!important;
        fill:#071827!important;
        opacity:1!important;
    }
    [data-testid="stImage"] [data-testid="stElementToolbar"] button{
        background:#EAF6FB!important;
        border:1px solid #9FC7D8!important;
        border-radius:7px!important;
    }
    </style>''',unsafe_allow_html=True)
    title('Image Vessel Detection','Upload a maritime image and inspect AI detections in real time.')
    settings,result_area=st.columns([1,3],gap='large')
    with settings:
        st.markdown('<div class="section">Detection Configuration</div>',unsafe_allow_html=True)
        classes,conf,iou,box,text_scale,label_palette,box_palette=detection_controls('img')
    with result_area:
        upload=st.file_uploader('Upload maritime image',type=['jpg','jpeg','png'],key='image_upload')
        if upload:
            image=Image.open(upload).convert('RGB')
            if st.button('🚀 Run Object Detection',type='primary',width='stretch'):
                with st.spinner('Sea Sentinel is analysing the image...'):
                    try:
                        preds=run_workflow(image,classes,conf,iou,text_scale,box,box_palette,label_palette)
                        st.session_state.image_predictions=preds; st.session_state.image_obj=image; st.session_state.image_name=upload.name
                        save_history('Image', upload.name, predictions=preds, threshold=conf)
                        st.success('Detection completed and saved to Detection History.')
                    except Exception as e: st.error(f'Roboflow workflow error: {e}')
        if st.session_state.image_obj is not None:
            image=st.session_state.image_obj; preds=st.session_state.image_predictions; annotated=draw_detections(image,preds,box)
            a,b=st.columns(2)
            with a: st.markdown('<div class="section">Input</div>',unsafe_allow_html=True); st.image(image,width='stretch')
            with b: st.markdown('<div class="section">Output</div>',unsafe_allow_html=True); st.image(annotated,width='stretch'); st.download_button('⬇ Download Detected Image',png_bytes(annotated),'sea_sentinel_detection.png','image/png',width='stretch')
            confidences=[p.get('confidence',0) for p in preds]; classes_found=[pretty(p.get('class','unknown')) for p in preds]
            metrics=st.columns(3); vals=[('Vessels Detected',len(preds)),('Vessel Types',len(set(classes_found))),('Highest Confidence',f'{max(confidences)*100:.1f}%' if confidences else '0.0%')]
            for c,(n,v) in zip(metrics,vals):
                with c:kpi(n,v,'Current uploaded image')
            show_detection_results(preds,'Detection Results',image_page=True)
    history_preview(6, 'Recent Saved Detections')

# ---------------- VIDEO ----------------
elif page=='Video Detection':
    title('Video Vessel Detection','Analyse maritime video frames using the same Sea Sentinel detection workflow.')
    left,right=st.columns([1,2.5],gap='large')
    with left:
        classes,conf,iou,box,text_scale,label_palette,box_palette=detection_controls('vid')
        frame_skip=st.slider('Process every Nth frame',1,30,10,key='frame_skip')
    with right:
        video=st.file_uploader('Upload maritime video',type=['mp4','avi','mov','mkv'],key='video_upload')
        if video:
            video_size_mb=len(video.getvalue())/(1024*1024)
            safe_video_name=str(video.name).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
            st.markdown(
                f'''<div class="panel" style="text-align:center;padding:30px 24px;margin:14px 0 16px 0;">
                    <div style="font-size:34px;line-height:1;margin-bottom:10px;">🎬</div>
                    <div style="font-size:19px;font-weight:800;color:#FFFFFF;letter-spacing:.4px;">VIDEO READY FOR ANALYSIS</div>
                    <div style="margin-top:8px;color:#22D3EE;font-weight:700;">{safe_video_name}</div>
                    <div style="margin-top:4px;color:#BFD7E3;font-size:13px;">{video_size_mb:.1f} MB • Upload successful</div>
                    <div style="max-width:650px;margin:14px auto 0 auto;color:#E8F3F8;line-height:1.55;">
                        Select the detection settings and click <b>Analyse Video</b> to begin YOLO26 Nano vessel detection.
                        Processed frames will appear in the Live Detection Preview below.
                    </div>
                    <div style="margin-top:14px;color:#7DE7F5;font-weight:700;">✓ Ready for YOLO26 Analysis</div>
                </div>''',
                unsafe_allow_html=True
            )
        process=st.button('▶ Analyse Video',type='primary',width='stretch',disabled=video is None)
    if video and process:
        suffix=os.path.splitext(video.name)[1]
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f: f.write(video.getvalue()); input_path=f.name
        cap=cv2.VideoCapture(input_path)
        if not cap.isOpened(): st.error('Unable to open uploaded video.')
        else:
            fps=cap.get(cv2.CAP_PROP_FPS) or 25; total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_path=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4').name
            writer=cv2.VideoWriter(output_path,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
            st.markdown('<div class="section">Live Detection Preview</div>',unsafe_allow_html=True)
            st.caption('Processed frames are displayed immediately below while Sea Sentinel analyses the uploaded video. This avoids browser video-codec limitations during the live demonstration.')
            live_left, live_right = st.columns(2, gap='large')
            with live_left:
                st.markdown('### Current Original Frame')
                original_frame_slot = st.empty()
            with live_right:
                st.markdown('### Current AI Detection')
                detected_frame_slot = st.empty()
                live_result_slot = st.empty()

            progress=st.progress(0); status=st.empty(); frame_no=processed=detections=0; confs=[]; class_counts=Counter(); timeline=[]; last_annotated=None
            while True:
                ok,frame=cap.read()
                if not ok: break
                out_frame=frame
                if frame_no % frame_skip==0:
                    processed+=1; status.text(f'Analysing frame {frame_no+1} / {max(total,1)}')
                    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                    original_frame_slot.image(pil, width=500)
                    try:
                        preds=run_workflow(pil,classes,conf,iou,text_scale,box,box_palette,label_palette); detections+=len(preds)
                        frame_confs=[float(p.get('confidence',0) or 0) for p in preds]
                        confs.extend(frame_confs); class_counts.update([pretty(p.get('class','unknown')) for p in preds]); timeline.append({'Frame':frame_no+1,'Detections':len(preds)})
                        last_annotated=draw_video_detections(pil,preds,max(5,box)); out_frame=cv2.cvtColor(np.array(last_annotated),cv2.COLOR_RGB2BGR)
                        detected_frame_slot.image(last_annotated, width=500)
                        st.session_state.video_last_original = pil
                        st.session_state.video_last_annotated = last_annotated
                        st.session_state.video_last_predictions = preds
                        if preds:
                            top=max(preds,key=lambda x:float(x.get('confidence',0) or 0))
                            result_text=f"Frame {frame_no+1}: {len(preds)} detection(s) • Top: {pretty(top.get('class','unknown'))} {float(top.get('confidence',0) or 0)*100:.1f}%"
                            st.session_state.video_last_result=('success', result_text)
                            live_result_slot.success(result_text)
                        else:
                            result_text=f'Frame {frame_no+1}: No detections above the selected threshold.'
                            st.session_state.video_last_result=('info', result_text)
                            live_result_slot.info(result_text)
                    except Exception as e:
                        detected_frame_slot.image(pil, width=500)
                        live_result_slot.warning(f'Frame {frame_no+1}: inference skipped ({e})')
                writer.write(out_frame); frame_no+=1
                if total>0: progress.progress(min(frame_no/total,1.0))
            cap.release(); writer.release(); progress.progress(1.0); status.success('Video analysis completed.')
            # Convert the OpenCV-generated MP4 to browser-friendly H.264 when FFmpeg is available.
            # Keep the original OpenCV MP4 as a safe fallback so detection never fails just because
            # a local FFmpeg installation is unavailable.
            playback_path = output_path
            ffmpeg_exe = shutil.which('ffmpeg')
            if ffmpeg_exe:
                h264_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                try:
                    subprocess.run([
                        ffmpeg_exe, '-y', '-i', output_path,
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        '-an', h264_path
                    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if os.path.exists(h264_path) and os.path.getsize(h264_path) > 0:
                        playback_path = h264_path
                except Exception:
                    try:
                        if os.path.exists(h264_path): os.remove(h264_path)
                    except Exception:
                        pass

            with open(playback_path,'rb') as f: video_bytes=f.read()
            # Store the uploaded bytes too. Streamlit can infer/play the original container directly.
            st.session_state.video_output=video_bytes; st.session_state.video_original=video.getvalue(); st.session_state.video_original_name=video.name; st.session_state.video_stats={'processed':processed,'detections':detections,'avg':float(np.mean(confs)) if confs else 0,'latest_conf':(max([float(p.get('confidence',0) or 0) for p in st.session_state.video_last_predictions],default=0.0) if st.session_state.video_last_predictions else 0.0),'skip':frame_skip,'classes':dict(class_counts),'timeline':timeline}
            save_history('Video', video.name, threshold=conf, detections=detections, confidences=confs, class_counts=class_counts, details=timeline)
            st.success('Video analysis completed and saved to Detection History.')
    if st.session_state.video_stats:
        if (not process) and st.session_state.video_last_annotated is not None:
            st.markdown('<div class="section">Latest Video Detection Result</div>',unsafe_allow_html=True)
            pleft, pright = st.columns(2, gap='large')
            with pleft:
                st.markdown('### Last Original Frame')
                if st.session_state.video_last_original is not None:
                    st.image(st.session_state.video_last_original, width=500)
            with pright:
                st.markdown('### Last AI Detection')
                st.image(st.session_state.video_last_annotated, width=500)
                if st.session_state.video_last_result:
                    kind, msg = st.session_state.video_last_result
                    if kind == 'success': st.success(msg)
                    else: st.info(msg)
        s=st.session_state.video_stats; cols=st.columns(4)
        detection_conf=s.get('latest_conf',s.get('avg',0))
        for c,(n,v,sub) in zip(cols,[('Frames Processed',s['processed'],'Video analysis'),('Total Detections',s['detections'],'Across processed frames'),('Detection Confidence',f"{detection_conf*100:.1f}%",'Latest AI detection'),('Frame Sampling',f"1/{s['skip']}",'Video analysis')]):
            with c:kpi(n,v,sub)
        if st.session_state.video_output:
            st.markdown('<div class="section">Processed Video Export</div>',unsafe_allow_html=True)
            st.caption('The live frame-by-frame preview above is the primary in-dashboard result. The processed MP4 remains available for export.')
            st.download_button('⬇ Download Detected Video',st.session_state.video_output,'sea_sentinel_video.mp4','video/mp4',width='stretch')
        if s['timeline']:
            fig=px.line(pd.DataFrame(s['timeline']),x='Frame',y='Detections',markers=True,title='Detections Across Processed Frames'); st.plotly_chart(transparent(fig,300),width='stretch',config={'displayModeBar':False})
        st.caption('“Total Detections” counts detection instances across sampled frames; it is not an estimate of unique physical vessels tracked through the video.')
        show_detection_results(st.session_state.video_last_predictions,'Latest Processed Frame Results')

# ---------------- LIVE DETECTION ----------------
elif page=='Live Detection':
    title('Live Vessel Detection','Capture a live webcam feed and run frame-by-frame Sea Sentinel vessel detection.')
    st.markdown('<div class="panel"><b>Live camera inference</b><br><span class="soft">This page captures frames from a webcam connected to the computer running Sea Sentinel and sends selected frames through the existing YOLO26 Nano Roboflow workflow. The displayed processing rate includes camera capture, network and cloud-inference latency.</span></div>',unsafe_allow_html=True)

    control_col, live_col = st.columns([1,2.5],gap='large')
    with control_col:
        st.markdown('<div class="section">Live Configuration</div>',unsafe_allow_html=True)
        live_classes=st.multiselect('Vessel classes',VESSEL_CLASSES,default=VESSEL_CLASSES,format_func=pretty,key='live_classes')
        live_conf=st.slider('Confidence threshold',0.10,1.00,0.40,0.05,key='live_conf')
        live_iou=st.slider('IoU threshold',0.10,0.90,0.30,0.05,key='live_iou',help='Class-aware non-maximum suppression threshold for overlapping detections.')
        live_box=st.slider('Bounding box thickness',1,6,3,key='live_box')
        camera_index=st.number_input('Camera index',min_value=0,max_value=5,value=0,step=1,key='live_camera_index',help='0 is normally the built-in/default webcam. Try 1 for a USB camera.')
        live_duration=st.slider('Demo duration (seconds)',5,60,15,5,key='live_duration')
        inference_every=st.slider('Run AI every Nth camera frame',1,30,5,key='live_every',help='Use 1 for every captured frame. Higher values reduce cloud API calls and may make the preview smoother.')
        start_live=st.button('🔴 Start Live Detection',type='primary',width='stretch')
        st.caption('The session stops automatically after the selected duration. This prevents a camera loop from locking the Streamlit page.')

    with live_col:
        st.markdown('<div class="section">Live Camera Feed</div>',unsafe_allow_html=True)
        live_frame_slot=st.empty(); live_message_slot=st.empty(); live_metrics_slot=st.empty()

    if start_live:
        if not live_classes:
            st.warning('Select at least one vessel class before starting live detection.')
        else:
            backend=cv2.CAP_DSHOW if os.name=='nt' else cv2.CAP_ANY
            cap=cv2.VideoCapture(int(camera_index),backend)
            if not cap.isOpened():
                st.error('Unable to open the selected webcam. Close other apps using the camera, confirm camera permission, or try another camera index.')
            else:
                started=time.perf_counter(); captured=0; inferred=0; total_detections=0; inference_times=[]; latest_preds=[]
                try:
                    while time.perf_counter()-started < live_duration:
                        ok,frame=cap.read()
                        if not ok:
                            live_message_slot.error('Camera frame could not be read.'); break
                        captured+=1
                        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); display_image=pil
                        if (captured-1) % inference_every==0:
                            t0=time.perf_counter()
                            try:
                                latest_preds=run_workflow(pil,live_classes,live_conf,live_iou,0.7,live_box,'ROBOFLOW','Matplotlib Pastel1')
                                st.session_state.live_last_predictions=latest_preds
                                latency=time.perf_counter()-t0; inference_times.append(latency); inferred+=1; total_detections+=len(latest_preds)
                                display_image=draw_detections(pil,latest_preds,live_box)
                                if latest_preds:
                                    top=max(latest_preds,key=lambda x:float(x.get('confidence',0) or 0))
                                    live_message_slot.success(f"● LIVE • {len(latest_preds)} detection(s) • Top: {pretty(top.get('class','unknown'))} {float(top.get('confidence',0) or 0)*100:.1f}%")
                                else: live_message_slot.info('● LIVE • No detections above the selected threshold.')
                            except Exception as e:
                                live_message_slot.warning(f'Live inference request failed: {e}')
                        elif latest_preds:
                            display_image=draw_detections(pil,latest_preds,live_box)
                        live_frame_slot.image(display_image,width='stretch',channels='RGB')
                        elapsed=max(time.perf_counter()-started,0.001); observed_rate=inferred/elapsed; avg_latency=float(np.mean(inference_times))*1000 if inference_times else 0.0
                        live_metrics_slot.markdown(f'<div class="panel"><b>● LIVE CAMERA</b> &nbsp; | &nbsp; AI Frames: <b>{inferred}</b> &nbsp; | &nbsp; Detections: <b>{total_detections}</b> &nbsp; | &nbsp; Avg Inference: <b>{avg_latency:.0f} ms</b> &nbsp; | &nbsp; Observed AI Rate: <b>{observed_rate:.2f} FPS</b></div>',unsafe_allow_html=True)
                finally:
                    cap.release()
                elapsed=max(time.perf_counter()-started,0.001); observed_rate=inferred/elapsed; avg_latency=float(np.mean(inference_times))*1000 if inference_times else 0.0
                live_message_slot.success(f'Live detection session completed after {elapsed:.1f} seconds.')
                m=st.columns(4)
                values=[('AI Frames',inferred,'Frames submitted for inference'),('Frame Detections',total_detections,'Across inferred frames'),('Avg Inference Latency',f'{avg_latency:.0f} ms','Includes network/cloud latency'),('Observed AI Rate',f'{observed_rate:.2f} FPS','Measured during this session')]
                for c,(n,v,note) in zip(m,values):
                    with c:kpi(n,v,note)
                st.caption('Live Detection currently uses the Roboflow serverless workflow, so it requires internet access. The FPS value is the observed end-to-end AI processing rate, not the webcam hardware frame rate.')
                show_detection_results(st.session_state.live_last_predictions,'Latest Live Detection Results')

# ---------------- DETECTION ANALYTICS ----------------
elif page=='Detection Analytics':
    title('Detection Analytics','Persistent history of Sea Sentinel image and video analyses.')
    hist=load_history()
    if hist.empty:
        st.info('No saved detection history yet. Run Image Detection or Video Detection and the results will be stored here automatically.')
    else:
        total_analyses=len(hist); total_detections=int(hist['detections'].sum()); avg_conf=float(hist['avg_confidence'].mean()*100); image_runs=int((hist['source_type']=='Image').sum())
        cols=st.columns(4)
        for c,(n,v,note) in zip(cols,[('Saved Analyses',total_analyses,'Persistent records'),('Total Detections',total_detections,'Across saved analyses'),('Average Confidence',f'{avg_conf:.1f}%','Across analyses'),('Image Analyses',image_runs,'Saved image runs')]):
            with c:kpi(n,v,note)
        chart_df=hist.copy(); chart_df['timestamp']=pd.to_datetime(chart_df['timestamp']); chart_df=chart_df.sort_values('timestamp'); chart_df['Average Confidence (%)']=chart_df['avg_confidence']*100
        c1,c2=st.columns([1.35,1],gap='large')
        with c1:
            fig=px.line(chart_df,x='timestamp',y='detections',markers=True,color='source_type',title='Detection Activity Over Time',labels={'timestamp':'Date / Time','detections':'Detections','source_type':'Source'})
            st.plotly_chart(transparent(fig,330),width='stretch',config={'displayModeBar':False})
        with c2:
            agg=Counter()
            for raw in hist['class_counts'].fillna('{}'):
                try: agg.update(json.loads(raw))
                except Exception: pass
            if agg:
                cc=pd.DataFrame({'Vessel Type':list(agg.keys()),'Detections':list(agg.values())})
                fig=px.pie(cc,names='Vessel Type',values='Detections',hole=.58,title='Historical Vessel Distribution')
                st.plotly_chart(transparent(fig,330),width='stretch',config={'displayModeBar':False})
            else: st.info('Class-level history will appear after detections are recorded.')
        st.markdown('<div class="section">Saved Detection History</div>',unsafe_allow_html=True)
        display=hist[['timestamp','source_type','filename','detections','avg_confidence','max_confidence','vessel_types','threshold']].copy()
        display['timestamp']=pd.to_datetime(display['timestamp']).dt.strftime('%d %b %Y, %H:%M:%S')
        display['avg_confidence']=(display['avg_confidence']*100).round(1).astype(str)+'%'
        display['max_confidence']=(display['max_confidence']*100).round(1).astype(str)+'%'
        display['threshold']=(display['threshold']*100).round(0).astype(int).astype(str)+'%'
        display.columns=['Date / Time','Source','File','Detections','Avg Confidence','Highest Confidence','Vessel Types','Threshold']
        st.dataframe(display,width='stretch',hide_index=True)
        st.download_button('⬇ Export History CSV',display.to_csv(index=False).encode('utf-8'),'sea_sentinel_detection_history.csv','text/csv')
        with st.expander('History management'):
            st.warning('Clearing history permanently deletes the locally stored detection records.')
            confirm=st.checkbox('I understand and want to clear the saved history',key='confirm_clear_history')
            if st.button('Clear Detection History',disabled=not confirm):
                clear_history(); st.success('Detection history cleared.'); st.rerun()
    preds=st.session_state.image_predictions
    if preds:
        st.markdown('<div class="section">Latest Image Detection Details</div>',unsafe_allow_html=True)
        rows=[]
        for i,p in enumerate(preds,1): rows.append({'#':i,'Vessel Type':pretty(p.get('class','unknown')),'Confidence (%)':round(p.get('confidence',0)*100,1),'X':round(p.get('x',0),1),'Y':round(p.get('y',0),1),'Width':round(p.get('width',0),1),'Height':round(p.get('height',0),1)})
        st.dataframe(pd.DataFrame(rows),width='stretch',hide_index=True)

# ---------------- DATASET ----------------
elif page=='Dataset Analytics':
    title('Dataset Analytics','Exploration of the maritime vessel dataset used by Sea Sentinel.')
    cols=st.columns(5)
    for c,(n,v) in zip(cols,[('Total Images','2,241'),('Vessel Classes','8'),('Training','1,569'),('Validation','403'),('Testing','269')]):
        with c:kpi(n,v,'Dataset composition')
    c1,c2=st.columns([1.4,1],gap='large')
    with c1:
        df=pd.DataFrame({'Vessel':DATASET_COUNTS.keys(),'Images':DATASET_COUNTS.values()}); fig=px.bar(df,x='Vessel',y='Images',text='Images',title='Class Distribution'); fig.update_traces(textposition='outside'); st.plotly_chart(transparent(fig),width='stretch',config={'displayModeBar':False})
    with c2:
        split=pd.DataFrame({'Split':['Training','Validation','Testing'],'Images':[1569,403,269]}); fig=px.pie(split,names='Split',values='Images',hole=.62,title='Dataset Split'); st.plotly_chart(transparent(fig),width='stretch',config={'displayModeBar':False})
        st.caption('The displayed split is 1,569 training, 403 validation and 269 testing images.')

# ---------------- MODEL ----------------
elif page=='Model Performance':
    title('Model Performance','Evaluation results for YOLO26 Nano on the maritime vessel dataset.')
    cols=st.columns(4)
    for c,(n,v) in zip(cols,MODEL.items()):
        with c:kpi(n,f'{v:.1f}%','Validation metric')
    df=pd.DataFrame({'Class':CLASS_MAP50.keys(),'mAP@50 (%)':CLASS_MAP50.values()})
    fig=px.bar(df,x='mAP@50 (%)',y='Class',orientation='h',text='mAP@50 (%)',title='Per-Class Performance')
    fig.update_xaxes(range=[0,100])
    st.plotly_chart(transparent(fig,420),width='stretch',config={'displayModeBar':False})
    st.info('Model-performance values above are project-level evaluation metrics. Live image/video confidence values are kept separate on the detection pages.')

    st.markdown('<div class="section">Model Resilience to Visual Degradation</div>',unsafe_allow_html=True)
    st.markdown('''<div class="panel"><b>Controlled resilience test</b><br><span class="soft">Upload one maritime test image, create a degraded copy, and process both versions using the same YOLO26 Nano workflow. This test does not retrain or modify the model. Results below are live inference results, not dataset-level validation metrics.</span></div>''',unsafe_allow_html=True)

    control_col, preview_col = st.columns([1,2.25],gap='large')
    with control_col:
        resilience_upload=st.file_uploader('Upload resilience test image',type=['jpg','jpeg','png'],key='resilience_upload')
        degradation=st.selectbox('Visual degradation',['Gaussian Noise','Blur','Low Light','Fog / Haze'],key='resilience_degradation')
        severity=st.slider('Degradation severity',1,5,3,key='resilience_severity',help='1 = mild, 5 = severe')
        severity_name={1:'Mild',2:'Mild–Moderate',3:'Moderate',4:'Strong',5:'Severe'}[severity]
        st.caption(f'Selected severity: {severity_name}')
        resilience_conf=st.slider('Inference confidence threshold',0.10,1.00,0.40,0.05,key='resilience_conf')
        resilience_iou=st.slider('IoU threshold',0.10,0.90,0.30,0.05,key='resilience_iou',help='Class-aware non-maximum suppression threshold applied equally to original and degraded predictions.')
        run_resilience=st.button('🧪 Run Resilience Test',type='primary',width='stretch',disabled=resilience_upload is None)

    if resilience_upload is not None:
        original_resilience=Image.open(resilience_upload).convert('RGB')
        degraded_resilience=apply_visual_degradation(original_resilience,degradation,severity)
        with preview_col:
            p1,p2=st.columns(2)
            with p1:
                st.markdown('<div class="section">Original Preview</div>',unsafe_allow_html=True)
                st.image(original_resilience,width='stretch')
            with p2:
                st.markdown(f'<div class="section">{degradation} Preview</div>',unsafe_allow_html=True)
                st.image(degraded_resilience,width='stretch')

        if run_resilience:
            with st.spinner('Running the same YOLO26 Nano workflow on the original and degraded images...'):
                try:
                    original_preds=run_workflow(original_resilience,VESSEL_CLASSES,resilience_conf,resilience_iou,0.7,3,'ROBOFLOW','Matplotlib Pastel1')
                    degraded_preds=run_workflow(degraded_resilience,VESSEL_CLASSES,resilience_conf,resilience_iou,0.7,3,'ROBOFLOW','Matplotlib Pastel1')
                    st.session_state.resilience_result={
                        'filename':resilience_upload.name,
                        'degradation':degradation,
                        'severity':severity,
                        'severity_name':severity_name,
                        'threshold':resilience_conf,
                        'iou_threshold':resilience_iou,
                        'original_image':original_resilience,
                        'degraded_image':degraded_resilience,
                        'original_preds':original_preds,
                        'degraded_preds':degraded_preds
                    }
                    st.success('Resilience test completed. Results are shown below and are not added to Detection History.')
                except Exception as e:
                    st.error(f'Resilience test workflow error: {e}')

    rr=st.session_state.resilience_result
    if rr:
        st.markdown('<div class="section">Resilience Test Results</div>',unsafe_allow_html=True)
        osum=summarize_predictions(rr['original_preds']); dsum=summarize_predictions(rr['degraded_preds'])
        original_annotated=draw_detections(rr['original_image'],rr['original_preds'],3)
        degraded_annotated=draw_detections(rr['degraded_image'],rr['degraded_preds'],3)
        r1,r2=st.columns(2,gap='large')
        with r1:
            st.markdown('### Original Image')
            st.image(original_annotated,width='stretch')
            st.caption(f"Top prediction: {osum['top_class']} • {osum['top_confidence']*100:.1f}% confidence")
        with r2:
            st.markdown(f"### {rr['degradation']} · {rr['severity_name']}")
            st.image(degraded_annotated,width='stretch')
            st.caption(f"Top prediction: {dsum['top_class']} • {dsum['top_confidence']*100:.1f}% confidence")

        # Use the displayed one-decimal confidence values for a visually consistent change value.
        original_top_pct=round(osum['top_confidence']*100,1)
        degraded_top_pct=round(dsum['top_confidence']*100,1)
        confidence_change=round(degraded_top_pct-original_top_pct,1)
        detection_change=dsum['detections']-osum['detections']
        same_class=osum['top_class']==dsum['top_class'] and osum['top_class']!='No Detection'
        metrics=st.columns(5)
        result_metrics=[
            ('Original Top Confidence',f"{original_top_pct:.1f}%",osum['top_class']),
            ('Degraded Top Confidence',f"{degraded_top_pct:.1f}%",dsum['top_class']),
            ('Confidence Change',f"{confidence_change:+.1f} pp",f"{rr['degradation']} · {rr['severity_name']}"),
            ('Detection Count Change',f"{detection_change:+d}",f"{osum['detections']} → {dsum['detections']} detections"),
            ('Top Class Retained','Yes' if same_class else 'No','Original vs degraded')
        ]
        for c,(n,v,note) in zip(metrics,result_metrics):
            with c:kpi(n,v,note)

        comparison=pd.DataFrame({
            'Condition':['Original',f"{rr['degradation']} ({rr['severity_name']})"],
            'Top Class':[osum['top_class'],dsum['top_class']],
            'Top Conf. (%)':[original_top_pct,degraded_top_pct],
            'Detections':[osum['detections'],dsum['detections']],
            'Avg Conf. (%)':[round(osum['avg_confidence']*100,1),round(dsum['avg_confidence']*100,1)]
        })
        st.markdown('<div class="section">Prediction Comparison</div>',unsafe_allow_html=True)
        st.dataframe(comparison,width='stretch',hide_index=True,column_config={
            'Condition':st.column_config.TextColumn('Image Condition',width='medium'),
            'Top Class':st.column_config.TextColumn('Top Class',width='medium'),
            'Top Conf. (%)':st.column_config.NumberColumn('Top Confidence (%)',format='%.1f'),
            'Detections':st.column_config.NumberColumn('Detections',format='%d'),
            'Avg Conf. (%)':st.column_config.NumberColumn('Average Confidence (%)',format='%.1f')
        })

        def prediction_rows(predictions):
            ordered=sorted(predictions or [],key=lambda x:float(x.get('confidence',0) or 0),reverse=True)
            return pd.DataFrame([{
                '#':i,
                'Detected Class':pretty(pred.get('class','unknown')),
                'Confidence (%)':round(float(pred.get('confidence',0) or 0)*100,1)
            } for i,pred in enumerate(ordered,1)])

        st.markdown('<div class="section">Technical Detection Results</div>',unsafe_allow_html=True)
        tleft,tright=st.columns(2,gap='large')
        with tleft:
            show_detection_results(rr['original_preds'],'Original Detection Results',clean_table=True)
        with tright:
            show_detection_results(rr['degraded_preds'],f"{rr['degradation']} Detection Results",clean_table=True)

        chart_df=comparison[['Condition','Top Conf. (%)']].copy()
        fig=px.bar(chart_df,x='Condition',y='Top Conf. (%)',text='Top Conf. (%)',title='Top Detection Confidence')
        fig.update_yaxes(range=[0,100],title='Top Detection Confidence (%)')
        fig.update_xaxes(title='Image Condition')
        fig.update_traces(texttemplate='%{text:.1f}%',textposition='outside')
        st.plotly_chart(transparent(fig,340),width='stretch',config={'displayModeBar':False})

        if detection_change>0:
            st.warning(f"The degraded image produced {detection_change} additional detection(s). Review the 'All Detected Classes' table above; degradation can introduce additional predictions and they should not automatically be treated as additional physical vessels.")
        elif detection_change<0:
            st.warning(f"The degraded image produced {abs(detection_change)} fewer detection(s) than the original image.")

        st.caption('Interpretation note: this is a controlled single-image resilience comparison. It does not by itself establish overall model robustness. Aggregate resilience claims should be based on multiple test images and conditions.')

# ---------------- METHOD ----------------
elif page=='Methodology':
    title('Methodology','Sea Sentinel computer-vision workflow from data to live detection.')
    steps=[('1','Dataset Collection','2,241 maritime images across 8 vessel classes.'),('2','Data Preparation','Images organised into training, validation and testing splits.'),('3','Model Training','YOLO26 Nano trained through the project workflow.'),('4','Evaluation','mAP@50, precision, recall, F1 and per-class performance.'),('5','Live Inference','Uploaded images and sampled video frames are sent to Roboflow Workflows.'),('6','Visual Analytics','Bounding boxes, confidence values, class distribution and detection tables are presented.')]
    for n,h,d in steps: st.markdown(f'<div class="panel"><span class="pill">STEP {n}</span><h3>{h}</h3><div class="soft">{d}</div></div>',unsafe_allow_html=True)

# ---------------- ABOUT ----------------
else:
    title('About Sea Sentinel','Project information and competition presentation summary.')
    st.markdown('''<div class="panel"><h2>🌊 Sea Sentinel</h2><p>Sea Sentinel is an AI-powered maritime vision dashboard designed to detect and classify vessels from uploaded imagery and video while presenting the results through an analytics-focused interface.</p><p class="soft">The dashboard separates dataset statistics, model-evaluation results and live inference analytics so each type of evidence is presented in the correct context.</p></div>''',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    with a:kpi('Model','YOLO26 Nano','Object detection')
    with b:kpi('Classes','8','Maritime vessel categories')
    with c:kpi('Dataset','2,241','Images')
    st.markdown('### Vessel Classes')
    st.markdown(' '.join([f'<span class="pill">{pretty(x)}</span>' for x in VESSEL_CLASSES]),unsafe_allow_html=True)

st.markdown('<br><hr><div style="text-align:center;color:#B8CED9;font-size:.72rem;padding:12px">SEA SENTINEL • MARITIME VISION INTELLIGENCE • AI-POWERED VESSEL OBJECT DETECTION</div>',unsafe_allow_html=True)
