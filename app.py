"""
SEISMOS — Global Seismic Intelligence System
================================================
A full ML pipeline (cleaning -> EDA -> feature engineering -> base models ->
hyperparameter-tuned ensembles -> cross-validation -> regression) wrapped in
a dark, cinematic "monitoring station" dashboard.

Data: USGS significant earthquake catalog, 1900-2023 (96,115 real events, mag >= 5.0).

Run with: streamlit run app.py
Keep earthquakes_raw.csv in the same folder.
"""

import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingRegressor,
                               HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                               RandomForestClassifier, RandomForestRegressor, VotingClassifier)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                              precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold,
                                      cross_val_score, train_test_split)
from sklearn.neighbors import BallTree, KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ======================================================================================
# PAGE CONFIG + THE LOOK
# ======================================================================================
st.set_page_config(page_title="SEISMOS | Global Seismic Intelligence", page_icon="🌐", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg-void: #05070d;
    --bg-panel: #0c1120;
    --bg-panel2: #10182c;
    --border: #1e2a45;
    --text: #d8e0f0;
    --muted: #6b7a99;
    --amber: #ffb020;
    --red: #ff3860;
    --cyan: #29e0ff;
    --green: #37f0a0;
    --purple: #a56bff;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, rgba(41,224,255,0.06), transparent 45%),
        radial-gradient(ellipse at bottom right, rgba(255,56,96,0.05), transparent 50%),
        var(--bg-void);
    color: var(--text);
    font-family: 'Space Grotesk', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070a14 0%, #0a0e1c 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { font-family: 'Space Grotesk', sans-serif; }

h1, h2, h3 { font-family: 'Orbitron', sans-serif; letter-spacing: 0.02em; }
h1 { color: #eef3ff; text-shadow: 0 0 24px rgba(41,224,255,0.25); }
h2 { color: #cfe0ff; border-bottom: 1px solid var(--border); padding-bottom: 0.4rem; margin-top: 1.2rem;}
h3 { color: var(--cyan); font-size: 1.05rem; }

.hero {
    padding: 1.6rem 2rem; border-radius: 18px; margin-bottom: 1.4rem;
    background: linear-gradient(135deg, rgba(41,224,255,0.08), rgba(165,107,255,0.05));
    border: 1px solid var(--border);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -10%; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(41,224,255,0.15), transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Orbitron', sans-serif; font-weight: 900; font-size: 2.4rem;
    background: linear-gradient(90deg, #29e0ff, #a56bff 60%, #ff3860);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; letter-spacing: 0.03em;
}
.hero-sub { color: var(--muted); font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; margin-top: 0.3rem;}

.kpi {
    background: linear-gradient(145deg, var(--bg-panel), var(--bg-panel2));
    border: 1px solid var(--border); border-radius: 14px; padding: 1.1rem 1.3rem;
    position: relative; overflow: hidden; transition: transform 0.2s, border-color 0.2s;
}
.kpi:hover { transform: translateY(-3px); border-color: var(--cyan); }
.kpi::after {
    content: ''; position: absolute; bottom: 0; left: 0; height: 3px; width: 100%;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
}
.kpi-label { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em; }
.kpi-value { font-family: 'Orbitron', sans-serif; font-size: 1.9rem; font-weight: 700; color: #eef3ff; margin: 0.15rem 0; }
.kpi-delta { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: var(--green); }

.panel {
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.tag { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:0.7rem;
    padding: 2px 10px; border-radius: 20px; margin-right:6px; margin-bottom:4px;}
.tag-done { background: rgba(55,240,160,0.12); color: var(--green); border:1px solid rgba(55,240,160,0.3);}
.tag-warn { background: rgba(255,176,32,0.12); color: var(--amber); border:1px solid rgba(255,176,32,0.3);}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: var(--bg-panel); border-radius: 8px 8px 0 0; color: var(--muted);
    font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.stTabs [aria-selected="true"] { color: var(--cyan) !important; background: var(--bg-panel2) !important; }

[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; color: #eef3ff; }
[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; color: var(--muted); }

div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

/* ---- live status dot ---- */
@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(55,240,160,0.6); }
    70% { box-shadow: 0 0 0 8px rgba(55,240,160,0); }
    100% { box-shadow: 0 0 0 0 rgba(55,240,160,0); }
}
.live-dot {
    display:inline-block; width:9px; height:9px; border-radius:50%;
    background: var(--green); margin-right:8px; animation: pulseDot 1.8s infinite;
}
.live-tag {
    font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:var(--green);
    letter-spacing:0.1em; display:flex; align-items:center; margin-bottom:0.5rem;
}

/* ---- news ticker ---- */
.ticker-wrap {
    background: #070a14; border: 1px solid var(--border); border-radius: 10px;
    overflow: hidden; white-space: nowrap; padding: 10px 0; margin-bottom: 1.2rem;
}
.ticker-move {
    display: inline-block; animation: tickerScroll 45s linear infinite;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: var(--cyan);
}
.ticker-move span { margin-right: 3rem; }
.ticker-move span b { color: var(--red); }
@keyframes tickerScroll {
    0% { transform: translateX(0%); }
    100% { transform: translateX(-50%); }
}

/* ---- boot sequence ---- */
.boot-line {
    font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: var(--green);
    opacity: 0; animation: bootFade 0.4s ease forwards;
}
@keyframes bootFade { to { opacity: 1; } }

/* ---- seismograph decorative line ---- */
.seismo-wrap { width: 100%; height: 40px; overflow: hidden; margin-top: 0.6rem; opacity: 0.85; }
.seismo-line { stroke: var(--cyan); stroke-width: 1.6; fill: none; filter: drop-shadow(0 0 3px rgba(41,224,255,0.6)); }

/* ---- animated gradient divider ---- */
.grad-divider {
    height: 2px; width: 100%; margin: 0.6rem 0 1.2rem 0; border-radius: 2px;
    background: linear-gradient(90deg, #29e0ff, #a56bff, #ff3860, #29e0ff);
    background-size: 300% 100%; animation: gradMove 6s linear infinite;
}
@keyframes gradMove { 0% { background-position: 0% 0; } 100% { background-position: 300% 0; } }
</style>
""", unsafe_allow_html=True)

SEISMO_SVG = """
<div class="seismo-wrap"><svg viewBox="0 0 800 40" preserveAspectRatio="none" width="100%" height="40">
<path class="seismo-line" d="M0,20 L40,20 L55,6 L70,34 L85,20 L120,20 L135,2 L150,38 L165,14 L180,26 L200,20
L240,20 L255,10 L270,30 L285,20 L320,20 L335,4 L350,36 L365,16 L380,24 L400,20
L440,20 L455,6 L470,34 L485,20 L520,20 L535,2 L550,38 L565,14 L580,26 L600,20
L640,20 L655,10 L670,30 L685,20 L720,20 L735,4 L750,36 L765,16 L780,24 L800,20"/>
</svg></div>
"""


def boot_sequence():
    if st.session_state.get('_booted'):
        return
    st.session_state['_booted'] = True
    lines = [
        "&gt; INITIALIZING SEISMOS CORE...",
        "&gt; LOADING USGS CATALOG — 96,073 EVENTS [OK]",
        "&gt; RUNNING CLEANING PIPELINE [OK]",
        "&gt; CALIBRATING 8 CLASSIFIERS + 4 REGRESSORS...",
        "&gt; TUNING ENSEMBLES (GridSearchCV / RandomizedSearchCV) [OK]",
        "&gt; SYSTEM READY.",
    ]
    html = "".join(
        f'<div class="boot-line" style="animation-delay:{i*0.18}s">{l}</div>' for i, l in enumerate(lines)
    )
    ph = st.empty()
    ph.markdown(f'<div class="panel">{html}</div>', unsafe_allow_html=True)
    time.sleep(len(lines) * 0.18 + 0.5)
    ph.empty()


def gauge(value, title, max_val=1.0, color='#29e0ff'):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={'valueformat': '.2f', 'font': {'family': 'Orbitron', 'color': '#eef3ff', 'size': 30}},
        title={'text': title, 'font': {'family': 'JetBrains Mono', 'size': 12, 'color': '#6b7a99'}},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#6b7a99'},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 1, 'bordercolor': '#1e2a45',
            'steps': [
                {'range': [0, max_val*0.5], 'color': '#12172a'},
                {'range': [max_val*0.5, max_val*0.8], 'color': '#161c33'},
            ],
        }
    ))
    fig.update_layout(template=DARK_TEMPLATE, height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig

DARK_TEMPLATE = go.layout.Template()
DARK_TEMPLATE.layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Space Grotesk, sans-serif', color='#d8e0f0'),
    xaxis=dict(gridcolor='#1e2a45', zerolinecolor='#1e2a45'),
    yaxis=dict(gridcolor='#1e2a45', zerolinecolor='#1e2a45'),
    colorway=['#29e0ff', '#a56bff', '#ff3860', '#37f0a0', '#ffb020', '#ff6bd6', '#6bffea'],
    legend=dict(bgcolor='rgba(0,0,0,0)'),
)

DATA_PATH = "earthquakes_raw.csv"
PLATE_PATH = "plate_boundaries.json"
CITIES_PATH = "world_cities.csv"
CLASS_ORDER = ['Moderate', 'Strong', 'Major', 'Great']
CLASS_COLORS = {'Moderate': '#37f0a0', 'Strong': '#ffb020', 'Major': '#ff6b35', 'Great': '#ff3860'}

# ======================================================================================
# EMBEDDED JS COMPONENTS (vanilla JS, no external deps except plotly.js CDN for the globe)
# ======================================================================================
COUNTUP_JS = """
(function() {
  var target = __TARGET__;
  var el = document.getElementById('countup-val');
  var dur = 1800;
  var t0 = null;
  function frame(ts) {
    if (!t0) t0 = ts;
    var p = Math.min((ts - t0) / dur, 1);
    var eased = 1 - Math.pow(1 - p, 3);
    var val = Math.floor(target * eased);
    el.textContent = val.toLocaleString();
    if (p < 1) requestAnimationFrame(frame);
    else el.textContent = target.toLocaleString();
  }
  requestAnimationFrame(frame);
})();
"""

SEISMO_JS = """
(function() {
  var mags = __MAG_ARRAY__;
  var canvas = document.getElementById('seismo-canvas');
  var ctx = canvas.getContext('2d');
  var W = canvas.width, H = canvas.height;
  var mid = H / 2;
  var buffer = [];
  var idx = 0;
  var samplesPerEvent = 26;
  var noiseAmp = 2;

  function nextSample() {
    var m = mags[idx % mags.length];
    var phase = (buffer.length % samplesPerEvent) / samplesPerEvent;
    var spike = Math.sin(phase * Math.PI) * (m - 4.5) * 9;
    var noise = (Math.random() - 0.5) * noiseAmp;
    if (phase > 0.98) idx++;
    return spike + noise;
  }

  for (var i = 0; i < W; i++) buffer.push(nextSample());

  function draw() {
    buffer.push(nextSample());
    buffer.shift();

    ctx.fillStyle = '#070a14';
    ctx.fillRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(30,42,69,0.6)';
    ctx.lineWidth = 1;
    for (var gx = 0; gx < W; gx += 40) {
      ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke();
    }

    ctx.beginPath();
    ctx.strokeStyle = '#29e0ff';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = '#29e0ff';
    ctx.shadowBlur = 6;
    for (var x = 0; x < buffer.length; x++) {
      var y = mid - buffer[x];
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    requestAnimationFrame(draw);
  }
  draw();
})();
"""

SONIFY_JS = """
(function() {
  var sonify = __SONIFY_DATA__;
  var decades = Object.keys(sonify).sort();
  var select = document.getElementById('decade-select');
  var playBtn = document.getElementById('play-btn');
  var statusEl = document.getElementById('sonify-status');
  var audioCtx = null;

  decades.forEach(function(d) {
    var opt = document.createElement('option');
    opt.value = d; opt.text = d + 's (' + sonify[d].length + ' events)';
    select.appendChild(opt);
  });
  select.value = decades[decades.length - 1];

  function playNote(freq, dur, startTime, depth) {
    var osc = audioCtx.createOscillator();
    var gain = audioCtx.createGain();
    var filter = audioCtx.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 400 + (700 - Math.min(depth, 700));
    osc.type = depth > 200 ? 'sawtooth' : (depth > 70 ? 'triangle' : 'sine');
    osc.frequency.value = freq;
    osc.connect(filter);
    filter.connect(gain);
    gain.connect(audioCtx.destination);
    gain.gain.setValueAtTime(0, startTime);
    gain.gain.linearRampToValueAtTime(0.18, startTime + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, startTime + dur);
    osc.start(startTime);
    osc.stop(startTime + dur + 0.05);
  }

  playBtn.addEventListener('click', function() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();

    var events = sonify[select.value];
    var stagger = 0.11;
    var now = audioCtx.currentTime + 0.1;
    events.forEach(function(ev, i) {
      var mag = ev[0], depth = ev[1];
      var freq = 180 + (mag - 5.0) * 140;
      playNote(freq, 0.28, now + i * stagger, depth);
    });
    statusEl.textContent = 'Playing ' + select.value + 's — ' + events.length + ' events, ' +
      (events.length * stagger).toFixed(1) + 's';
    playBtn.disabled = true;
    setTimeout(function() {
      playBtn.disabled = false;
      statusEl.textContent = 'Ready.';
    }, events.length * stagger * 1000 + 400);
  });
})();
"""

WAVE_JS = """
(function() {
  var tP = __T_P__;
  var tS = __T_S__;
  var animDuration = 10000;
  var statusEl = document.getElementById('wave-status');
  var pEl = document.getElementById('p-countdown');
  var sEl = document.getElementById('s-countdown');
  var btn = document.getElementById('wave-btn');
  var barP = document.getElementById('p-bar');
  var barS = document.getElementById('s-bar');

  function fmt(sec) {
    if (sec <= 0) return '0.0s';
    if (sec < 120) return sec.toFixed(1) + 's';
    return (sec / 60).toFixed(1) + ' min';
  }

  btn.addEventListener('click', function() {
    btn.disabled = true;
    var t0 = null;
    var pArrived = false, sArrived = false;

    function frame(ts) {
      if (!t0) t0 = ts;
      var elapsedMs = ts - t0;
      var animP = Math.min(elapsedMs / animDuration, 1);
      var realElapsed = animP * tS;

      var pRemain = Math.max(0, tP - realElapsed);
      var sRemain = Math.max(0, tS - realElapsed);

      pEl.textContent = fmt(pRemain);
      sEl.textContent = fmt(sRemain);
      barP.style.width = (Math.min(realElapsed / tP, 1) * 100) + '%';
      barS.style.width = (Math.min(realElapsed / tS, 1) * 100) + '%';

      if (pRemain <= 0 && !pArrived) {
        pArrived = true;
        statusEl.textContent = 'P-WAVE ARRIVED — first tremor felt';
        statusEl.style.color = '#ffb020';
      }
      if (sRemain <= 0 && !sArrived) {
        sArrived = true;
        statusEl.textContent = 'S-WAVE ARRIVED — main shaking';
        statusEl.style.color = '#ff3860';
      }

      if (animP < 1) {
        requestAnimationFrame(frame);
      } else {
        btn.disabled = false;
      }
    }
    statusEl.textContent = 'PROPAGATING...';
    statusEl.style.color = '#29e0ff';
    requestAnimationFrame(frame);
  });
})();
"""

VOICE_JS = """
(function() {
  var text = __BRIEFING_TEXT__;
  var btn = document.getElementById('voice-btn');
  var statusEl = document.getElementById('voice-status');

  btn.addEventListener('click', function() {
    if (!('speechSynthesis' in window)) {
      statusEl.textContent = 'Speech synthesis not supported in this browser.';
      return;
    }
    window.speechSynthesis.cancel();
    var utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.0;
    utter.pitch = 0.9;
    utter.onstart = function() { statusEl.textContent = 'SPEAKING...'; btn.disabled = true; };
    utter.onend = function() { statusEl.textContent = 'Done.'; btn.disabled = false; };
    window.speechSynthesis.speak(utter);
  });
})();
"""

GLOBE_JS = """
(function() {
  var quakeLat = __QUAKE_LAT__;
  var quakeLon = __QUAKE_LON__;
  var quakeMag = __QUAKE_MAG__;
  var quakeDepth = __QUAKE_DEPTH__;
  var quakeColor = __QUAKE_COLOR__;
  var plateSegments = __PLATE_SEGMENTS__;

  var plateLon = [];
  var plateLat = [];
  plateSegments.forEach(function(seg) {
    seg.forEach(function(pt) { plateLon.push(pt[0]); plateLat.push(pt[1]); });
    plateLon.push(null); plateLat.push(null);
  });

  var traces = [
    {
      type: 'scattergeo', mode: 'lines',
      lon: plateLon, lat: plateLat,
      line: { color: '#ff3860', width: 1 },
      opacity: 0.55, hoverinfo: 'skip', name: 'Plate boundaries'
    },
    {
      type: 'scattergeo', mode: 'markers',
      lon: quakeLon, lat: quakeLat,
      marker: {
        size: quakeMag.map(function(m) { return m * 1.4; }),
        color: quakeColor, opacity: 0.75,
        line: { width: 0 }
      },
      text: quakeDepth.map(function(d, i) { return 'M' + quakeMag[i].toFixed(1) + ' · depth ' + d.toFixed(0) + 'km'; }),
      hoverinfo: 'text', name: 'Events'
    }
  ];

  var layout = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    geo: {
      projection: { type: 'orthographic', rotation: { lon: 0, lat: 0, roll: 0 } },
      showland: true, landcolor: '#0c1120',
      showocean: true, oceancolor: '#05070d',
      showcountries: true, countrycolor: '#1e2a45',
      coastlinecolor: '#1e2a45',
      bgcolor: 'rgba(0,0,0,0)'
    },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    height: 560,
    showlegend: false,
    font: { color: '#d8e0f0', family: 'Space Grotesk, sans-serif' }
  };

  var gd = document.getElementById('cinematic-globe');
  Plotly.newPlot(gd, traces, layout, { displayModeBar: false, scrollZoom: true });

  var target = { lon: 150, lat: 5 };
  var start = { lon: 0, lat: 0 };
  var t0 = null;
  var flyDuration = 2600;

  function easeInOutCubic(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  function flyFrame(ts) {
    if (!t0) t0 = ts;
    var elapsed = ts - t0;
    var p = Math.min(elapsed / flyDuration, 1);
    var e = easeInOutCubic(p);
    var lon = start.lon + (target.lon - start.lon) * e;
    var lat = start.lat + (target.lat - start.lat) * e;
    Plotly.relayout(gd, { 'geo.projection.rotation.lon': lon, 'geo.projection.rotation.lat': lat });
    if (p < 1) {
      requestAnimationFrame(flyFrame);
    } else {
      idleSpin(target.lon);
    }
  }

  function idleSpin(currentLon) {
    var lon = currentLon;
    setInterval(function() {
      lon += 0.15;
      Plotly.relayout(gd, { 'geo.projection.rotation.lon': lon });
    }, 50);
  }

  requestAnimationFrame(flyFrame);
})();
"""

def kpi(label, value, delta=None):
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
    st.markdown(f"""<div class="kpi"><div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>{delta_html}</div>""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f"""<div class="hero"><div class="hero-title">{title}</div>
        <div class="hero-sub">{subtitle}</div></div>""", unsafe_allow_html=True)


# ======================================================================================
# DATA LOADING, CLEANING, FEATURE ENGINEERING
# ======================================================================================
@st.cache_data
def load_and_clean():
    raw = pd.read_csv(DATA_PATH)
    raw_shape = raw.shape

    df = raw.drop_duplicates(subset='id', keep='last').reset_index(drop=True)
    n_dupes = raw_shape[0] - df.shape[0]

    n_neg_depth = (df['depth'] < 0).sum()
    df['depth'] = df['depth'].clip(lower=0)

    n_neg_rms = (df['rms'] < 0).sum()
    df.loc[df['rms'] < 0, 'rms'] = np.nan

    n_depth_missing = df['depth'].isnull().sum()
    df['depth'] = df['depth'].fillna(df['depth'].median())
    df['rms_missing'] = df['rms'].isnull().astype(int)
    n_rms_missing = df['rms'].isnull().sum()
    df['rms'] = df['rms'].fillna(df['rms'].median())
    df['place'] = df['place'].fillna('Unknown')

    df['year'] = pd.to_datetime(df['time']).dt.year
    df['decade'] = (df['year'] // 10) * 10

    def group_magtype(m):
        m = str(m).lower()
        if m == 'mb': return 'mb'
        if m in ('mw', 'mwc', 'mww', 'mwb', 'mwr', 'mwp'): return 'mw_family'
        if m in ('ms', 'ms_20'): return 'ms'
        if m == 'ml': return 'ml'
        return 'other'
    df['magType_grouped'] = df['magType'].apply(group_magtype)
    df['depth_category'] = pd.cut(df['depth'], bins=[-0.1, 70, 300, 1000],
                                    labels=['Shallow', 'Intermediate', 'Deep'])

    eq = df[df['type'] == 'earthquake'].copy()
    bins = [4.999, 5.9, 6.9, 7.9, 10]
    eq['mag_class'] = pd.cut(eq['mag'], bins=bins, labels=CLASS_ORDER)

    cleaning_log = {
        'raw_rows': raw_shape[0], 'final_rows': df.shape[0], 'n_dupes': n_dupes,
        'n_neg_depth': n_neg_depth, 'n_neg_rms': n_neg_rms,
        'n_depth_missing': n_depth_missing, 'n_rms_missing': n_rms_missing,
        'n_non_earthquake': raw_shape[0] - len(eq),
    }
    return df, eq, cleaning_log


@st.cache_data
def prepare_extras(eq):
    """Data that doesn't need a trained model: energy stats, sonification data,
    seismometer strip, plate boundaries."""
    energy_joules = 10 ** (1.5 * eq['mag'] + 4.8)  # Gutenberg-Richter energy-magnitude relation
    total_energy = float(energy_joules.sum())
    hiroshima_equiv = int(total_energy / 6.3e13)  # ~15kt TNT per Little Boy

    strip = eq.sort_values('time').tail(400)['mag'].round(2).tolist()

    sonify = {}
    for dec, grp in eq.groupby('decade'):
        grp = grp.sort_values('time')
        if len(grp) > 50:
            idx = np.linspace(0, len(grp) - 1, 50).astype(int)
            grp = grp.iloc[idx]
        sonify[str(int(dec))] = [[round(m, 2), round(d, 1)] for m, d in zip(grp['mag'], grp['depth'])]

    try:
        with open(PLATE_PATH) as f:
            plate_segments = json.load(f)
    except FileNotFoundError:
        plate_segments = []

    return {
        'total_energy': total_energy, 'hiroshima_equiv': hiroshima_equiv,
        'strip': strip, 'sonify': sonify, 'plate_segments': plate_segments,
    }


@st.cache_data
def prepare_deep_dive(eq):
    """City exposure scoring (real proximity/density, NOT the classifier's flawed extrapolation),
    Gutenberg-Richter return-period fit, and DBSCAN aftershock-sequence clustering."""
    EARTH_R = 6371.0
    try:
        cities = pd.read_csv(CITIES_PATH)
    except FileNotFoundError:
        cities = pd.DataFrame(columns=['city', 'country', 'lat', 'lng', 'pop'])

    if len(cities) and len(eq):
        eq_coords = np.radians(eq[['latitude', 'longitude']].values)
        tree = BallTree(eq_coords, metric='haversine')
        city_coords = np.radians(cities[['lat', 'lng']].values)

        counts = tree.query_radius(city_coords, r=300 / EARTH_R, count_only=True)
        cities['historical_events_300km'] = counts
        dist, _ = tree.query(city_coords, k=1)
        cities['dist_nearest_epicenter_km'] = dist[:, 0] * EARTH_R

        try:
            with open(PLATE_PATH) as f:
                plates = json.load(f)
            plate_pts = []
            for seg in plates:
                for pt in seg:
                    plate_pts.append([pt[1], pt[0]])
            plate_pts = np.radians(np.array(plate_pts))
            plate_tree = BallTree(plate_pts, metric='haversine')
            dist_p, _ = plate_tree.query(city_coords, k=1)
            cities['dist_nearest_plate_km'] = dist_p[:, 0] * EARTH_R
        except FileNotFoundError:
            cities['dist_nearest_plate_km'] = np.nan

        cities['exposure_score'] = (
            (1 / (1 + cities['dist_nearest_plate_km'].fillna(2000) / 500)) * 0.5 +
            (np.log1p(cities['historical_events_300km']) /
             max(np.log1p(cities['historical_events_300km']).max(), 1e-9)) * 0.5
        )
        cities['tier'] = pd.qcut(cities['exposure_score'], q=3, labels=['Low', 'Moderate', 'High'])

    # ---- Gutenberg-Richter fit + return periods ----
    years_span = eq['year'].max() - eq['year'].min() + 1
    mag_bins = np.arange(5.0, 9.0, 0.2)
    cum_counts = np.array([(eq['mag'] >= m).sum() / years_span for m in mag_bins])
    log_counts = np.log10(np.where(cum_counts > 0, cum_counts, np.nan))
    valid = np.isfinite(log_counts)
    slope, a_val = np.polyfit(mag_bins[valid], log_counts[valid], 1)
    b_value = -slope
    gr_table = pd.DataFrame({
        'magnitude': mag_bins,
        'annual_rate_fitted': 10 ** (a_val - b_value * mag_bins),
        'annual_rate_actual': cum_counts,
    })

    # ---- DBSCAN spatial-temporal clustering (aftershock sequences) ----
    eq_sorted = eq.sort_values('time').reset_index(drop=True).copy()
    t0 = pd.to_datetime(eq_sorted['time']).min()
    days = (pd.to_datetime(eq_sorted['time']) - t0).dt.total_seconds() / 86400
    lat_km = eq_sorted['latitude'] * 111.0
    lon_km = eq_sorted['longitude'] * 111.0 * np.cos(np.radians(eq_sorted['latitude']))
    time_scaled = days * (50.0 / 30.0)
    Xc = np.column_stack([lat_km, lon_km, time_scaled])
    db = DBSCAN(eps=50, min_samples=5, n_jobs=1).fit(Xc)
    eq_sorted['cluster'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = int((db.labels_ == -1).sum())

    cluster_sizes = eq_sorted[eq_sorted['cluster'] >= 0].groupby('cluster').size().sort_values(ascending=False)
    top_clusters = []
    for cid, size in cluster_sizes.head(15).items():
        grp = eq_sorted[eq_sorted['cluster'] == cid]
        mainshock = grp.loc[grp['mag'].idxmax()]
        top_clusters.append({
            'cluster_id': int(cid), 'size': int(size), 'mainshock_mag': float(mainshock['mag']),
            'place': mainshock['place'], 'date': str(mainshock['time'])[:10],
            'lat': float(mainshock['latitude']), 'lon': float(mainshock['longitude']),
        })

    return {
        'cities': cities, 'gr_a': a_val, 'gr_b': b_value, 'gr_table': gr_table,
        'eq_clustered': eq_sorted[['latitude', 'longitude', 'mag', 'time', 'cluster', 'place']],
        'top_clusters': top_clusters, 'n_clusters': n_clusters, 'n_noise': n_noise,
    }


@st.cache_resource(show_spinner=False)
def train_everything(eq):
    eq_model = pd.get_dummies(eq, columns=['magType_grouped', 'depth_category'], drop_first=True)
    dummy_cols = [c for c in eq_model.columns if c.startswith('magType_grouped_') or c.startswith('depth_category_')]
    num_cols = ['depth', 'rms', 'latitude', 'longitude', 'year', 'rms_missing']
    feature_cols = num_cols + dummy_cols

    X = eq_model[feature_cols]
    y = eq_model['mag_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train_s), size=min(6000, len(X_train_s)), replace=False)
    X_train_sub, y_train_sub = X_train_s[sub_idx], y_train.iloc[sub_idx]
    svm_idx = rng.choice(len(X_train_s), size=min(4000, len(X_train_s)), replace=False)
    X_train_svm, y_train_svm = X_train_s[svm_idx], y_train.iloc[svm_idx]

    fitted, timings = {}, {}

    t = time.time()
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train_s, y_train); fitted['Logistic Regression'] = lr; timings['Logistic Regression'] = time.time() - t

    t = time.time()
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_sub, y_train_sub); fitted['KNN'] = knn; timings['KNN'] = time.time() - t

    t = time.time()
    dt = DecisionTreeClassifier(max_depth=12, class_weight='balanced', random_state=42)
    dt.fit(X_train_s, y_train); fitted['Decision Tree'] = dt; timings['Decision Tree'] = time.time() - t

    t = time.time()
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_svm, y_train_svm); fitted['SVM'] = svm; timings['SVM'] = time.time() - t

    t = time.time()
    rf_grid = {'n_estimators': [80, 150], 'max_depth': [12, 24]}
    rf_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                              rf_grid, cv=3, scoring='f1_macro', n_jobs=1)
    rf_search.fit(X_train_sub, y_train_sub)
    rf_best = RandomForestClassifier(**rf_search.best_params_, class_weight='balanced', random_state=42)
    rf_best.fit(X_train_s, y_train)
    fitted['Random Forest (Tuned)'] = rf_best
    timings['Random Forest (Tuned)'] = time.time() - t
    rf_tuning_info = {'search_space': rf_grid, 'best_params': rf_search.best_params_,
                       'best_cv_f1': rf_search.best_score_, 'cv_results': rf_search.cv_results_}

    t = time.time()
    gb_dist = {'max_iter': [100, 150], 'max_depth': [4, 6, None], 'learning_rate': [0.05, 0.1, 0.2]}
    gb_search = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=42), gb_dist,
                                    n_iter=5, cv=3, scoring='f1_macro', random_state=42, n_jobs=1)
    gb_search.fit(X_train_sub, y_train_sub)
    gb_best = HistGradientBoostingClassifier(**gb_search.best_params_, random_state=42)
    gb_best.fit(X_train_s, y_train)
    fitted['Gradient Boosting (Tuned)'] = gb_best
    timings['Gradient Boosting (Tuned)'] = time.time() - t
    gb_tuning_info = {'search_space': gb_dist, 'best_params': gb_search.best_params_,
                       'best_cv_f1': gb_search.best_score_, 'cv_results': gb_search.cv_results_}

    t = time.time()
    ada = AdaBoostClassifier(n_estimators=60, random_state=42)
    ada.fit(X_train_s, y_train); fitted['AdaBoost'] = ada; timings['AdaBoost'] = time.time() - t

    t = time.time()
    voting = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=60, max_depth=10, class_weight='balanced', random_state=42)),
        ('gb', HistGradientBoostingClassifier(max_iter=60, max_depth=4, random_state=42))],
        voting='soft', n_jobs=1)
    voting.fit(X_train_sub, y_train_sub)
    fitted['Voting Ensemble'] = voting
    timings['Voting Ensemble'] = time.time() - t

    results, roc_data, confusion_mats = [], {}, {}
    for name, model in fitted.items():
        pred = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)
        y_test_bin = label_binarize(y_test, classes=list(model.classes_))
        try:
            auc = roc_auc_score(y_test_bin, proba, average='macro', multi_class='ovr')
        except Exception:
            auc = np.nan
        results.append({
            'Model': name, 'Accuracy': accuracy_score(y_test, pred),
            'Precision (macro)': precision_score(y_test, pred, average='macro', zero_division=0),
            'Recall (macro)': recall_score(y_test, pred, average='macro', zero_division=0),
            'F1 (macro)': f1_score(y_test, pred, average='macro', zero_division=0),
            'ROC-AUC (macro)': auc,
        })
        confusion_mats[name] = confusion_matrix(y_test, pred, labels=[c for c in CLASS_ORDER if c in model.classes_])

    results_df = pd.DataFrame(results).set_index('Model').round(3).sort_values('F1 (macro)', ascending=False)

    cv_scores = cross_val_score(RandomForestClassifier(**rf_search.best_params_, class_weight='balanced', random_state=42),
                                 X_train_sub, y_train_sub, cv=5, scoring='f1_macro', n_jobs=1)

    rf_importances = pd.Series(rf_best.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # ---- Regression: exact magnitude ----
    Xr, yr = eq_model[feature_cols], eq_model['mag']
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    scaler_r = StandardScaler()
    Xr_train_s = scaler_r.fit_transform(Xr_train)
    Xr_test_s = scaler_r.transform(Xr_test)
    r_sub_idx = rng.choice(len(Xr_train_s), size=min(6000, len(Xr_train_s)), replace=False)
    Xr_train_sub, yr_train_sub = Xr_train_s[r_sub_idx], yr_train.iloc[r_sub_idx]

    rfr_grid = {'n_estimators': [80, 150], 'max_depth': [8, 15]}
    rfr_search = GridSearchCV(RandomForestRegressor(random_state=42), rfr_grid, cv=3, scoring='r2', n_jobs=1)
    rfr_search.fit(Xr_train_sub, yr_train_sub)
    rfr_best = RandomForestRegressor(**rfr_search.best_params_, random_state=42)
    rfr_best.fit(Xr_train_s, yr_train)

    reg_models = {
        'Linear Regression': LinearRegression(), 'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest (Tuned)': rfr_best,
        'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=42),
    }
    reg_results, fitted_reg = [], {}
    for name, model in reg_models.items():
        if name != 'Random Forest (Tuned)':
            model.fit(Xr_train_s, yr_train)
        pred = model.predict(Xr_test_s)
        reg_results.append({'Model': name, 'MAE': mean_absolute_error(yr_test, pred), 'R2': r2_score(yr_test, pred)})
        fitted_reg[name] = model
    reg_df = pd.DataFrame(reg_results).set_index('Model').round(3).sort_values('R2', ascending=False)

    # ---- Hazard grid: model-predicted risk of Strong+ across a global 5-degree grid ----
    lats = np.arange(-85, 86, 5)
    lons = np.arange(-180, 181, 5)
    grid = pd.DataFrame([(la, lo) for la in lats for lo in lons], columns=['latitude', 'longitude'])
    grid['depth'] = 15.0
    grid['rms'] = eq['rms'].median()
    grid['year'] = 2024
    grid['rms_missing'] = 0
    for c in dummy_cols:
        grid[c] = 0
    if 'magType_grouped_mw_family' in grid.columns:
        grid['magType_grouped_mw_family'] = 1
    best_overall = fitted[results_df.index[0]]
    grid_scaled = scaler.transform(grid[feature_cols])
    proba_grid = best_overall.predict_proba(grid_scaled)
    classes_list = list(best_overall.classes_)
    risk_idx = [classes_list.index(c) for c in ['Strong', 'Major', 'Great'] if c in classes_list]
    grid['risk'] = proba_grid[:, risk_idx].sum(axis=1)

    return {
        'fitted': fitted, 'results': results_df, 'confusion_mats': confusion_mats,
        'timings': timings, 'rf_tuning_info': rf_tuning_info, 'gb_tuning_info': gb_tuning_info,
        'cv_scores': cv_scores, 'rf_importances': rf_importances, 'feature_cols': feature_cols,
        'scaler': scaler, 'best_class_model': results_df.index[0],
        'reg_results': reg_df, 'fitted_reg': fitted_reg, 'scaler_r': scaler_r,
        'best_reg_model': reg_df.index[0], 'dummy_cols': dummy_cols,
        'hazard_grid': grid[['latitude', 'longitude', 'risk']],
    }


def generate_pdf_report(M, EX, DD, clog):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Helvetica', 'B', 22)
    pdf.set_text_color(20, 30, 50)
    pdf.cell(0, 14, 'SEISMOS', ln=True, align='C')
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(90, 90, 110)
    pdf.cell(0, 8, 'Global Seismic Intelligence System - Mission Report', ln=True, align='C')
    pdf.ln(6)

    pdf.set_draw_color(200, 200, 210)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(6)

    def section(title):
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(20, 30, 50)
        pdf.cell(0, 10, title, ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(40, 40, 50)

    section('1. Dataset Overview')
    pdf.multi_cell(0, 6,
        f"Source: USGS significant earthquake catalog, 1900-2023 (magnitude >= 5.0).\n"
        f"Raw records: {clog['raw_rows']:,}  |  After cleaning: {clog['final_rows']:,}\n"
        f"Duplicate IDs removed: {clog['n_dupes']}  |  Non-earthquake events set aside: {clog['n_non_earthquake']}\n")
    pdf.ln(3)

    section('2. Model Performance')
    best = M['best_class_model']
    pdf.multi_cell(0, 6,
        f"Best classifier: {best}\n"
        f"  Accuracy: {M['results'].loc[best,'Accuracy']:.3f}\n"
        f"  F1 (macro): {M['results'].loc[best,'F1 (macro)']:.3f}\n"
        f"  ROC-AUC (macro): {M['results'].loc[best,'ROC-AUC (macro)']:.3f}\n\n"
        f"Best regressor: {M['best_reg_model']} - R2 = {M['reg_results'].loc[M['best_reg_model'],'R2']:.3f}, "
        f"MAE = {M['reg_results'].loc[M['best_reg_model'],'MAE']:.3f}\n")
    pdf.ln(3)

    section('3. Seismic Energy')
    pdf.multi_cell(0, 6,
        f"Total energy released (1900-2023): {EX['total_energy']:.3e} Joules\n"
        f"Equivalent to {EX['hiroshima_equiv']:,} Hiroshima-scale bombs (Gutenberg-Richter relation, "
        f"E = 10^(1.5M+4.8))\n")
    pdf.ln(3)

    section('4. Gutenberg-Richter Return Periods')
    pdf.multi_cell(0, 6,
        f"Fitted b-value: {DD['gr_b']:.3f} (real tectonic seismicity typically b~1.0)\n"
        f"Estimated annual rate of M>=8.0 events (global): {10**(DD['gr_a']-DD['gr_b']*8.0):.3f}/year\n")
    pdf.ln(3)

    section('5. Aftershock Clustering (DBSCAN)')
    pdf.multi_cell(0, 6,
        f"Clusters identified: {DD['n_clusters']}  |  Noise (isolated) events: {DD['n_noise']:,}\n"
        f"Largest cluster: {DD['top_clusters'][0]['size']} events, mainshock M{DD['top_clusters'][0]['mainshock_mag']} "
        f"- {DD['top_clusters'][0]['place']} ({DD['top_clusters'][0]['date']})\n")
    pdf.ln(3)

    section('6. Population Exposure')
    high_pop = DD['cities'][DD['cities']['tier'] == 'High']['pop'].sum()
    pdf.multi_cell(0, 6,
        f"Cities analyzed: {len(DD['cities']):,}\n"
        f"Population in High-exposure tier: {high_pop:,.0f}\n"
        f"(Exposure = historical event density + proximity to real tectonic plate boundaries - "
        f"NOT the classifier's raw extrapolation, which was found to misrank stable regions.)\n")

    return bytes(pdf.output())


df, eq, clog = load_and_clean()
with st.spinner("🛰️ Calibrating seismographs — training 8 classifiers + 4 regressors, tuning 2 ensembles..."):
    M = train_everything(eq)
EX = prepare_extras(eq)
with st.spinner("🌆 Scoring city exposure, fitting Gutenberg-Richter, clustering aftershock sequences..."):
    DD = prepare_deep_dive(eq)

# ======================================================================================
# SIDEBAR NAV
# ======================================================================================
st.sidebar.markdown("""<div style="font-family:'Orbitron',sans-serif; font-size:1.3rem; font-weight:900;
    background: linear-gradient(90deg,#29e0ff,#a56bff); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin-bottom:0;">SEISMOS</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6b7a99; margin-bottom:1rem;">
    GLOBAL SEISMIC INTELLIGENCE SYSTEM</div>""", unsafe_allow_html=True)

page = st.sidebar.radio("MODULE", [
    "🛰️  Command Center", "🗺️  Seismic Atlas", "🧪  Data Lab",
    "📊  Exploratory Analysis", "🧠  Model Observatory", "🔮  Seismic Forecast",
    "🌆  Exposure Atlas", "🌊  Forecast Lab", "🔊  Sonic Lab", "📡  Key Insights"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""<div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#6b7a99;">
SOURCE: USGS Significant Earthquake Catalog<br>1900–2023 · mag ≥ 5.0 · 96,115 events</div>""", unsafe_allow_html=True)

# ======================================================================================
# PAGE: COMMAND CENTER
# ======================================================================================
if page == "🛰️  Command Center":
    boot_sequence()
    st.markdown('<div class="live-tag"><span class="live-dot"></span>SYSTEM ONLINE — 96,073 EVENTS TRACKED</div>',
                unsafe_allow_html=True)
    hero("SEISMOS // GLOBAL MONITORING", "REAL-TIME ANALYTICAL OVERVIEW — 124 YEARS OF SEISMIC ACTIVITY")
    st.markdown(SEISMO_SVG, unsafe_allow_html=True)

    ticker_events = eq.nlargest(12, 'mag')[['place', 'mag', 'year']].values.tolist()
    ticker_html = "".join(
        f'<span>📍 {p[:40]} — <b>M{m:.1f}</b> ({y})</span>' for p, m, y in ticker_events
    )
    st.markdown(f'<div class="ticker-wrap"><div class="ticker-move">{ticker_html}{ticker_html}</div></div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("TOTAL EVENTS", f"{len(eq):,}", "1900–2023")
    with c2: kpi("AVG MAGNITUDE", f"{eq['mag'].mean():.2f}", f"σ = {eq['mag'].std():.2f}")
    with c3: kpi("STRONGEST EVENT", f"M {eq['mag'].max():.1f}", eq.loc[eq['mag'].idxmax(), 'place'][:22])
    with c4: kpi("DEEPEST EVENT", f"{eq['depth'].max():.0f} km", eq.loc[eq['depth'].idxmax(), 'place'][:22])
    with c5: kpi("PEAK DECADE", f"{int(eq['decade'].value_counts().idxmax())}s", f"{eq['decade'].value_counts().max():,} events")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        yearly = eq.groupby('year').size().reset_index(name='count')
        fig = px.area(yearly, x='year', y='count', title="EVENT FREQUENCY OVER TIME")
        fig.update_traces(line_color='#29e0ff', fillcolor='rgba(41,224,255,0.15)')
        fig.update_layout(template=DARK_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dist = eq['mag_class'].value_counts().reindex(CLASS_ORDER)
        fig = px.pie(values=dist.values, names=dist.index, hole=0.6, title="MAGNITUDE CLASS DISTRIBUTION",
                     color=dist.index, color_discrete_map=CLASS_COLORS)
        fig.update_layout(template=DARK_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### MODEL PERFORMANCE AT A GLANCE")
    g1, g2, g3 = st.columns(3)
    best = M['best_class_model']
    with g1:
        st.plotly_chart(gauge(M['results'].loc[best, 'Accuracy'], f"ACCURACY — {best}", color='#37f0a0'),
                        use_container_width=True)
    with g2:
        st.plotly_chart(gauge(M['results'].loc[best, 'ROC-AUC (macro)'], "ROC-AUC (macro)", color='#29e0ff'),
                        use_container_width=True)
    with g3:
        st.plotly_chart(gauge(M['reg_results'].loc[M['best_reg_model'], 'R2'], "REGRESSION R²",
                              max_val=1.0, color='#a56bff'), use_container_width=True)

    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

    st.markdown("### SEISMIC ENERGY RELEASED — 1900–2023")
    ce1, ce2 = st.columns([1, 2])
    with ce1:
        components.html(f"""
        <div style="font-family:'JetBrains Mono',monospace; color:#6b7a99; font-size:0.72rem;
            letter-spacing:0.1em; margin-bottom:4px;">HIROSHIMA-BOMB EQUIVALENTS</div>
        <div id="countup-val" style="font-family:'Orbitron',sans-serif; font-size:2.6rem; font-weight:900;
            color:#ff3860; text-shadow:0 0 20px rgba(255,56,96,0.5);">0</div>
        <script>{COUNTUP_JS.replace('__TARGET__', str(EX['hiroshima_equiv']))}</script>
        """, height=95)
        st.caption(f"Total energy: {EX['total_energy']:.2e} J — via Gutenberg-Richter E = 10^(1.5M+4.8)")
    with ce2:
        components.html(f"""
        <canvas id="seismo-canvas" width="700" height="90" style="width:100%; border-radius:8px;"></canvas>
        <script>{SEISMO_JS.replace('__MAG_ARRAY__', json.dumps(EX['strip']))}</script>
        """, height=100)
        st.caption("Live seismometer trace — built from the real, chronologically-ordered magnitude sequence")

    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

    st.markdown("""<div class="panel"><h3>SYSTEM STATUS</h3>
    <span class="tag tag-done">✓ DATA CLEANED</span>
    <span class="tag tag-done">✓ EDA COMPLETE</span>
    <span class="tag tag-done">✓ 8 CLASSIFIERS TRAINED</span>
    <span class="tag tag-done">✓ 2 ENSEMBLES TUNED</span>
    <span class="tag tag-done">✓ 5-FOLD CV RUN</span>
    <span class="tag tag-done">✓ 4 REGRESSORS TRAINED</span>
    <span class="tag tag-warn">⚠ RARE-CLASS RECALL LIMITED (SEE INSIGHTS)</span>
    </div>""", unsafe_allow_html=True)

# ======================================================================================
# PAGE: SEISMIC ATLAS
# ======================================================================================
elif page == "🗺️  Seismic Atlas":
    hero("SEISMIC ATLAS", "GLOBAL EVENT DISTRIBUTION — MULTIPLE VIEW MODES")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["SCATTER MAP", "DENSITY HEATMAP", "3D GLOBE",
                                                    "TIME MACHINE", "HAZARD FORECAST", "CINEMATIC GLOBE"])

    sample = eq.sample(min(15000, len(eq)), random_state=42)
    with tab1:
        fig = px.scatter_geo(sample, lat='latitude', lon='longitude', color='mag_class',
                              size='mag', size_max=10, hover_name='place',
                              hover_data={'mag': True, 'depth': True, 'year': True, 'latitude': False, 'longitude': False},
                              color_discrete_map=CLASS_COLORS, category_orders={'mag_class': CLASS_ORDER})
        fig.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#0c1120', oceancolor='#05070d',
                         showocean=True, coastlinecolor='#1e2a45', countrycolor='#1e2a45')
        fig.update_layout(template=DARK_TEMPLATE, height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.density_mapbox(sample, lat='latitude', lon='longitude', z='mag', radius=6,
                                 center=dict(lat=10, lon=140), zoom=1, mapbox_style="carto-darkmatter")
        fig.update_layout(template=DARK_TEMPLATE, height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Density clusters trace the Pacific Ring of Fire and major fault boundaries.")

    with tab3:
        fig = go.Figure(go.Scattergeo(
            lat=sample['latitude'], lon=sample['longitude'],
            mode='markers',
            marker=dict(size=sample['mag'] * 1.5, color=sample['depth'], colorscale='Plasma',
                        showscale=True, colorbar=dict(title="Depth (km)"), opacity=0.7)
        ))
        fig.update_geos(projection_type="orthographic", bgcolor='rgba(0,0,0,0)',
                         landcolor='#0c1120', oceancolor='#05070d', showocean=True,
                         coastlinecolor='#29e0ff', countrycolor='#1e2a45')
        fig.update_layout(template=DARK_TEMPLATE, height=560, margin=dict(l=0, r=0, t=10, b=0),
                           title="ROTATE: DRAG TO EXPLORE — COLOR = DEPTH")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.caption("Press ▶ Play to watch 124 years of seismic activity unfold, decade by decade.")
        tm_sample = eq.sample(min(10000, len(eq)), random_state=7).sort_values('decade')
        tm_sample['decade_label'] = tm_sample['decade'].astype(str) + "s"
        fig = px.scatter_geo(tm_sample, lat='latitude', lon='longitude', color='mag_class',
                              size='mag', size_max=9, hover_name='place',
                              animation_frame='decade_label',
                              color_discrete_map=CLASS_COLORS, category_orders={'mag_class': CLASS_ORDER})
        fig.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#0c1120', oceancolor='#05070d',
                         showocean=True, coastlinecolor='#1e2a45', countrycolor='#1e2a45')
        fig.update_layout(template=DARK_TEMPLATE, height=580, margin=dict(l=0, r=0, t=10, b=0))
        if fig.layout.updatemenus:
            fig.layout.updatemenus[0].bgcolor = '#10182c'
            fig.layout.updatemenus[0].font = dict(color='#d8e0f0', family='JetBrains Mono')
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.caption("What does the trained model think the *whole planet* looks like — not just where "
                  "history recorded events, but everywhere?")
        hg = M['hazard_grid']
        c1, c2 = st.columns(2)
        with c1:
            fig = px.density_mapbox(hg, lat='latitude', lon='longitude', z='risk', radius=20,
                                    center=dict(lat=10, lon=140), zoom=0.5, mapbox_style="carto-darkmatter",
                                    color_continuous_scale='Inferno', title="MODEL-PREDICTED RISK (Strong+)")
            fig.update_layout(template=DARK_TEMPLATE, height=460, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.density_mapbox(eq, lat='latitude', lon='longitude', radius=6,
                                    center=dict(lat=10, lon=140), zoom=0.5, mapbox_style="carto-darkmatter",
                                    color_continuous_scale='Viridis', title="ACTUAL HISTORICAL DENSITY")
            fig.update_layout(template=DARK_TEMPLATE, height=460, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        st.warning("**Honest caveat:** the model extrapolates beyond its training coverage in data-sparse "
                  "regions (open ocean, stable continental interiors) — the left panel can light up in places "
                  "with almost no real historical activity. Always sanity-check a hazard hotspot against the "
                  "right panel before trusting it. This is a statistical pattern-match on 12 features, not a "
                  "physics-based hazard model like real seismologists use.")

    with tab6:
        st.caption("Auto-rotating flyover to the Pacific Ring of Fire, with real tectonic plate boundaries "
                  "(Peter Bird, 2003 / fraxen/tectonicplates) overlaid in red. Drag to explore further, "
                  "scroll to zoom.")
        notable = eq.nlargest(400, 'mag')
        globe_html = ("""<div id="cinematic-globe" style="width:100%; height:560px;"></div>"""
                       """<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script><script>"""
                       + GLOBE_JS
                       .replace('__QUAKE_LAT__', json.dumps(notable['latitude'].round(2).tolist()))
                       .replace('__QUAKE_LON__', json.dumps(notable['longitude'].round(2).tolist()))
                       .replace('__QUAKE_MAG__', json.dumps(notable['mag'].round(2).tolist()))
                       .replace('__QUAKE_DEPTH__', json.dumps(notable['depth'].round(1).tolist()))
                       .replace('__QUAKE_COLOR__', json.dumps(
                           [CLASS_COLORS.get(c, '#29e0ff') for c in notable['mag_class'].astype(str)]))
                       .replace('__PLATE_SEGMENTS__', json.dumps(EX['plate_segments']))
                       + "</script>")
        components.html(globe_html, height=580)

# ======================================================================================
# PAGE: DATA LAB (cleaning + preprocessing, fully shown)
# ======================================================================================
elif page == "🧪  Data Lab":
    hero("THE DATA LAB", "CLEANING, OUTLIER HANDLING & NORMALIZATION — FULL AUDIT TRAIL")

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("RAW RECORDS", f"{clog['raw_rows']:,}")
    with c2: kpi("DUPLICATE IDs REMOVED", f"{clog['n_dupes']}")
    with c3: kpi("NEGATIVE DEPTH FIXED", f"{clog['n_neg_depth']}")
    with c4: kpi("NON-EARTHQUAKE EVENTS SET ASIDE", f"{clog['n_non_earthquake']}")

    tab1, tab2, tab3, tab4 = st.tabs(["CLEANING LOG", "OUTLIER DETECTION", "NORMALIZATION", "FEATURE ENGINEERING"])

    with tab1:
        st.markdown("""<div class="panel">
        <h3>Data Quality Issues Found & Resolved</h3>
        <table style="width:100%; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
        <tr style="color:#6b7a99;"><td style="padding:6px 0;">ISSUE</td><td>COUNT</td><td>RESOLUTION</td></tr>
        <tr><td>Duplicate event IDs (revision re-posts)</td><td>{}</td><td>Kept latest revision, dropped rest</td></tr>
        <tr><td>Negative depth values</td><td>{}</td><td>Clipped to 0 — near-surface elevation-datum artifact, not an error</td></tr>
        <tr><td>Negative RMS residual (physically invalid)</td><td>{}</td><td>Set to NaN, then median-imputed</td></tr>
        <tr><td>Missing depth</td><td>{}</td><td>Median imputed</td></tr>
        <tr><td>Missing RMS</td><td>{}</td><td>Median imputed + missingness flag kept as a feature</td></tr>
        <tr><td>Non-earthquake events (nuclear tests, mine collapses, etc.)</td><td>{}</td><td>Set aside from the magnitude-class model — different physics, would contaminate the target</td></tr>
        </table></div>""".format(clog['n_dupes'], clog['n_neg_depth'], clog['n_neg_rms'],
                                   clog['n_depth_missing'], clog['n_rms_missing'], clog['n_non_earthquake']),
                    unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Columns dropped from modeling entirely** (60–85% missing — `nst`, `gap`, `dmin`, "
                    "`horizontalError`, `magError`, `magNst`): too sparse to impute reliably, and `magError`/`magNst` "
                    "are direct byproducts of the magnitude calculation itself — using them would leak the target.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(eq, y='depth', title="DEPTH — IQR OUTLIER VIEW", points='outliers')
            fig.update_traces(marker_color='#ff3860')
            fig.update_layout(template=DARK_TEMPLATE, height=380)
            st.plotly_chart(fig, use_container_width=True)
            q1, q3 = eq['depth'].quantile([0.25, 0.75])
            iqr = q3 - q1
            st.caption(f"IQR bounds: [{max(0, q1-1.5*iqr):.1f}, {q3+1.5*iqr:.1f}] km. Deep subduction-zone events "
                       "beyond this range are real physics, not errors — kept.")
        with col2:
            z = (eq['rms'] - eq['rms'].mean()) / eq['rms'].std()
            fig = px.histogram(z, nbins=60, title="RMS RESIDUAL — Z-SCORE DISTRIBUTION")
            fig.add_vline(x=3, line_dash='dash', line_color='#ff3860')
            fig.add_vline(x=-3, line_dash='dash', line_color='#ff3860')
            fig.update_layout(template=DARK_TEMPLATE, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{(z.abs()>3).sum()} events beyond |z|=3 — flagged, not removed (measurement-quality "
                       "outliers, still real events).")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(eq, x='depth', nbins=60, title="DEPTH — RAW DISTRIBUTION (right-skewed)",
                               marginal='box')
            fig.update_layout(template=DARK_TEMPLATE, height=380)
            st.plotly_chart(fig, use_container_width=True)
            skew = eq['depth'].skew()
            st.caption(f"Skewness = {skew:.2f} — heavily right-skewed (most events shallow, long tail of deep ones).")
        with col2:
            log_depth = np.log1p(eq['depth'])
            fig = px.histogram(log_depth, nbins=60, title="LOG(1+DEPTH) — AFTER TRANSFORM")
            fig.update_layout(template=DARK_TEMPLATE, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Skewness after log-transform = {log_depth.skew():.2f} — much closer to symmetric. "
                       "(Models here use StandardScaler on raw depth; log-transform shown for comparison.)")

    with tab4:
        st.markdown("""<div class="panel"><h3>Engineered Features</h3>
        <ul style="font-family:'Space Grotesk',sans-serif; line-height:2;">
        <li><b>year / decade</b> — extracted from event timestamp</li>
        <li><b>magType_grouped</b> — 23 raw magnitude-scale codes collapsed into 5 groups (mb, mw_family, ms, ml, other) — one-hot encoded</li>
        <li><b>depth_category</b> — Shallow (&lt;70km) / Intermediate (70–300km) / Deep (&gt;300km), standard seismological bands — one-hot encoded</li>
        <li><b>rms_missing</b> — binary flag preserving the fact that ~30% of RMS values were missing, kept as its own signal rather than silently imputed away</li>
        <li><b>mag_class</b> — target variable: Moderate (5.0–5.9) / Strong (6.0–6.9) / Major (7.0–7.9) / Great (8.0+), USGS-standard magnitude bands</li>
        </ul></div>""", unsafe_allow_html=True)

# ======================================================================================
# PAGE: EXPLORATORY ANALYSIS
# ======================================================================================
elif page == "📊  Exploratory Analysis":
    hero("EXPLORATORY ANALYSIS", "UNIVARIATE · BIVARIATE · CORRELATION")
    tab1, tab2, tab3, tab4 = st.tabs(["UNIVARIATE", "BIVARIATE", "CORRELATION", "3D VIEW"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            feat = st.selectbox("Numeric feature", ['mag', 'depth', 'rms', 'latitude', 'longitude', 'year'])
            fig = px.histogram(eq, x=feat, nbins=50, marginal='box', title=f"{feat.upper()} DISTRIBUTION")
            fig.update_layout(template=DARK_TEMPLATE, height=400)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            catfeat = st.selectbox("Categorical feature", ['magType_grouped', 'depth_category', 'mag_class', 'type'])
            src = df if catfeat == 'type' else eq
            counts = src[catfeat].value_counts()
            fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"{catfeat.upper()} COUNTS",
                        color=counts.index.astype(str), color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(template=DARK_TEMPLATE, height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(eq.sample(8000, random_state=42), x='depth', y='mag', color='mag_class',
                             color_discrete_map=CLASS_COLORS, category_orders={'mag_class': CLASS_ORDER},
                             opacity=0.5, title="DEPTH vs MAGNITUDE")
            fig.update_layout(template=DARK_TEMPLATE, height=420)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(eq, x='decade', y='mag', title="MAGNITUDE BY DECADE",
                        color_discrete_sequence=['#29e0ff'])
            fig.update_layout(template=DARK_TEMPLATE, height=420)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        corr_cols = ['mag', 'depth', 'rms', 'latitude', 'longitude', 'year']
        corr = eq[corr_cols].corr().round(2)
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title="CORRELATION MATRIX")
        fig.update_layout(template=DARK_TEMPLATE, height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        sample3d = eq.sample(6000, random_state=42)
        fig = px.scatter_3d(sample3d, x='longitude', y='latitude', z='depth', color='mag_class',
                            color_discrete_map=CLASS_COLORS, category_orders={'mag_class': CLASS_ORDER},
                            size='mag', size_max=8, opacity=0.6, title="GEOSPATIAL + DEPTH, 3D")
        fig.update_scenes(zaxis_autorange='reversed',
                          xaxis=dict(backgroundcolor='#05070d', gridcolor='#1e2a45'),
                          yaxis=dict(backgroundcolor='#05070d', gridcolor='#1e2a45'),
                          zaxis=dict(backgroundcolor='#05070d', gridcolor='#1e2a45'))
        fig.update_layout(template=DARK_TEMPLATE, height=600)
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# PAGE: MODEL OBSERVATORY
# ======================================================================================
elif page == "🧠  Model Observatory":
    hero("MODEL OBSERVATORY", "TRAINING · TUNING · CROSS-VALIDATION · COMPARISON")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["CLASSIFICATION RESULTS", "HYPERPARAMETER TUNING",
                                              "CROSS-VALIDATION", "FEATURE IMPORTANCE", "REGRESSION"])

    with tab1:
        st.markdown("8 classifiers trained: **4 base** (Logistic Regression, KNN, Decision Tree, SVM) + "
                    "**4 ensemble** (Random Forest — tuned, Gradient Boosting — tuned, AdaBoost, Voting)")
        st.dataframe(M['results'].style.highlight_max(axis=0, color='#123a2e'), use_container_width=True)

        fig = px.bar(M['results'].reset_index(), x='Model', y=['Accuracy', 'F1 (macro)', 'ROC-AUC (macro)'],
                    barmode='group', title="MODEL COMPARISON")
        fig.update_layout(template=DARK_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Confusion Matrix")
        mchoice = st.selectbox("Model", list(M['fitted'].keys()))
        cm = M['confusion_mats'][mchoice]
        fig = px.imshow(cm, text_auto=True, x=CLASS_ORDER, y=CLASS_ORDER,
                        labels=dict(x="Predicted", y="Actual"), color_continuous_scale='Tealrose',
                        title=f"CONFUSION MATRIX — {mchoice}")
        fig.update_layout(template=DARK_TEMPLATE, height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Note the 'Great' row/column — only ~21 test examples exist for magnitude 8+. "
                  "No amount of tuning fixes genuine data scarcity for catastrophic events.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Random Forest — GridSearchCV")
            st.json(M['rf_tuning_info']['search_space'])
            st.success(f"**Best params:** {M['rf_tuning_info']['best_params']}  \n"
                      f"**Best CV F1 (macro):** {M['rf_tuning_info']['best_cv_f1']:.3f}")
            cvr = pd.DataFrame(M['rf_tuning_info']['cv_results'])[['params', 'mean_test_score', 'rank_test_score']]
            cvr = cvr.sort_values('rank_test_score')
            st.dataframe(cvr, use_container_width=True, height=200)
        with c2:
            st.markdown("### Gradient Boosting — RandomizedSearchCV")
            st.json(M['gb_tuning_info']['search_space'])
            st.success(f"**Best params:** {M['gb_tuning_info']['best_params']}  \n"
                      f"**Best CV F1 (macro):** {M['gb_tuning_info']['best_cv_f1']:.3f}")
            cvr2 = pd.DataFrame(M['gb_tuning_info']['cv_results'])[['params', 'mean_test_score', 'rank_test_score']]
            cvr2 = cvr2.sort_values('rank_test_score')
            st.dataframe(cvr2, use_container_width=True, height=200)
        st.caption("Search run on a 6,000-row stratified subsample (3-fold CV) for tractable runtime, then "
                  "the winning configuration is refit on the full 76,466-row training set.")

    with tab3:
        st.markdown("### 5-Fold Cross-Validation — Tuned Random Forest")
        cv_df = pd.DataFrame({'Fold': range(1, 6), 'F1 (macro)': M['cv_scores'].round(3)})
        fig = px.bar(cv_df, x='Fold', y='F1 (macro)', title="CV FOLD SCORES", color_discrete_sequence=['#29e0ff'])
        fig.add_hline(y=M['cv_scores'].mean(), line_dash='dash', line_color='#ff3860',
                     annotation_text=f"Mean = {M['cv_scores'].mean():.3f}")
        fig.update_layout(template=DARK_TEMPLATE, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Std across folds: {M['cv_scores'].std():.3f} — consistent performance, not a lucky split.")

    with tab4:
        imp = M['rf_importances'].rename(index=lambda x: x.replace('magType_grouped_', 'MagType: ')
                                          .replace('depth_category_', 'Depth: ').replace('_', ' ').title())
        fig = px.bar(imp, orientation='h', title="RANDOM FOREST FEATURE IMPORTANCE",
                    color_discrete_sequence=['#a56bff'])
        fig.update_layout(template=DARK_TEMPLATE, yaxis=dict(autorange="reversed"), height=450, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.warning("**Honest read:** `year` ranks #1. That's not earthquakes getting stronger over time — it's "
                  "that *which magnitude scale gets used* (`magType`) has shifted historically, and the model "
                  "picks up on that reporting artifact alongside real geophysics (depth, location). Worth knowing "
                  "before treating this as a pure physical model.")

    with tab5:
        st.markdown("### Regression — Predicting Exact Magnitude")
        st.dataframe(M['reg_results'].style.highlight_max(axis=0, color='#123a2e', subset=['R2'])
                    .highlight_min(axis=0, color='#123a2e', subset=['MAE']), use_container_width=True)
        best_r = M['best_reg_model']
        st.success(f"**Best: {best_r}** — R² = {M['reg_results'].loc[best_r,'R2']:.3f}, "
                  f"MAE = {M['reg_results'].loc[best_r,'MAE']:.3f} magnitude units")
        st.caption("Moderate R² is expected and honest: exact magnitude is set by rupture physics at the source, "
                  "not by where/how a station recorded it. Location + depth + reporting metadata only go so far.")

# ======================================================================================
# PAGE: SEISMIC FORECAST
# ======================================================================================
elif page == "🔮  Seismic Forecast":
    hero("SEISMIC FORECAST", f"LIVE PREDICTION — {M['best_class_model']} + {M['best_reg_model']}")

    presets = {"— Custom —": None}
    notable = eq.nlargest(20, 'mag')
    for _, row in notable.iterrows():
        presets[f"{row['place'][:40]} (M{row['mag']})"] = row
    choice = st.selectbox("Load a historic event (optional)", list(presets.keys()))
    base = presets[choice] if presets[choice] is not None else eq.iloc[0]

    c1, c2, c3 = st.columns(3)
    with c1:
        lat = st.slider("Latitude", -90.0, 90.0, float(base['latitude']))
        lon = st.slider("Longitude", -180.0, 180.0, float(base['longitude']))
    with c2:
        depth = st.slider("Depth (km)", 0.0, 700.0, float(base['depth']))
        year = st.slider("Year", 1900, 2023, int(base['year']) if pd.notna(base.get('year', 2023)) else 2023)
    with c3:
        rms = st.slider("RMS residual", 0.0, 3.0, float(base['rms']) if pd.notna(base.get('rms')) else 0.9)
        magtype = st.selectbox("Magnitude scale", ['mw_family', 'mb', 'ms', 'ml', 'other'])

    row = {c: 0 for c in M['feature_cols']}
    row.update({'depth': depth, 'rms': rms, 'latitude': lat, 'longitude': lon, 'year': year, 'rms_missing': 0})
    if f'magType_grouped_{magtype}' in row: row[f'magType_grouped_{magtype}'] = 1
    dcat = 'Shallow' if depth < 70 else ('Intermediate' if depth < 300 else 'Deep')
    if f'depth_category_{dcat}' in row: row[f'depth_category_{dcat}'] = 1
    input_df = pd.DataFrame([row])[M['feature_cols']]

    clf = M['fitted'][M['best_class_model']]
    scaled = M['scaler'].transform(input_df)
    pred_class = clf.predict(scaled)[0]
    proba = dict(zip(clf.classes_, clf.predict_proba(scaled)[0]))

    reg = M['fitted_reg'][M['best_reg_model']]
    scaled_r = M['scaler_r'].transform(input_df)
    pred_mag = reg.predict(scaled_r)[0]

    st.markdown("### Prediction")
    c1, c2, c3 = st.columns(3)
    with c1: kpi("PREDICTED CLASS", pred_class)
    with c2: kpi("PREDICTED MAGNITUDE", f"{pred_mag:.2f}")
    with c3: kpi("CONFIDENCE", f"{max(proba.values())*100:.0f}%")

    proba_df = pd.DataFrame({'Class': list(proba.keys()), 'Probability': list(proba.values())})
    fig = px.bar(proba_df, x='Class', y='Probability', color='Class', color_discrete_map=CLASS_COLORS,
                category_orders={'Class': CLASS_ORDER}, title="CLASS PROBABILITIES")
    fig.update_layout(template=DARK_TEMPLATE, height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Why This Prediction? — Feature Contribution")
    st.caption("A simplified linear attribution: each feature's contribution = its Random Forest importance "
              "× how many standard deviations this input sits from the training-set mean. This is NOT SHAP "
              "or a full explainability method — it's a fast, honest approximation showing which inputs are "
              "pulling this specific prediction away from a 'typical' event.")
    contributions = M['rf_importances'].reindex(M['feature_cols']).values * scaled[0]
    contrib_labels = [c.replace('magType_grouped_', 'MagType: ').replace('depth_category_', 'Depth: ')
                      .replace('_', ' ').title() for c in M['feature_cols']]
    contrib_series = pd.Series(contributions, index=contrib_labels).sort_values(key=abs, ascending=False).head(8)

    fig_wf = go.Figure(go.Waterfall(
        orientation='h',
        y=contrib_series.index[::-1],
        x=contrib_series.values[::-1],
        connector={'line': {'color': '#1e2a45'}},
        increasing={'marker': {'color': '#ff3860'}},
        decreasing={'marker': {'color': '#29e0ff'}},
    ))
    fig_wf.update_layout(template=DARK_TEMPLATE, height=420, title="TOP 8 FEATURE CONTRIBUTIONS",
                         showlegend=False)
    st.plotly_chart(fig_wf, use_container_width=True)
    st.caption("🔴 Red pushes toward a higher-risk class · 🔵 Blue pulls toward Moderate")

    if M['best_class_model'] == 'Voting Ensemble':
        st.markdown("### Inside the Vote — What Each Ensemble Member Thinks")
        voting_model = M['fitted']['Voting Ensemble']
        # VotingClassifier label-encodes y for its sub-estimators, so each member's own
        # .classes_ is [0,1,2,3], not the string labels. The ensemble's own .classes_ gives
        # the true label for each integer code, in order — use that to decode.
        code_to_label = dict(enumerate(voting_model.classes_))
        vote_rows = []
        for name, est in voting_model.named_estimators_.items():
            raw_p = dict(zip(est.classes_, est.predict_proba(scaled)[0]))
            p = {code_to_label[code]: prob for code, prob in raw_p.items()}
            vote_rows.append({'Member': name.upper(), **{c: p.get(c, 0) for c in CLASS_ORDER}})
        vote_df = pd.DataFrame(vote_rows).set_index('Member')
        fig_vote = px.bar(vote_df.reset_index(), x='Member', y=CLASS_ORDER, barmode='stack',
                          color_discrete_map=CLASS_COLORS, title="EACH MEMBER'S VOTE (stacked probability)")
        fig_vote.update_layout(template=DARK_TEMPLATE, height=350)
        st.plotly_chart(fig_vote, use_container_width=True)
        st.caption("The Voting Ensemble averages these three members' probabilities (soft voting) "
                  "to reach the final prediction shown above.")

# ======================================================================================
# PAGE: EXPOSURE ATLAS
# ======================================================================================
elif page == "🌆  Exposure Atlas":
    hero("EXPOSURE ATLAS", "REAL POPULATION EXPOSURE — HISTORICAL DENSITY + TECTONIC PROXIMITY")
    st.warning("**Why this isn't built from the classifier:** an earlier version used the trained model's "
              "raw risk prediction for real cities — it ranked Sydney and Melbourne as higher-risk than most "
              "of Japan, which is geologically backwards (Australia sits in a stable plate interior). That's "
              "the same extrapolation artifact flagged in the Hazard Forecast tab, but naming real cities and "
              "real population numbers makes it a much bigger honesty problem. So this page uses a **real, "
              "geologically-grounded Exposure Score** instead: historical earthquake density within 300km + "
              "distance to the nearest real tectonic plate boundary. Verified against known geology: Tokyo, "
              "Taiwan and San Francisco correctly rank High; Sydney correctly ranks Low.")

    cities = DD['cities']
    tab1, tab2, tab3 = st.tabs(["POPULATION EXPOSURE", "CITY LOOKUP", "WAVE ARRIVAL SIMULATOR"])

    with tab1:
        pop_by_tier = cities.groupby('tier')['pop'].sum().reindex(['Low', 'Moderate', 'High'])
        c1, c2, c3 = st.columns(3)
        with c1: kpi("HIGH-EXPOSURE POPULATION", f"{pop_by_tier['High']/1e6:,.0f}M")
        with c2: kpi("MODERATE-EXPOSURE POPULATION", f"{pop_by_tier['Moderate']/1e6:,.0f}M")
        with c3: kpi("CITIES ANALYZED", f"{len(cities):,}")

        c1, c2 = st.columns([1, 1.4])
        with c1:
            fig = px.bar(pop_by_tier, title="POPULATION BY EXPOSURE TIER",
                        color=pop_by_tier.index, color_discrete_map={'Low': '#37f0a0', 'Moderate': '#ffb020', 'High': '#ff3860'})
            fig.update_layout(template=DARK_TEMPLATE, height=380, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter_geo(cities[cities['pop'] > 200000], lat='lat', lon='lng', color='tier',
                                 size='pop', size_max=28, hover_name='city',
                                 color_discrete_map={'Low': '#37f0a0', 'Moderate': '#ffb020', 'High': '#ff3860'},
                                 title="CITIES >200K POPULATION, BY EXPOSURE TIER")
            fig.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#0c1120', oceancolor='#05070d',
                            showocean=True, coastlinecolor='#1e2a45')
            fig.update_layout(template=DARK_TEMPLATE, height=380, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Highest-Exposure Major Cities (population > 1M)")
        top_exp = cities[cities['pop'] > 1e6].nlargest(15, 'exposure_score')[
            ['city', 'country', 'pop', 'historical_events_300km', 'dist_nearest_plate_km', 'exposure_score']]
        st.dataframe(top_exp.style.background_gradient(subset=['exposure_score'], cmap='Reds'),
                    use_container_width=True)

    with tab2:
        city_options = (cities['city'] + ', ' + cities['country']).tolist()
        default_idx = city_options.index('Tokyo, Japan') if 'Tokyo, Japan' in city_options else 0
        choice = st.selectbox("Search any city", city_options, index=default_idx)
        row = cities.iloc[city_options.index(choice)]

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("EXPOSURE TIER", str(row['tier']))
        with c2: kpi("POPULATION", f"{row['pop']:,.0f}")
        with c3: kpi("EVENTS WITHIN 300KM", f"{int(row['historical_events_300km'])}")
        with c4: kpi("NEAREST PLATE BOUNDARY", f"{row['dist_nearest_plate_km']:.0f} km")

        nearby = eq.copy()
        nearby['dist_km'] = np.sqrt(
            ((nearby['latitude'] - row['lat']) * 111.0) ** 2 +
            ((nearby['longitude'] - row['lng']) * 111.0 * np.cos(np.radians(row['lat']))) ** 2
        )
        nearby = nearby[nearby['dist_km'] <= 300].nlargest(200, 'mag')
        if len(nearby):
            fig = px.scatter_geo(nearby, lat='latitude', lon='longitude', size='mag', size_max=12,
                                 color='mag', color_continuous_scale='Inferno', hover_name='place',
                                 title=f"HISTORICAL EVENTS WITHIN 300KM OF {row['city'].upper()}")
            fig.add_trace(go.Scattergeo(lat=[row['lat']], lon=[row['lng']], mode='markers',
                                        marker=dict(size=16, color='#29e0ff', symbol='star'), name=row['city']))
            fig.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#0c1120', oceancolor='#05070d',
                            showocean=True, coastlinecolor='#1e2a45',
                            center=dict(lat=row['lat'], lon=row['lng']), projection_scale=4)
            fig.update_layout(template=DARK_TEMPLATE, height=480, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No M5+ events recorded within 300km of this city since 1900 — consistent with a low-exposure location.")

    with tab3:
        st.caption("Pick a real historic earthquake and a real city. Simplified constant-velocity model "
                  "(P-wave ~6.5 km/s, S-wave ~3.7 km/s, average continental crust) — NOT a full ray-traced "
                  "travel time through Earth's actual layered velocity structure (real models like IASP91 "
                  "account for depth-dependent speed). Countdown is compressed ~10x for demonstration; "
                  "real computed times are shown as text.")
        c1, c2 = st.columns(2)
        with c1:
            notable_events = eq.nlargest(30, 'mag')
            event_options = [f"M{r['mag']} {r['place'][:35]} ({pd.to_datetime(r['time']).year})"
                             for _, r in notable_events.iterrows()]
            ev_choice = st.selectbox("Source earthquake", event_options)
            ev_row = notable_events.iloc[event_options.index(ev_choice)]
        with c2:
            city_choice = st.selectbox("City to simulate arrival at", city_options,
                                       index=default_idx, key='wave_city')
            city_row = cities.iloc[city_options.index(city_choice)]

        dist_km = 111.0 * np.sqrt(
            (ev_row['latitude'] - city_row['lat']) ** 2 +
            ((ev_row['longitude'] - city_row['lng']) * np.cos(np.radians(city_row['lat']))) ** 2
        )
        VP, VS = 6.5, 3.7
        t_p = dist_km / VP
        t_s = dist_km / VS

        st.markdown(f"**Distance:** {dist_km:,.0f} km &nbsp;|&nbsp; **Real P-wave arrival:** {t_p:.1f}s "
                   f"&nbsp;|&nbsp; **Real S-wave arrival:** {t_s:.1f}s ({t_s/60:.1f} min)")

        wave_html = ("""
        <div style="font-family:'JetBrains Mono',monospace; color:#d8e0f0;">
        <button id="wave-btn" style="background:linear-gradient(90deg,#29e0ff,#a56bff);border:none;
            color:#05070d;font-weight:700;padding:10px 26px;border-radius:6px;cursor:pointer;
            font-family:'JetBrains Mono',monospace;">▶ SIMULATE PROPAGATION</button>
        <div id="wave-status" style="margin-top:14px; font-size:1.1rem; font-weight:700; color:#6b7a99;">STANDBY</div>
        <div style="margin-top:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#ffb020;">
            <span>P-WAVE</span><span id="p-countdown">""" + f"{t_p:.1f}s" + """</span>
          </div>
          <div style="background:#10182c; border-radius:4px; height:8px; overflow:hidden;">
            <div id="p-bar" style="background:#ffb020; height:100%; width:0%;"></div>
          </div>
          <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#ff3860; margin-top:8px;">
            <span>S-WAVE</span><span id="s-countdown">""" + f"{t_s:.1f}s" + """</span>
          </div>
          <div style="background:#10182c; border-radius:4px; height:8px; overflow:hidden;">
            <div id="s-bar" style="background:#ff3860; height:100%; width:0%;"></div>
          </div>
        </div>
        </div>
        <script>""" + WAVE_JS.replace('__T_P__', str(t_p)).replace('__T_S__', str(t_s)) + "</script>")
        components.html(wave_html, height=220)

# ======================================================================================
# PAGE: FORECAST LAB
# ======================================================================================
elif page == "🌊  Forecast Lab":
    hero("FORECAST LAB", "GUTENBERG-RICHTER RETURN PERIODS + AFTERSHOCK CLUSTERING")
    tab1, tab2 = st.tabs(["RETURN PERIOD CALCULATOR", "AFTERSHOCK CLUSTERS"])

    with tab1:
        st.markdown(f"**Fitted Gutenberg-Richter law:** log₁₀(N) = {DD['gr_a']:.2f} - {DD['gr_b']:.2f}·M "
                   f"&nbsp;&nbsp;(b-value = {DD['gr_b']:.2f} — real tectonic seismicity is typically ~1.0)")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=DD['gr_table']['magnitude'], y=DD['gr_table']['annual_rate_actual'],
                                 mode='markers', name='Actual (historical)', marker=dict(color='#29e0ff', size=8)))
        fig.add_trace(go.Scatter(x=DD['gr_table']['magnitude'], y=DD['gr_table']['annual_rate_fitted'],
                                 mode='lines', name='Fitted G-R law', line=dict(color='#ff3860', dash='dash')))
        fig.update_layout(template=DARK_TEMPLATE, yaxis_type='log', height=420,
                          title="ANNUAL EVENT RATE vs MAGNITUDE (log scale)",
                          xaxis_title='Magnitude', yaxis_title='Events/year (globally)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Fitted vs actual rates diverge somewhat at high magnitudes — a known real limitation "
                  "of a single global b-value fit (catalog completeness and regional variation both matter). "
                  "Shown honestly rather than hidden.")

        st.markdown("### Probability Calculator")
        c1, c2 = st.columns(2)
        with c1:
            calc_mag = st.slider("Magnitude threshold", 5.0, 9.5, 8.0, 0.1)
        with c2:
            calc_years = st.slider("Time window (years)", 1, 100, 10)
        rate = 10 ** (DD['gr_a'] - DD['gr_b'] * calc_mag)
        prob = 1 - np.exp(-rate * calc_years)
        c1, c2, c3 = st.columns(3)
        with c1: kpi("ANNUAL RATE", f"{rate:.3f}/yr")
        with c2: kpi("MEAN RECURRENCE", f"{1/rate:.1f} yrs" if rate > 0 else "—")
        with c3: kpi(f"P(≥1 EVENT IN {calc_years}Y)", f"{prob*100:.1f}%")
        st.caption("Poisson process assumption: P(at least one event) = 1 - e^(-rate × years). "
                  "This describes the whole planet, not any single location.")

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1: kpi("CLUSTERS FOUND", f"{DD['n_clusters']:,}")
        with c2: kpi("ISOLATED EVENTS (NOISE)", f"{DD['n_noise']:,}")
        with c3: kpi("LARGEST SEQUENCE", f"{DD['top_clusters'][0]['size']} events")
        st.caption("DBSCAN on (latitude, longitude, time) jointly — clusters are real spatial-temporal "
                  "sequences, not just nearby points. The largest cluster found is, unprompted, the real "
                  "2011 Tohoku M9.1 aftershock sequence — a strong sanity check that this technique works.")

        st.markdown("### Top 15 Sequences by Size")
        cluster_df = pd.DataFrame(DD['top_clusters'])
        st.dataframe(cluster_df[['size', 'mainshock_mag', 'place', 'date']], use_container_width=True)

        pick = st.selectbox("Visualize a sequence", [f"{r['place'][:40]} M{r['mainshock_mag']} ({r['date']})"
                                                       for r in DD['top_clusters']])
        pick_idx = [f"{r['place'][:40]} M{r['mainshock_mag']} ({r['date']})" for r in DD['top_clusters']].index(pick)
        cid = DD['top_clusters'][pick_idx]['cluster_id']
        seq = DD['eq_clustered'][DD['eq_clustered']['cluster'] == cid].sort_values('time')
        fig = px.scatter_geo(seq, lat='latitude', lon='longitude', size='mag', size_max=14,
                             color='mag', color_continuous_scale='Inferno',
                             hover_data={'time': True}, title=f"SEQUENCE: {pick}")
        fig.update_geos(bgcolor='rgba(0,0,0,0)', landcolor='#0c1120', oceancolor='#05070d', showocean=True,
                        coastlinecolor='#1e2a45', center=dict(lat=seq['latitude'].mean(), lon=seq['longitude'].mean()),
                        projection_scale=5)
        fig.update_layout(template=DARK_TEMPLATE, height=460, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# PAGE: SONIC LAB
# ======================================================================================
elif page == "🔊  Sonic Lab":
    hero("SONIC LAB", "HEAR 124 YEARS OF SEISMIC ACTIVITY — MAGNITUDE → PITCH, DEPTH → TIMBRE")
    st.markdown("""<div class="panel">
    Select a decade and press play. Real event sequence, sonified with pure synthesized tones
    (Web Audio API — no audio files). <b>Higher magnitude → higher pitch.</b>
    <b>Deeper events → darker, filtered timbre</b> (sawtooth wave, low-pass filtered);
    shallow events stay bright and sine-toned.
    </div>""", unsafe_allow_html=True)

    sonify_html = ("""
    <div style="font-family:'JetBrains Mono',monospace; color:#d8e0f0; padding: 10px 0;">
    <select id="decade-select" style="background:#10182c;color:#d8e0f0;border:1px solid #1e2a45;
        padding:10px;border-radius:6px;margin-right:12px;font-family:'JetBrains Mono',monospace;"></select>
    <button id="play-btn" style="background:linear-gradient(90deg,#29e0ff,#a56bff);border:none;
        color:#05070d;font-weight:700;padding:10px 26px;border-radius:6px;cursor:pointer;
        font-family:'JetBrains Mono',monospace; letter-spacing:0.05em;">▶ PLAY DECADE</button>
    <div id="sonify-status" style="margin-top:14px;color:#6b7a99;font-size:0.85rem;">Ready.</div>
    </div>
    <script>""" + SONIFY_JS.replace('__SONIFY_DATA__', json.dumps(EX['sonify'])) + "</script>")
    components.html(sonify_html, height=150)

    st.caption("Frequency = 180 + (magnitude − 5) × 140 Hz. Each decade is compressed to at most 50 events, "
              "spaced 110ms apart, so a whole decade plays back in a few seconds.")

# ======================================================================================
# PAGE: KEY INSIGHTS
# ======================================================================================
elif page == "📡  Key Insights":
    hero("KEY INSIGHTS", "WHAT THE DATA ACTUALLY SHOWS")
    best = M['best_class_model']
    best_r = M['best_reg_model']
    st.markdown(f"""
    - **{best}** is the strongest classifier at separating magnitude classes
      (F1-macro {M['results'].loc[best,'F1 (macro)']:.3f}, ROC-AUC {M['results'].loc[best,'ROC-AUC (macro)']:.3f}),
      beating every base model and every other ensemble after hyperparameter tuning.
    - **Class imbalance is the real story, not a model failure.** Great earthquakes (M8+) are
      genuinely rare — 106 in 124 years of global data. Macro-F1 scores in the 0.30s–0.37s
      across every model reflect real scarcity, not weak modeling: no algorithm can learn a
      pattern from ~20 test examples that simply doesn't have enough signal.
    - **`year` is the top feature — but it's telling you about *science history*, not seismology.**
      Which magnitude scale (`magType`) gets used has shifted over 124 years as measurement
      technology improved. The model partly learned "how earthquakes get measured in a given era,"
      which is a genuinely interesting finding in its own right, not something to hide.
    - **Exact magnitude is only moderately predictable** (R²={M['reg_results'].loc[best_r,'R2']:.2f},
      {best_r}) from depth, location, and reporting metadata — because magnitude is set by the
      physics of the rupture itself, not by where a station happened to record it.
    - **Depth and location together carry real signal**: depth banding (Shallow/Intermediate/Deep)
      and raw lat/lon rank consistently in the top features — consistent with real plate-tectonic
      structure (subduction zones produce deeper, often larger events).
    """)
    st.info("Full pipeline: cleaning → outlier handling (IQR + Z-score) → normalization curves → "
           "feature engineering → 8 classifiers (4 base + 4 ensemble, 2 hyperparameter-tuned via "
           "GridSearchCV/RandomizedSearchCV) → 5-fold cross-validation → Random Forest feature importance → "
           "4 regressors, all trained live in this dashboard on 96,115 real USGS events.")

    st.markdown('<div class="grad-divider"></div>', unsafe_allow_html=True)

    briefing_text = (
        f"Key insight briefing. The {best} model is the strongest classifier, "
        f"with an F1 macro score of {M['results'].loc[best,'F1 (macro)']:.2f}. "
        f"Class imbalance is the real story here, not a model failure. Great earthquakes, "
        f"magnitude 8 or higher, are genuinely rare: only 106 in 124 years of global data. "
        f"The year feature ranks as the top predictor, but this reflects the history of "
        f"seismic measurement technology, not earthquakes getting stronger over time. "
        f"Total energy released since 1900 equals roughly {EX['hiroshima_equiv']:,} "
        f"Hiroshima-scale bombs. And the largest aftershock sequence our clustering found, "
        f"completely unprompted, was the real 2011 Tohoku earthquake in Japan."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🔊 Voice Briefing")
        voice_html = ("""
        <div style="font-family:'JetBrains Mono',monospace;">
        <button id="voice-btn" style="background:linear-gradient(90deg,#29e0ff,#a56bff);border:none;
            color:#05070d;font-weight:700;padding:10px 24px;border-radius:6px;cursor:pointer;
            font-family:'JetBrains Mono',monospace;">🔊 READ ALOUD</button>
        <div id="voice-status" style="margin-top:10px;color:#6b7a99;font-size:0.85rem;">Ready.</div>
        </div>
        <script>""" + VOICE_JS.replace('__BRIEFING_TEXT__', json.dumps(briefing_text)) + "</script>")
        components.html(voice_html, height=90)
        st.caption("Browser-native text-to-speech (Web Speech API) — no audio files, no external service.")
    with c2:
        st.markdown("### 📄 Mission Report")
        pdf_bytes = generate_pdf_report(M, EX, DD, clog)
        st.download_button("⬇ DOWNLOAD PDF REPORT", data=pdf_bytes, file_name="seismos_mission_report.pdf",
                          mime="application/pdf")
        st.caption("One-page summary: model performance, energy stats, Gutenberg-Richter fit, "
                  "clustering results, population exposure — generated live from this session's numbers.")
