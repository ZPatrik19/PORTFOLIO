from __future__ import annotations

import streamlit as st


THEME_CSS = """
<style>
:root {
    --rag-border: rgba(126, 146, 184, 0.26);
    --rag-border-strong: rgba(126, 176, 255, 0.36);
    --rag-primary: #67adff;
    --rag-primary-strong: #2f80ff;
    --rag-primary-soft: rgba(103, 173, 255, 0.16);
    --rag-accent: #9d7eff;
    --rag-accent-soft: rgba(157, 126, 255, 0.14);
    --rag-success: #22b07d;
    --rag-warning: #dca84d;
    --rag-surface: rgba(10, 16, 28, 0.72);
    --rag-surface-2: rgba(18, 27, 43, 0.92);
    --rag-surface-3: rgba(17, 26, 40, 0.82);
    --rag-card-bg: linear-gradient(180deg, rgba(19, 28, 44, 0.94), rgba(10, 16, 28, 0.96));
    --rag-card-bg-light: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(245,248,252,0.98));
    --rag-hero-bg: radial-gradient(circle at top left, rgba(84, 155, 255, 0.20), transparent 34%), linear-gradient(135deg, rgba(18, 30, 50, 0.98), rgba(12, 19, 31, 0.96));
    --rag-text: #f5f9ff;
    --rag-text-soft: rgba(231, 239, 250, 0.94);
    --rag-text-muted: rgba(188, 203, 224, 0.86);
    --rag-text-dark: #162132;
    --rag-text-dark-soft: #4a5b74;
}
html, body, [class*="css"]  { letter-spacing: 0.01em; }
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 3rem;
    max-width: 1580px;
}
main .block-container > div:first-child { gap: 1rem; }
[data-testid="stSidebar"] {
    border-right: 1px solid var(--rag-border);
    background: linear-gradient(180deg, rgba(12,18,30,0.98), rgba(12,18,30,0.90));
}
[data-testid="stSidebar"] section { padding-top: .5rem; }
.rag-hero {
    padding: 1.15rem 1.30rem;
    border: 1px solid var(--rag-border-strong);
    border-radius: 22px;
    background: var(--rag-hero-bg);
    margin-bottom: 1.05rem;
    box-shadow: 0 18px 36px rgba(9, 16, 27, 0.22);
    color: var(--rag-text);
}
.rag-hero h1, .rag-hero h2, .rag-hero h3 {
    margin: 0 0 .42rem 0;
    color: var(--rag-text);
    line-height: 1.15;
    letter-spacing: -0.01em;
}
.rag-hero p { margin: 0; line-height: 1.62; color: var(--rag-text-soft); max-width: 1100px; }
.rag-small { font-size: .80rem; color: var(--rag-text-muted); margin-bottom: .3rem; letter-spacing: .06em; text-transform: uppercase; }
.rag-section-title { margin-top: 1.15rem; margin-bottom: .40rem; font-size: 1.18rem; font-weight: 700; }
.rag-section-subtitle { color: var(--rag-text-muted); margin-bottom: .85rem; font-size: .96rem; line-height: 1.55; }
.rag-card {
    min-height: 156px;
    padding: 1.05rem 1.10rem;
    border: 1px solid var(--rag-border);
    border-radius: 18px;
    background: var(--rag-card-bg);
    color: var(--rag-text);
    margin-bottom: .90rem;
    box-shadow: 0 12px 28px rgba(0, 0, 0, .16);
    transition: transform .12s ease, border-color .12s ease;
}
.rag-card:hover { transform: translateY(-1px); border-color: rgba(103, 173, 255, 0.32); }
.rag-card h4 { margin: 0 0 .48rem 0; font-size: 1.04rem; color: var(--rag-text); line-height: 1.3; }
.rag-card p { margin: 0; line-height: 1.64; color: var(--rag-text-soft); }
.rag-card .rag-kicker {
    display: inline-block;
    font-size: .72rem;
    padding: .16rem .48rem;
    border-radius: 999px;
    background: rgba(103,173,255,.12);
    color: var(--rag-text);
    border: 1px solid rgba(255,255,255,.12);
    margin-bottom: .58rem;
}
.rag-kpi {
    border: 1px solid var(--rag-border);
    border-radius: 18px;
    padding: 1rem 1.05rem;
    background: linear-gradient(180deg, rgba(20,30,48,.94), rgba(13,21,34,.98));
    margin-bottom: .85rem;
    box-shadow: 0 10px 24px rgba(0,0,0,.12);
}
.rag-kpi .label { color: var(--rag-text-muted); font-size: .80rem; text-transform: uppercase; letter-spacing: .05em; }
.rag-kpi .value { color: var(--rag-text); font-size: 1.30rem; font-weight: 700; margin-top: .18rem; line-height: 1.2; }
.rag-kpi .sub { color: var(--rag-text-soft); font-size: .90rem; margin-top: .40rem; line-height: 1.45; }

.rag-status-card {
    min-height: 132px;
    padding: 1rem 1.05rem;
    border: 1px solid var(--rag-border);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(18,27,43,.94), rgba(11,18,30,.98));
    box-shadow: 0 10px 24px rgba(0,0,0,.12);
    margin-bottom: .85rem;
    overflow: hidden;
}
.rag-status-head {
    display: flex;
    align-items: center;
    gap: .48rem;
    color: var(--rag-text-muted);
    font-size: .82rem;
    line-height: 1.35;
    margin-bottom: .42rem;
}
.rag-status-dot {
    width: .58rem;
    height: .58rem;
    min-width: .58rem;
    border-radius: 999px;
    box-shadow: 0 0 0 4px rgba(255,255,255,.025);
}
.rag-status-dot.ok { background: #40d99b; }
.rag-status-dot.warn { background: #f1b65b; }
.rag-status-dot.off { background: #8190aa; }
.rag-status-dot.info { background: #69a7ff; }
.rag-status-value {
    color: var(--rag-text);
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.32;
    white-space: normal;
    overflow-wrap: anywhere;
}
.rag-status-detail {
    color: var(--rag-text-soft);
    font-size: .88rem;
    line-height: 1.48;
    margin-top: .42rem;
    white-space: normal;
    overflow-wrap: anywhere;
}

.rag-note {
    border: 1px dashed rgba(103,173,255,.36);
    border-radius: 16px;
    padding: .95rem 1rem;
    background: rgba(103,173,255,.07);
    margin-bottom: .85rem;
}
.rag-note strong { color: var(--rag-text); }
.rag-step {
    padding: .80rem 1rem;
    border-left: 4px solid rgba(103,173,255,.96);
    background: rgba(103,173,255,.08);
    border-radius: 0 12px 12px 0;
    margin: .50rem 0;
    color: var(--rag-text-soft);
}
.rag-step b { color: var(--rag-text); }
.rag-status-ok { color: #19ad6e; font-weight: 700; }
.rag-status-warn { color: #d38b05; font-weight: 700; }
.rag-status-off { color: #9aa7b8; font-weight: 700; }
.rag-chip {
    display:inline-block; padding:.24rem .62rem; margin:.12rem .20rem .12rem 0;
    border:1px solid var(--rag-border); border-radius:999px; font-size:.78rem;
    background: rgba(103,173,255,.08); color: var(--rag-text-soft);
}
.rag-evidence {
    border: 1px solid var(--rag-border);
    border-radius:14px; padding:.9rem 1rem; margin:.45rem 0;
    background: rgba(248,250,252,.05); color: var(--rag-text-soft);
}
[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(19,28,44,.92), rgba(12,18,30,.96));
    border: 1px solid var(--rag-border);
    border-radius: 18px;
    padding: .85rem .95rem;
    box-shadow: 0 8px 22px rgba(0,0,0,.12);
}
[data-testid="stMetricLabel"] { color: var(--rag-text-muted); }
[data-testid="stMetricValue"] { color: var(--rag-text); }
[data-testid="stMetricDelta"] { font-weight: 600; }
.stButton > button, .stDownloadButton > button {
    border-radius: 12px;
    border: 1px solid var(--rag-border);
    padding: .55rem 1rem;
    min-height: 2.55rem;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(61,130,255,1), rgba(115,114,255,1));
    border: none;
}
.stButton > button:hover { border-color: rgba(103,173,255,.45); }
.stTabs [data-baseweb="tab-list"] {
    gap: .35rem;
    border-bottom: 1px solid var(--rag-border);
    padding-bottom: .25rem;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(18,27,43,.58);
    border-radius: 12px 12px 0 0;
    border: 1px solid transparent;
    padding-inline: .9rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(103,173,255,.12);
    border-color: rgba(103,173,255,.25);
}
.streamlit-expanderHeader {
    border-radius: 12px;
    border: 1px solid var(--rag-border);
    background: rgba(18,27,43,.64);
}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}
[data-testid="stPlotlyChart"] > div {
    border: 1px solid rgba(126,146,184,.20);
    border-radius: 16px;
    padding: .20rem .30rem;
    background: rgba(12,19,31,.72);
    box-shadow: 0 8px 20px rgba(0,0,0,.10);
}
[data-testid="stDataFrame"] > div {
    border: 1px solid var(--rag-border);
    border-radius: 16px;
}
.stSlider [data-baseweb="slider"] { padding-inline: .15rem; }
div[data-testid="stRadio"] label p, .stSelectbox label p, .stSlider label p, .stTextArea label p, .stTextInput label p { font-size: .92rem; }
@media (prefers-color-scheme: light) {
    .rag-hero { color: var(--rag-text-dark); background: linear-gradient(135deg, rgba(96,169,255,.16), rgba(157,126,255,.10)); }
    .rag-hero h1, .rag-hero h2, .rag-hero h3 { color: var(--rag-text-dark); }
    .rag-hero p, .rag-small, .rag-section-subtitle { color: var(--rag-text-dark-soft); }
    .rag-card, .rag-kpi { background: var(--rag-card-bg-light); color: var(--rag-text-dark); }
    .rag-card h4, .rag-kpi .value { color: var(--rag-text-dark); }
    .rag-card p, .rag-kpi .sub, .rag-kpi .label { color: var(--rag-text-dark-soft); }
    .rag-card .rag-kicker { background: rgba(90,162,255,.10); color: var(--rag-text-dark); border-color: rgba(90,162,255,.16); }

    .rag-status-card { background: rgba(255,255,255,.98); }
    .rag-status-value { color: var(--rag-text-dark); }
    .rag-status-head, .rag-status-detail { color: var(--rag-text-dark-soft); }
    .rag-step { color: var(--rag-text-dark); }
    [data-testid="stMetric"] { background: rgba(255,255,255,.96); }
    [data-testid="stPlotlyChart"] > div { background: rgba(255,255,255,.98); }
}

.rag-empty {
    display:flex; align-items:flex-start; gap:.9rem;
    border:1px dashed rgba(126,146,184,.28);
    border-radius:16px; padding:1rem 1.05rem;
    background:rgba(16,24,38,.50); margin:.65rem 0 1rem;
}
.rag-empty-icon { font-size:1.45rem; line-height:1; color:var(--rag-primary); padding-top:.08rem; }
.rag-empty-title { color:var(--rag-text); font-weight:700; font-size:1rem; margin-bottom:.2rem; }
.rag-empty-text { color:var(--rag-text-soft); line-height:1.5; }
.rag-empty-hint { color:var(--rag-text-muted); font-size:.86rem; margin-top:.32rem; }

.rag-live-run {
    border:1px solid rgba(103,173,255,.32); border-radius:18px;
    padding:1rem 1.05rem; margin:.6rem 0 1rem;
    background:linear-gradient(180deg, rgba(17,29,48,.94), rgba(10,17,29,.98));
    box-shadow:0 10px 24px rgba(0,0,0,.12);
}
.rag-live-run-head { display:flex; align-items:center; gap:.5rem; color:var(--rag-text); }
.rag-live-dot { width:.55rem; height:.55rem; border-radius:999px; background:#35C98E; box-shadow:0 0 0 5px rgba(53,201,142,.09); }
.rag-live-percent { margin-left:auto; color:var(--rag-primary); font-weight:700; }
.rag-live-current { color:var(--rag-text-soft); font-size:.9rem; margin:.55rem 0 .75rem; overflow-wrap:anywhere; }
.rag-live-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; }
.rag-live-grid div { border:1px solid rgba(126,146,184,.18); border-radius:12px; padding:.7rem .75rem; background:rgba(255,255,255,.025); }
.rag-live-grid span { display:block; color:var(--rag-text-muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; }
.rag-live-grid b { display:block; color:var(--rag-text); margin-top:.15rem; font-size:.98rem; }

.rag-evidence-card {
    min-height:210px; border:1px solid var(--rag-border); border-radius:17px;
    padding:.95rem 1rem; margin-bottom:.8rem;
    background:linear-gradient(180deg, rgba(18,28,45,.92), rgba(11,18,30,.98));
}
.rag-evidence-top { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.rag-rank-chip { display:inline-flex; align-items:center; justify-content:center; min-width:2.15rem; height:1.75rem; border-radius:999px; background:rgba(79,142,247,.16); color:#cfe0ff; font-weight:700; font-size:.78rem; }
.rag-evidence-status { border-radius:999px; padding:.18rem .5rem; font-size:.72rem; border:1px solid rgba(255,255,255,.10); }
.rag-evidence-status.cited { background:rgba(53,201,142,.12); color:#89e6bd; }
.rag-evidence-status.uncited { background:rgba(143,162,190,.10); color:#b7c3d5; }
.rag-evidence-title { color:var(--rag-text); font-weight:700; margin-top:.65rem; line-height:1.35; }
.rag-evidence-meta { color:var(--rag-text-muted); font-size:.78rem; margin-top:.25rem; overflow-wrap:anywhere; }
.rag-score-track { width:100%; height:.34rem; border-radius:999px; background:rgba(143,162,190,.14); margin:.7rem 0 .65rem; overflow:hidden; }
.rag-score-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#4F8EF7,#35C98E); }
.rag-evidence-excerpt { color:var(--rag-text-soft); font-size:.88rem; line-height:1.5; }

.rag-presentation-note {
    border:1px solid rgba(155,123,247,.28); background:rgba(155,123,247,.08);
    border-radius:12px; padding:.65rem .75rem; color:var(--rag-text-soft); font-size:.84rem;
}


.rag-reference-legend {
    display:flex; flex-wrap:wrap; gap:.55rem 1rem; align-items:center;
    border:1px solid var(--rag-border); border-radius:14px;
    padding:.72rem .85rem; margin:.2rem 0 1rem;
    background:rgba(18,27,43,.52); color:var(--rag-text-soft); font-size:.84rem;
}
.rag-reference-topic {
    min-height:150px; border:1px solid var(--rag-border); border-radius:17px;
    padding:.95rem 1rem; margin-bottom:.8rem;
    background:linear-gradient(180deg, rgba(18,28,45,.90), rgba(11,18,30,.96));
    box-shadow:0 8px 20px rgba(0,0,0,.10);
}
.rag-reference-topic-count {
    display:inline-block; padding:.18rem .48rem; border-radius:999px;
    background:rgba(103,173,255,.11); color:#bcd8ff; font-size:.72rem;
    border:1px solid rgba(103,173,255,.18); margin-bottom:.55rem;
}
.rag-reference-topic-title { color:var(--rag-text); font-weight:750; font-size:1rem; line-height:1.3; }
.rag-reference-topic-text { color:var(--rag-text-soft); font-size:.87rem; line-height:1.48; margin-top:.38rem; }
.rag-reference-detail-head { display:flex; flex-wrap:wrap; align-items:center; gap:.5rem; margin-bottom:.7rem; }
.rag-reference-kind, .rag-reference-direction, .rag-reference-chip {
    display:inline-flex; align-items:center; border-radius:999px; padding:.18rem .5rem;
    font-size:.72rem; line-height:1.25; border:1px solid rgba(255,255,255,.10);
}
.rag-reference-kind { background:rgba(103,173,255,.12); color:#cfe0ff; }
.rag-reference-direction.up { background:rgba(53,201,142,.11); color:#91e8c1; }
.rag-reference-direction.down { background:rgba(242,184,86,.11); color:#f2cd8f; }
.rag-reference-direction.context { background:rgba(155,123,247,.11); color:#cdbdff; }
.rag-reference-chip { background:rgba(143,162,190,.09); color:var(--rag-text-soft); margin:.12rem .18rem .12rem 0; }
.rag-reference-definition {
    color:var(--rag-text); font-size:1rem; line-height:1.58; font-weight:560;
    border-left:3px solid rgba(103,173,255,.72); padding:.2rem 0 .2rem .75rem; margin-bottom:.8rem;
}

@media (max-width: 1100px) {
    .rag-live-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}

</style>
"""


def apply_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)
