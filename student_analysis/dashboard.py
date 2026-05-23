import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Space+Mono:wght@400;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg:          #0d0f14;
    --surface:     #13161e;
    --surface2:    #1a1e2a;
    --border:      #252a38;
    --accent:      #5b6af7;
    --accent2:     #f7695b;
    --accent3:     #46d9ac;
    --text:        #e8eaf0;
    --muted:       #6b7280;
    --dropout:     #f7695b;
    --enrolled:    #46d9ac;
    --graduate:    #5b6af7;
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background-color: var(--bg) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stButton > button {
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ── Main Header ── */
.main-header {
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.main-title {
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--text);
    margin: 0;
    line-height: 1.1;
}
.main-subtitle {
    font-size: 0.95rem;
    color: var(--muted);
    margin-top: 0.5rem;
    font-weight: 300;
}
.mono-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    background: rgba(91,106,247,0.12);
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.08em;
    display: inline-block;
    margin-bottom: 0.8rem;
}

/* ── Section Headers ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 2.5rem 0 1.2rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.metric-card.blue::before  { background: var(--graduate); }
.metric-card.red::before   { background: var(--dropout); }
.metric-card.green::before { background: var(--enrolled); }
.metric-card.purple::before { background: #c084fc; }
.metric-card.gold::before  { background: #fbbf24; }
.metric-card:hover { border-color: rgba(91,106,247,0.4); }
.metric-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text);
    line-height: 1;
    letter-spacing: -0.02em;
}
.metric-badge {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.5rem;
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
}
.metric-badge.danger { background: rgba(247,105,91,0.15); color: var(--dropout); }
.metric-badge.safe   { background: rgba(70,217,172,0.15); color: var(--enrolled); }
.metric-badge.info   { background: rgba(91,106,247,0.15); color: var(--graduate); }

/* ── Chart Cards ── */
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.chart-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
}

/* ── Insight Cards ── */
.insight-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.insight-card.warn { border-left-color: var(--accent2); }
.insight-card.good { border-left-color: var(--accent3); }

/* ── Recommendation List ── */
.rec-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 1rem 1.2rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.5;
}
.rec-num {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: var(--accent);
    background: rgba(91,106,247,0.12);
    padding: 3px 8px;
    border-radius: 4px;
    flex-shrink: 0;
    margin-top: 1px;
}

/* ── Legend dots ── */
.legend {
    display: flex;
    gap: 1.2rem;
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.5rem;
    align-items: center;
}
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 4px;
}
.dot.dropout  { background: var(--dropout); }
.dot.enrolled { background: var(--enrolled); }
.dot.graduate { background: var(--graduate); }

/* ── Override Streamlit widgets to dark ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.2rem 1.4rem !important;
}
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
div[data-testid="stCheckbox"] label { color: var(--muted) !important; font-size: 0.85rem; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Upload widget ── */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Success/warning/info text in sidebar ── */
.stSuccess, .stWarning, .stInfo, .stError {
    border-radius: 8px !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#9ca3af", size=11),
    title_font=dict(family="DM Sans, sans-serif", color="#e8eaf0", size=13),
    title_x=0,
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", size=10),
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    margin=dict(l=0, r=0, t=36, b=0),
    xaxis=dict(gridcolor="#1e2330", linecolor="#252a38", tickfont=dict(color="#6b7280")),
    yaxis=dict(gridcolor="#1e2330", linecolor="#252a38", tickfont=dict(color="#6b7280")),
    height=360,
)

COLOR_MAP = {
    "Dropout":  "#f7695b",
    "Enrolled": "#46d9ac",
    "Graduate": "#5b6af7",
}

# ─── DATA LOADING ────────────────────────────────────────────────────────────────
@st.cache_data
def load_csv_from_path():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    paths = ["data.csv", os.path.join(script_dir, "data.csv"), "./data.csv",
             "dataset/data.csv", "data/data.csv"]
    try:
        for f in os.listdir('.'):
            if f.endswith('.csv'): paths.insert(0, f)
    except: pass
    for p in paths:
        try:
            if os.path.exists(p):
                return pd.read_csv(p), p
        except: continue
    return None, None


@st.cache_data
def create_sample_data():
    np.random.seed(42)
    n = 1000
    ages = np.random.normal(20, 2, n).clip(17, 35).astype(int)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.45, 0.55])
    scholarship_prob = np.where(ages < 22, 0.4, 0.2)
    scholarships = np.random.binomial(1, scholarship_prob, n)
    base_grade = 12 + scholarships * 2 + np.random.normal(0, 2, n)
    grade_1st = base_grade.clip(0, 20)
    grade_2nd = (base_grade + np.random.normal(0, 1, n)).clip(0, 20)
    dropout_prob = (0.1 + 0.3 * (grade_1st < 10) + 0.2 * (scholarships == 0) + 0.1 * (ages > 25)).clip(0, 1)
    statuses = []
    for p in dropout_prob:
        r = np.random.random()
        statuses.append('Dropout' if r < p else 'Enrolled' if r < p + 0.4 else 'Graduate')
    return pd.DataFrame({
        'Student_ID': range(1, n + 1),
        'Age_at_enrollment': ages, 'Gender': genders,
        'Application_mode': np.random.choice([1,2,17,18,39,42,43], n),
        'Fathers_qualification': np.random.choice([1,2,3,4,5,19,34,35], n),
        'Mothers_qualification': np.random.choice([1,2,3,4,5,19,34,35], n),
        'Tuition_fees_up_to_date': np.random.choice([0,1], n, p=[0.15,0.85]),
        'Scholarship_holder': scholarships,
        'Curricular_units_1st_sem_grade': grade_1st,
        'Curricular_units_2nd_sem_grade': grade_2nd,
        'Daytime_evening_attendance': np.random.choice([0,1], n, p=[0.3,0.7]),
        'Target': statuses,
    })

# ─── CHART HELPERS ───────────────────────────────────────────────────────────────
def stacked_bar(df, group_col, title):
    grp = df.groupby([group_col, 'Target']).size().unstack(fill_value=0)
    pct = grp.div(grp.sum(axis=1), axis=0) * 100
    fig = px.bar(pct.reset_index(), x=group_col,
                 y=[c for c in ['Dropout','Enrolled','Graduate'] if c in pct.columns],
                 color_discrete_map=COLOR_MAP, barmode='stack',
                 labels={'value': 'Percentage (%)', group_col: ''},
                 title=title)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def pie_chart(df):
    vc = df['Target'].value_counts()
    fig = go.Figure(go.Pie(
        labels=vc.index, values=vc.values,
        marker_colors=[COLOR_MAP.get(l, '#999') for l in vc.index],
        hole=0.55,
        textinfo='percent',
        textfont=dict(size=11, color='#e8eaf0'),
        hovertemplate='<b>%{label}</b><br>%{value} students<br>%{percent}<extra></extra>',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=340,
                      showlegend=True, title='Status Distribution',
                      annotations=[dict(text=f"<b>{len(df):,}</b><br><span style='font-size:10px'>students</span>",
                                        x=0.5, y=0.5, font_size=18, showarrow=False,
                                        font_color='#e8eaf0')])
    return fig


def grade_chart(df, col):
    df2 = df.copy()
    df2['Grade_Band'] = pd.cut(df2[col], bins=[0, 10, 14, 17, 20],
                                labels=['0–10 · Low', '10–14 · Avg', '14–17 · Good', '17–20 · Top'])
    return stacked_bar(df2, 'Grade_Band', 'Academic Performance vs Status')

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="mono-tag">STUDENT ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("### Data Source")

    df, file_path = load_csv_from_path()
    if df is not None:
        st.success(f"Auto-loaded: `{os.path.basename(file_path)}`")
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("File uploaded successfully")
            except Exception as e:
                st.error(f"Parse error: {e}")
        else:
            st.info("No file found — showing sample data")
            df = create_sample_data()

    # Normalize columns
    if df is not None:
        if 'Status' in df.columns:
            df = df.rename(columns={'Status': 'Target'})
        if 'Age' not in df.columns and 'Age_at_enrollment' in df.columns:
            df['Age'] = df['Age_at_enrollment']

        st.markdown("---")
        st.markdown("### Dataset")
        st.caption(f"**{df.shape[0]:,}** rows  ·  **{df.shape[1]}** columns")

        status_vc = df['Target'].value_counts()
        for status, cnt in status_vc.items():
            pct = cnt / len(df) * 100
            color = {"Dropout": "#f7695b", "Enrolled": "#46d9ac", "Graduate": "#5b6af7"}.get(status, "#999")
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:6px 10px;background:#1a1e2a;border-radius:6px;margin:4px 0;">
              <span style="font-size:0.8rem;color:#9ca3af;">{status}</span>
              <span style="font-family:'Space Mono',monospace;font-size:0.75rem;color:{color};">
                {cnt:,} · {pct:.1f}%
              </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Columns")
        cols_html = "".join([
            f'<span style="font-family:Space Mono;font-size:0.65rem;background:#1a1e2a;'
            f'color:#6b7280;padding:2px 7px;border-radius:4px;margin:2px;display:inline-block;">'
            f'{c}</span>' for c in df.columns
        ])
        st.markdown(cols_html, unsafe_allow_html=True)

# ─── MAIN CONTENT ────────────────────────────────────────────────────────────────
if df is None or 'Target' not in df.columns:
    st.error("Cannot load data. Please upload a valid CSV file with a `Target` or `Status` column.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
  <div class="mono-tag">EDUCATIONAL DATA INTELLIGENCE</div>
  <h1 class="main-title">Student Dropout Analysis</h1>
  <p class="main-subtitle">Comprehensive analysis of dropout risk factors · Academic year overview</p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ──────────────────────────────────────────────────────────────────
total = len(df)
dropout_n  = len(df[df['Target'] == 'Dropout'])
enrolled_n = len(df[df['Target'] == 'Enrolled'])
grad_n     = len(df[df['Target'] == 'Graduate'])
dr = dropout_n / total * 100
rr = (total - dropout_n) / total * 100

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card blue">
    <div class="metric-label">Total Students</div>
    <div class="metric-value">{total:,}</div>
    <span class="metric-badge info">All records</span>
  </div>
  <div class="metric-card red">
    <div class="metric-label">Dropout</div>
    <div class="metric-value">{dropout_n:,}</div>
    <span class="metric-badge danger">{dr:.1f}% rate</span>
  </div>
  <div class="metric-card green">
    <div class="metric-label">Enrolled</div>
    <div class="metric-value">{enrolled_n:,}</div>
    <span class="metric-badge safe">Active</span>
  </div>
  <div class="metric-card purple">
    <div class="metric-label">Graduate</div>
    <div class="metric-value">{grad_n:,}</div>
    <span class="metric-badge info">Completed</span>
  </div>
  <div class="metric-card gold">
    <div class="metric-label">Retention Rate</div>
    <div class="metric-value">{rr:.1f}<span style="font-size:1rem">%</span></div>
    <span class="metric-badge safe">Non-dropout</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Section: Overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.6])

with col1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(pie_chart(df), use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if 'Age' in df.columns:
        df2 = df.copy()
        df2['Age_Group'] = pd.cut(df2['Age'], bins=[16,20,23,26,100],
                                   labels=['17–20', '21–23', '24–26', '27+'])
        fig = stacked_bar(df2, 'Age_Group', 'Dropout Rate by Age Group (%)')
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Age data not available")

# ── Section: Demographics ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Demographics & Financial</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    if 'Gender' in df.columns:
        fig = stacked_bar(df, 'Gender', 'Gender vs Status (%)')
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

with c2:
    if 'Scholarship_holder' in df.columns:
        df3 = df.copy()
        df3['Beasiswa'] = df3['Scholarship_holder'].map({1: 'Scholarship', 0: 'No Scholarship'})
        fig = stacked_bar(df3, 'Beasiswa', 'Scholarship Impact (%)')
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

with c3:
    if 'Tuition_fees_up_to_date' in df.columns:
        df4 = df.copy()
        df4['SPP'] = df4['Tuition_fees_up_to_date'].map({1: 'Up-to-date', 0: 'Overdue'})
        fig = stacked_bar(df4, 'SPP', 'Tuition Status vs Dropout (%)')
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

# ── Section: Academic ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Academic Performance</div>', unsafe_allow_html=True)

grade_col = next((c for c in df.columns if 'grade' in c.lower() or 'Grade' in c), None)
attend_col = 'Daytime_evening_attendance' if 'Daytime_evening_attendance' in df.columns else None

gc1, gc2 = st.columns(2)
with gc1:
    if grade_col:
        fig = grade_chart(df, grade_col)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Grade data not available")

with gc2:
    if attend_col:
        df5 = df.copy()
        df5['Session'] = df5[attend_col].map({1: 'Daytime', 0: 'Evening'})
        fig = stacked_bar(df5, 'Session', 'Attendance Session vs Status (%)')
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Attendance data not available")

# ── Section: Family Background ────────────────────────────────────────────────
parent_col = next((c for c in df.columns if 'qualification' in c.lower()), None)
if parent_col:
    st.markdown('<div class="section-title">Family Background</div>', unsafe_allow_html=True)
    qual_map = {1:'Elementary',2:'Secondary',3:'High School',4:'Diploma',5:'Bachelor',
                19:'Master',34:'Doctorate',35:'Other'}
    df6 = df.copy()
    df6['Parent_Edu'] = df6[parent_col].map(qual_map).fillna('Other')
    fig = stacked_bar(df6, 'Parent_Edu', "Parent's Education Level vs Student Status (%)")
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)

# ── Section: Insights ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

insights = []
insights.append(("info",
    f"<b>{dr:.1f}%</b> of students drop out — "
    f"that's <b>{dropout_n:,}</b> students who need intervention strategies."))

if 'Age' in df.columns:
    d_age = df[df['Target']=='Dropout']['Age'].mean()
    g_age = df[df['Target']=='Graduate']['Age'].mean()
    direction = "older" if d_age > g_age else "younger"
    insights.append(("warn",
        f"Dropout students tend to be <b>{direction}</b> on average: "
        f"<b>{d_age:.1f} yrs</b> (dropout) vs <b>{g_age:.1f} yrs</b> (graduate)."))

if 'Scholarship_holder' in df.columns:
    r_s = len(df[(df['Target']=='Dropout')&(df['Scholarship_holder']==1)]) / max(1, len(df[df['Scholarship_holder']==1])) * 100
    r_n = len(df[(df['Target']=='Dropout')&(df['Scholarship_holder']==0)]) / max(1, len(df[df['Scholarship_holder']==0])) * 100
    cls = "good" if r_s < r_n else "warn"
    insights.append((cls,
        f"Scholarship holders drop out at <b>{r_s:.1f}%</b> vs <b>{r_n:.1f}%</b> for non-holders — "
        f"{'scholarship programs are effective ✓' if r_s < r_n else 'scholarship programs need re-evaluation ⚠'}."))

if grade_col:
    avg_d = df[df['Target']=='Dropout'][grade_col].mean()
    avg_g = df[df['Target']=='Graduate'][grade_col].mean()
    insights.append(("good",
        f"Academic performance is a strong predictor: graduate avg grade <b>{avg_g:.1f}</b> "
        f"vs dropout avg <b>{avg_d:.1f}</b> — a <b>{avg_g - avg_d:.1f} point</b> gap."))

for cls, text in insights:
    st.markdown(f'<div class="insight-card {cls}">{text}</div>', unsafe_allow_html=True)

# ── Section: Recommendations ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Action Recommendations</div>', unsafe_allow_html=True)

recs = [
    ("01", "🎓 Academic Support", "Prioritize early intervention for students with low first-semester grades. Assign tutors before the second semester begins."),
    ("02", "💰 Financial Aid", "Expand scholarship programs targeting high-risk demographic groups — particularly older, non-scholarship students."),
    ("03", "👥 Peer Mentoring", "Build a structured mentoring network pairing at-risk students with graduates in similar programs."),
    ("04", "📊 Early Warning System", "Implement an automated risk score dashboard tracking GPA, tuition status, and attendance in real time."),
    ("05", "🏫 Family Engagement", "Launch workshops for parents on how to support students academically — especially first-generation college families."),
]

for num, title, desc in recs:
    st.markdown(f"""
    <div class="rec-item">
      <span class="rec-num">{num}</span>
      <div><b style="color:#e8eaf0;">{title}</b><br>
      <span style="color:#6b7280;font-size:0.83rem;">{desc}</span></div>
    </div>""", unsafe_allow_html=True)

# ── Data Preview ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Raw Data</div>', unsafe_allow_html=True)
if st.checkbox("Show data preview (first 100 rows)", value=False):
    st.dataframe(df.head(100), use_container_width=True)
