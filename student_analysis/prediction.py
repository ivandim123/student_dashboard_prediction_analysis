import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="EduPredict · Student Status",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Serif+Display:ital@0;1&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Page background ── */
.stApp {
    background: #F7F6F2;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1200px !important;
}

/* ── Custom top banner ── */
.edu-header {
    background: #1A1A2E;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.edu-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(99, 102, 241, 0.15);
}
.edu-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 200px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(16, 185, 129, 0.1);
}
.edu-title {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem;
    color: #FFFFFF;
    margin: 0;
    line-height: 1.1;
}
.edu-subtitle {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.55);
    margin: 0.4rem 0 0;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.edu-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: rgba(255,255,255,0.8);
    font-size: 0.75rem;
    padding: 6px 14px;
    border-radius: 100px;
    letter-spacing: 0.05em;
    font-weight: 500;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E8E6E0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="metric-container"] label {
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    color: #888 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 600 !important;
    color: #1A1A2E !important;
    line-height: 1.2 !important;
}

/* ── Section headers ── */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #1A1A2E !important;
}
h2 { font-size: 1.6rem !important; margin-top: 0.5rem !important; }
h3 { font-size: 1.2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1A1A2E !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    padding: 0.6rem 1rem !important;
    margin-bottom: 0.4rem !important;
    transition: all 0.2s ease !important;
    font-size: 0.875rem !important;
    color: rgba(255,255,255,0.8) !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.12) !important;
    border-color: rgba(255,255,255,0.25) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
}
[data-testid="stSidebarHeader"] {
    padding-top: 1.5rem !important;
}

/* ── Info/success/warning/error boxes ── */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-size: 0.875rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #1A1A2E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 1.75rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: #2D2D4E !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(26,26,46,0.25) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #1A1A2E !important;
    border: 1.5px solid #D0CEC8 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    border-color: #1A1A2E !important;
    background: rgba(26,26,46,0.04) !important;
}

/* ── Input labels ── */
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stSlider label, .stFileUploader label {
    color: #2C2C3E !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 2px !important;
}

/* ── Form inputs ── */
.stTextInput input, .stNumberInput input {
    border-radius: 8px !important;
    border: 1.5px solid #C8C6C0 !important;
    font-family: 'DM Sans', sans-serif !important;
    background: #F7F6F2 !important;
    color: #1A1A2E !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #6366F1 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextInput input::placeholder, .stNumberInput input::placeholder {
    color: #9999AA !important;
}

/* ── Number input stepper buttons ── */
[data-testid="stNumberInput"] button {
    background: #EEECEA !important;
    border: 1px solid #D0CEC8 !important;
    color: #444 !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    border-radius: 8px !important;
    border: 1.5px solid #C8C6C0 !important;
    background: #F7F6F2 !important;
    color: #1A1A2E !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: #6366F1 !important;
    background: #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem !important;
}
.stSlider [data-testid="stThumbValue"] {
    background: #1A1A2E !important;
    color: white !important;
    border-radius: 6px !important;
    font-size: 0.75rem !important;
}

/* ── Form container — light gray so inputs (F7F6F2) contrast against it ── */
[data-testid="stForm"] {
    background: #EEECEA !important;
    border: 1.5px solid #D0CEC8 !important;
    border-radius: 14px !important;
    padding: 1.75rem !important;
}

/* ── Section divider inside form ── */
.form-section {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #7777AA;
    border-bottom: 1.5px solid #D0CEC8;
    padding-bottom: 6px;
    margin: 1.25rem 0 0.75rem;
}
.form-section:first-child { margin-top: 0; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #FFFFFF !important;
    border: 1px solid #E8E6E0 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #1A1A2E !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid #E8E6E0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed #D0CEC8 !important;
    border-radius: 12px !important;
    background: #FAFAF8 !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #6366F1 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    gap: 8px !important;
    border-bottom: 1.5px solid #E8E6E0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #888 !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #1A1A2E !important;
    border-bottom: 2px solid #1A1A2E !important;
}

/* ── Divider ── */
hr { border-color: #E8E6E0 !important; margin: 1.5rem 0 !important; }

/* ── Prediction result cards ── */
.pred-card {
    background: #FFFFFF;
    border: 1px solid #E8E6E0;
    border-radius: 14px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}
.pred-card-graduate { border-left: 4px solid #10B981; }
.pred-card-dropout  { border-left: 4px solid #EF4444; }
.pred-card-enrolled { border-left: 4px solid #F59E0B; }

.pill-graduate { background:#D1FAE5; color:#065F46; padding:4px 12px; border-radius:100px; font-size:0.8rem; font-weight:600; }
.pill-dropout  { background:#FEE2E2; color:#991B1B; padding:4px 12px; border-radius:100px; font-size:0.8rem; font-weight:600; }
.pill-enrolled { background:#FEF3C7; color:#92400E; padding:4px 12px; border-radius:100px; font-size:0.8rem; font-weight:600; }

/* ── Section card ── */
.section-card {
    background: #FFFFFF;
    border: 1px solid #E8E6E0;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Code blocks ── */
.stCodeBlock {
    border-radius: 10px !important;
    font-size: 0.8rem !important;
}

/* ── Spinner ── */
.stSpinner > div > div {
    border-top-color: #1A1A2E !important;
}

/* ── Column labels ── */
.col-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─── HEADER BANNER ───────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-header">
    <div>
        <p class="edu-title">EduPredict</p>
        <p class="edu-subtitle">Student Status Prediction · Random Forest & Decision Tree</p>
    </div>
    <span class="edu-badge">PROTOTYPE v1.0</span>
</div>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('student_analysis/random_forest_model.pkl')
        dt_model = joblib.load('student_analysis/decision_tree_model.pkl')
        if not hasattr(rf_model, 'feature_names_in_') or not hasattr(dt_model, 'feature_names_in_'):
            st.error("❌ Model tidak memiliki atribut 'feature_names_in_'.")
            st.stop()
        return rf_model, dt_model
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan! {str(e)}")
        return None, None
    except Exception as e:
        st.error(f"❌ Error memuat model: {str(e)}")
        return None, None

def get_status_mapping():
    return {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}

def preprocess_data(df_raw, scaler_path='submission/scaler.pkl', contains_status_col=True):
    df_clean = df_raw.copy()
    if contains_status_col and 'Status' in df_clean.columns:
        df_clean['Status_Original'] = df_clean['Status']
        df_clean = df_clean.dropna().reset_index(drop=True)
    elif not contains_status_col:
        model_features = rf_model.feature_names_in_ if rf_model else []
        df_clean = df_clean.dropna(subset=[c for c in model_features if c in df_clean.columns]).reset_index(drop=True)
    capping_features = [
        'Age_at_enrollment','Admission_grade','Curricular_units_1st_sem_grade',
        'Previous_qualification_grade','Course','Curricular_units_2nd_sem_grade'
    ]
    temp = df_clean.copy()
    existing = [c for c in capping_features if c in temp.columns]
    if existing:
        for col in existing:
            Q1, Q3 = temp[col].quantile(0.25), temp[col].quantile(0.75)
            IQR = Q3 - Q1
            temp[col] = np.clip(temp[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
        scaler = None
        try:
            scaler = joblib.load(scaler_path)
            cols_to_scale = [f for f in existing if f in scaler.feature_names_in_]
            temp[cols_to_scale] = scaler.transform(temp[cols_to_scale])
        except FileNotFoundError:
            scaler = StandardScaler()
            temp[existing] = scaler.fit_transform(temp[existing])
        except Exception:
            scaler = StandardScaler()
            temp[existing] = scaler.fit_transform(temp[existing])
        for col in existing:
            df_clean[col] = temp[col]
    return df_clean, scaler if 'scaler' in dir() else None, existing

def make_predictions(df_processed, rf_model, dt_model, include_real_status=False):
    try:
        model_features = rf_model.feature_names_in_
        missing = [f for f in model_features if f not in df_processed.columns]
        if missing:
            st.error(f"❌ Fitur yang hilang: {', '.join(missing)}")
            return None, None
        X = df_processed[model_features]
        status_col = df_processed['Status_Original'].values if (include_real_status and 'Status_Original' in df_processed.columns) else ['N/A'] * len(df_processed)
        rf_pred = rf_model.predict(X)
        dt_pred = dt_model.predict(X)
        data = {'ID': range(1, len(df_processed)+1)}
        if include_real_status:
            data['Status_Asli'] = status_col
        data['Random_Forest_Prediction'] = rf_pred
        data['Decision_Tree_Prediction'] = dt_pred
        results = pd.DataFrame(data)
        mapping = get_status_mapping()
        for col in ['Random_Forest_Prediction','Decision_Tree_Prediction']:
            if results[col].dtype in ['int64','float64']:
                results[col] = results[col].map(mapping)
        return results, X
    except Exception as e:
        st.error(f"Error prediksi: {str(e)}")
        return None, None

def highlight_predictions(row):
    colors = []
    for col_name in row.index:
        if col_name in ['Random_Forest_Prediction','Decision_Tree_Prediction','Status_Asli']:
            val = row[col_name]
            if val == 'Graduate':   colors.append('background-color:#D1FAE5;color:#065F46;font-weight:600')
            elif val == 'Dropout':  colors.append('background-color:#FEE2E2;color:#991B1B;font-weight:600')
            elif val == 'Enrolled': colors.append('background-color:#FEF3C7;color:#92400E;font-weight:600')
            else:                   colors.append('')
        else:
            colors.append('')
    return colors

def render_prediction_cards(rf_val, dt_val):
    def card_class(v):
        return 'graduate' if v == 'Graduate' else 'dropout' if v == 'Dropout' else 'enrolled'
    def pill(v):
        css = card_class(v)
        return f'<span class="pill-{css}">{v}</span>'
    def icon(v):
        return '🎓' if v=='Graduate' else '⚠️' if v=='Dropout' else '📚'

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown(f"""
        <div class="pred-card pred-card-{card_class(rf_val)}">
            <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#888;text-transform:uppercase;margin-bottom:8px;">Random Forest</div>
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:2rem;">{icon(rf_val)}</span>
                <div>
                    <div style="font-size:1.4rem;font-weight:700;color:#1A1A2E;">{rf_val}</div>
                    {pill(rf_val)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="pred-card pred-card-{card_class(dt_val)}">
            <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;color:#888;text-transform:uppercase;margin-bottom:8px;">Decision Tree</div>
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:2rem;">{icon(dt_val)}</span>
                <div>
                    <div style="font-size:1.4rem;font-weight:700;color:#1A1A2E;">{dt_val}</div>
                    {pill(dt_val)}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if rf_val == dt_val:
        st.success("✅ Kedua model **sepakat** dalam prediksi ini.")
    else:
        st.warning("⚡ Model memberikan **prediksi berbeda**. Pertimbangkan untuk meninjau ulang nilai input.")

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem;">
        <div style="font-size:1.1rem;font-weight:600;color:white;letter-spacing:-0.01em;">⚙️ Panel Kontrol</div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.4);margin-top:4px;">Pilih mode prediksi</div>
    </div>
    """, unsafe_allow_html=True)

    prediction_mode = st.radio(
        "Mode Prediksi:",
        ["✏️  Input Manual (1 Siswa)",
         "📤  Unggah CSV (Batch)"],
        label_visibility="collapsed"
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.7rem;color:rgba(255,255,255,0.3);line-height:1.6;">
        <b style="color:rgba(255,255,255,0.5);">Model</b><br>
        Random Forest<br>Decision Tree<br><br>
        <b style="color:rgba(255,255,255,0.5);">Output</b><br>
        Graduate · Dropout · Enrolled
    </div>
    """, unsafe_allow_html=True)

# ─── LOAD MODELS ──────────────────────────────────────────────────────────────
rf_model, dt_model = load_models()
if rf_model is None or dt_model is None:
    st.stop()
model_expected_features = list(rf_model.feature_names_in_)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 · INPUT MANUAL
# ══════════════════════════════════════════════════════════════════════════════
if "Input Manual" in prediction_mode:
    st.markdown("## ✏️  Prediksi Satu Siswa")
    st.info("Isi formulir di bawah ini dengan data siswa, lalu klik **Dapatkan Prediksi**.", icon="ℹ️")

    input_data = {}
    MARITAL_STATUS_OPTIONS  = {1:'Single',2:'Married',3:'Widower',4:'Divorced',5:'Facto union',6:'Legally separated'}
    DAYTIME_EVENING_OPTIONS = {1:'Daytime (Pagi)',0:'Evening (Malam)'}
    YES_NO_OPTIONS          = {1:'Ya',0:'Tidak'}
    GENDER_OPTIONS          = {1:'Laki-laki',0:'Perempuan'}

    def cat_input(label, feature, opts, default_key, help_text=""):
        disp = list(opts.values())
        sel = st.selectbox(label, disp, index=disp.index(opts[default_key]),
                           help=help_text, key=f"manual_{feature}")
        return {v:k for k,v in opts.items()}[sel]

    with st.form("manual_form"):

        # ── SEKSI 1: Data Pribadi ──────────────────────────────────────────
        st.markdown('<div class="form-section">👤 Data Pribadi Siswa</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            input_data['Marital_status'] = cat_input(
                "Status Pernikahan", 'Marital_status', MARITAL_STATUS_OPTIONS, 1,
                "Status pernikahan siswa saat mendaftar")
        with c2:
            input_data['Gender'] = cat_input(
                "Jenis Kelamin", 'Gender', GENDER_OPTIONS, 1, "")
        with c3:
            input_data['Age_at_enrollment'] = st.number_input(
                "Usia saat Mendaftar", min_value=15, max_value=80, value=20,
                help="Usia siswa dalam tahun saat pertama kali mendaftar",
                key="m_Age_at_enrollment")
        c4, c5, c6 = st.columns(3, gap="medium")
        with c4:
            input_data['Nacionality'] = st.number_input(
                "Kewarganegaraan (Kode)", min_value=1, max_value=109, value=1, step=1,
                help="Kode negara: 1=Portugal, 2=Jerman, dst. Lihat dokumentasi untuk daftar lengkap.",
                key="m_Nacionality")
        with c5:
            input_data['International'] = cat_input(
                "Mahasiswa Internasional?", 'International', YES_NO_OPTIONS, 0,
                "Apakah siswa berasal dari luar negeri?")
        with c6:
            input_data['Displaced'] = cat_input(
                "Pindahan (Displaced)?", 'Displaced', YES_NO_OPTIONS, 0,
                "Apakah siswa merupakan siswa pindahan dari daerah lain?")

        # ── SEKSI 2: Latar Belakang Pendidikan ───────────────────────────
        st.markdown('<div class="form-section">🎒 Latar Belakang Pendidikan</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            input_data['Previous_qualification'] = st.number_input(
                "Kualifikasi Sebelumnya (Kode)", min_value=1, max_value=43, value=1, step=1,
                help="Jenjang pendidikan sebelumnya: 1=SMA, 2=Sarjana, 3=Magister, dst.",
                key="m_Previous_qualification")
        with c2:
            pq_key = next((f for f in model_expected_features if 'Previous_qualification' in f and 'grade' in f.lower()), None)
            if pq_key:
                input_data[pq_key] = st.number_input(
                    "Nilai Kualifikasi Sebelumnya", min_value=0.0, max_value=200.0, value=120.0, step=0.1,
                    help="Nilai rata-rata pada jenjang pendidikan sebelumnya (skala 0–200)",
                    key=f"m_{pq_key}")
        with c3:
            input_data['Admission_grade'] = st.number_input(
                "Nilai Masuk (Admission Grade)", min_value=0.0, max_value=200.0, value=120.0, step=0.1,
                help="Nilai yang diperoleh saat seleksi masuk perguruan tinggi (skala 0–200)",
                key="m_Admission_grade")
        c4, c5 = st.columns([1,2], gap="medium")
        with c4:
            input_data['Educational_special_needs'] = cat_input(
                "Kebutuhan Pendidikan Khusus?", 'Educational_special_needs', YES_NO_OPTIONS, 0,
                "Apakah siswa memiliki kebutuhan pendidikan khusus (difabel, dsb.)?")
        with c5:
            input_data['Mother\'s_qualification'] = st.number_input(
                "Pendidikan Terakhir Ibu (Kode)", min_value=1, max_value=44, value=1, step=1,
                help="Kode tingkat pendidikan ibu: 1=SD, 2=SMP, 3=SMA, 4=Sarjana, dst.",
                key="m_Mother_qual")
            input_data['Father\'s_qualification'] = st.number_input(
                "Pendidikan Terakhir Ayah (Kode)", min_value=1, max_value=44, value=1, step=1,
                help="Kode tingkat pendidikan ayah: 1=SD, 2=SMP, 3=SMA, 4=Sarjana, dst.",
                key="m_Father_qual")

        # ── SEKSI 3: Pekerjaan Orang Tua ──────────────────────────────────
        st.markdown('<div class="form-section">💼 Pekerjaan Orang Tua</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            input_data['Mother\'s_occupation'] = st.number_input(
                "Pekerjaan Ibu (Kode)", min_value=0, max_value=194, value=99, step=1,
                help="Kode pekerjaan ibu. 99 = tidak diketahui/tidak bekerja. Lihat dokumentasi untuk kode lainnya.",
                key="m_Mother_occ")
        with c2:
            input_data['Father\'s_occupation'] = st.number_input(
                "Pekerjaan Ayah (Kode)", min_value=0, max_value=195, value=99, step=1,
                help="Kode pekerjaan ayah. 99 = tidak diketahui/tidak bekerja. Lihat dokumentasi untuk kode lainnya.",
                key="m_Father_occ")

        # ── SEKSI 4: Data Pendaftaran & Keuangan ──────────────────────────
        st.markdown('<div class="form-section">📋 Pendaftaran & Keuangan</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            input_data['Course'] = st.number_input(
                "Program Studi (Kode)", min_value=33, max_value=9991, value=9119, step=1,
                help="Kode program studi yang diambil. Contoh: 9119=Agronomi, 9254=Desain Komunikasi, dst.",
                key="m_Course")
        with c2:
            input_data['Application_mode'] = st.number_input(
                "Jalur Pendaftaran (Kode)", min_value=1, max_value=57, value=1, step=1,
                help="Jalur penerimaan mahasiswa: 1=Umum (1st fase), 5=Khusus (Azores), 7=Pemegang Gelar Lain, dst.",
                key="m_Application_mode")
        with c3:
            input_data['Application_order'] = st.number_input(
                "Urutan Pilihan Prodi", min_value=0, max_value=9, value=0, step=1,
                help="Urutan pilihan program studi ini saat mendaftar (0=pilihan pertama, 9=pilihan kesembilan)",
                key="m_Application_order")
        c4, c5, c6 = st.columns(3, gap="medium")
        with c4:
            input_data['Daytime/evening_attendance'] = cat_input(
                "Waktu Kuliah", 'Daytime/evening_attendance', DAYTIME_EVENING_OPTIONS, 1,
                "Apakah siswa mengikuti kuliah pagi atau malam hari?")
        with c5:
            input_data['Tuition_fees_up_to_date'] = cat_input(
                "SPP Lunas?", 'Tuition_fees_up_to_date', YES_NO_OPTIONS, 1,
                "Apakah pembayaran SPP siswa saat ini sudah lunas / up-to-date?")
        with c6:
            input_data['Debtor'] = cat_input(
                "Memiliki Tunggakan?", 'Debtor', YES_NO_OPTIONS, 0,
                "Apakah siswa memiliki tunggakan pembayaran kepada institusi?")
        c7, c8 = st.columns(2, gap="medium")
        with c7:
            input_data['Scholarship_holder'] = cat_input(
                "Penerima Beasiswa?", 'Scholarship_holder', YES_NO_OPTIONS, 0,
                "Apakah siswa saat ini menerima beasiswa?")

        # ── SEKSI 5: Akademik Semester 1 ─────────────────────────────────
        st.markdown('<div class="form-section">📚 Akademik — Semester 1</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
        sem1_fields = [
            ('Curricular_units_1st_sem_credited',    "SKS Diakui",     0, 30, 0,   "Jumlah SKS yang diakui/dibebaskan dari jenjang sebelumnya"),
            ('Curricular_units_1st_sem_enrolled',    "Mata Kuliah Diambil", 0, 30, 6, "Jumlah mata kuliah yang diambil semester 1"),
            ('Curricular_units_1st_sem_evaluations', "Jumlah Evaluasi",0, 30, 6,   "Jumlah evaluasi/ujian yang diikuti semester 1"),
            ('Curricular_units_1st_sem_approved',    "Mata Kuliah Lulus",0,30, 5,  "Jumlah mata kuliah yang berhasil lulus semester 1"),
        ]
        for col_widget, (feat, lbl, mn, mx, dv, hlp) in zip([c1,c2,c3,c4], sem1_fields):
            with col_widget:
                input_data[feat] = st.number_input(lbl, min_value=mn, max_value=mx, value=dv, step=1, help=hlp, key=f"m_{feat}")
        with c5:
            g1_key = next((f for f in model_expected_features if '1st_sem_grade' in f), None)
            if g1_key:
                input_data[g1_key] = st.number_input(
                    "Nilai Rata-rata", min_value=0.0, max_value=20.0, value=12.0, step=0.1,
                    help="Nilai rata-rata semester 1 (skala 0–20)", key=f"m_{g1_key}")

        # ── SEKSI 6: Akademik Semester 2 ─────────────────────────────────
        st.markdown('<div class="form-section">📖 Akademik — Semester 2</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            g2_key = next((f for f in model_expected_features if '2nd_sem_grade' in f), None)
            if g2_key:
                input_data[g2_key] = st.number_input(
                    "Nilai Rata-rata Semester 2", min_value=0.0, max_value=20.0, value=12.0, step=0.1,
                    help="Nilai rata-rata semester 2 (skala 0–20)", key=f"m_{g2_key}")

        # ── Fitur lain yang belum tercakup ────────────────────────────────
        covered = {
            'Marital_status','Gender','Age_at_enrollment','Nacionality','International','Displaced',
            'Previous_qualification','Admission_grade','Educational_special_needs',
            "Mother's_qualification","Father's_qualification","Mother's_occupation","Father's_occupation",
            'Course','Application_mode','Application_order','Daytime/evening_attendance',
            'Tuition_fees_up_to_date','Debtor','Scholarship_holder',
            'Curricular_units_1st_sem_credited','Curricular_units_1st_sem_enrolled',
            'Curricular_units_1st_sem_evaluations','Curricular_units_1st_sem_approved',
        }
        if pq_key: covered.add(pq_key)
        if g1_key: covered.add(g1_key)
        if g2_key: covered.add(g2_key)

        remaining = [f for f in model_expected_features if f not in covered]
        if remaining:
            st.markdown('<div class="form-section">⚙️ Fitur Tambahan</div>', unsafe_allow_html=True)
            cols_r = st.columns(3, gap="medium")
            for i, feat in enumerate(remaining):
                with cols_r[i % 3]:
                    input_data[feat] = st.number_input(
                        feat.replace('_',' ').title(), value=0.0, key=f"m_{feat}")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮  Dapatkan Prediksi", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_data])
        with st.spinner("Memproses..."):
            df_proc, _, _ = preprocess_data(input_df, contains_status_col=False)
        if df_proc.empty:
            st.error("❌ Preprocessing menghasilkan data kosong.")
        else:
            pred_res, _ = make_predictions(df_proc, rf_model, dt_model, include_real_status=False)
            if pred_res is not None:
                st.markdown("### 🎯 Hasil Prediksi")
                render_prediction_cards(
                    pred_res['Random_Forest_Prediction'].iloc[0],
                    pred_res['Decision_Tree_Prediction'].iloc[0]
                )
                with st.expander("🔎  Lihat Data yang Diproses"):
                    st.dataframe(df_proc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 · BATCH CSV
# ══════════════════════════════════════════════════════════════════════════════
elif "Unggah CSV" in prediction_mode:
    st.markdown("## 📤  Prediksi Batch via CSV")
    st.info("Unggah file CSV berisi fitur siswa **tanpa** kolom 'Status' untuk prediksi massal.", icon="ℹ️")

    uploaded = st.file_uploader("Pilih file CSV:", type=['csv'])

    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded, low_memory=False)
            st.success(f"✅ File berhasil diunggah — **{len(df_batch):,}** baris.")

            if 'Status' in df_batch.columns:
                st.warning("⚠️ Kolom 'Status' ditemukan dan akan diabaikan.", icon="⚠️")
                df_batch = df_batch.drop(columns=['Status'])

            with st.spinner("Memproses..."):
                df_proc, _, _ = preprocess_data(df_batch, contains_status_col=False)

            if df_proc.empty:
                st.error("❌ Data kosong setelah preprocessing.")
            else:
                res, _ = make_predictions(df_proc, rf_model, dt_model, include_real_status=False)

            if res is not None:
                st.success(f"✅ Prediksi selesai untuk **{len(res):,}** siswa.")

                final = df_batch.copy()
                final['Random_Forest_Prediction'] = res['Random_Forest_Prediction']
                final['Decision_Tree_Prediction']  = res['Decision_Tree_Prediction']

                st.dataframe(final.style.apply(highlight_predictions, axis=1), use_container_width=True)

                st.markdown("#### Ringkasan Prediksi")
                c1, c2 = st.columns(2, gap="medium")
                for col_i, (title, key) in enumerate(zip(['Random Forest','Decision Tree'],['Random_Forest_Prediction','Decision_Tree_Prediction'])):
                    with [c1,c2][col_i]:
                        st.markdown(f"<div class='col-label'>{title}</div>", unsafe_allow_html=True)
                        cnt = res[key].value_counts().reindex(['Graduate','Dropout','Enrolled'], fill_value=0)
                        for lbl, val in cnt.items():
                            c = 'graduate' if lbl=='Graduate' else 'dropout' if lbl=='Dropout' else 'enrolled'
                            st.markdown(f'<span class="pill-{c}">{lbl}: {val}</span><br>', unsafe_allow_html=True)

                buf = io.StringIO()
                final.to_csv(buf, index=False)
                st.download_button(
                    "📥  Unduh Hasil (CSV)",
                    data=buf.getvalue(),
                    file_name=f"batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error memproses file: {str(e)}")

    st.markdown("---")
    st.markdown("#### 📝 Format Data yang Diharapkan")

    with st.expander("Lihat spesifikasi fitur lengkap"):
        st.markdown("""
**Fitur yang diproses (capping + standarisasi):**
`Age_at_enrollment` · `Admission_grade` · `Curricular_units_1st_sem_grade` · `Previous_qualification_grade` · `Course` · `Curricular_units_2nd_sem_grade`

**Fitur kategorikal (nilai numerik):**

| Fitur | Nilai |
|---|---|
| Marital_status | 1=Single, 2=Married, 3=Widower, 4=Divorced, 5=Facto union, 6=Legally separated |
| Daytime/evening_attendance | 0=Evening, 1=Daytime |
| Displaced / Educational_special_needs / Debtor / Tuition_fees_up_to_date / Scholarship_holder / International | 0=No, 1=Yes |
| Gender | 0=Female, 1=Male |

**Output prediksi:** `Graduate` · `Dropout` · `Enrolled`
        """)
        if model_expected_features:
            st.markdown("**Semua fitur yang diharapkan model:**")
            st.code(", ".join(model_expected_features))

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#bbb;font-size:0.75rem;letter-spacing:0.05em;">
    EduPredict · Prototype · Random Forest &amp; Decision Tree
</div>
""", unsafe_allow_html=True)
