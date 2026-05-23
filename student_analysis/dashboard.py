import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Analisis Dropout Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_MAP = {"Dropout": "#f7695b", "Enrolled": "#46d9ac", "Graduate": "#5b6af7"}

# ── Data Loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    candidates = ["data.csv", os.path.join(script_dir, "data.csv")]
    try:
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                candidates.insert(0, f)
    except:
        pass
    for p in candidates:
        try:
            if os.path.exists(p):
                return pd.read_csv(p), p
        except:
            continue
    return None, None


@st.cache_data
def sample_data():
    np.random.seed(42)
    n = 1000
    ages = np.random.normal(20, 2, n).clip(17, 35).astype(int)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.45, 0.55])
    scholarships = np.random.binomial(1, np.where(ages < 22, 0.4, 0.2), n)
    base = 12 + scholarships * 2 + np.random.normal(0, 2, n)
    grade1 = base.clip(0, 20)
    grade2 = (base + np.random.normal(0, 1, n)).clip(0, 20)
    drop_p = (0.1 + 0.3*(grade1 < 10) + 0.2*(scholarships == 0) + 0.1*(ages > 25)).clip(0, 1)
    statuses = ['Dropout' if r < p else ('Enrolled' if r < p+0.4 else 'Graduate')
                for r, p in zip(np.random.random(n), drop_p)]
    return pd.DataFrame({
        'Age_at_enrollment': ages, 'Gender': genders,
        'Scholarship_holder': scholarships,
        'Tuition_fees_up_to_date': np.random.choice([0, 1], n, p=[0.15, 0.85]),
        'Curricular_units_1st_sem_grade': grade1,
        'Daytime_evening_attendance': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'Fathers_qualification': np.random.choice([1,2,3,4,5,19,34], n),
        'Mothers_qualification': np.random.choice([1,2,3,4,5,19,34], n),
        'Target': statuses,
    })


def stacked_pct_bar(df, group_col, title, x_label=""):
    grp = df.groupby([group_col, 'Target']).size().unstack(fill_value=0)
    pct = (grp.div(grp.sum(axis=1), axis=0) * 100).reset_index()
    cols = [c for c in ['Dropout', 'Enrolled', 'Graduate'] if c in pct.columns]
    fig = px.bar(pct, x=group_col, y=cols, title=title,
                 color_discrete_map=COLOR_MAP, barmode='stack',
                 labels={'value': 'Persentase (%)', group_col: x_label})
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af', title_font_color='#e8eaf0',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font_color='#9ca3af', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=40, b=0), height=340,
        xaxis=dict(gridcolor='#1e2330', linecolor='#252a38'),
        yaxis=dict(gridcolor='#1e2330', linecolor='#252a38'),
    )
    fig.update_traces(marker_line_width=0)
    return fig


def donut_chart(df):
    vc = df['Target'].value_counts().reset_index()
    vc.columns = ['Status', 'Jumlah']
    fig = px.pie(vc, values='Jumlah', names='Status', hole=0.55,
                 color='Status', color_discrete_map=COLOR_MAP,
                 title='Distribusi Status Siswa')
    fig.update_traces(textinfo='percent+label', textfont_color='#e8eaf0')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af', title_font_color='#e8eaf0',
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5,
                    font_color='#9ca3af', bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=40, b=0), height=340,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Analytics")
    st.markdown("---")

    df, path = load_csv()
    if df is not None:
        st.success(f"✅ File: `{os.path.basename(path)}`")
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ File berhasil diupload")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("📊 Menggunakan data sampel")
            df = sample_data()

    # Normalize
    if df is not None:
        if 'Status' in df.columns:
            df = df.rename(columns={'Status': 'Target'})
        if 'Age' not in df.columns and 'Age_at_enrollment' in df.columns:
            df['Age'] = df['Age_at_enrollment']

    if df is not None and 'Target' in df.columns:
        st.markdown("---")
        st.markdown("**Ringkasan Dataset**")
        st.caption(f"{df.shape[0]:,} baris · {df.shape[1]} kolom")

        for status, color in COLOR_MAP.items():
            n = len(df[df['Target'] == status])
            pct = n / len(df) * 100
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:5px 8px;'
                f'background:#1a1e2a;border-radius:6px;margin:3px 0;border-left:3px solid {color};">'
                f'<span style="font-size:0.8rem;color:#9ca3af;">{status}</span>'
                f'<span style="font-size:0.8rem;color:{color};font-weight:600;">{n:,} ({pct:.1f}%)</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        with st.expander(f"📋 {df.shape[1]} Kolom"):
            for i, col in enumerate(df.columns, 1):
                st.caption(f"{i}. `{col}` — *{df[col].dtype}*")


# ── Guard ──────────────────────────────────────────────────────────────────────
if df is None or 'Target' not in df.columns:
    st.error("Tidak bisa memuat data. Pastikan file CSV memiliki kolom `Target` atau `Status`.")
    st.stop()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎓 Dashboard Analisis Dropout Siswa")
st.caption("Analisis komprehensif faktor-faktor yang mempengaruhi dropout siswa")
st.divider()

# ── KPI Metrics ────────────────────────────────────────────────────────────────
total      = len(df)
dropout_n  = len(df[df['Target'] == 'Dropout'])
enrolled_n = len(df[df['Target'] == 'Enrolled'])
grad_n     = len(df[df['Target'] == 'Graduate'])
dr = dropout_n / total * 100
rr = (total - dropout_n) / total * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👥 Total Siswa",    f"{total:,}")
m2.metric("❌ Dropout",        f"{dropout_n:,}",  f"{dr:.1f}% dari total")
m3.metric("📚 Enrolled",       f"{enrolled_n:,}")
m4.metric("🎓 Graduate",       f"{grad_n:,}")
m5.metric("📈 Retention Rate", f"{rr:.1f}%")

st.divider()

# ── Overview ───────────────────────────────────────────────────────────────────
st.subheader("📊 Overview")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(donut_chart(df), use_container_width=True)

with col2:
    if 'Age' in df.columns:
        d = df.copy()
        d['Kelompok Usia'] = pd.cut(d['Age'], bins=[16,20,23,26,100],
                                    labels=['17–20', '21–23', '24–26', '27+'])
        st.plotly_chart(
            stacked_pct_bar(d, 'Kelompok Usia', 'Status Siswa per Kelompok Usia (%)'),
            use_container_width=True
        )
    else:
        st.info("Data usia tidak tersedia")

st.divider()

# ── Demografi & Keuangan ───────────────────────────────────────────────────────
st.subheader("👥 Demografi & Keuangan")
c1, c2, c3 = st.columns(3)

with c1:
    if 'Gender' in df.columns:
        st.plotly_chart(
            stacked_pct_bar(df, 'Gender', 'Status per Gender (%)'),
            use_container_width=True
        )

with c2:
    if 'Scholarship_holder' in df.columns:
        d = df.copy()
        d['Beasiswa'] = d['Scholarship_holder'].map({1: 'Ada Beasiswa', 0: 'Tidak Ada'})
        st.plotly_chart(
            stacked_pct_bar(d, 'Beasiswa', 'Pengaruh Beasiswa (%)'),
            use_container_width=True
        )

with c3:
    if 'Tuition_fees_up_to_date' in df.columns:
        d = df.copy()
        d['Status SPP'] = d['Tuition_fees_up_to_date'].map({1: 'Lunas', 0: 'Menunggak'})
        st.plotly_chart(
            stacked_pct_bar(d, 'Status SPP', 'Status Pembayaran SPP (%)'),
            use_container_width=True
        )

st.divider()

# ── Akademik ───────────────────────────────────────────────────────────────────
st.subheader("📚 Performa Akademik")
grade_col = next((c for c in df.columns if 'grade' in c.lower()), None)
attend_col = 'Daytime_evening_attendance' if 'Daytime_evening_attendance' in df.columns else None

g1, g2 = st.columns(2)
with g1:
    if grade_col:
        d = df.copy()
        d['Kategori Nilai'] = pd.cut(d[grade_col], bins=[-0.01, 10, 14, 17, 20],
                                     labels=['0–10 Rendah', '10–14 Sedang', '14–17 Baik', '17–20 Sangat Baik'])
        st.plotly_chart(
            stacked_pct_bar(d, 'Kategori Nilai', 'Prestasi Akademik vs Status (%)'),
            use_container_width=True
        )
    else:
        st.info("Data nilai tidak tersedia")

with g2:
    if attend_col:
        d = df.copy()
        d['Sesi'] = d[attend_col].map({1: 'Pagi', 0: 'Malam'})
        st.plotly_chart(
            stacked_pct_bar(d, 'Sesi', 'Waktu Kehadiran vs Status (%)'),
            use_container_width=True
        )
    else:
        st.info("Data kehadiran tidak tersedia")

st.divider()

# ── Latar Belakang Keluarga ────────────────────────────────────────────────────
parent_col = next((c for c in df.columns if 'qualification' in c.lower()), None)
if parent_col:
    st.subheader("👨‍👩‍👧 Pendidikan Orang Tua")
    qual_map = {1:'SD', 2:'SMP', 3:'SMA', 4:'Diploma', 5:'S1', 19:'S2', 34:'S3', 35:'Lainnya'}
    d = df.copy()
    d['Pendidikan OT'] = d[parent_col].map(qual_map).fillna('Lainnya')
    st.plotly_chart(
        stacked_pct_bar(d, 'Pendidikan OT', 'Tingkat Pendidikan Orang Tua vs Status Siswa (%)'),
        use_container_width=True
    )
    st.divider()

# ── Key Insights ───────────────────────────────────────────────────────────────
st.subheader("💡 Key Insights")

insights = [(
    "info",
    f"**Tingkat Dropout: {dr:.1f}%** — {dropout_n:,} siswa perlu strategi intervensi segera."
)]

if 'Age' in df.columns:
    d_age = df[df['Target']=='Dropout']['Age'].mean()
    g_age = df[df['Target']=='Graduate']['Age'].mean()
    arah = "lebih tua" if d_age > g_age else "lebih muda"
    insights.append(("warning",
        f"Siswa dropout cenderung **{arah}**: rata-rata **{d_age:.1f} tahun** "
        f"vs graduate **{g_age:.1f} tahun**."))

if 'Scholarship_holder' in df.columns:
    r_s = len(df[(df['Target']=='Dropout') & (df['Scholarship_holder']==1)]) / max(1, len(df[df['Scholarship_holder']==1])) * 100
    r_n = len(df[(df['Target']=='Dropout') & (df['Scholarship_holder']==0)]) / max(1, len(df[df['Scholarship_holder']==0])) * 100
    if r_s < r_n:
        insights.append(("success",
            f"**Beasiswa efektif**: dropout penerima beasiswa **{r_s:.1f}%** vs non-penerima **{r_n:.1f}%**."))
    else:
        insights.append(("warning",
            f"**Beasiswa perlu dievaluasi**: dropout penerima beasiswa **{r_s:.1f}%** vs non-penerima **{r_n:.1f}%**."))

if grade_col:
    avg_d = df[df['Target']=='Dropout'][grade_col].mean()
    avg_g = df[df['Target']=='Graduate'][grade_col].mean()
    insights.append(("success",
        f"**Nilai akademik berpengaruh besar**: rata-rata graduate **{avg_g:.1f}** vs dropout **{avg_d:.1f}** "
        f"(selisih {avg_g - avg_d:.1f} poin)."))

for kind, text in insights:
    if kind == "info":
        st.info(text)
    elif kind == "warning":
        st.warning(text)
    else:
        st.success(text)

st.divider()

# ── Rekomendasi ────────────────────────────────────────────────────────────────
st.subheader("🎯 Rekomendasi Aksi")

recs = [
    ("🎓", "Program Dukungan Akademik",
     "Fokus pada siswa dengan nilai rendah di semester pertama. Tugaskan tutor sebelum semester kedua dimulai."),
    ("💰", "Perluasan Bantuan Keuangan",
     "Perluas program beasiswa untuk kelompok berisiko tinggi — khususnya siswa lebih tua tanpa beasiswa."),
    ("👥", "Program Mentoring",
     "Bangun jaringan mentoring terstruktur yang memasangkan siswa berisiko dengan mahasiswa senior atau alumni."),
    ("📊", "Sistem Early Warning",
     "Implementasikan skor risiko otomatis yang memantau IPK, status SPP, dan kehadiran secara real-time."),
    ("🏫", "Keterlibatan Orang Tua",
     "Adakan workshop untuk orang tua tentang pentingnya dukungan akademik — terutama untuk keluarga pertama yang kuliah."),
]

for i, (icon, title, desc) in enumerate(recs, 1):
    with st.expander(f"{icon} {i}. {title}"):
        st.write(desc)

st.divider()

# ── Data Preview ───────────────────────────────────────────────────────────────
st.subheader("📋 Data Preview")
if st.checkbox("Tampilkan 100 baris pertama"):
    st.dataframe(df.head(100), use_container_width=True)
