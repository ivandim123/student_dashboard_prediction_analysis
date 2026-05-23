import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Analisis Dropout Siswa — Jaya Jaya Institut",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

COLOR_MAP = {"Dropout": "#f7695b", "Enrolled": "#46d9ac", "Graduate": "#5b6af7"}

# Mapping kode → label untuk kolom-kolom kategorikal dataset UCI
APPLICATION_MODE_MAP = {
    1: "1st phase - general",
    2: "Ordinance No. 612/93",
    5: "1st phase - special (Azores)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student",
    16: "1st phase - special (Madeira)",
    17: "2nd phase - general",
    18: "3rd phase - general",
    26: "Ordinance No. 533-A/99 (b2)",
    27: "Ordinance No. 533-A/99 (b3)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Tech. diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution (intl)",
}
QUALIFICATION_MAP = {
    1: "Secondary Education",
    2: "Bachelor's Degree",
    3: "Degree",
    4: "Master's",
    5: "Doctorate",
    6: "Frequency of Higher Ed",
    9: "12th Year (not completed)",
    10: "11th Year (not completed)",
    11: "7th Year (Old)",
    12: "Other - 11th Year",
    14: "10th Year",
    18: "General commerce course",
    19: "Basic Ed. 3rd Cycle (9th yr)",
    22: "Technical-professional course",
    26: "7th Year of schooling",
    27: "2nd Cycle of basic schooling",
    29: "9th Year (not completed)",
    30: "8th Year",
    34: "Unknown",
    35: "Can't read/write",
    36: "Can read, no 4th year",
    37: "Basic Ed. 1st Cycle (4th yr)",
    38: "Basic Ed. 2nd Cycle (6th yr)",
    39: "Technological course",
    40: "Higher Ed. (1st cycle)",
    41: "Specialized higher study",
    42: "Professional higher tech.",
    43: "Higher Ed. (2nd cycle)",
    44: "Higher Ed. (3rd cycle)",
}
MARITAL_MAP = {1:"Single", 2:"Married", 3:"Widower", 4:"Divorced", 5:"Facto union", 6:"Legally separated"}
ATTENDANCE_MAP = {1: "Pagi (Daytime)", 0: "Malam (Evening)"}
GENDER_MAP = {1: "Male", 0: "Female"}
SCHOLARSHIP_MAP = {1: "Ada Beasiswa", 0: "Tidak Ada"}
TUITION_MAP = {1: "Lunas", 0: "Menunggak"}
DISPLACED_MAP = {1: "Ya (Displaced)", 0: "Tidak"}
INTERNATIONAL_MAP = {1: "Internasional", 0: "Domestik"}
DEBTOR_MAP = {1: "Ya (Debtor)", 0: "Tidak"}


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
    scholarships = np.random.binomial(1, np.where(ages < 22, 0.4, 0.2), n)
    base = 12 + scholarships * 2 + np.random.normal(0, 2, n)
    grade1 = base.clip(0, 20)
    grade2 = (base + np.random.normal(0, 1, n)).clip(0, 20)
    drop_p = (0.1 + 0.3*(grade1 < 10) + 0.2*(scholarships == 0) + 0.1*(ages > 25)).clip(0, 1)
    statuses = ['Dropout' if r < p else ('Enrolled' if r < p+0.4 else 'Graduate')
                for r, p in zip(np.random.random(n), drop_p)]
    return pd.DataFrame({
        'Age_at_enrollment': ages,
        'Gender': np.random.choice([0, 1], n),
        'Scholarship_holder': scholarships,
        'Tuition_fees_up_to_date': np.random.choice([0, 1], n, p=[0.15, 0.85]),
        'Curricular_units_1st_sem_(grade)': grade1,
        'Curricular_units_2nd_sem_(grade)': grade2,
        'Daytime_evening_attendance': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        'Father\'s_qualification': np.random.choice([1,2,3,4,5,19,34,36], n),
        'Mother\'s_qualification': np.random.choice([1,2,3,4,5,19,34,36], n),
        'Application_mode': np.random.choice([1,2,17,18,39,42,43], n),
        'Marital_status': np.random.choice([1,2,3,4,5,6], n, p=[0.7,0.15,0.03,0.07,0.03,0.02]),
        'Displaced': np.random.choice([0, 1], n, p=[0.6, 0.4]),
        'International': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'Debtor': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'Target': statuses,
    })


# ── Chart helpers ──────────────────────────────────────────────────────────────
def stacked_pct_bar(df, group_col, title, height=340, sort_by_dropout=False):
    d = df[[group_col, 'Target']].dropna(subset=[group_col])
    grp = d.groupby([group_col, 'Target']).size().unstack(fill_value=0)
    pct = (grp.div(grp.sum(axis=1), axis=0) * 100).reset_index()
    if sort_by_dropout and 'Dropout' in pct.columns:
        pct = pct.sort_values('Dropout', ascending=False)
    cols = [c for c in ['Dropout', 'Enrolled', 'Graduate'] if c in pct.columns]
    fig = px.bar(pct, x=group_col, y=cols, title=title,
                 color_discrete_map=COLOR_MAP, barmode='stack',
                 labels={'value': 'Persentase (%)', group_col: '', 'variable': 'Status'})
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af', title_font_color='#e8eaf0', title_font_size=13,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font_color='#9ca3af', bgcolor='rgba(0,0,0,0)', title_text=''),
        margin=dict(l=0, r=0, t=44, b=0), height=height,
        xaxis=dict(gridcolor='#1e2330', linecolor='#252a38', tickfont_color='#6b7280'),
        yaxis=dict(gridcolor='#1e2330', linecolor='#252a38', tickfont_color='#6b7280'),
        bargap=0.25,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def grade_bar(df, grade_col, title):
    """Buat bins grade otomatis berdasarkan range data aktual (0–20 atau 0–200)."""
    d = df[[grade_col, 'Target']].dropna()
    vmin, vmax = d[grade_col].min(), d[grade_col].max()

    # Skala 0–20 (sistem Eropa/Portugal)
    if vmax <= 20:
        bins   = [-0.01, 10, 14, 17, 20.01]
        labels = ['0–10 (Rendah)', '10–14 (Cukup)', '14–17 (Baik)', '17–20 (Sangat Baik)']
    # Skala 0–200
    elif vmax <= 200:
        bins   = [-0.01, 100, 140, 170, 200.01]
        labels = ['0–100 (Rendah)', '100–140 (Cukup)', '140–170 (Baik)', '170–200 (Sangat Baik)']
    else:
        # Quartile-based fallback
        q1, q2, q3 = d[grade_col].quantile([0.25, 0.5, 0.75])
        bins   = [vmin - 0.01, q1, q2, q3, vmax + 0.01]
        labels = ['Q1 (Rendah)', 'Q2 (Cukup)', 'Q3 (Baik)', 'Q4 (Sangat Baik)']

    d2 = d.copy()
    d2['Kategori Nilai'] = pd.cut(d2[grade_col], bins=bins, labels=labels)
    d2 = d2.dropna(subset=['Kategori Nilai'])
    return stacked_pct_bar(d2, 'Kategori Nilai', title)


def donut_chart(df):
    vc = df['Target'].value_counts().reset_index()
    vc.columns = ['Status', 'Jumlah']
    fig = px.pie(vc, values='Jumlah', names='Status', hole=0.55,
                 color='Status', color_discrete_map=COLOR_MAP,
                 title='Distribusi Status Siswa')
    fig.update_traces(textinfo='percent+label', textfont_color='white',
                      textfont_size=12)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#9ca3af', title_font_color='#e8eaf0', title_font_size=13,
        showlegend=False,
        margin=dict(l=0, r=0, t=44, b=0), height=340,
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Jaya Jaya Institut")
    st.caption("Dashboard Analisis Dropout Siswa")
    st.divider()

    df, path = load_csv()
    if df is not None:
        st.success(f"✅ `{os.path.basename(path)}`")
    else:
        uploaded = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ File berhasil diupload")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("📊 Menggunakan data sampel")
            df = sample_data()

    # Normalize kolom
    if df is not None:
        if 'Status' in df.columns:
            df = df.rename(columns={'Status': 'Target'})
        if 'Age' not in df.columns and 'Age_at_enrollment' in df.columns:
            df['Age'] = df['Age_at_enrollment']

    if df is not None and 'Target' in df.columns:
        st.divider()
        st.markdown("**Ringkasan Dataset**")
        st.caption(f"{df.shape[0]:,} baris · {df.shape[1]} kolom")
        st.write("")

        total_s = len(df)
        for status, color in COLOR_MAP.items():
            n = len(df[df['Target'] == status])
            pct = n / total_s * 100
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:6px 10px;background:#1a1e2a;border-radius:6px;margin:3px 0;'
                f'border-left:3px solid {color};">'
                f'<span style="font-size:0.82rem;color:#9ca3af;">{status}</span>'
                f'<span style="font-size:0.82rem;color:{color};font-weight:600;">'
                f'{n:,} &nbsp;·&nbsp; {pct:.1f}%</span></div>',
                unsafe_allow_html=True
            )

        st.divider()
        with st.expander(f"📋 {df.shape[1]} Kolom Tersedia"):
            for i, col in enumerate(df.columns, 1):
                st.caption(f"{i}. `{col}` — *{df[col].dtype}*")


# ── Guard ──────────────────────────────────────────────────────────────────────
if df is None or 'Target' not in df.columns:
    st.error("❌ Tidak bisa memuat data. Pastikan file CSV memiliki kolom `Target` atau `Status`.")
    st.stop()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("🎓 Dashboard Analisis Dropout Siswa")
st.caption("Jaya Jaya Institut · Analisis faktor risiko dropout berbasis data historis mahasiswa")
st.divider()

# ── KPI Cards ──────────────────────────────────────────────────────────────────
total      = len(df)
dropout_n  = len(df[df['Target'] == 'Dropout'])
enrolled_n = len(df[df['Target'] == 'Enrolled'])
grad_n     = len(df[df['Target'] == 'Graduate'])
dr = dropout_n / total * 100
rr = (total - dropout_n) / total * 100

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("👥 Total Siswa",     f"{total:,}")
m2.metric("❌ Dropout",         f"{dropout_n:,}",  f"{dr:.1f}% dari total", delta_color="inverse")
m3.metric("📚 Enrolled",        f"{enrolled_n:,}", f"{enrolled_n/total*100:.1f}%")
m4.metric("🎓 Graduate",        f"{grad_n:,}",     f"{grad_n/total*100:.1f}%")
m5.metric("📈 Retention Rate",  f"{rr:.1f}%")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Overview Distribusi")
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(donut_chart(df), use_container_width=True)

with col2:
    if 'Age' in df.columns:
        d = df.copy()
        d['Kelompok Usia'] = pd.cut(d['Age'],
                                    bins=[16, 20, 23, 26, 100],
                                    labels=['17–20', '21–23', '24–26', '27+'])
        st.plotly_chart(
            stacked_pct_bar(d, 'Kelompok Usia',
                            'Status Siswa per Kelompok Usia (%)'),
            use_container_width=True
        )
    else:
        st.info("Kolom usia tidak tersedia")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – DEMOGRAFI & KEUANGAN
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("👥 Demografi & Status Keuangan")
c1, c2, c3 = st.columns(3)

with c1:
    gender_col = 'Gender'
    if gender_col in df.columns:
        d = df.copy()
        if d[gender_col].dtype in [int, float, np.int64, np.float64]:
            d[gender_col] = d[gender_col].map(GENDER_MAP).fillna(d[gender_col].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, gender_col, 'Status per Gender (%)'),
            use_container_width=True
        )
    else:
        st.info("Data gender tidak tersedia")

with c2:
    if 'Scholarship_holder' in df.columns:
        d = df.copy()
        d['Beasiswa'] = d['Scholarship_holder'].map(SCHOLARSHIP_MAP).fillna('?')
        st.plotly_chart(
            stacked_pct_bar(d, 'Beasiswa', 'Pengaruh Beasiswa (%)'),
            use_container_width=True
        )
    else:
        st.info("Data beasiswa tidak tersedia")

with c3:
    if 'Tuition_fees_up_to_date' in df.columns:
        d = df.copy()
        d['Status SPP'] = d['Tuition_fees_up_to_date'].map(TUITION_MAP).fillna('?')
        st.plotly_chart(
            stacked_pct_bar(d, 'Status SPP', 'Status Pembayaran SPP (%)'),
            use_container_width=True
        )
    else:
        st.info("Data pembayaran SPP tidak tersedia")

# Baris kedua demografi
d1, d2, d3 = st.columns(3)

with d1:
    if 'Marital_status' in df.columns:
        d = df.copy()
        d['Status Nikah'] = d['Marital_status'].map(MARITAL_MAP).fillna(d['Marital_status'].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Status Nikah', 'Status Pernikahan vs Dropout (%)'),
            use_container_width=True
        )

with d2:
    if 'Displaced' in df.columns:
        d = df.copy()
        d['Displaced'] = d['Displaced'].map(DISPLACED_MAP).fillna(d['Displaced'].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Displaced', 'Status Displaced vs Dropout (%)'),
            use_container_width=True
        )

with d3:
    if 'Debtor' in df.columns:
        d = df.copy()
        d['Debitur'] = d['Debtor'].map(DEBTOR_MAP).fillna(d['Debtor'].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Debitur', 'Status Debitur vs Dropout (%)'),
            use_container_width=True
        )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – PERFORMA AKADEMIK
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📚 Performa Akademik")

# Cari semua kolom grade — dataset UCI pakai format "Curricular_units_Xst_sem_(grade)"
grade_cols = [c for c in df.columns if 'grade' in c.lower()]
attend_col = 'Daytime_evening_attendance' if 'Daytime_evening_attendance' in df.columns else None

if not grade_cols and not attend_col:
    st.info("Data akademik tidak tersedia dalam dataset ini.")
else:
    g1, g2 = st.columns(2)

    with g1:
        if grade_cols:
            # Pilih kolom grade semester 1 jika ada, kalau tidak ambil yang pertama
            sem1 = next((c for c in grade_cols if '1st' in c or '1_sem' in c), grade_cols[0])
            # Debug info kecil
            non_zero = (df[sem1] > 0).sum()
            st.caption(f"Kolom: `{sem1}` · {non_zero:,} data non-zero · "
                       f"range {df[sem1].min():.1f}–{df[sem1].max():.1f}")
            if non_zero > 10:
                st.plotly_chart(
                    grade_bar(df, sem1, f'Nilai Semester 1 vs Status (%)'),
                    use_container_width=True
                )
            else:
                st.warning(f"Kolom `{sem1}` hampir semua nol, tidak bisa diplot.")
        else:
            st.info("Kolom grade tidak ditemukan")

    with g2:
        if attend_col:
            d = df.copy()
            d['Sesi Kuliah'] = d[attend_col].map(ATTENDANCE_MAP).fillna(d[attend_col].astype(str))
            st.plotly_chart(
                stacked_pct_bar(d, 'Sesi Kuliah', 'Sesi Kuliah vs Status (%)'),
                use_container_width=True
            )
        elif len(grade_cols) > 1:
            sem2 = next((c for c in grade_cols if '2nd' in c or '2_sem' in c), grade_cols[1])
            non_zero2 = (df[sem2] > 0).sum()
            if non_zero2 > 10:
                st.plotly_chart(
                    grade_bar(df, sem2, f'Nilai Semester 2 vs Status (%)'),
                    use_container_width=True
                )
        else:
            st.info("Data kehadiran tidak tersedia")

    # Kalau ada 2 grade cols, tampilkan sem 2 dan kehadiran di bawah
    if len(grade_cols) > 1 and attend_col:
        g3, g4 = st.columns(2)
        sem2 = next((c for c in grade_cols if '2nd' in c or '2_sem' in c), grade_cols[1])
        with g3:
            non_zero2 = (df[sem2] > 0).sum()
            if non_zero2 > 10:
                st.caption(f"Kolom: `{sem2}` · {non_zero2:,} data non-zero · "
                           f"range {df[sem2].min():.1f}–{df[sem2].max():.1f}")
                st.plotly_chart(
                    grade_bar(df, sem2, f'Nilai Semester 2 vs Status (%)'),
                    use_container_width=True
                )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – JALUR PENDAFTARAN
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📝 Jalur & Fase Pendaftaran")
a1, a2 = st.columns(2)

with a1:
    if 'Application_mode' in df.columns:
        d = df.copy()
        d['Jalur Pendaftaran'] = d['Application_mode'].map(APPLICATION_MODE_MAP).fillna(
            d['Application_mode'].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Jalur Pendaftaran',
                            'Jalur Pendaftaran vs Status (%) — Urutkan by Dropout',
                            height=420, sort_by_dropout=True),
            use_container_width=True
        )
    else:
        st.info("Kolom Application_mode tidak tersedia")

with a2:
    if 'Application_order' in df.columns:
        d = df.copy()
        d['Fase Pendaftaran'] = d['Application_order'].astype(str)
        st.plotly_chart(
            stacked_pct_bar(d, 'Fase Pendaftaran',
                            'Fase/Urutan Pendaftaran vs Status (%)', height=420),
            use_container_width=True
        )
    else:
        st.info("Kolom Application_order tidak tersedia")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – LATAR BELAKANG ORANG TUA
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("👨‍👩‍👧 Latar Belakang Pendidikan Orang Tua")

# Cari kolom qualification ayah & ibu (dataset UCI pakai apostrophe)
father_col = next((c for c in df.columns if 'father' in c.lower() and 'qual' in c.lower()), None)
mother_col = next((c for c in df.columns if 'mother' in c.lower() and 'qual' in c.lower()), None)

p1, p2 = st.columns(2)

with p1:
    if father_col:
        d = df.copy()
        d['Pendidikan Ayah'] = d[father_col].map(QUALIFICATION_MAP).fillna(d[father_col].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Pendidikan Ayah',
                            "Pendidikan Ayah vs Status (%) — Urutkan by Dropout",
                            height=420, sort_by_dropout=True),
            use_container_width=True
        )
    else:
        st.info("Kolom pendidikan ayah tidak tersedia")

with p2:
    if mother_col:
        d = df.copy()
        d['Pendidikan Ibu'] = d[mother_col].map(QUALIFICATION_MAP).fillna(d[mother_col].astype(str))
        st.plotly_chart(
            stacked_pct_bar(d, 'Pendidikan Ibu',
                            "Pendidikan Ibu vs Status (%) — Urutkan by Dropout",
                            height=420, sort_by_dropout=True),
            use_container_width=True
        )
    else:
        st.info("Kolom pendidikan ibu tidak tersedia")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("💡 Key Insights")

st.info(f"**Tingkat Dropout: {dr:.1f}%** — {dropout_n:,} dari {total:,} mahasiswa tidak menyelesaikan studi mereka.")

if 'Age' in df.columns:
    d_age = df[df['Target']=='Dropout']['Age'].mean()
    g_age = df[df['Target']=='Graduate']['Age'].mean()
    arah = "lebih tua" if d_age > g_age else "lebih muda"
    st.warning(f"**Usia berpengaruh:** Mahasiswa dropout rata-rata **{arah}** "
               f"({d_age:.1f} thn) dibanding yang lulus ({g_age:.1f} thn).")

if 'Tuition_fees_up_to_date' in df.columns:
    tun = df[df['Tuition_fees_up_to_date'] == 0]
    if len(tun) > 0:
        pct_tun = len(tun[tun['Target']=='Dropout']) / len(tun) * 100
        st.error(f"**Status SPP kritis:** Mahasiswa yang menunggak SPP memiliki dropout rate **{pct_tun:.1f}%**.")

if 'Scholarship_holder' in df.columns:
    r_s = len(df[(df['Target']=='Dropout') & (df['Scholarship_holder']==1)]) / max(1, len(df[df['Scholarship_holder']==1])) * 100
    r_n = len(df[(df['Target']=='Dropout') & (df['Scholarship_holder']==0)]) / max(1, len(df[df['Scholarship_holder']==0])) * 100
    if r_s < r_n:
        st.success(f"**Beasiswa efektif:** Dropout penerima beasiswa **{r_s:.1f}%** vs non-penerima **{r_n:.1f}%**.")
    else:
        st.warning(f"**Beasiswa perlu dievaluasi:** Dropout penerima beasiswa **{r_s:.1f}%** vs non-penerima **{r_n:.1f}%**.")

if grade_cols:
    sem1 = next((c for c in grade_cols if '1st' in c or '1_sem' in c), grade_cols[0])
    if (df[sem1] > 0).sum() > 10:
        avg_d = df[df['Target']=='Dropout'][sem1].mean()
        avg_g = df[df['Target']=='Graduate'][sem1].mean()
        st.success(f"**Nilai Sem. 1 prediktif:** Graduate rata-rata **{avg_g:.1f}** vs dropout **{avg_d:.1f}** "
                   f"(selisih {abs(avg_g - avg_d):.1f} poin).")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – REKOMENDASI
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Rekomendasi Aksi")

recs = [
    ("🔔", "Sistem Early Warning",
     "Terapkan model prediksi (Random Forest) untuk mendeteksi mahasiswa berisiko tinggi sejak akhir semester pertama. Mahasiswa yang teridentifikasi langsung dimasukkan ke program pendampingan."),
    ("💰", "Intervensi Finansial Terarah",
     "Berikan skema cicilan atau beasiswa darurat bagi mahasiswa yang menunggak SPP. Mahasiswa dengan tunggakan SPP adalah kelompok paling berisiko dropout."),
    ("📋", "Evaluasi Jalur Pendaftaran",
     "Tinjau ulang proses seleksi dan onboarding khusus untuk jalur Ordinance No. 612/93 dan fase pendaftaran ketiga yang menunjukkan korelasi tinggi dengan dropout."),
    ("👨‍👩‍👧", "Program Keterlibatan Keluarga",
     "Luncurkan program komunikasi aktif dengan orang tua, terutama bagi mahasiswa dengan latar belakang pendidikan orang tua yang rendah."),
    ("🎓", "Mentoring & Dukungan Akademik",
     "Bangun program tutoring peer-to-peer khusus mahasiswa semester 1 dengan nilai rendah, karena nilai semester 1 adalah prediktor dropout yang sangat kuat."),
]

for icon, title, desc in recs:
    with st.expander(f"{icon} {title}"):
        st.write(desc)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 – DATA PREVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("📋 Data Preview")
if st.checkbox("Tampilkan 100 baris pertama"):
    st.dataframe(df.head(100), use_container_width=True)
