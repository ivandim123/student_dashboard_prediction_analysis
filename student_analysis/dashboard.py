import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Dashboard Analisis Dropout Siswa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-whitegrid')

# Dashboard title
st.title("🎓 Dashboard Analisis Dropout Siswa")
st.markdown("""
Dashboard ini memberikan analisis komprehensif tentang faktor-faktor yang mempengaruhi dropout siswa.
Visualisasi dirancang untuk memberikan insight yang mudah dipahami oleh berbagai kalangan.
""")

@st.cache_data
def load_csv_from_path():
    """Try to load CSV from various paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    
    possible_paths = [
        "data.csv",
        os.path.join(script_dir, "data.csv"),
        "./data.csv", 
        "dataset/data.csv",
        "data/data.csv",
        os.path.join(script_dir, "dataset", "data.csv"),
        os.path.join(script_dir, "data", "data.csv")
    ]
    
    
    # Check for any CSV file in current directory
    try:
        current_dir_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if current_dir_files:
            for csv_file in current_dir_files:
                possible_paths.insert(0, csv_file)
    except:
        pass
    
    # Check script directory for CSV files
    try:
        script_dir_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
        if script_dir_files:
            for csv_file in script_dir_files:
                possible_paths.insert(0, os.path.join(script_dir, csv_file))
    except:
        pass
    
    st.sidebar.write("**🔍 Mencari file CSV:**")
    for path in possible_paths[:5]:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Found: {path}")
                return df, path
            else:
                st.sidebar.write(f"❌ {path}")
        except Exception as e:
            st.sidebar.write(f"❌ Error: {path}")
            continue

    return None, None

@st.cache_data
def create_sample_data():
    """Create comprehensive sample data for demonstration"""
    np.random.seed(42)
    n = 1000
    
    # Create more realistic sample data
    ages = np.random.normal(20, 2, n).clip(17, 35).astype(int)
    genders = np.random.choice(['Male', 'Female'], n, p=[0.45, 0.55])
    
    # Create correlated data for more realistic analysis
    scholarship_prob = np.where(ages < 22, 0.4, 0.2)  # Younger students more likely to have scholarship
    scholarships = np.random.binomial(1, scholarship_prob, n)
    
    # Grades correlated with scholarship and inversely with dropout probability
    base_grade = 12 + scholarships * 2 + np.random.normal(0, 2, n)
    grade_1st = base_grade.clip(0, 20)
    grade_2nd = (base_grade + np.random.normal(0, 1, n)).clip(0, 20)
    
    # Dropout probability based on multiple factors
    dropout_prob = 0.1 + 0.3 * (grade_1st < 10) + 0.2 * (scholarships == 0) + 0.1 * (ages > 25)
    dropout_prob = dropout_prob.clip(0, 1)
    
    # Generate status based on probability
    status_rand = np.random.random(n)
    statuses = []
    for i in range(n):
        if status_rand[i] < dropout_prob[i]:
            statuses.append('Dropout')
        elif status_rand[i] < dropout_prob[i] + 0.4:
            statuses.append('Enrolled')
        else:
            statuses.append('Graduate')
    
    data = {
        'Student_ID': range(1, n+1),
        'Age': ages,
        'Gender': genders,
        'Application_mode': np.random.choice([1, 2, 17, 18, 39, 42, 43], n),
        'Fathers_qualification': np.random.choice([1, 2, 3, 4, 5, 19, 34, 35], n),
        'Mothers_qualification': np.random.choice([1, 2, 3, 4, 5, 19, 34, 35], n),
        'Tuition_fees_up_to_date': np.random.choice([0, 1], n, p=[0.15, 0.85]),
        'Scholarship_holder': scholarships,
        'Grade_1st_semester': grade_1st,
        'Grade_2nd_semester': grade_2nd,
        'Curricular_units_enrolled': np.random.randint(4, 8, n),
        'Curricular_units_approved': np.random.randint(2, 8, n),
        'Marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.8, 0.15, 0.05]),
        'Status': statuses
    }
    
    return pd.DataFrame(data)

def create_overview_metrics(df):
    """Create overview metrics cards"""
    total_students = len(df)
    dropout_count = len(df[df['Target'] == 'Dropout'])
    enrolled_count = len(df[df['Target'] == 'Enrolled'])
    graduate_count = len(df[df['Target'] == 'Graduate'])
    
    dropout_rate = (dropout_count / total_students) * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("👥 Total Siswa", f"{total_students:,}")
    
    with col2:
        st.metric("❌ Dropout", f"{dropout_count:,}", 
                 f"{dropout_rate:.1f}%")
    
    with col3:
        st.metric("📚 Enrolled", f"{enrolled_count:,}")
    
    with col4:
        st.metric("🎓 Graduate", f"{graduate_count:,}")
    
    with col5:
        retention_rate = ((total_students - dropout_count) / total_students) * 100
        st.metric("📈 Retention Rate", f"{retention_rate:.1f}%")

def create_status_distribution_pie(df):
    """Create pie chart for status distribution"""
    status_counts = df['Target'].value_counts()
    
    fig = px.pie(values=status_counts.values, 
                 names=status_counts.index,
                 title="Distribusi Status Siswa",
                 color_discrete_map={
                     'Dropout': '#ff6b6b',
                     'Enrolled': '#4ecdc4', 
                     'Graduate': '#45b7d1'
                 })
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=True, height=400)
    
    return fig

def create_age_dropout_analysis(df):
    """Create age vs dropout analysis"""
    if 'Age_at_enrollment' not in df.columns:
        return None

    df['Age'] = df['Age_at_enrollment']
    df['Age_Group'] = pd.cut(df['Age'], 
                              bins=[16, 20, 23, 26, 100], 
                              labels=['17-20', '21-23', '24-26', '27+'])

    age_status = df.groupby(['Age_Group', 'Target']).size().unstack(fill_value=0)
    age_status_pct = age_status.div(age_status.sum(axis=1), axis=0) * 100

    fig = px.bar(age_status_pct.reset_index(), 
                 x='Age_Group', 
                 y=['Dropout', 'Enrolled', 'Graduate'],
                 title="Distribusi Status Siswa berdasarkan Kelompok Usia (%)",
                 labels={'value': 'Persentase (%)', 'Age_Group': 'Kelompok Usia'},
                 color_discrete_map={
                     'Dropout': '#ff6b6b',
                     'Enrolled': '#4ecdc4', 
                     'Graduate': '#45b7d1'
                 })

    fig.update_layout(barmode='stack', height=400)
    return fig

def create_scholarship_analysis(df):
    """Create scholarship vs dropout analysis"""
    if 'Scholarship_holder' in df.columns:
        scholarship_map = {0: 'Tidak Ada Beasiswa', 1: 'Ada Beasiswa'}
        df['Scholarship_Status'] = df['Scholarship_holder'].map(scholarship_map)
        
        scholarship_status = df.groupby(['Scholarship_Status', 'Target']).size().unstack(fill_value=0)
        scholarship_status_pct = scholarship_status.div(scholarship_status.sum(axis=1), axis=0) * 100
        
        fig = px.bar(scholarship_status_pct.reset_index(), 
                     x='Scholarship_Status', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Pengaruh Beasiswa terhadap Status Siswa (%)",
                     labels={'value': 'Persentase (%)', 'Scholarship_Status': 'Status Beasiswa'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })
        
        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_gender_analysis(df):
    """Create gender vs dropout analysis"""
    if 'Gender' in df.columns:
        gender_status = df.groupby(['Gender', 'Target']).size().unstack(fill_value=0)
        gender_status_pct = gender_status.div(gender_status.sum(axis=1), axis=0) * 100
        
        fig = px.bar(gender_status_pct.reset_index(), 
                     x='Gender', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Distribusi Status Siswa berdasarkan Gender (%)",
                     labels={'value': 'Persentase (%)', 'Gender': 'Jenis Kelamin'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })
        
        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_grades_analysis(df):
    """Create grades analysis"""
    grade_cols = ['Curricular_units_1st_sem_grade'] if 'Curricular_units_1st_sem_grade' in df.columns else []

    if grade_cols:
        grade_col = grade_cols[0]

        df['Grade_Category'] = pd.cut(df[grade_col], 
                                     bins=[0, 10, 14, 17, 20], 
                                     labels=['Rendah (0-10)', 'Sedang (10-14)', 
                                             'Baik (14-17)', 'Sangat Baik (17-20)'])

        grade_status = df.groupby(['Grade_Category', 'Target']).size().unstack(fill_value=0)
        grade_status_pct = grade_status.div(grade_status.sum(axis=1), axis=0) * 100

        fig = px.bar(grade_status_pct.reset_index(), 
                     x='Grade_Category', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title=f"Pengaruh Prestasi Akademik terhadap Status Siswa (%)",
                     labels={'value': 'Persentase (%)', 'Grade_Category': 'Kategori Nilai'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })

        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_tuition_analysis(df):
    """Create tuition fees analysis"""
    if 'Tuition_fees_up_to_date' in df.columns:
        tuition_map = {0: 'Tidak Up-to-date', 1: 'Up-to-date'}
        df['Tuition_Status'] = df['Tuition_fees_up_to_date'].map(tuition_map)
        
        tuition_status = df.groupby(['Tuition_Status', 'Target']).size().unstack(fill_value=0)
        tuition_status_pct = tuition_status.div(tuition_status.sum(axis=1), axis=0) * 100
        
        fig = px.bar(tuition_status_pct.reset_index(), 
                     x='Tuition_Status', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Pengaruh Status Pembayaran SPP terhadap Dropout (%)",
                     labels={'value': 'Persentase (%)', 'Tuition_Status': 'Status Pembayaran SPP'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })
        
        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_parents_qualification_analysis(df):
    """Create parents qualification analysis"""
    # Check for possible parent qualification columns with different naming conventions
    parent_cols = []
    possible_names = [
        'Fathers_qualification', 'Mothers_qualification', 
        "Father's_qualification", "Mother's_qualification",
        'fathers_qualification', 'mothers_qualification',
        'parent_qualification', 'qualification'
    ]
    
    for col_name in possible_names:
        if col_name in df.columns:
            parent_cols.append(col_name)
    
    if parent_cols:
        # Use first available parent qualification column
        parent_col = parent_cols[0]
        
        # Map qualification codes to meaningful labels
        qual_map = {1: 'Dasar', 2: 'Menengah', 3: 'Menengah Atas', 4: 'Diploma', 5: 'Sarjana', 
                   19: 'Magister', 34: 'Doktor', 35: 'Lainnya'}
        
        # Apply mapping, keep original value if not in map
        df['Parent_Qualification'] = df[parent_col].map(qual_map).fillna('Lainnya')
        
        qual_status = df.groupby(['Parent_Qualification', 'Target']).size().unstack(fill_value=0)
        qual_status_pct = qual_status.div(qual_status.sum(axis=1), axis=0) * 100
        
        fig = px.bar(qual_status_pct.reset_index(), 
                     x='Parent_Qualification', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Pengaruh Pendidikan Orang Tua terhadap Status Siswa (%)",
                     labels={'value': 'Persentase (%)', 'Parent_Qualification': 'Tingkat Pendidikan Orang Tua'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })
        
        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_attendance_analysis(df):
    """Create attendance analysis"""
    if 'Daytime_evening_attendance' in df.columns:
        df['Attendance_Label'] = df['Daytime_evening_attendance'].map({1: 'Daytime', 0: 'Evening'})

        attendance_status = df.groupby(['Attendance_Label', 'Target']).size().unstack(fill_value=0)
        attendance_status_pct = attendance_status.div(attendance_status.sum(axis=1), axis=0) * 100

        fig = px.bar(attendance_status_pct.reset_index(), 
                     x='Attendance_Label', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Pengaruh Waktu Kehadiran terhadap Status Siswa (%)",
                     labels={'value': 'Persentase (%)', 'Attendance_Label': 'Waktu Kehadiran'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })

        fig.update_layout(barmode='stack', height=400)
        return fig
    return None

def create_parents_background_analysis(df):
    """Create parents background analysis"""
    # Check for possible mother qualification columns
    mother_cols = []
    possible_names = [
        'Mothers_qualification', "Mother's_qualification", 
        'mothers_qualification', 'mother_qualification'
    ]
    
    for col_name in possible_names:
        if col_name in df.columns:
            mother_cols.append(col_name)
    
    if mother_cols:
        mother_col = mother_cols[0]
        
        # Create qualification groups
        df['Mother_Qual_Group'] = pd.cut(df[mother_col],
                                        bins=[0, 2, 4, 10, 50], 
                                        labels=['Basic', 'Secondary', 'Higher', 'Advanced'])
        
        qual_status = df.groupby(['Mother_Qual_Group', 'Target']).size().unstack(fill_value=0)
        qual_status_pct = qual_status.div(qual_status.sum(axis=1), axis=0) * 100
        
        fig = px.bar(qual_status_pct.reset_index(), 
                     x='Mother_Qual_Group', 
                     y=['Dropout', 'Enrolled', 'Graduate'],
                     title="Pengaruh Latar Belakang Pendidikan Ibu (%)",
                     labels={'value': 'Persentase (%)', 'Mother_Qual_Group': 'Tingkat Pendidikan Ibu'},
                     color_discrete_map={
                         'Dropout': '#ff6b6b',
                         'Enrolled': '#4ecdc4', 
                         'Graduate': '#45b7d1'
                     })
        
        fig.update_layout(barmode='stack', height=400)
        return fig
    
    return None

def create_insights_summary(df):
    """Create insights summary"""
    insights = []
    
    # Calculate dropout rate
    dropout_rate = (len(df[df['Target'] == 'Dropout']) / len(df)) * 100
    insights.append(f"📊 **Tingkat Dropout**: {dropout_rate:.1f}% dari total siswa")
    
    # Age analysis
    if 'Age' in df.columns:
        dropout_by_age = df[df['Target'] == 'Dropout']['Age'].mean()
        overall_age = df['Age'].mean()
        if dropout_by_age > overall_age:
            insights.append(f"👴 Siswa yang dropout cenderung **lebih tua** (rata-rata {dropout_by_age:.1f} tahun vs {overall_age:.1f} tahun)")
        else:
            insights.append(f"👶 Siswa yang dropout cenderung **lebih muda** (rata-rata {dropout_by_age:.1f} tahun vs {overall_age:.1f} tahun)")
    
    # Scholarship analysis
    if 'Scholarship_holder' in df.columns:
        dropout_with_scholarship = df[(df['Target'] == 'Dropout') & (df['Scholarship_holder'] == 1)].shape[0]
        total_with_scholarship = df[df['Scholarship_holder'] == 1].shape[0]
        dropout_without_scholarship = df[(df['Target'] == 'Dropout') & (df['Scholarship_holder'] == 0)].shape[0]
        total_without_scholarship = df[df['Scholarship_holder'] == 0].shape[0]
        
        if total_with_scholarship > 0 and total_without_scholarship > 0:
            dropout_rate_with = (dropout_with_scholarship / total_with_scholarship) * 100
            dropout_rate_without = (dropout_without_scholarship / total_without_scholarship) * 100
            
            if dropout_rate_with < dropout_rate_without:
                insights.append(f"🎓 **Beasiswa efektif**: Dropout rate dengan beasiswa ({dropout_rate_with:.1f}%) lebih rendah daripada tanpa beasiswa ({dropout_rate_without:.1f}%)")
            else:
                insights.append(f"⚠️ **Beasiswa perlu evaluasi**: Dropout rate dengan beasiswa ({dropout_rate_with:.1f}%) tidak lebih rendah dari tanpa beasiswa ({dropout_rate_without:.1f}%)")
    
    # Grade analysis
    grade_cols = [col for col in df.columns if 'Grade' in col or 'grade' in col]
    if grade_cols:
        grade_col = grade_cols[0]
        avg_grade_dropout = df[df['Target'] == 'Dropout'][grade_col].mean()
        avg_grade_graduate = df[df['Target'] == 'Graduate'][grade_col].mean()
        
        insights.append(f"📚 **Prestasi akademik berpengaruh**: Rata-rata nilai graduate ({avg_grade_graduate:.1f}) lebih tinggi dari dropout ({avg_grade_dropout:.1f})")
    
    return insights

def main():
    # Main application logic
    st.sidebar.header("📁 Data Source")

    # Load data
    df, file_path = load_csv_from_path()

    if df is not None:
        st.sidebar.success(f"✅ Data loaded: {file_path}")
    else:
        st.sidebar.warning("⚠️ No CSV file found automatically")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ Data loaded from upload")
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
                df = None
        else:
            st.sidebar.info("📊 Using sample data for demo")
            df = create_sample_data()

    # Process data
    if df is not None:
        # Handle Status/Target column
        if 'Status' in df.columns:
            df = df.rename(columns={'Status': 'Target'})
            st.sidebar.success("✅ Column 'Status' renamed to 'Target'")
        elif 'Target' not in df.columns:
            st.error("❌ Column 'Status' or 'Target' not found in dataset")
            st.stop()
        # Handle missing 'Age' column by using 'Age_at_enrollment'
        if 'Age' not in df.columns and 'Age_at_enrollment' in df.columns:
            df['Age'] = df['Age_at_enrollment']
            st.sidebar.success("✅ 'Age' column created from 'Age_at_enrollment'")

        # Display dataset info
        st.sidebar.subheader("📊 Dataset Info")
        st.sidebar.write(f"Rows: {df.shape[0]:,}")
        st.sidebar.write(f"Columns: {df.shape[1]}")
        
        # Show available columns for debugging
        st.sidebar.subheader("📋 Available Columns")
        st.sidebar.write(list(df.columns))
        
        # Main dashboard content
        st.markdown("---")
        
        # Overview Metrics
        st.header("📈 Overview Metrics")
        create_overview_metrics(df)
        
        st.markdown("---")
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Distribusi Status Siswa")
            fig_pie = create_status_distribution_pie(df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            st.subheader("👥 Analisis Berdasarkan Usia")
            fig_age = create_age_dropout_analysis(df)
            if fig_age:
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Data usia tidak tersedia")
        
        st.markdown("---")
        
        # Additional analysis
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("🎓 Pengaruh Beasiswa")
            fig_scholarship = create_scholarship_analysis(df)
            if fig_scholarship:
                st.plotly_chart(fig_scholarship, use_container_width=True)
            else:
                st.info("Data beasiswa tidak tersedia")
        
        with col4:
            st.subheader("👫 Analisis Gender")
            fig_gender = create_gender_analysis(df)
            if fig_gender:
                st.plotly_chart(fig_gender, use_container_width=True)
            else:
                st.info("Data gender tidak tersedia")
        
        st.markdown("---")
        
        # Academic and financial factors
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("📚 Prestasi Akademik")
            fig_grades = create_grades_analysis(df)
            if fig_grades:
                st.plotly_chart(fig_grades, use_container_width=True)
            else:
                st.info("Data nilai tidak tersedia")
        
        with col6:
            st.subheader("💰 Status Pembayaran SPP")
            fig_tuition = create_tuition_analysis(df)
            if fig_tuition:
                st.plotly_chart(fig_tuition, use_container_width=True)
            else:
                st.info("Data pembayaran SPP tidak tersedia")
        
        st.markdown("---")
        
        # Parents education and attendance
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("👨‍👩‍👧‍👦 Pendidikan Orang Tua")
            fig_parents = create_parents_qualification_analysis(df)
            if fig_parents:
                st.plotly_chart(fig_parents, use_container_width=True)
            else:
                st.info("Data pendidikan orang tua tidak tersedia")
        
        with col8:
            st.subheader("🕐 Waktu Kehadiran")
            fig_attendance = create_attendance_analysis(df)
            if fig_attendance:
                st.plotly_chart(fig_attendance, use_container_width=True)
            else:
                st.info("Data waktu kehadiran tidak tersedia")
        
        st.markdown("---")
        
        # Parents background
        st.subheader("👩‍🎓 Latar Belakang Pendidikan Ibu")
        fig_parents_bg = create_parents_background_analysis(df)
        if fig_parents_bg:
            st.plotly_chart(fig_parents_bg, use_container_width=True)
        else:
            st.info("Data latar belakang pendidikan ibu tidak tersedia")
        
        st.markdown("---")
        
        # Key Insights
        st.header("💡 Key Insights & Rekomendasi")
        insights = create_insights_summary(df)
        
        for insight in insights:
            st.markdown(insight)
        
        st.markdown("### 🎯 Rekomendasi Aksi:")
        st.markdown("""
        1. **🎓 Program Dukungan Akademik**: Fokus pada siswa dengan nilai rendah semester pertama
        2. **💰 Bantuan Keuangan**: Perluas program beasiswa untuk siswa berisiko tinggi
        3. **👥 Mentoring Program**: Buat program pendampingan khusus berdasarkan profil risiko
        4. **📊 Early Warning System**: Implementasikan sistem deteksi dini berdasarkan faktor-faktor kunci
        5. **🏫 Dukungan Keluarga**: Program edukasi untuk orang tua tentang pentingnya dukungan akademik
        """)
        
        # Data preview section
        if st.checkbox("📋 Tampilkan Preview Data"):
            st.subheader("Data Preview")
            st.dataframe(df.head(100))

    else:
        st.error("❌ Cannot load data. Please ensure CSV file is available.")
        st.markdown("""
        **Troubleshooting:**
        1. Ensure CSV file is in the same directory as the script
        2. Check file name (case-sensitive)
        3. Make sure file is not open in another application
        4. Try uploading the file using the sidebar uploader
        """)

if __name__ == "__main__":
    main()
