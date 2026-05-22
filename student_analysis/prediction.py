import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import io
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# PENTING: st.set_page_config() HARUS DIPANGGIL PERTAMA KALI, SEBELUM IMPORT LAINNYA
st.set_page_config(
    page_title="Student Status Prediction",
    page_icon="🎓",
    layout="wide"
)

# Cache untuk loading model dan preprocessing
@st.cache_resource
def load_models():
    """Load pre-trained models from submission folder"""
    try:
        # Model ada di folder submission/
        rf_model = joblib.load('student_analysis/random_forest_model.pkl')
        dt_model = joblib.load('student_analysis/decision_tree_model.pkl')
        
        # Asumsikan feature_names_in_ ada di model
        if not hasattr(rf_model, 'feature_names_in_') or not hasattr(dt_model, 'feature_names_in_'):
            st.error("❌ Model tidak memiliki atribut 'feature_names_in_'. Pastikan model disimpan dengan benar.")
            st.stop() # Hentikan jika model tidak dimuat dengan benar atau tidak memiliki atribut yang diharapkan

        return rf_model, dt_model
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan! Error: {str(e)}")
        st.error("Pastikan 'random_forest_model.pkl' dan 'decision_tree_model.pkl' berada di folder 'submission/'.")
        
        # Debug info
        st.write("**Info Debug:**")
        st.write(f"Direktori kerja saat ini: {os.getcwd()}")
        st.write(f"File di direktori saat ini: {os.listdir('.')}")
        if os.path.exists('submission'):
            st.write(f"File di folder 'submission': {os.listdir('submission/')}")
        else:
            st.write("❌ Folder 'submission' tidak ditemukan!")
            
        return None, None
    except Exception as e:
        st.error(f"❌ Error saat memuat model: {str(e)}")
        return None, None

@st.cache_data
def load_default_data():
    """Load default dataset from submission folder"""
    try:
        # Data juga ada di folder submission/
        df = pd.read_csv('submission/data.csv', low_memory=False)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset default 'data.csv' tidak ditemukan di folder submission!")
        return None

# Fungsi baru untuk mendapatkan pemetaan status
def get_status_mapping(df_with_status):
    """
    Secara dinamis membuat pemetaan status dari data asli.
    Digunakan untuk mengonversi prediksi numerik kembali ke label string.
    """
    if 'Status' in df_with_status.columns:
        unique_statuses = sorted(df_with_status['Status'].unique())
        # Pastikan ada 3 status yang diharapkan: Dropout, Enrolled, Graduate
        if len(unique_statuses) == 3 and 'Dropout' in unique_statuses and 'Enrolled' in unique_statuses and 'Graduate' in unique_statuses:
            # Tetapkan mapping yang konsisten berdasarkan urutan alfabet (jika model juga menggunakan ini)
            # Atau sesuaikan dengan encoding yang pasti digunakan saat training model
            return {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'} 
        elif len(unique_statuses) >= 2: # Minimal 2 status unik diperlukan untuk pemetaan
            # Fallback jika tidak sesuai 3 status yang diharapkan, tapi tetap buat mapping
            # Perhatian: Ini mungkin tidak cocok jika urutan encoding model berbeda
            return {i: status for i, status in enumerate(unique_statuses)}
    # Pemetaan default jika kolom 'Status' tidak ditemukan atau ada masalah
    # Ini harus sesuai dengan bagaimana label di-encode saat model dilatih
    return {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'} # Fallback umum

def preprocess_data(df_raw, scaler_path='submission/scaler.pkl', contains_status_col=True):
    """
    Melakukan preprocessing data serupa dengan preprocessing pelatihan.
    Dapat menangani data dengan atau tanpa kolom 'Status'.
    contains_status_col: True jika data mengandung kolom 'Status' (label asli).
    """
    df_clean = df_raw.copy()
    
    # Salin kolom 'Status' asli ke 'Status_Original' jika ada
    if contains_status_col and 'Status' in df_clean.columns:
        df_clean['Status_Original'] = df_clean['Status']
        # Drop baris dengan NaN di semua kolom (jika ini yang dilakukan saat training)
        df_clean = df_clean.dropna().reset_index(drop=True)
    elif not contains_status_col: # Untuk data baru, drop NaN hanya pada fitur yang relevan
        model_features = rf_model.feature_names_in_ if rf_model else []
        df_clean = df_clean.dropna(subset=[col for col in model_features if col in df_clean.columns]).reset_index(drop=True)
    
    # Define features for capping and standardization (must match training features)
    capping_features = [
        'Age_at_enrollment',
        'Admission_grade',
        'Curricular_units_1st_sem_grade',
        'Previous_qualification_grade',
        'Course',
        'Curricular_units_2nd_sem_grade'
    ]
    
    # Buat salinan sementara untuk preprocessing agar tidak mengubah df_clean terlalu dini
    temp_df_for_processing = df_clean.copy() 
    
    existing_capped_features = [col for col in capping_features if col in temp_df_for_processing.columns]

    if existing_capped_features:
        # Terapkan capping terlebih dahulu (jika ini urutan saat training)
        # CATATAN PENTING: Untuk produksi, lebih baik menyimpan batas Q1/Q3/IQR dari data training asli
        # dan menggunakannya di sini, bukan menghitung ulang dari data input.
        # Namun, mengikuti pola kode asli Anda yang menghitung ulang.
        for col in existing_capped_features:
            Q1 = temp_df_for_processing[col].quantile(0.25)
            Q3 = temp_df_for_processing[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            temp_df_for_processing[col] = np.clip(temp_df_for_processing[col], lower_bound, upper_bound)
        
        # Sekarang, terapkan scaling. Coba muat scaler yang sudah dilatih.
        scaler = None
        try:
            scaler = joblib.load(scaler_path)
            st.sidebar.info("✅ Scaler yang sudah dilatih dimuat untuk preprocessing.")
            # Hanya transform fitur yang scaler tahu
            features_to_transform_by_scaler = [f for f in existing_capped_features if f in scaler.feature_names_in_]
            temp_df_for_processing[features_to_transform_by_scaler] = scaler.transform(temp_df_for_processing[features_to_transform_by_scaler])
        except FileNotFoundError:
            st.warning("⚠️ File scaler tidak ditemukan. Melatih scaler baru pada data yang disediakan. Pastikan preprocessing konsisten dengan pelatihan.")
            scaler = StandardScaler()
            temp_df_for_processing[existing_capped_features] = scaler.fit_transform(temp_df_for_processing[existing_capped_features])
        except Exception as e:
            st.error(f"❌ Error saat memuat atau menerapkan scaler: {e}. Melatih scaler baru.")
            scaler = StandardScaler()
            temp_df_for_processing[existing_capped_features] = scaler.fit_transform(temp_df_for_processing[existing_capped_features])
        
        # Perbarui df_clean dengan fitur yang sudah diproses dari temp_df_for_processing
        for col in existing_capped_features:
            df_clean[col] = temp_df_for_processing[col]

    return df_clean, scaler, existing_capped_features


def make_predictions(df_processed, rf_model, dt_model, include_real_status=False):
    """
    Melakukan prediksi pada data yang sudah diproses.
    Jika include_real_status True, ia mengharapkan kolom 'Status_Original' dan menyertakannya.
    Jika tidak, ia mengasumsikan data baru tanpa label yang diketahui.
    """
    try:
        model_features = rf_model.feature_names_in_ # Fitur yang diharapkan oleh model
        
        # Pastikan semua fitur yang diperlukan ada di DataFrame yang diproses
        missing_features = [f for f in model_features if f not in df_processed.columns]
        if missing_features:
            st.error(f"❌ Fitur yang diperlukan untuk prediksi tidak ada: {', '.join(missing_features)}. Pastikan data input Anda mengandung semua kolom yang diperlukan.")
            return None, None

        X_predict = df_processed[model_features]
        
        # Dapatkan status asli hanya jika diminta dan tersedia
        status_original_col = []
        if include_real_status and 'Status_Original' in df_processed.columns:
            status_original_col = df_processed['Status_Original'].values
        else:
            status_original_col = ['N/A'] * len(df_processed) # Untuk data baru, status asli tidak diketahui

        rf_pred = rf_model.predict(X_predict)
        dt_pred = dt_model.predict(X_predict)
        
        # Buat hasil
        results_data = {
            'ID': range(1, len(df_processed) + 1),
        }
        
        if include_real_status:
            results_data['Status_Asli'] = status_original_col # Mengubah nama kolom menjadi 'Status_Asli'

        results_data['Random_Forest_Prediction'] = rf_pred
        results_data['Decision_Tree_Prediction'] = dt_pred

        results = pd.DataFrame(results_data)
        
        # Ambil mapping dari data default
        default_data_for_mapping = load_default_data()
        status_mapping = get_status_mapping(default_data_for_mapping)
            
        # Konversi prediksi numerik ke label jika diperlukan
        if results['Random_Forest_Prediction'].dtype in ['int64', 'float64']:
            results['Random_Forest_Prediction'] = results['Random_Forest_Prediction'].map(status_mapping)
        if results['Decision_Tree_Prediction'].dtype in ['int64', 'float64']:
            results['Decision_Tree_Prediction'] = results['Decision_Tree_Prediction'].map(status_mapping)
            
        return results, X_predict
            
    except Exception as e:
        st.error(f"Error selama prediksi: {str(e)}")
        return None, None

def create_visualizations(results):
    """Membuat visualisasi untuk prediksi"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribusi Prediksi")
        
        # Menghitung prediksi untuk kedua model
        rf_counts = results['Random_Forest_Prediction'].value_counts()
        dt_counts = results['Decision_Tree_Prediction'].value_counts()
        
        # Pastikan semua kategori ada
        all_categories = ['Graduate', 'Dropout', 'Enrolled']
        rf_counts = rf_counts.reindex(all_categories, fill_value=0)
        dt_counts = dt_counts.reindex(all_categories, fill_value=0)
        
        fig = go.Figure(data=[
            go.Bar(name='Random Forest', x=rf_counts.index, y=rf_counts.values, marker_color='#2E8B57'),
            go.Bar(name='Decision Tree', x=dt_counts.index, y=dt_counts.values, marker_color='#4682B4')
        ])
        fig.update_layout(
            barmode='group', 
            title="Prediksi Status Siswa berdasarkan Model",
            xaxis_title="Status",
            yaxis_title="Jumlah"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Kesepakatan Model")
        
        # Memeriksa kesepakatan antara model
        agreement = (results['Random_Forest_Prediction'] == results['Decision_Tree_Prediction']).sum()
        total = len(results)
        
        fig = go.Figure(data=[go.Pie(
            labels=['Setuju', 'Tidak Setuju'],
            values=[agreement, total - agreement],
            hole=0.3,
            marker_colors=['#2E8B57', '#DC143C']
        )])
        fig.update_layout(title=f"Kesepakatan Model: {agreement}/{total} ({agreement/total*100:.1f}%)")
        st.plotly_chart(fig, use_container_width=True)

def show_status_distribution(df):
    """Menampilkan distribusi Status dalam dataset (hanya jika kolom 'Status' ada)"""
    if 'Status' in df.columns:
        st.subheader("📈 Distribusi Status Dataset (dari Data Default)")
        
        status_counts = df['Status'].value_counts()
        
        # Membuat pie chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.3,
            marker_colors=colors
        )])
        fig.update_layout(title="Distribusi Status Siswa dalam Dataset yang Dimuat")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("**Jumlah Status:**")
            for status, count in status_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {status}: {count} ({percentage:.1f}%)")
    else:
        st.info("Dataset yang diunggah tidak memiliki kolom 'Status', sehingga distribusi status data asli tidak ditampilkan.")

# Header aplikasi
st.title("🎓 Prototipe Prediksi Status Siswa")
st.markdown("*Gunakan prototipe ini untuk memprediksi apakah siswa baru akan Lulus, Drop Out, atau tetap Terdaftar.*")
st.markdown("---")

# Memuat model
rf_model, dt_model = load_models()

if rf_model is None or dt_model is None:
    st.stop()

# Dapatkan daftar fitur yang diharapkan oleh model
model_expected_features = list(rf_model.feature_names_in_)

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Pengaturan Prediksi")

# Pilihan mode input
prediction_mode = st.sidebar.radio(
    "Pilih Mode Prediksi:",
    ["Prediksi untuk Satu Siswa (Input Manual)", "Prediksi untuk Sekelompok Siswa (Unggah CSV)", "Lihat Analisis Data Default"]
)

df_raw = None # Inisialisasi df_raw di luar blok kondisional

if prediction_mode == "Lihat Analisis Data Default":
    st.header("📋 Informasi & Analisis Dataset Default")
    df_raw = load_default_data()
    if df_raw is not None:
        st.info(f"✅ Data default dimuat: {len(df_raw)} baris. Data ini mengandung label yang diketahui untuk tujuan analitis.")
        
        # Informasi dataset default
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Total Baris", len(df_raw))
        with col2: st.metric("Total Kolom", len(df_raw.columns))
        with col3: st.metric("Nilai Hilang", df_raw.isnull().sum().sum())
        with col4: st.metric("Kategori Status", df_raw['Status'].nunique() if 'Status' in df_raw.columns else "N/A")
        
        show_status_distribution(df_raw) # Tampilkan distribusi untuk data default
        
        with st.expander("Lihat Sampel Data Mentah (Default)"):
            st.dataframe(df_raw.head())
        
        st.markdown("---")
        st.header("🔮 Contoh Prediksi pada Data Default")
        st.info("Bagian ini menunjukkan bagaimana model memprediksi pada sampel dataset default, termasuk status asli untuk perbandingan.")

        max_samples = min(50, len(df_raw))
        num_samples_default = st.slider(
            "Jumlah sampel untuk diprediksi dari Data Default:",
            min_value=1,
            max_value=max_samples,
            value=min(10, max_samples),
            key="default_samples_slider"
        )

        if st.button("🚀 Jalankan Contoh Prediksi pada Data Default", key="run_default_predictions_button"):
            with st.spinner("Memproses dan memprediksi pada data default..."):
                # Preprocessing data default, termasuk 'Status_Original'
                df_processed_default, _, _ = preprocess_data(df_raw, contains_status_col=True)
                
                # Ambil sampel acak dari data default yang sudah diproses untuk tampilan prediksi
                sample_indices = random.sample(range(len(df_processed_default)), min(num_samples_default, len(df_processed_default)))
                df_sample_for_pred = df_processed_default.iloc[sample_indices].copy()
                
                # Buat prediksi termasuk status asli
                results_default, _ = make_predictions(df_sample_for_pred, rf_model, dt_model, include_real_status=True)

            if results_default is not None:
                st.success(f"✅ Contoh prediksi selesai untuk {len(results_default)} siswa dari data default!")
                st.subheader("📋 Hasil Prediksi Contoh (Data Default)")
                
                def highlight_predictions(row):
                    colors = []
                    for col_name in row.index:
                        if col_name in ['Random_Forest_Prediction', 'Decision_Tree_Prediction', 'Status_Asli']:
                            if row[col_name] == 'Graduate':
                                colors.append('background-color: #d4edda; color: #155724')
                            elif row[col_name] == 'Dropout':
                                colors.append('background-color: #f8d7da; color: #721c24')
                            elif row[col_name] == 'Enrolled':
                                colors.append('background-color: #fff3cd; color: #856404')
                            else:
                                colors.append('')
                        else:
                            colors.append('')
                    return colors
                
                styled_results_default = results_default.style.apply(highlight_predictions, axis=1)
                st.dataframe(styled_results_default, use_container_width=True)

                create_visualizations(results_default) # Gunakan hasil default untuk visualisasi
                st.subheader("📊 Ringkasan Prediksi Contoh (Data Default)")
                
                col1, col2, col3 = st.columns(3)
                with col1: st.write("**Prediksi Random Forest:**")
                with col2: st.write("**Prediksi Decision Tree:**")
                with col3: st.write("**Status Asli:**")
                
                rf_counts = results_default['Random_Forest_Prediction'].value_counts().reindex(['Graduate', 'Dropout', 'Enrolled'], fill_value=0)
                dt_counts = results_default['Decision_Tree_Prediction'].value_counts().reindex(['Graduate', 'Dropout', 'Enrolled'], fill_value=0)
                real_counts = results_default['Status_Asli'].value_counts().reindex(['Graduate', 'Dropout', 'Enrolled'], fill_value=0)

                st.write(f"- Lulus: {rf_counts.get('Graduate', 0)}")
                st.write(f"- Drop Out: {rf_counts.get('Dropout', 0)}")
                st.write(f"- Terdaftar: {rf_counts.get('Enrolled', 0)}")
                
                st.write(f"- Lulus: {dt_counts.get('Graduate', 0)}")
                st.write(f"- Drop Out: {dt_counts.get('Dropout', 0)}")
                st.write(f"- Terdaftar: {dt_counts.get('Enrolled', 0)}")
                
                st.write(f"- Lulus: {real_counts.get('Graduate', 0)}")
                st.write(f"- Drop Out: {real_counts.get('Dropout', 0)}")
                st.write(f"- Terdaftar: {real_counts.get('Enrolled', 0)}")

                csv_buffer = io.StringIO()
                results_default.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                st.download_button(
                    label="📥 Unduh Hasil Contoh (Data Default) sebagai CSV",
                    data=csv_data,
                    file_name=f"default_data_predictions_sample_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Dataset default tidak dapat dimuat. Silakan periksa jalur file.")

elif prediction_mode == "Prediksi untuk Satu Siswa (Input Manual)":
    st.header("📝 Input Manual untuk Prediksi Status Siswa Tunggal")
    st.info("Masukkan karakteristik siswa di bawah ini untuk mendapatkan prediksi status mereka.")

    # Buat formulir input untuk fitur
    input_data = {}
    with st.form("manual_prediction_form"):
        st.write("Silakan masukkan detail siswa:")
        
        # Atur input dalam kolom untuk tata letak yang lebih baik
        cols_per_row = 3
        current_cols = st.columns(cols_per_row)
        col_idx = 0
        
        # Definisi mapping untuk fitur kategorikal
        MARITAL_STATUS_OPTIONS = {
            1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 
            5: 'Facto union', 6: 'Legally separated'
        }
        DAYTIME_EVENING_OPTIONS = {1: 'Daytime', 0: 'Evening'}
        YES_NO_OPTIONS = {1: 'Yes', 0: 'No'}
        GENDER_OPTIONS = {1: 'Male', 0: 'Female'}

        # Untuk fitur kategorikal dengan banyak pilihan, gunakan selectbox
        # Untuk fitur kategorikal biner, bisa pakai radio atau selectbox
        # Untuk numerik, pakai number_input dengan min/max

        # Fungsi pembantu untuk membuat selectbox/radio dengan label dan nilai
        def create_categorical_input(feature_name, options_dict, default_value_key, col):
            display_options = list(options_dict.values())
            default_index = display_options.index(options_dict[default_value_key])
            selected_display_value = col.selectbox(
                f"{feature_name.replace('_', ' ').title()}:",
                options=display_options,
                index=default_index,
                key=f"manual_{feature_name}"
            )
            # Konversi kembali ke nilai numerik untuk model
            return {v: k for k, v in options_dict.items()}[selected_display_value]

        # Loop melalui fitur yang diharapkan model untuk membuat input
        for feature in model_expected_features:
            with current_cols[col_idx % cols_per_row]:
                if feature == 'Marital_status':
                    input_data[feature] = create_categorical_input(feature, MARITAL_STATUS_OPTIONS, 1, current_cols[col_idx % cols_per_row])
                elif feature == 'Application_mode':
                    # Angka dari 1 sampai 57, terlalu banyak untuk dropdown yang eksplisit
                    # Mungkin better sebagai number input, atau hanya list pilihan yang sering muncul
                    # Untuk prototype, number input dengan help text bisa jadi solusi
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=1, max_value=57, value=1, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == 'Application_order':
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0, max_value=9, value=0, step=1, key=f"manual_{feature}")
                elif feature == 'Course':
                    # Sama seperti Application Mode, Course memiliki banyak ID
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=33, max_value=9991, value=9119, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == 'Daytime/evening_attendance':
                    input_data[feature] = create_categorical_input(feature, DAYTIME_EVENING_OPTIONS, 1, current_cols[col_idx % cols_per_row])
                elif feature == 'Previous_qualification':
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=1, max_value=43, value=1, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == 'Previous_qualification_(grade)': # Perhatikan underscore/spasi
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, max_value=200.0, value=120.0, step=0.1, key=f"manual_{feature}")
                elif feature == 'Nacionality':
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=1, max_value=109, value=1, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == "Mother's_qualification":
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=1, max_value=44, value=1, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == "Father's_qualification":
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=1, max_value=44, value=1, step=1, 
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == "Mother's_occupation":
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=0, max_value=194, value=99, step=1, # 99 is blank
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == "Father's_occupation":
                    input_data[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}:", 
                        min_value=0, max_value=195, value=99, step=1, # 99 is blank
                        help="Lihat 'Format Data yang Diharapkan' di bawah untuk daftar kode.",
                        key=f"manual_{feature}"
                    )
                elif feature == 'Admission_grade': # Sudah ada di capping_features
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, max_value=200.0, value=120.0, step=0.1, key=f"manual_{feature}")
                elif feature == 'Displaced':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 0, current_cols[col_idx % cols_per_row])
                elif feature == 'Educational_special_needs':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 0, current_cols[col_idx % cols_per_row])
                elif feature == 'Debtor':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 0, current_cols[col_idx % cols_per_row])
                elif feature == 'Tuition_fees_up_to_date':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 1, current_cols[col_idx % cols_per_row])
                elif feature == 'Gender':
                    input_data[feature] = create_categorical_input(feature, GENDER_OPTIONS, 1, current_cols[col_idx % cols_per_row])
                elif feature == 'Scholarship_holder':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 0, current_cols[col_idx % cols_per_row])
                elif feature == 'Age_at_enrollment': # Sudah ada di capping_features
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=15, max_value=80, value=20, key=f"manual_{feature}")
                elif feature == 'International':
                    input_data[feature] = create_categorical_input(feature, YES_NO_OPTIONS, 0, current_cols[col_idx % cols_per_row])
                elif feature == 'Curricular_units_1st_sem_credited':
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0, max_value=30, value=0, step=1, key=f"manual_{feature}")
                elif feature == 'Curricular_units_1st_sem_enrolled':
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0, max_value=30, value=6, step=1, key=f"manual_{feature}")
                elif feature == 'Curricular_units_1st_sem_evaluations':
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0, max_value=30, value=6, step=1, key=f"manual_{feature}")
                elif feature == 'Curricular_units_1st_sem_approved':
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0, max_value=30, value=5, step=1, key=f"manual_{feature}")
                elif feature == 'Curricular_units_2nd_sem_grade': # Sudah ada di capping_features
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}:", min_value=0.0, max_value=200.0, value=120.0, step=0.1, key=f"manual_{feature}")
                else:
                    # Generic number input for any other numeric feature not explicitly listed
                    input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()} (Other Numeric):", value=0.0, key=f"manual_{feature}")
            col_idx += 1
        
        submitted = st.form_submit_button("Dapatkan Prediksi")

    if submitted:
        input_df = pd.DataFrame([input_data])
        
        with st.spinner("Memprediksi status siswa..."):
            # Preprocessing satu baris input
            # contains_status_col=False karena ini adalah data baru, tidak berlabel
            df_processed_single, _, _ = preprocess_data(input_df, contains_status_col=False)
            
            if df_processed_single.empty:
                st.error("❌ Preprocessing menghasilkan data kosong. Silakan periksa nilai input Anda.")
            else:
                # Buat prediksi (tanpa status asli)
                single_prediction_results, _ = make_predictions(df_processed_single, rf_model, dt_model, include_real_status=False)
            
                if single_prediction_results is not None:
                    st.success("✅ Prediksi selesai!")
                    st.subheader("🔮 Status Siswa yang Diprediksi")
                    
                    rf_pred_val = single_prediction_results['Random_Forest_Prediction'].iloc[0]
                    dt_pred_val = single_prediction_results['Decision_Tree_Prediction'].iloc[0]

                    st.write(f"**Random Forest Memprediksi:** <span style='font-size: 24px; color: {'#155724' if rf_pred_val == 'Graduate' else '#721c24' if rf_pred_val == 'Dropout' else '#856404'};'>**{rf_pred_val}**</span>", unsafe_allow_html=True)
                    st.write(f"**Decision Tree Memprediksi:** <span style='font-size: 24px; color: {'#155724' if dt_pred_val == 'Graduate' else '#721c24' if dt_pred_val == 'Dropout' else '#856404'};'>**{dt_pred_val}**</span>", unsafe_allow_html=True)
                    
                    if rf_pred_val == dt_pred_val:
                        st.info("Kedua model sepakat dalam prediksi!")
                    else:
                        st.warning("Model memberikan prediksi yang berbeda. Pertimbangkan untuk meninjau input.")

                    with st.expander("Lihat Data Input yang Diproses untuk Prediksi"):
                        st.dataframe(df_processed_single)
else: # prediction_mode == "Prediksi untuk Sekelompok Siswa (Unggah CSV)"
    st.header("📤 Unggah CSV untuk Prediksi Sekelompok Siswa")
    st.info("Unggah file CSV yang mengandung fitur siswa (tanpa kolom 'Status') untuk mendapatkan prediksi untuk beberapa siswa.")

    uploaded_file_batch = st.file_uploader(
        "Pilih file CSV untuk prediksi sekelompok siswa:",
        type=['csv'],
        help="Unggah file CSV yang mengandung fitur siswa. JANGAN sertakan kolom 'Status' dalam file ini."
    )
    
    if uploaded_file_batch is not None:
        try:
            df_batch_raw = pd.read_csv(uploaded_file_batch, low_memory=False)
            st.success(f"✅ File berhasil diunggah: {len(df_batch_raw)} baris.")

            if 'Status' in df_batch_raw.columns:
                st.warning("⚠️ CSV yang diunggah mengandung kolom 'Status'. Kolom ini akan diabaikan untuk prediksi karena mode ini untuk inferensi data baru yang tidak berlabel.")
                # Hapus kolom 'Status' secara eksplisit untuk inferensi
                df_batch_raw = df_batch_raw.drop(columns=['Status'])

            with st.spinner("Memproses data sekelompok siswa dan membuat prediksi..."):
                # Preprocessing data sekelompok siswa
                df_processed_batch, _, _ = preprocess_data(df_batch_raw, contains_status_col=False)
                
                if df_processed_batch.empty:
                    st.error("❌ Preprocessing menghasilkan data kosong setelah menghilangkan NaN. Silakan periksa kualitas data file yang Anda unggah.")
                else:
                    # Buat prediksi sekelompok siswa (tanpa status asli)
                    batch_prediction_results, _ = make_predictions(df_processed_batch, rf_model, dt_model, include_real_status=False)

            if batch_prediction_results is not None:
                st.success(f"✅ Prediksi sekelompok siswa selesai untuk {len(batch_prediction_results)} siswa!")
                
                st.subheader("📋 Hasil Prediksi Sekelompok Siswa")
                # Tampilkan fitur input bersama dengan prediksi
                final_results_df = df_batch_raw.copy()
                final_results_df['Random_Forest_Prediction'] = batch_prediction_results['Random_Forest_Prediction']
                final_results_df['Decision_Tree_Prediction'] = batch_prediction_results['Decision_Tree_Prediction']

                # Sorot prediksi dalam dataframe
                def highlight_batch_predictions(row):
                    colors = [''] * len(row) # Default tanpa warna
                    for i, col_name in enumerate(row.index):
                        if col_name == 'Random_Forest_Prediction':
                            if row[col_name] == 'Graduate': colors[i] = 'background-color: #d4edda; color: #155724'
                            elif row[col_name] == 'Dropout': colors[i] = 'background-color: #f8d7da; color: #721c24'
                            elif row[col_name] == 'Enrolled': colors[i] = 'background-color: #fff3cd; color: #856404'
                        elif col_name == 'Decision_Tree_Prediction':
                            if row[col_name] == 'Graduate': colors[i] = 'background-color: #d4edda; color: #155724'
                            elif row[col_name] == 'Dropout': colors[i] = 'background-color: #f8d7da; color: #721c24'
                            elif row[col_name] == 'Enrolled': colors[i] = 'background-color: #fff3cd; color: #856404'
                    return colors

                st.dataframe(final_results_df.style.apply(highlight_batch_predictions, axis=1), use_container_width=True)
                
                # Visualisasi
                create_visualizations(batch_prediction_results)
                
                st.subheader("📊 Ringkasan Prediksi Sekelompok Siswa")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Prediksi Random Forest:**")
                    rf_counts = batch_prediction_results['Random_Forest_Prediction'].value_counts().reindex(['Graduate', 'Dropout', 'Enrolled'], fill_value=0)
                    for status, count in rf_counts.items():
                        st.metric(f"RF - {status}", count)
                with col2:
                    st.write("**Prediksi Decision Tree:**")
                    dt_counts = batch_prediction_results['Decision_Tree_Prediction'].value_counts().reindex(['Graduate', 'Dropout', 'Enrolled'], fill_value=0)
                    for status, count in dt_counts.items():
                        st.metric(f"DT - {status}", count)
                
                # Unduh hasil
                csv_buffer = io.StringIO()
                final_results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                st.download_button(
                    label="📥 Unduh Hasil Prediksi Sekelompok Siswa sebagai CSV",
                    data=csv_data,
                    file_name=f"student_status_batch_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error saat memproses file yang diunggah: {str(e)}")
            st.warning("Pastikan file CSV Anda diformat dengan benar dan mengandung fitur yang diharapkan.")
            
    # Tampilkan informasi tentang format data yang diharapkan untuk inferensi
    st.markdown("---")
    st.header("📝 Format Data yang Diharapkan untuk Prediksi Baru")
    st.markdown("""
    Saat memberikan data baru (baik secara manual atau melalui unggahan CSV) untuk prediksi, input Anda **TIDAK BOLEH** mengandung kolom 'Status'.
    
    Model mengharapkan fitur numerik berikut:
    """)
    
    st.subheader("Fitur Penting yang Diproses (Capping + Standarisasi):")
    st.markdown("""
    - **`Age_at_enrollment`**: Usia siswa saat pendaftaran (numerik)
    - **`Admission_grade`**: Nilai yang diperoleh saat pendaftaran (numerik, 0-200)
    - **`Curricular_units_1st_sem_grade`**: Nilai semester pertama (numerik, 0-200)
    - **`Previous_qualification_grade`**: Nilai kualifikasi sebelumnya (numerik, 0-200)
    - **`Course`**: Pengidentifikasi mata kuliah (numerik, lihat daftar kode di bawah)
    - **`Curricular_units_2nd_sem_grade`**: Nilai semester kedua (numerik, 0-200)
    """)

    st.subheader("Fitur Kategorikal (dengan kode numerik):")
    st.markdown("""
    - **`Marital_status`**: (1-6) 
        1 - single, 2 - married, 3 - widower, 4 - divorced, 5 - facto union, 6 - legally separated
    - **`Application_mode`**: (1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57)
    - **`Daytime/evening_attendance`**: (0-1) 
        0 - evening, 1 - daytime
    - **`Previous_qualification`**: (1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43)
    - **`Nacionality`**: (1, 2, 6, 11, 13, 14, 17, 21, 22, 24, 25, 26, 32, 41, 62, 100, 101, 103, 105, 108, 109)
    - **`Mother's_qualification`**: (Kode numerik, 1-44)
    - **`Father's_qualification`**: (Kode numerik, 1-44)
    - **`Mother's_occupation`**: (Kode numerik, 0-194)
    - **`Father's_occupation`**: (Kode numerik, 0-195)
    - **`Displaced`**: (0-1) 
        0 - no, 1 - yes
    - **`Educational_special_needs`**: (0-1) 
        0 - no, 1 - yes
    - **`Debtor`**: (0-1) 
        0 - no, 1 - yes
    - **`Tuition_fees_up_to_date`**: (0-1) 
        0 - no, 1 - yes
    - **`Gender`**: (0-1) 
        0 - female, 1 - male
    - **`Scholarship_holder`**: (0-1) 
        0 - no, 1 - yes
    - **`International`**: (0-1) 
        0 - no, 1 - yes
    """)

    st.subheader("Fitur Numerik Lain:")
    st.markdown("""
    - **`Application_order`**: (numerik, 0-9)
    - **`Curricular_units_1st_sem_credited`**: (numerik, 0-30)
    - **`Curricular_units_1st_sem_enrolled`**: (numerik, 0-30)
    - **`Curricular_units_1st_sem_evaluations`**: (numerik, 0-30)
    - **`Curricular_units_1st_sem_approved`**: (numerik, 0-30)
    """)
    
    st.markdown("""
    **Catatan Penting:**
    - Semua fitur input harus berupa numerik (bahkan untuk kategori, gunakan kodenya).
    - Nilai yang hilang dalam fitur input akan dihilangkan (baris yang mengandung NaN). Pastikan kualitas data.
    - Output prediksi akan menjadi salah satu dari: **Graduate** (Lulus), **Dropout** (Drop Out), atau **Enrolled** (Terdaftar).
    """)
    
    # Tampilkan daftar fitur yang diharapkan dari model (untuk debugging/referensi)
    if model_expected_features:
        st.subheader("Daftar Lengkap Fitur yang Diharapkan Model:")
        st.code(str(model_expected_features)) # Tampilkan sebagai string untuk menghindari pemformatan list yang panjang
    else:
        st.warning("Tidak dapat menentukan fitur yang diharapkan model.")


# Footer
st.markdown("---")
st.markdown("*Prototipe Prediksi Status Siswa - Model Random Forest & Decision Tree*")
