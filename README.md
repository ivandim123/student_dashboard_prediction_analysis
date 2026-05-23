# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah institusi pendidikan perguruan tinggi yang telah beroperasi selama lebih dari 23 tahun sejak didirikan pada tahun 2000. Selama periode tersebut, institusi ini telah membangun reputasi yang solid dalam menghasilkan lulusan berkualitas tinggi yang diakui di dunia kerja.  
Institut ini telah berhasil mencetak banyak alumni dengan reputasi sangat baik, menunjukkan kualitas pendidikan dan program akademik yang mereka tawarkan. Hal ini mencerminkan komitmen institusi terhadap excellence dalam pendidikan tinggi.  
  

### Permasalahan Bisnis
Meskipun memiliki track record yang baik dalam menghasilkan lulusan berkualitas, Jaya Jaya Institut menghadapi tantangan serius berupa tingginya angka dropout atau siswa yang tidak menyelesaikan pendidikan mereka.

### Cakupan Proyek
Di sini kita akan membuat dashboard untuk evaluasi berbagai parameter dan juga membuat model ML untuk memprediksi siswa yang akan dropout.

### Persiapan

Sumber data: [Jaya Jaya Institut Siswa](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)
Acknowledgements: [UCI Machine Learning Repository](https://doi.org/10.24432/C5MC89)
Deskripsi umum dataset: Sebuah dataset yang dibuat dari institusi pendidikan tinggi (diperoleh dari beberapa database terpisah) yang berkaitan dengan mahasiswa yang terdaftar dalam berbagai program sarjana, seperti agronomi, desain, pendidikan, keperawatan, jurnalisme, manajemen, layanan sosial, dan teknologi. Dataset ini mencakup informasi yang diketahui pada saat pendaftaran mahasiswa (jalur akademik, demografi, dan faktor sosial-ekonomi) serta kinerja akademik mahasiswa pada akhir semester pertama dan kedua. Data ini digunakan untuk membangun model klasifikasi untuk memprediksi dropout dan kesuksesan akademik mahasiswa.

Setup environment:
- Anaconda

```
conda create --name main-ds python=3.12
conda activate main-ds
pip install -r requirements.txt
```

- Shell/Terminal

```
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```


## Business Dashboard
Dashboard yang sudah saya buat memiliki beberapa fitur sebagai berikut:
- Terdapat judul proyek dan beberapa info jika flow dari akses dataset berhasil
- Di sini digunakan 2 buah plot sekaligus yang dipisahkan berdasarkan kolom attrition karena kita ingin menganalisis perbedaan antara karyawan attrition dengan yang tidak
- Plot dapat diubah-ubah berdasarkan filter yang diatur dari sidebar
- Plot akan beradaptasi sesuai jenis fitur yang dipilih, jika dipilih fitur numerik, akan diplot bar grafik dengan line trend, sedangkan untuk fitur kategorikal, akan diplot bar grafik berdasarkan unique value dari masing-masing fitur
- Ada fitur untuk menampilkan raw data dalam tabel (opsional)
- Di bagian bawah juga ada fitur statistik deskriptif dari plot data yang dipilih. Ini juga akan auto adjust jika filter diubah
- Pada sidebar terdapat beberapa info tambahan dan debug info  

Setelah menganalisis dashboard yang telah dibuat, berikut insight dari saya:
- Siswa yang DO banyak dari yang mahasiswa di atas 23 tahun, Ordinance No. 612/93, dan Fase pendaftaran umum yang ketiga
- Siswa yang DO banyak dari yang mahasiswa yang ayahnya masuk kualifikasi bisa baca tanpa pendidikan formal kelas 4, pendidikan dasar siklus 3 (kelas 9/10/11), dan sarjana
- Siswa yang DO banyak dari yang mahasiswa yang menunggak membayar biaya kuliah

Di sini saya menggunakan streamlit untuk membuat dashboard. Saya juga sudah deploy aplikasi dashboard ini ke streamlit dengan integrasi bersama github.  
Berikut link github dari aplikasi ini:  
[Github Dashboard](https://github.com/ivandim123/student_dashboard_prediction_analysis/)  
Berikut link dashboard dari aplikasi ini:  
[Streamlit Dashboard](https://studentdashboardpredictionanalysis-qy7rbnkmrwubma9tc2cjlk.streamlit.app/)  
Berikut link prediction dari aplikasi ini:  
[Streamlit Prediction](https://studentdashboardpredictionanalysis-qmvfjr7rewmeybhmj5tjxf.streamlit.app/)

## Menjalankan Sistem Machine Learning

🧭 Cara Menjalankan Prototype
1. Akses Aplikasi
Buka aplikasi melalui link:
📍 https://hwnzgahtjf9hetn29ennzf.streamlit.app/
atau bisa run saja secara lokal dengan: streamlit run prediksi.py

2. Pilih Sumber Data
Di sidebar sebelah kiri:

Pilih "Default Dataset (data.csv)" untuk memakai data bawaan, atau

Pilih "Upload New CSV" untuk mengunggah data Anda sendiri.

3. Lakukan Prediksi
Data yang dipilih akan diproses dan dibersihkan.

Model Random Forest dan Decision Tree akan membuat prediksi untuk beberapa sampel mahasiswa (default: 10 data).

Hasil prediksi ditampilkan dalam bentuk tabel dan grafik.

4. Visualisasi
Aplikasi menampilkan beberapa visual:

📊 Distribution of Predicted Status per model.

🎯 Agreement antara dua model (berapa banyak prediksi yang sama).

📈 Distribusi Status Asli dari dataset yang digunakan.

📁 Struktur Model & Data
Model: random_forest_model.pkl, decision_tree_model.pkl

Dataset Default: data.csv

Semuanya berada dalam folder submission/.

🛠️ Teknologi yang Digunakan
Bahasa Pemrograman: Python

Framework UI: Streamlit

Visualisasi: Plotly

Model ML: RandomForestClassifier, DecisionTreeClassifier dari scikit-learn

Preprocessing: StandardScaler, Outlier Capping

## Conclusion

Proyek ini berhasil mengidentifikasi dan menganalisis faktor-faktor yang berkontribusi terhadap tingginya angka dropout mahasiswa di Jaya Jaya Institut melalui pendekatan data science. Dengan menggabungkan business dashboard interaktif berbasis Streamlit dan model machine learning (Random Forest dan Decision Tree), institusi kini memiliki alat yang powerful untuk:  

- Mengevaluasi karakteristik demografis, sosial ekonomi, dan akademik mahasiswa.
- Mengidentifikasi pola dropout berdasarkan atribut seperti usia, status pendidikan orang tua, dan keterlambatan pembayaran biaya kuliah.
- Melakukan prediksi status mahasiswa (Dropout, Enrolled, Graduate) berdasarkan data historis dengan akurasi yang cukup tinggi.  

Berdasarkan analisis yang mendalam, faktor-faktor utama yang paling berpengaruh terhadap tingginya angka dropout adalah usia mahasiswa, tingkat pendidikan orang tua (khususnya ayah), dan status pembayaran biaya kuliah.  

Karakteristik umum mahasiswa yang cenderung dropout adalah sebagai berikut:  

- Usia: Mahasiswa yang berusia di atas 23 tahun memiliki kecenderungan lebih tinggi untuk dropout.
- Latar Belakang Pendidikan Orang Tua: Mahasiswa yang ayahnya memiliki kualifikasi pendidikan formal rendah (misalnya, hanya bisa membaca tanpa pendidikan formal kelas 4) atau sebaliknya, pendidikan tinggi (pendidikan dasar siklus 3 atau sarjana), menunjukkan korelasi dengan tingkat dropout. Ini menunjukkan bahwa ekstremitas dalam latar belakang pendidikan orang tua dapat menjadi faktor risiko.
- Status Keuangan: Mahasiswa yang menunggak pembayaran biaya kuliah sangat berisiko untuk dropout.
- Jalur Pendaftaran: Mahasiswa yang masuk melalui jalur Ordinance No. 612/93 dan pada fase pendaftaran umum ketiga juga menunjukkan tingkat dropout yang tinggi.  

Model Random Forest terbukti menjadi model terbaik dengan F1-score sebesar 0.7613, dan performa recall sebesar 0.764 untuk kelas Dropout, yang artinya model ini cukup andal dalam mendeteksi mahasiswa yang berpotensi keluar dari institusi.

Secara keseluruhan, proyek ini memberikan dasar analitik yang kuat untuk pengambilan keputusan berbasis data demi menurunkan tingkat dropout di masa mendatang dengan fokus pada faktor-faktor risiko yang telah teridentifikasi.


### Rekomendasi Action Items
Untuk mengurangi angka dropout dan meningkatkan retensi mahasiswa, berikut beberapa rekomendasi strategis yang dapat dilakukan oleh Jaya Jaya Institut:

- Implementasi Sistem Monitoring Dini
Terapkan sistem berbasis model Random Forest untuk mendeteksi mahasiswa berisiko tinggi dropout sejak awal semester. Mahasiswa yang terdeteksi dapat langsung dimasukkan ke dalam program pendampingan khusus.

- Intervensi Sosial dan Finansial Terarah
Berikan bantuan atau skema beasiswa bagi mahasiswa yang memiliki keterlambatan pembayaran biaya kuliah atau berasal dari latar belakang ekonomi rentan.

- Peningkatan Komunikasi Orang Tua/Wali
Berdasarkan analisis, latar belakang pendidikan orang tua berpengaruh signifikan. Program komunikasi aktif dengan orang tua atau wali bisa membantu mahasiswa mendapatkan dukungan lebih besar dari rumah.

- Evaluasi Jalur dan Fase Pendaftaran
Tinjau ulang proses pendaftaran, khususnya pada fase ketiga dan jalur Ordinance No. 612/93, yang menunjukkan korelasi tinggi terhadap risiko dropout.

- Integrasi Dashboard ke Sistem Akademik
Integrasikan dashboard evaluasi ke sistem internal institusi agar tim akademik dan konselor mahasiswa dapat terus memantau kondisi dan performa mahasiswa secara real-time.
