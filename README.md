# 📘 CompVis Course Projects

Proyek ini merupakan bagian dari pembelajaran Computer Vision dan Machine Learning yang terbagi menjadi dua bagian utama, masing-masing menggunakan model pembelajaran mesin klasik. Fokus utama diletakkan pada penerapan **K-Nearest Neighbors (KNN)** untuk klasifikasi citra dan **Naive Bayes** untuk klasifikasi data tabular.

## 🔍 Proyek

### 🦋 Butterfly Image Classification
- **Deskripsi:**  
  Klasifikasi jenis kupu-kupu berdasarkan gambar.
- **Model:**  
  Menggunakan algoritma **K-Nearest Neighbors (KNN)** untuk mengklasifikasikan gambar yang telah diekstraksi fitur-fiturnya.
- **Proses:**
  - Preprocessing gambar (resize dan ekstraksi fitur Color Moments).
  - Pembagian dataset menjadi train-test (contoh: 70:30).
  - Training dan evaluasi model KNN dengan nilai _k_ (jumlah tetangga terdekat).
- **Hasil Terbaik:**  
  Model KNN dengan _k = 3_ dan rasio train-test 0.7:0.3 menghasilkan akurasi tertinggi sebesar **0.98**.
- **Repo:**  
  🔗 [Butterfly Classification](https://github.com/nabilsaragih/compvis-course/tree/main/image-classification)

---

### ❤️ Heart Failure Prediction
- **Deskripsi:**  
  Prediksi kemungkinan gagal jantung berdasarkan data kesehatan pasien.
- **Model:**  
  Menggunakan algoritma **Naive Bayes** untuk klasifikasi data tabular.
- **Proses:**
  - Exploratory Data Analysis (EDA) dan preprocessing data (handling missing values, encoding, normalization).
  - Pembagian data menjadi train-test.
  - Training model Naive Bayes dan evaluasi performa (akurasi, precision, recall, confusion matrix).
- **Tujuan:**  
  Memberikan sistem prediksi awal risiko gagal jantung menggunakan pendekatan statistik sederhana namun efektif.
- **Repo:**  
  🔗 [Heart Failure Prediction](https://github.com/nabilsaragih/compvis-course/tree/main/tabular-classification)

---

## 🛠️ Tools & Libraries
- Python (Jupyter Notebook)
- Scikit-learn
- OpenCV / PIL (untuk image preprocessing)
- Pandas & NumPy
- Matplotlib / Seaborn

---

## 🎯 Tujuan Pembelajaran
- Memahami penerapan supervised learning pada dua jenis data yang berbeda.
- Mempelajari bagaimana preprocessing berbeda antara data citra dan data tabular.
- Menganalisis performa model KNN dan Naive Bayes dalam konteks yang sesuai.

---

## 📈 Hasil dan Analisis
- **KNN untuk Citra:**  
  Performa sangat baik dengan data citra terstruktur, terutama setelah preprocessing fitur visual.
- **Naive Bayes untuk Tabular:**  
  Cepat dan efisien, cocok untuk dataset kecil-menengah dengan asumsi distribusi probabilistik yang sesuai.
