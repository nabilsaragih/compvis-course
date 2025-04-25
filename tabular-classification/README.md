# Prediksi Penyakit Jantung

**Proyek ini** menggunakan *machine learning* untuk memprediksi risiko penyakit jantung berdasarkan data klinis pasien. Dibangun dengan Python dan *library* scikit-learn, model ini dapat membantu tenaga medis dalam melakukan penilaian awal pasien.

---

## **🔍 Deskripsi Projek**
Proyek ini bertujuan untuk memprediksi penyakit jantung menggunakan algoritma **Gaussian Naive Bayes** dengan *pipeline* preprocessing yang mencakup:
- **StandardScaler** → Normalisasi fitur numerik (usia, tekanan darah, kolesterol, dll.)
- **OneHotEncoder** → Mengubah fitur kategorikal (jenis kelamin, jenis nyeri dada, dll.)
- **PCA** → Reduksi dimensi untuk meningkatkan performa model
- **GaussianNB** → Model klasifikasi utama

---

## **📊 Struktur Data**
Dataset terdiri dari **fitur klinis pasien**, antara lain:
| Kolom            | Tipe Data  | Keterangan                          |
|------------------|------------|-------------------------------------|
| `Age`            | Numerik    | Usia pasien                         |
| `Sex`            | Kategorikal| Jenis kelamin (M/F)                 |
| `ChestPainType`  | Kategorikal| Jenis nyeri dada (ATA, NAP, dll.)   |
| `RestingBP`      | Numerik    | Tekanan darah istirahat             |
| `Cholesterol`    | Numerik    | Kadar kolesterol                    |
| `FastingBS`      | Numerik    | Gula darah puasa (0/1)              |
| `RestingECG`     | Kategorikal| Hasil EKG istirahat                 |
| `MaxHR`          | Numerik    | Denyut jantung maksimal             |
| `ExerciseAngina` | Kategorikal| Nyeri dada saat olahraga (Y/N)      |
| `Oldpeak`        | Numerik    | Depresi ST saat olahraga            |
| `ST_Slope`       | Kategorikal| Kemiringan segmen ST                |
| `HeartDisease`   | Target     | Diagnosis penyakit jantung (0/1)    |

---

## **🛠 Cara Menggunakan**
### **1. Instalasi**
Pastikan Python (≥3.10) dan *library* berikut terinstal:
```bash
pip install numpy pandas scikit-learn
```

### **2. Menjalankan Model**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

# Definisikan pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Preprocessing data
    ('pca', PCA(n_components=2)),   # Reduksi dimensi
    ('classifier', GaussianNB())     # Model prediksi
])

# Latih model
pipeline.fit(X_train, y_train)

# Prediksi
y_pred = pipeline.predict(X_test)
```

### **3. Evaluasi Model**
```python
from sklearn.metrics import accuracy_score, classification_report

print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```
**Contoh Output:**
```
Akurasi: 0.82
              precision    recall  f1-score   support
           0       0.85      0.78      0.81       100
           1       0.80      0.86      0.83       100
    accuracy                           0.82       200
```

---

## **📈 Hasil Evaluasi Model**
- **Akurasi**: 82% (dapat bervariasi tergantung data)
- **Kendala**:  
  - Jika akurasi tampak tidak konsisten, periksa:
    1. **Ketidakseimbangan kelas** (gunakan `class_weight` atau resampling)
    2. **Reduksi dimensi PCA** (coba tanpa PCA atau ubah `n_components`)
    3. **Preprocessing data** (pastikan tidak ada kebocoran data)

---

## **💻 Kebutuhan Sistem**
- **Python** 3.10+
- **Library**:
  - NumPy, Pandas
  - Scikit-learn
  - (Opsional) Matplotlib/Seaborn untuk visualisasi

---

## **🚀 Penyempurnaan di Masa Depan**
- [ ] **Optimasi Hyperparameter** (GridSearchCV untuk GaussianNB)
- [ ] **Coba Model Lain** (Random Forest, XGBoost)
- [ ] **Deploy sebagai Aplikasi Web** (Flask/Django)
- [ ] **Penanganan Data Tidak Seimbang** (SMOTE/Undersampling)

