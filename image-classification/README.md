# **Klasifikasi Spesies Kupu-Kupu**

## **Overview**
Proyek ini merupakan sistem klasifikasi gambar untuk mengidentifikasi spesies kupu-kupu, khususnya **Adonis**, **Clouded Sulphur**, dan **Scarce Swallow**. Sistem menggunakan teknik *machine learning* berbasis *image processing* untuk mengekstrak fitur warna dari gambar kupu-kupu dan memprediksi spesiesnya dengan algoritma K-Nearest Neighbors (KNN). Jenis kupu-kupu yang terdapat dalam dataset adalah:
![Spesies Kupu-Kupu](https://github.com/nabilsaragih/compvis-course/blob/main/image-classification/butterfly.png)

## **Fitur Utama**
- **Ekstraksi Fitur Warna**: Menggunakan *color moments* (mean, standar deviasi, dan skewness) dari gambar untuk representasi fitur.
- **Background Removal (Opsional)**: Dapat menghapus latar belakang gambar untuk fokus pada kupu-kupu saja.
- **Model KNN**: Menggunakan K-Nearest Neighbors untuk klasifikasi berdasarkan kemiripan fitur.

## **Cara Penggunaan**
1. **Prediksi Spesies**:
   ```python
   class_name, confidence, knn_model = predict_butterfly_class("path/to/image.jpg")
   print(f"Prediksi: {class_name}, Confidence: {confidence:.2f}")
   ```
2. **Pelatihan Model**:
   - Dataset harus dalam format yang sesuai (folder per kelas).
   - Ekstraksi fitur dilakukan menggunakan `ImageFeatureExtractor`.
   - Model disimpan dalam format `.joblib` untuk penggunaan selanjutnya.

## **Struktur Direktori**
```
project/
├── data/                # Dataset gambar kupu-kupu
├── model/               # Model dan labelmap
│   ├── butterfly_classifier.joblib
│   └── labelmap/
│       └── butterfly_classifier_labels.txt
└── temp_processed_images/  # Gambar sementara (jika background removal aktif)
```

## **Kebutuhan Sistem**
- Python 3.10+
- Library: `scikit-learn`, `Pillow`, `numpy`, `scipy`, `rembg` (untuk background removal), `joblib`

## **Catatan**
- Jika tidak memerlukan penghapusan latar belakang, set `remove_bg=False` pada `ImageFeatureExtractor`.
- Direktori `temp_processed_images` hanya diperlukan saat pelatihan dengan background removal.
- Dataset dapat diunduh dari [Butterfly Image Classification](https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification) atau [Butterfly Pattern](https://www.kaggle.com/datasets/fritze/butterfly-pattern)

Sistem ini cocok untuk identifikasi cepat spesies kupu-kupu berdasarkan fitur warna dan dapat dikembangkan untuk mendukung lebih banyak spesies.