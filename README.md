# Perbandingan Metode Hyperparameter Tuning LightGBM untuk Prediksi Konsumsi Energi pada Smart Grid

Repository ini berisi implementasi, dokumentasi, dataset pendukung, hasil eksperimen, dan implementasi model dari penelitian mengenai perbandingan metode *hyperparameter tuning* pada Light Gradient Boosting Machine (LightGBM) untuk prediksi konsumsi energi pada Smart Grid.

---

## 1. Deskripsi Penelitian

Penelitian ini berfokus pada prediksi konsumsi energi menggunakan model LightGBM dengan memanfaatkan data konsumsi energi dan variabel pendukung dari dataset HEAPO.

Data yang digunakan memiliki karakteristik *time series*, sehingga tahapan pengolahan data mempertahankan urutan waktu. Sebelum pemodelan, data melalui proses *data understanding*, *data integration*, Exploratory Data Analysis (EDA), preprocessing, dan *feature engineering*.

LightGBM digunakan sebagai model baseline. Selanjutnya, dilakukan optimasi *hyperparameter* menggunakan tiga metode, yaitu:

- Random Search
- Bayesian Optimization
- Optuna (TPE)

Perbandingan metode dilakukan berdasarkan performa prediksi, stabilitas model, dan waktu komputasi.

---

## 2. Tujuan Penelitian

Penelitian ini bertujuan untuk:

1. Membangun model LightGBM baseline untuk memprediksi konsumsi energi pada Smart Grid.
2. Membandingkan metode Random Search, Bayesian Optimization, dan Optuna dalam mengoptimalkan *hyperparameter* LightGBM.
3. Menentukan metode *hyperparameter tuning* yang memberikan hasil paling sesuai berdasarkan performa prediksi, stabilitas model, dan waktu komputasi.

---

## 3. Dataset

Penelitian menggunakan dataset HEAPO sebagai sumber data utama.

Dataset HEAPO merupakan dataset terbuka yang menyediakan data konsumsi listrik rumah tangga yang menggunakan *heat pump* dan *smart electricity meter*.

Dataset asli tidak disimpan langsung di repository karena ukuran data yang besar.

### Sumber Resmi Dataset

**HEAPO GitHub Repository**

https://github.com/tbrumue/heapo

**HEAPO Dataset – Zenodo**

https://zenodo.org/records/15056919

Informasi lebih lanjut mengenai dataset dapat dilihat pada dokumentasi di:

`data/raw/README.md`

---

## 4. Alur Penelitian

Alur penelitian dilakukan secara sistematis mulai dari pengumpulan dataset HEAPO hingga penentuan metode *hyperparameter tuning* terbaik.

![Alur Penelitian](docs/flowchart-alur-penelitian.png)

### Tahapan Penelitian

```text
MULAI
  │
  ▼
Pengumpulan Dataset HEAPO
  │
  ▼
Data Understanding
  │
  ▼
Data Integration
  │
  ▼
Exploratory Data Analysis (EDA)
  │
  ▼
Preprocessing Data
  │
  ├── Optimasi Tipe Data
  ├── Penanganan Missing Value
  ├── Penghapusan Fitur Tidak Digunakan
  ├── Encoding Variabel Kategorikal
  └── Pengurutan Berdasarkan Waktu
  │
  ▼
Feature Engineering
  │
  ├── Temporal Feature
  ├── Cyclical Encoding
  ├── Lag Feature
  └── Rolling Statistics
  │
  ▼
Dataset Final
  │
  ▼
Time Based Split
  │
  ├── Training Set (70%)
  ├── Validation Set
  └── Test Set
  │
  ▼
Model LightGBM Baseline
  │
  ▼
Evaluasi Baseline
  │
  ▼
Hyperparameter Tuning
  │
  ├── Random Search
  ├── Bayesian Optimization
  └── Optuna (TPE)
  │
  ▼
Evaluasi Model Hyperparameter Tuning
  │
  ├── MAE
  ├── RMSE
  ├── SMAPE
  ├── R²
  ├── Training Time
  └── Tuning Time
  │
  ▼
Time Series Cross Validation
  │
  ▼
Feature Importance
  │
  ▼
Perbandingan Hasil Model
  │
  ▼
Kesimpulan dan Rekomendasi
  │
  ▼
SELESAI
