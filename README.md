# Perbandingan Metode Hyperparameter Tuning LightGBM untuk Prediksi Konsumsi Energi pada Smart Grid

Repository ini berisi kode, dataset pendukung, hasil eksperimen, dan dokumentasi penelitian mengenai perbandingan metode *hyperparameter tuning* pada model Light Gradient Boosting Machine (LightGBM) untuk prediksi konsumsi energi pada Smart Grid.

## Deskripsi Penelitian

Penelitian ini menggunakan dataset HEAPO sebagai sumber data konsumsi energi rumah tangga. Data kemudian melalui tahapan *data understanding*, *data integration*, *exploratory data analysis*, *preprocessing*, *feature engineering*, dan *data splitting* sebelum digunakan dalam pemodelan.

Model LightGBM digunakan sebagai model baseline. Selanjutnya dilakukan optimasi *hyperparameter* menggunakan tiga metode, yaitu:

- Random Search
- Bayesian Optimization
- Optuna (TPE)

Performa model dievaluasi menggunakan:

- MAE (*Mean Absolute Error*)
- RMSE (*Root Mean Squared Error*)
- SMAPE (*Symmetric Mean Absolute Percentage Error*)
- R² (*Coefficient of Determination*)
- 5-fold Cross-Validation
- Waktu komputasi

Hasil penelitian menunjukkan bahwa ketiga metode optimasi meningkatkan performa LightGBM dibandingkan model baseline. Berdasarkan hasil *cross-validation*, performa prediksi, stabilitas, dan waktu *tuning*, Random Search dipilih sebagai metode *hyperparameter tuning* terbaik.

## Dataset

Penelitian menggunakan dataset **HEAPO**.

Dataset asli tidak disimpan langsung di repository karena ukuran file yang besar. Informasi sumber dan akses dataset tersedia pada:

**[Data Raw – HEAPO](data/raw/)**

Sumber resmi:

- [HEAPO GitHub Repository](https://github.com/tbrumue/heapo)
- [HEAPO Dataset – Zenodo](https://zenodo.org/records/15056919)

Dataset hasil preprocessing dan *feature engineering* tersedia secara dokumentatif pada:

**[Data Processed](data/Processed/)**

## Metode Penelitian

Alur utama penelitian:

1. Data Understanding
2. Data Integration
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Data Splitting
7. LightGBM Baseline
8. Hyperparameter Tuning
   - Random Search
   - Bayesian Optimization
   - Optuna (TPE)
9. Evaluasi Model
10. Analisis Stabilitas dengan 5-fold Cross-Validation
11. Analisis Feature Importance
12. Implementasi Model

## Hasil Utama

Model LightGBM baseline pada data *test* menghasilkan:

| Metrik | Baseline |
|---|---:|
| MAE | 0,187378 |
| RMSE | 0,272954 |
| SMAPE | 60,542612% |
| R² | 0,624751 |

Hasil *5-fold cross-validation* menunjukkan bahwa Random Search menghasilkan performa rata-rata terbaik:

| Metrik | Random Search |
|---|---:|
| MAE | 0,1212 |
| RMSE | 0,2138 |
| SMAPE | 52,1873% |
| R² | 0,7224 |

Waktu *tuning* Random Search sebesar **6.409,14 detik**.

Berdasarkan hasil evaluasi, Random Search dipilih sebagai metode *hyperparameter tuning* terbaik dalam penelitian ini.

## Struktur Repository

```text
.
├── data/
│   ├── raw/
│   │   └── README.md
│   │
│   └── Processed/
│       └── README.md
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_modeling_baseline.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_evaluation.ipynb
│
├── models/
│
├── results/
│
├── app/
│
├── README.md
└── requirements.txt
