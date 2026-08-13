# Processed Dataset

Folder ini berisi dataset hasil preprocessing, integrasi,
feature engineering, dan pembagian data yang digunakan
dalam penelitian prediksi konsumsi energi menggunakan LightGBM.

Karena ukuran dataset relatif besar, file dataset tidak
disimpan secara langsung di repository GitHub.

## Daftar Dataset

### Dataset Utama

- `master_dataset.parquet`
  Dataset utama hasil pengolahan data.

- `master_dataset_clean.parquet`
  Dataset utama setelah proses pembersihan data.

- `selected_households.csv`
  Daftar household yang digunakan dalam penelitian.

### Dataset Pembagian Data

- `train_dataset.parquet`
  Dataset training.

- `valid_dataset.parquet`
  Dataset validation.

- `test_dataset.parquet`
  Dataset testing.

### Dataset Final

- `train_final.parquet`
  Dataset final untuk proses training.

- `valid_final.parquet`
  Dataset final untuk proses validation.

- `test_final.parquet`
  Dataset final untuk proses testing.

### Dataset Fitur dan Target

- `X_train.parquet`
  Fitur/predictor untuk data training.

- `X_valid.parquet`
  Fitur/predictor untuk data validation.

- `X_test.parquet`
  Fitur/predictor untuk data testing.

- `y_train.parquet`
  Target untuk data training.

- `y_valid.parquet`
  Target untuk data validation.

- `y_test.parquet`
  Target untuk data testing.

## Akses Dataset

Dataset lengkap disimpan secara terpisah karena ukuran
file melebihi batas yang sesuai untuk repository GitHub.

Dataset dapat diakses melalui:

🔗 **https://drive.google.com/drive/folders/10a8tWZXBCcB4KtJA0qAPaRj0g7i0ft47?usp=sharing**

## Catatan

Dataset digunakan untuk penelitian:

**Prediksi Konsumsi Energi pada Smart Grid menggunakan
LightGBM dan Hyperparameter Tuning.**

Target prediksi:
`kWh_received_Total`

Model yang digunakan:
`LightGBM`

Metode hyperparameter tuning:
- Random Search
- Bayesian Optimization
- Optuna (TPE)
