# Raw Dataset – HEAPO

Folder ini berisi informasi mengenai dataset mentah HEAPO
(Heat Pump Optimization), yaitu dataset yang digunakan sebagai
sumber data awal dalam penelitian prediksi konsumsi energi pada
Smart Grid menggunakan LightGBM.

## Sumber Dataset

Dataset HEAPO merupakan dataset terbuka yang dikembangkan oleh
Tobias Brudermüller, Elgar Fleisch, Marina González Vayá, dan
Thorsten Staake.

Sumber resmi dataset:

- GitHub:
  https://github.com/tbrumue/heapo

- Zenodo:
  https://zenodo.org/records/15056919

- DOI:
  https://doi.org/10.5281/zenodo.15056919

## Deskripsi Dataset

HEAPO merupakan dataset terbuka yang menyediakan data konsumsi
listrik dari 1.408 rumah tangga yang menggunakan heat pump dan
smart electricity meter di Canton of Zurich, Switzerland.

Data konsumsi tersedia pada resolusi:

- 15 menit
- Harian

Periode pengamatan:

3 November 2018 – 21 Maret 2024

Dataset juga dilengkapi dengan:

- Household metadata
- Data cuaca dari 8 stasiun
- Data konsumsi listrik
- Data terkait heat pump
- Ground truth dari 410 field visit protocols
- Python-based data loader untuk membantu proses pengolahan
  dan eksplorasi data

## Ukuran Dataset

Pada sumber resmi GitHub, dataset HEAPO tersedia dalam bentuk
arsip `heapo_data.zip` dengan ukuran sekitar 485 MB dan akan
berukuran sekitar 5,26 GB setelah diekstraksi.

Pada Zenodo, dataset tersedia sebagai:

`heapo_data.zip`

dengan ukuran sekitar 458,3 MB.

## Penggunaan dalam Penelitian

Dataset HEAPO digunakan sebagai sumber data awal penelitian.
Data mentah kemudian melalui beberapa tahapan pengolahan,
meliputi:

1. Data Understanding
2. Data Integration
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Data Splitting
7. Pemodelan LightGBM
8. Hyperparameter Tuning

Tahap pengolahan menghasilkan dataset yang digunakan untuk
training, validation, dan testing pada penelitian.

## Dataset Hasil Pengolahan

Dataset yang telah melalui proses preprocessing dan
feature engineering disimpan pada:

`data/Processed/`

Dataset hasil pengolahan tersebut mencakup data training,
validation, testing, fitur (X), dan target (y).

## Catatan

Dataset HEAPO asli tidak disimpan secara langsung di repository
penelitian karena ukuran dataset relatif besar.

Dataset dapat diperoleh melalui sumber resmi berikut:

- HEAPO GitHub Repository:
  https://github.com/tbrumue/heapo

- HEAPO Dataset – Zenodo:
  https://zenodo.org/records/15056919

Penggunaan dataset dalam penelitian ini mengacu pada sumber
resmi HEAPO dan publikasi dataset yang menyertainya.

## Sitasi Dataset

Brudermüller, T., Fleisch, E., González Vayá, M., & Staake, T.
(2025). HEAPO – An Open Dataset for Heat Pump Optimization with
Smart Electricity Meter Data and On-Site Inspection Protocols.
Zenodo.

DOI:
10.5281/zenodo.15056919
