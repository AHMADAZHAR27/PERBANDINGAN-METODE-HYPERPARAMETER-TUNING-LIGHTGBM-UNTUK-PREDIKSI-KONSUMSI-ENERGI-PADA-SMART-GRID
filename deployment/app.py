# ==========================================================
# SMART ENERGY AI
# Aplikasi Prediksi Konsumsi Energi Smart Grid
# Model final deployment: LightGBM + Random Search
# ==========================================================

import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ==========================================================
# 1. KONFIGURASI
# ==========================================================

st.set_page_config(
    page_title="Smart Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==========================================================
# 2. TEMA UI
# ==========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #05070b;
    --panel: #0c1118;
    --panel2: #101722;
    --border: #202b39;
    --muted: #718096;
    --text: #f8fafc;
    --cyan: #22d3ee;
    --blue: #3b82f6;
    --purple: #a855f7;
    --green: #4ade80;
    --yellow: #facc15;
}

.stApp {
    background:
        radial-gradient(circle at 85% 0%, rgba(37,99,235,.16), transparent 28%),
        radial-gradient(circle at 8% 20%, rgba(34,211,238,.07), transparent 23%),
        var(--bg);
    color: var(--text);
    font-family: Inter, sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding: 24px 34px 50px;
}

#MainMenu, footer {
    visibility: hidden;
}

/* ---------- header ---------- */

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
    padding: 15px 0 18px;
    border-bottom: 1px solid #1b2430;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 43px;
    height: 43px;
    border-radius: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    box-shadow: 0 12px 35px rgba(37,99,235,.25);
    font-size: 21px;
}

.brand-name {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 800;
}

.brand-sub {
    color: #64748b;
    font-size: 9px;
    margin-top: 3px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 11px;
    border-radius: 999px;
    border: 1px solid #164e3b;
    background: #07160f;
    color: #86efac;
    font-size: 9px;
    font-weight: 700;
}

/* ---------- hero ---------- */

.hero {
    margin: 24px 0 18px;
    padding: 30px 32px;
    border-radius: 21px;
    border: 1px solid #202b39;
    background:
        linear-gradient(135deg, rgba(15,23,42,.96), rgba(8,12,18,.96));
    box-shadow: 0 18px 65px rgba(0,0,0,.25);
}

.hero-kicker {
    color: var(--cyan);
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
}

.hero-title {
    color: #f8fafc;
    font-size: 34px;
    line-height: 1.13;
    font-weight: 800;
    margin-top: 8px;
}

.hero-text {
    color: #94a3b8;
    font-size: 11px;
    line-height: 1.7;
    max-width: 720px;
    margin-top: 10px;
}

/* ---------- cards ---------- */

.card {
    background: rgba(12,17,24,.91);
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 18px;
    margin-bottom: 12px;
}

.card-title {
    color: #f1f5f9;
    font-size: 13px;
    font-weight: 750;
}

.card-sub {
    color: #64748b;
    font-size: 9px;
    line-height: 1.55;
    margin-top: 4px;
}

.section-title {
    color: #f8fafc;
    font-size: 21px;
    font-weight: 800;
    margin: 22px 0 5px;
}

.section-sub {
    color: #64748b;
    font-size: 10px;
    margin-bottom: 14px;
}

/* ---------- result ---------- */

.result-card {
    min-height: 260px;
    padding: 27px;
    border-radius: 20px;
    border: 1px solid rgba(34,211,238,.45);
    background:
        radial-gradient(circle at 80% 15%, rgba(168,85,247,.17), transparent 35%),
        radial-gradient(circle at 15% 90%, rgba(34,211,238,.12), transparent 40%),
        linear-gradient(145deg, #0b1620, #090e15);
}

.result-kicker {
    color: #67e8f9;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .12em;
}

.result-number {
    color: #f8fafc;
    font-size: 53px;
    line-height: 1;
    font-weight: 800;
    margin-top: 17px;
}

.result-unit {
    color: #94a3b8;
    font-size: 11px;
    margin-top: 7px;
}

.result-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 11px;
    border-radius: 999px;
    background: rgba(34,211,238,.08);
    border: 1px solid rgba(34,211,238,.25);
    color: #67e8f9;
    font-size: 10px;
    font-weight: 700;
}


/* ---------- enhanced result ---------- */

.result-meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.result-meta-item {
    padding: 7px 10px;
    border-radius: 999px;
    border: 1px solid rgba(148,163,184,.18);
    background: rgba(15,23,42,.55);
    color: #94a3b8;
    font-size: 9px;
}

.result-change {
    margin-top: 18px;
    padding: 12px 14px;
    border-radius: 12px;
    background: rgba(34,197,94,.06);
    border: 1px solid rgba(34,197,94,.20);
}

.result-change.down {
    background: rgba(34,197,94,.06);
    border-color: rgba(34,197,94,.20);
}

.result-change.up {
    background: rgba(239,68,68,.06);
    border-color: rgba(239,68,68,.20);
}

.result-change-title {
    color: #cbd5e1;
    font-size: 9px;
    font-weight: 700;
}

.result-change-value {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 800;
    margin-top: 3px;
}

.status-strip {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin: 10px 0 16px;
}

.status-pill {
    padding: 8px 11px;
    border-radius: 999px;
    border: 1px solid rgba(34,197,94,.25);
    background: rgba(34,197,94,.06);
    color: #a7f3d0;
    font-size: 9px;
    font-weight: 700;
}

.insight {
    min-height: 0;
    padding: 14px 15px;
    border: 1px solid var(--border);
    background: #0a1017;
    border-radius: 13px;
}

.insight + .insight {
    margin-top: 9px;
}

.insight-title {
    color: #f1f5f9;
    font-size: 10px;
    font-weight: 800;
    margin-top: 0;
}

.insight-text {
    color: #94a3b8;
    font-size: 9px;
    line-height: 1.65;
    margin-top: 5px;
}

/* ---------- KPI ---------- */

.kpi {
    min-height: 92px;
    background: #0b1016;
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 14px;
}

.kpi-label {
    color: #64748b;
    font-size: 8px;
    letter-spacing: .07em;
    text-transform: uppercase;
}

.kpi-value {
    color: #f8fafc;
    font-size: 20px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-help {
    color: #475569;
    font-size: 8px;
    margin-top: 4px;
}

/* ---------- insight ---------- */

.insight {
    min-height: 105px;
    padding: 16px;
    border: 1px solid var(--border);
    background: #0a1017;
    border-radius: 13px;
}

.insight-icon {
    font-size: 17px;
}

.insight-title {
    color: #e2e8f0;
    font-size: 11px;
    font-weight: 750;
    margin-top: 6px;
}

.insight-text {
    color: #718096;
    font-size: 9px;
    line-height: 1.6;
    margin-top: 5px;
}

/* ---------- alerts ---------- */

.good {
    border: 1px solid #155e43;
    background: #071811;
    color: #a7f3d0;
    border-radius: 10px;
    padding: 11px 13px;
    font-size: 10px;
    line-height: 1.6;
    margin-bottom: 12px;
}

.warn {
    border: 1px solid #66521b;
    background: #171407;
    color: #fde68a;
    border-radius: 10px;
    padding: 11px 13px;
    font-size: 10px;
    line-height: 1.6;
    margin-bottom: 12px;
}

.bad {
    border: 1px solid #662727;
    background: #1a0a0a;
    color: #fecaca;
    border-radius: 10px;
    padding: 11px 13px;
    font-size: 10px;
    line-height: 1.6;
    margin-bottom: 12px;
}

/* ---------- buttons ---------- */

.stButton > button {
    min-height: 46px;
    border-radius: 11px;
    font-weight: 750;
    border: 1px solid rgba(34,211,238,.35);
    background: linear-gradient(90deg, #0891b2, #2563eb);
    color: white;
}

.stButton > button:hover {
    border-color: #22d3ee;
    box-shadow: 0 0 25px rgba(34,211,238,.18);
}

/* ---------- inputs ---------- */

label {
    color: #cbd5e1 !important;
    font-size: 10px !important;
    font-weight: 650 !important;
}

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #080d14;
    border-color: #263243;
    border-radius: 9px;
}

div[data-baseweb="input"] input {
    color: #f8fafc;
}

div[data-baseweb="select"] * {
    color: #e2e8f0;
}

/* ---------- sidebar ---------- */

[data-testid="stSidebar"] {
    background: #080b10;
    border-right: 1px solid #1b2430;
}

/* ---------- footer ---------- */

.footer {
    border-top: 1px solid #1b2430;
    margin-top: 42px;
    padding-top: 13px;
    text-align: center;
    color: #475569;
    font-size: 8px;
}

/* ---------- mobile ---------- */

@media (max-width: 800px) {
    .block-container {
        padding: 15px;
    }

    .hero-title {
        font-size: 27px;
    }

    .result-number {
        font-size: 42px;
    }

    .topbar {
        align-items: flex-start;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# 3. PATH ARTIFACT
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent


def find_artifact(filename, preferred_folder=None):
    candidates = []

    if preferred_folder:
        candidates.append(BASE_DIR / preferred_folder / filename)

    candidates.extend([
        BASE_DIR / filename,
        BASE_DIR / "deployment" / filename,
        BASE_DIR / "deployment" / "models" / filename,
        BASE_DIR / "deployment" / "features" / filename,
        BASE_DIR / "deployment" / "evaluation" / filename,
        BASE_DIR / "deployment" / "parameters" / filename,
        BASE_DIR / "models" / filename,
        BASE_DIR / "features" / filename,
    ])

    for path in candidates:
        if path.exists():
            return path

    try:
        matches = list(BASE_DIR.rglob(filename))
        if matches:
            matches.sort(
                key=lambda p: (
                    0 if "deployment" in p.parts else 1,
                    len(p.parts),
                )
            )
            return matches[0]
    except OSError:
        pass

    return None


MODEL_PATH = find_artifact(
    "lightgbm_random_search.pkl",
    "deployment/models",
)

FEATURE_PATH = find_artifact(
    "feature_columns.pkl",
    "deployment/features",
)

TEST_PATH = find_artifact(
    "test_final.parquet",
    "data/processed",
)

if TEST_PATH is None:
    TEST_PATH = find_artifact("test_dataset.parquet")

if TEST_PATH is None:
    TEST_PATH = find_artifact("X_test.parquet")


# ==========================================================
# 4. LOAD ARTIFACT
# ==========================================================

@st.cache_resource
def load_model(path):
    if path is None:
        return None, "Model Random Search tidak ditemukan."

    try:
        return joblib.load(path), None
    except Exception as exc:
        return None, f"Model gagal dimuat: {exc}"


@st.cache_resource
def load_features(path):
    if path is None:
        return None, "feature_columns.pkl tidak ditemukan."

    try:
        value = joblib.load(path)

        if isinstance(value, pd.Index):
            value = value.tolist()
        elif isinstance(value, np.ndarray):
            value = value.tolist()

        return list(value), None
    except Exception as exc:
        return None, f"Feature columns gagal dimuat: {exc}"


@st.cache_data
def load_test(path):
    if path is None:
        return None

    try:
        if str(path).lower().endswith(".parquet"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except Exception:
        return None


model, model_error = load_model(MODEL_PATH)
feature_columns, feature_error = load_features(FEATURE_PATH)
test_data = load_test(TEST_PATH)

# Fallback dari model LightGBM bila feature_columns.pkl tidak tersedia.
if feature_columns is None and model is not None:
    try:
        if hasattr(model, "feature_name_"):
            feature_columns = list(model.feature_name_)
        elif hasattr(model, "booster_"):
            feature_columns = list(model.booster_.feature_name())
    except Exception:
        pass

# Validasi artifact sebelum aplikasi dipakai.
MODEL_FEATURE_NAMES = None
if model is not None:
    try:
        if hasattr(model, "feature_name_"):
            MODEL_FEATURE_NAMES = list(model.feature_name_)
        elif hasattr(model, "booster_"):
            MODEL_FEATURE_NAMES = list(model.booster_.feature_name())
    except Exception:
        MODEL_FEATURE_NAMES = None

if (
    MODEL_FEATURE_NAMES is not None
    and feature_columns is not None
    and list(MODEL_FEATURE_NAMES) != list(feature_columns)
):
    model_error = (
        "Urutan/nama feature_columns.pkl tidak sama dengan feature "
        "yang tersimpan pada model Random Search."
    )


# ==========================================================
# 5. HELPER
# ==========================================================

def show_kpi(label, value, help_text=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalize_name(value):
    text = str(value).lower().strip()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        "(": "",
        ")": "",
        "%": "",
        "°": "",
        ".": "_",
        ",": "_",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_")


def get_target_column(data):
    if data is None:
        return None

    candidates = [
        "kWh_received_Total",
        "total_kwh",
        "Total",
        "total",
        "energy_consumption",
        "consumption",
        "target",
    ]

    for col in candidates:
        if col in data.columns:
            return col

    return None


def get_reference_series():
    target = get_target_column(test_data)

    if target is None:
        return None

    series = pd.to_numeric(
        test_data[target],
        errors="coerce",
    ).dropna()

    return series if len(series) else None


def format_number(value, digits=4):
    return f"{float(value):,.{digits}f}"


def forecast_values(prediction):
    # Model target digunakan sebagai konsumsi 15 menit.
    # Proyeksi berikut menggunakan asumsi nilai 15 menit tetap konstan.
    return {
        "15 Menit": prediction,
        "1 Jam": prediction * 4,
        "1 Hari": prediction * 96,
        "1 Minggu": prediction * 672,
        "1 Bulan": prediction * 2880,
    }


def comparison_insight(prediction, last_energy):
    if last_energy is None or last_energy <= 0:
        return (
            "Belum ada pembanding perubahan karena konsumsi terakhir "
            "bernilai nol atau belum tersedia.",
            None,
        )

    change = ((prediction - last_energy) / last_energy) * 100

    if change > 0:
        text = (
            f"Estimasi konsumsi berikutnya sekitar {abs(change):.1f}% "
            "lebih tinggi dibandingkan konsumsi terakhir."
        )
    elif change < 0:
        text = (
            f"Estimasi konsumsi berikutnya sekitar {abs(change):.1f}% "
            "lebih rendah dibandingkan konsumsi terakhir."
        )
    else:
        text = (
            "Estimasi konsumsi berikutnya sama dengan konsumsi terakhir."
        )

    return text, change


def distribution_insight(prediction, reference):
    if reference is None or len(reference) < 10:
        return (
            "Data pembanding belum tersedia dalam jumlah yang cukup "
            "untuk menentukan posisi prediksi terhadap distribusi historis."
        )

    q25 = float(reference.quantile(.25))
    q50 = float(reference.quantile(.50))
    q75 = float(reference.quantile(.75))

    if prediction < q25:
        return (
            "Prediksi berada di bawah kuartil pertama data pembanding, "
            "sehingga relatif berada pada sisi konsumsi rendah."
        )

    if prediction > q75:
        return (
            "Prediksi berada di atas kuartil ketiga data pembanding, "
            "sehingga relatif berada pada sisi konsumsi tinggi."
        )

    if prediction >= q50:
        return (
            "Prediksi berada pada rentang tengah hingga atas "
            "dari distribusi data pembanding."
        )

    return (
        "Prediksi berada pada rentang tengah hingga bawah "
        "dari distribusi data pembanding."
    )


# ==========================================================
# 6. HEADER
# ==========================================================

header_left, header_right = st.columns([.78, .22])

with header_left:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">⚡</div>
            <div>
                <div class="brand-name">Smart Energy AI</div>
                <div class="brand-sub">
                    Prediksi Konsumsi Energi Smart Grid
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with header_right:
    if model is not None:
        st.markdown(
            '<div style="text-align:right;margin-top:4px;">'
            '<span class="status-pill">● Model siap</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right;margin-top:4px;">'
            '<span class="status-pill" style="color:#fca5a5;border-color:#642727;background:#190b0b;">'
            '● Model bermasalah</span>'
            '</div>',
            unsafe_allow_html=True,
        )


# ==========================================================
# 7. BERANDA / HERO
# ==========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Prediksi Energi Berbasis AI</div>
        <div class="hero-title">
            Ketahui perkiraan konsumsi energi Anda.
        </div>
        <div class="hero-text">
            Masukkan informasi rumah, kondisi lingkungan, waktu,
            dan konsumsi terakhir. Sistem akan memproses input
            tersebut menggunakan model LightGBM dengan Random Search
            yang dipilih sebagai model terbaik penelitian.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 8. INPUT + HASIL
# ==========================================================

input_col, result_col = st.columns(
    [1.0, 1.65],
    gap="large",
)



# ==========================================================
# SESSION STATE & USER-FACING HELPERS
# ==========================================================

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_input" not in st.session_state:
    st.session_state["last_input"] = None
if "last_model_input" not in st.session_state:
    st.session_state["last_model_input"] = None


def _norm_name(value):
    text = str(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _numeric_series(df, col):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").dropna()


def target_column(df):
    if df is None:
        return None
    candidates = [
        "kWh_received_Total",
        "total_kwh",
        "Total",
        "total",
        "energy_consumption",
        "consumption",
        "target",
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def reference_values():
    col = target_column(test_data)
    if col is None:
        return pd.Series(dtype=float)
    return _numeric_series(test_data, col)


def input_warnings(user):
    warnings = []

    basic = [
        ("Luas bangunan", user["luas_bangunan"] > 0),
        ("Jumlah penghuni", user["jumlah_penghuni"] >= 1),
        ("Kelembapan", 0 <= user["kelembapan"] <= 100),
        ("Temperatur", -20 <= user["temperatur"] <= 60),
        ("Kecepatan angin", user["kecepatan_angin"] >= 0),
        ("Curah hujan", user["curah_hujan"] >= 0),
        ("Konsumsi terakhir", user["konsumsi_terakhir"] >= 0),
    ]

    for name, ok in basic:
        if not ok:
            warnings.append(f"{name} berada di luar rentang input yang wajar.")

    # Empirical range warning when the corresponding raw feature exists.
    aliases = {
        "luas_bangunan": [
            "luas_bangunan",
            "survey_building_livingarea",
            "building_area",
        ],
        "jumlah_penghuni": [
            "jumlah_penghuni",
            "survey_building_residents",
            "occupants",
        ],
        "temperatur": [
            "temperatur",
            "temperature",
            "temperature_avg_hourly",
        ],
        "kelembapan": [
            "kelembapan",
            "humidity",
            "humidity_avg_hourly",
        ],
        "kecepatan_angin": [
            "kecepatan_angin",
            "wind_speed",
            "windspeed_hourly",
        ],
        "curah_hujan": [
            "curah_hujan",
            "precipitation",
            "precipitation_total_hourly",
        ],
    }

    if test_data is not None:
        normalized_columns = {
            _norm_name(c): c for c in test_data.columns
        }

        for key, names in aliases.items():
            found = None
            for name in names:
                if _norm_name(name) in normalized_columns:
                    found = normalized_columns[_norm_name(name)]
                    break

            if found is None:
                continue

            series = _numeric_series(test_data, found)
            if len(series) < 10:
                continue

            value = float(user[key])
            lo = float(series.min())
            hi = float(series.max())

            if value < lo or value > hi:
                warnings.append(
                    f"{key.replace('_', ' ').title()} "
                    f"({value:g}) berada di luar rentang data pembanding "
                    f"({lo:g}–{hi:g}). Hasil dapat kurang representatif."
                )

    return warnings


def build_insights(prediction, last_energy, user=None):
    """Insight yang langsung menjawab arti hasil untuk user awam."""
    result = []

    last_energy = float(last_energy or 0)

    if last_energy > 0:
        change = ((prediction - last_energy) / last_energy) * 100

        if change < -0.5:
            result.append((
                "Perubahan",
                f"Prediksi sekitar {abs(change):.1f}% lebih rendah "
                f"dari konsumsi terakhir ({last_energy:.2f} kWh)."
            ))
        elif change > 0.5:
            result.append((
                "Perubahan",
                f"Prediksi sekitar {change:.1f}% lebih tinggi "
                f"dari konsumsi terakhir ({last_energy:.2f} kWh)."
            ))
        else:
            result.append((
                "Perubahan",
                "Prediksi relatif dekat dengan konsumsi terakhir."
            ))

    # Insight kondisi rumah selalu tersedia, tanpa bergantung pada dataset test.
    if user:
        building = user.get("building_type", "Rumah")
        building_label = (
            "Apartemen" if building == "Apartemen"
            else "Rumah" if building == "Rumah"
            else str(building)
        )
        area = float(user.get("luas_bangunan", 0))
        residents = int(user.get("jumlah_penghuni", 0))
        temp = float(user.get("temperatur", 0))
        humidity = float(user.get("kelembapan", 0))
        pv = user.get("pv", "Tidak")
        ev = user.get("ev", "Tidak")

        result.append((
            "Kondisi yang dianalisis",
            f"{building_label} seluas {area:.0f} m² dengan {residents} penghuni. "
            f"Temperatur {temp:.1f}°C dan kelembapan {humidity:.0f}%. "
            f"PV: {pv}; kendaraan listrik: {ev}."
        ))

    # Perbandingan terhadap dataset hanya sebagai insight tambahan.
    reference = reference_values()

    if len(reference) >= 10:
        q25 = float(reference.quantile(.25))
        q50 = float(reference.quantile(.50))
        q75 = float(reference.quantile(.75))

        if prediction < q25:
            position = "relatif rendah"
        elif prediction > q75:
            position = "relatif tinggi"
        elif prediction >= q50:
            position = "menengah hingga menengah-atas"
        else:
            position = "menengah hingga menengah-bawah"

        result.append((
            "Posisi terhadap data pembanding",
            f"Estimasi {prediction:.3f} kWh berada pada kelompok {position} "
            "jika dibandingkan dengan data penelitian yang tersedia."
        ))

    # Insight operasional yang mudah dipahami.
    if prediction >= 1.0:
        result.append((
            "Perhatian",
            "Estimasi konsumsi per 15 menit cukup tinggi. Jika hasil ini "
            "tidak sesuai kondisi rumah, periksa kembali konsumsi terakhir "
            "dan kondisi lingkungan yang dimasukkan."
        ))
    elif prediction < 0.25:
        result.append((
            "Perhatian",
            "Estimasi relatif rendah. Pastikan nilai konsumsi terakhir "
            "dan kondisi lingkungan yang dimasukkan sudah sesuai."
        ))
    else:
        result.append((
            "Interpretasi",
            "Model memperkirakan konsumsi untuk kondisi yang Anda masukkan. "
            "Hasil ini adalah estimasi, bukan pembacaan meter listrik langsung."
        ))

    return result



def _dummy_suffixes(prefix):
    if feature_columns is None:
        return []
    prefix_text = prefix + "_"
    return [
        str(col)[len(prefix_text):]
        for col in feature_columns
        if str(col).startswith(prefix_text)
        and len(str(col)) > len(prefix_text)
    ]


def categorical_options(prefix):
    if prefix == "Survey_Building_Type":
        return ["Rumah", "Apartemen"]

    suffixes = _dummy_suffixes(prefix)
    return ["Kategori dasar"] + suffixes if suffixes else ["Kategori dasar"]


def map_building_type(selected):
    suffixes = _dummy_suffixes("Survey_Building_Type")
    normalized = {
        re.sub(r"[^a-z0-9]+", "", str(s).lower()): s
        for s in suffixes
    }

    if selected == "Rumah":
        candidates = [
            "House", "house", "SingleFamilyHouse",
            "single_family_house", "DetachedHouse"
        ]
    else:
        candidates = [
            "Apartment", "apartment", "Flat", "flat",
            "MultiFamilyHouse", "multi_family_house"
        ]

    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in normalized:
            return normalized[key]

    return "Kategori dasar"


def season_name(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def _set_dummy(frame, prefix, selected):
    if feature_columns is None or selected == "Kategori dasar":
        return

    prefix_text = prefix + "_"
    for col in feature_columns:
        col_text = str(col)
        if col_text.startswith(prefix_text):
            suffix = col_text[len(prefix_text):]
            if suffix == str(selected):
                frame.loc[0, col] = 1.0


def save_prediction(user, prediction, model_input=None):
    """
    Menyimpan hasil prediksi ke session_state.
    Penyimpanan hanya di memori sesi agar deployment Streamlit
    tidak gagal karena masalah penulisan file.
    """
    prediction = float(prediction)

    record = dict(user)
    record["waktu_prediksi"] = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    record["prediksi_kWh"] = prediction

    st.session_state["last_prediction"] = prediction
    st.session_state["last_input"] = dict(user)

    if model_input is not None:
        st.session_state["last_model_input"] = model_input.copy()

    history = st.session_state.setdefault("prediction_history", [])
    history.append(record)

    # Maksimal 50 hasil dalam satu sesi.
    if len(history) > 50:
        del history[:-50]


def build_model_input(user):
    """
    Membentuk 38 feature untuk model deployment dari input user.
    Mode cold-start menggunakan konsumsi terakhir sebagai histori t-1;
    lag yang belum tersedia mengikuti fillna(0).
    """
    if feature_columns is None:
        raise ValueError("feature_columns.pkl tidak tersedia.")

    last = float(user["konsumsi_terakhir"])
    if last < 0:
        raise ValueError("Konsumsi terakhir tidak boleh negatif.")

    tanggal = pd.Timestamp(user["tanggal_lengkap"])
    hour = int(user["jam"])
    day = int(user["tanggal"])
    month = int(user["bulan"])
    dow = int(user["hari_dalam_minggu"])

    X = pd.DataFrame([{col: 0.0 for col in feature_columns}])

    def put(name, value):
        if name in X.columns:
            X.loc[0, name] = float(value)

    # Profil rumah.
    put("Survey_Building_LivingArea", user["luas_bangunan"])
    put("Survey_Building_Residents", user["jumlah_penghuni"])

    # Instalasi / survey.
    boolean_features = {
        "Installation_HasPVSystem": int(user["pv"] == "Ya"),
        "Survey_Installation_HasElectricVehicle": int(user["ev"] == "Ya"),
        "Survey_HeatDistribution_System_Radiator": 0,
        "Survey_HeatDistribution_System_FloorHeating": 0,
        "Survey_DHW_Production_ByHeatPump": 0,
        "Survey_DHW_Production_ByElectricWaterHeater": 0,
        "Survey_DHW_Production_BySolar": 0,
        "Survey_Installation_HasDryer": 0,
        "Survey_Installation_HasFreezer": 0,
    }
    for name, value in boolean_features.items():
        put(name, value)

    # Waktu.
    put("hour", hour)
    put("day", day)
    put("dayofweek", dow)
    put("month", month)
    put("quarter", tanggal.quarter)
    put("is_weekend", int(dow >= 5))

    # Cyclic.
    put("hour_sin", np.sin(2 * np.pi * hour / 24))
    put("hour_cos", np.cos(2 * np.pi * hour / 24))
    put("month_sin", np.sin(2 * np.pi * month / 12))
    put("month_cos", np.cos(2 * np.pi * month / 12))

    # Cuaca.
    put("Temperature_avg_hourly", user["temperatur"])
    put("Humidity_avg_hourly", user["kelembapan"])
    put("Precipitation_total_hourly", user["curah_hujan"])
    put("WindSpeed_hourly", user["kecepatan_angin"])

    # Histori cold-start.
    put("lag_1", last)
    put("lag_4", 0)
    put("lag_96", 0)
    put("lag_672", 0)

    put("rolling_mean_1h", last)
    put("rolling_std_1h", 0)
    put("rolling_mean_24h", last)
    put("rolling_std_24h", 0)

    # Kategori.
    _set_dummy(
        X,
        "Survey_Building_Type",
        map_building_type(user["building_type"]),
    )

    _set_dummy(
        X,
        "Survey_HeatPump_Installation_Type",
        user["heatpump_type"],
    )

    _set_dummy(
        X,
        "season",
        season_name(month),
    )

    X = X.loc[:, feature_columns].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    invalid = X.columns[X.isna().any()].tolist()
    if invalid:
        raise ValueError(
            "Feature tidak valid: " + ", ".join(map(str, invalid))
        )

    if X.shape[1] != 38:
        raise ValueError(
            f"Model memiliki {X.shape[1]} feature, sedangkan "
            "deployment penelitian harus menggunakan 38 feature."
        )

    return X


def kpi(label, value, caption=""):
    """Kartu KPI reusable untuk ringkasan hasil prediksi."""
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# INPUT PANEL
# ==========================================================

user = {}

with st.form("smart_energy_prediction_form"):

    left, right = st.columns(2, gap="large")

    with left:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🏠 Profil Rumah</div>
                <div class="card-sub">
                    Informasi yang paling mudah diketahui pengguna.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user["luas_bangunan"] = st.number_input(
            "Luas Bangunan (m²)",
            min_value=1.0,
            max_value=10000.0,
            value=100.0,
            step=1.0,
        )

        user["jumlah_penghuni"] = st.number_input(
            "Jumlah Penghuni",
            min_value=1,
            max_value=100,
            value=4,
            step=1,
        )

        user["pv"] = st.selectbox(
            "Memiliki PV / Solar System?",
            ["Tidak", "Ya"],
        )

        user["ev"] = st.selectbox(
            "Memiliki Electric Vehicle?",
            ["Tidak", "Ya"],
        )

        
        user["building_type"] = st.selectbox(
            "Jenis Bangunan",
            categorical_options("Survey_Building_Type"),
            help="Pilih Rumah atau Apartemen sesuai kondisi pengguna.",
        )

        heatpump_suffixes = _dummy_suffixes(
            "Survey_HeatPump_Installation_Type"
        )

        heatpump_labels = ["Tidak ada"]
        heatpump_map = {"Tidak ada": "None"}

        for suffix in heatpump_suffixes:
            normalized = re.sub(r"[^a-z0-9]+", "", str(suffix).lower())

            if normalized in {"none", "no", "nonesystem"}:
                continue

            if normalized in {"radiator", "heatingradiator"}:
                label = "Radiator"
            elif normalized in {"floorheating", "underfloorheating"}:
                label = "Floor Heating"
            else:
                label = str(suffix).replace("_", " ")

            if label not in heatpump_labels:
                heatpump_labels.append(label)
                heatpump_map[label] = suffix

        selected_heatpump = st.selectbox(
            "Tipe Instalasi Heat Pump",
            heatpump_labels,
            help="Pilih jenis instalasi jika rumah Anda memilikinya.",
        )
        user["heatpump_type"] = heatpump_map.get(
            selected_heatpump,
            selected_heatpump,
        )
        st.markdown(
            """
            <div class="card" style="margin-top:14px;">
                <div class="card-title">⚡ Riwayat Konsumsi</div>
                <div class="card-sub">
                    Masukkan konsumsi terakhir yang Anda ketahui.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user["konsumsi_terakhir"] = st.number_input(
            "Konsumsi Energi Terakhir (kWh)",
            min_value=0.0,
            max_value=100000.0,
            value=0.50,
            step=0.01,
            format="%.2f",
        )

    with right:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">🌤️ Kondisi Lingkungan</div>
                <div class="card-sub">
                    Kondisi pada waktu yang ingin diprediksi.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        user["temperatur"] = st.number_input(
            "Temperatur (°C)",
            min_value=-20.0,
            max_value=60.0,
            value=27.0,
            step=0.1,
        )

        user["kelembapan"] = st.number_input(
            "Kelembapan (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=0.1,
        )

        user["kecepatan_angin"] = st.number_input(
            "Kecepatan Angin (m/s)",
            min_value=0.0,
            max_value=100.0,
            value=3.0,
            step=0.1,
        )

        user["curah_hujan"] = st.number_input(
            "Curah Hujan (mm)",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=0.1,
        )

        st.markdown(
            """
            <div class="card" style="margin-top:14px;">
                <div class="card-title">🕐 Waktu Prediksi</div>
                <div class="card-sub">
                    Pilih tanggal dan jam. Hari dalam minggu dihitung otomatis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prediction_date = st.date_input(
            "Tanggal Prediksi",
            value=pd.Timestamp.now().date(),
        )

        prediction_hour = st.slider(
            "Jam Prediksi",
            min_value=0,
            max_value=23,
            value=12,
            format="%d:00",
        )

        user["tanggal"] = prediction_date.day
        user["bulan"] = prediction_date.month
        user["jam"] = prediction_hour
        user["tanggal_lengkap"] = str(prediction_date)
        user["hari_dalam_minggu"] = int(
            pd.Timestamp(prediction_date).dayofweek
        )

    st.write("")

    submit = st.form_submit_button(
        "⚡ HITUNG PREDIKSI KONSUMSI",
        type="primary",
        width='stretch',
    )


# ==========================================================
# 9. PREDICTION ENGINE
# ==========================================================

if submit:

    warnings = input_warnings(user)

    if warnings:
        st.markdown(
            "<div class='warn'><b>Periksa beberapa input</b><br>"
            + "<br>".join(f"• {w}" for w in warnings)
            + "</div>",
            unsafe_allow_html=True,
        )

    with st.spinner("Menganalisis kondisi dan menjalankan model..."):

        try:
            model_input = build_model_input(user)
        except Exception as exc:
            model_input = None
            st.error(f"Input belum dapat diproses: {exc}")

    if model_input is None:
        st.markdown(
            """
            <div class="bad">
                <b>Prediksi belum dapat dijalankan.</b><br>
                Pipeline input belum dapat membentuk seluruh feature
                yang dibutuhkan model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        # Jangan pernah diam-diam mengisi feature model yang hilang.
        missing_features = [
            col for col in feature_columns
            if col not in model_input.columns
        ]

        if missing_features:

            st.markdown(
                f"""
                <div class="bad">
                    <b>Data belum lengkap untuk model.</b><br>
                    {len(missing_features)} feature belum dapat dibentuk.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Lihat feature yang belum tersedia"):
                st.dataframe(
                    pd.DataFrame({"Feature": missing_features}),
                    width='stretch',
                    hide_index=True,
                )

            st.info(
                "Aplikasi tidak mengisi feature yang hilang dengan angka "
                "0 karena hal tersebut dapat mengubah makna prediksi."
            )

        else:

            try:

                # Pastikan urutan feature sama dengan model training.
                model_input = model_input.loc[:, feature_columns].copy()

                for col in model_input.columns:
                    model_input[col] = pd.to_numeric(
                        model_input[col],
                        errors="coerce",
                    )

                invalid = model_input.columns[
                    model_input.isna().any()
                ].tolist()

                if invalid:
                    raise ValueError(
                        "Terdapat nilai kosong/non-numerik pada feature: "
                        + ", ".join(map(str, invalid))
                    )

                prediction = model.predict(model_input)
                prediction = float(
                    np.asarray(prediction).reshape(-1)[0]
                )

                # Konsumsi energi tidak boleh negatif.
                prediction = max(prediction, 0.0)

                save_prediction(
                    user,
                    prediction,
                    model_input,
                )

                st.markdown(
                    """
                    <div class="good">
                        <b>✓ Analisis selesai.</b><br>
                        Model berhasil menghasilkan estimasi konsumsi energi.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            except Exception as exc:

                st.markdown(
                    f"""
                    <div class="bad">
                        <b>Prediksi gagal.</b><br>
                        {exc}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==========================================================
# 10. HASIL PANEL
# ==========================================================

prediction = st.session_state.get("last_prediction")
last_input = st.session_state.get("last_input")

if prediction is not None and last_input is not None:

    st.markdown(
        '<div class="section-title">Hasil Prediksi</div>',
        unsafe_allow_html=True,
    )

    insights = build_insights(
        prediction,
        float(last_input.get("konsumsi_terakhir", 0)),
        last_input,
    )

    st.markdown(
        """
        <div class="status-strip">
            <span class="status-pill">✓ Prediksi berhasil</span>
            <span class="status-pill">✓ 38 feature terbentuk</span>
            <span class="status-pill">✓ Random Search aktif</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


    result_left, result_right = st.columns(
        [1.1, .9],
        gap="large",
    )

    with result_left:

        last_energy = float(last_input.get("konsumsi_terakhir", 0) or 0)
        delta = prediction - last_energy
        pct = (delta / last_energy * 100) if last_energy > 0 else None

        if pct is None:
            change_class = ""
            change_title = "Perbandingan"
            change_value = "Belum tersedia"
        elif pct < 0:
            change_class = "down"
            change_title = "Dibanding konsumsi terakhir"
            change_value = f"↓ {abs(pct):.1f}%"
        elif pct > 0:
            change_class = "up"
            change_title = "Dibanding konsumsi terakhir"
            change_value = f"↑ {pct:.1f}%"
        else:
            change_class = "down"
            change_title = "Dibanding konsumsi terakhir"
            change_value = "≈ 0%"

        result_html = f"""
<div class="result-card">
    <div class="result-kicker">ESTIMASI KONSUMSI ENERGI</div>

    <div class="result-number">{prediction:,.4f}</div>

    <div class="result-unit">kWh per 15 menit</div>

    <div class="result-change {change_class}">
        <div class="result-change-title">{change_title}</div>
        <div class="result-change-value">{change_value}</div>
    </div>

    <div class="result-meta">
        <span class="result-meta-item">Model: LightGBM</span>
        <span class="result-meta-item">Optimasi: Random Search</span>
        <span class="result-meta-item">Feature: {len(feature_columns)}</span>
    </div>
</div>
"""

        # Render kartu hasil sebagai HTML native Streamlit.
        # Jangan gunakan st.markdown untuk blok ini karena pada beberapa
        # versi Streamlit HTML dapat ditampilkan sebagai teks/kode.
        st.html(result_html)

    with result_right:


        st.markdown(
            '<div class="card-title">💡 Insight</div>',
            unsafe_allow_html=True,
        )

        # Setiap insight dibuat sebagai blok terpisah agar selalu
        # terlihat dan tidak bergantung pada nesting HTML.
        if insights:
            for title, text in insights:
                st.markdown(
                    f"""
                    <div class="insight" style="margin-bottom:10px;">
                        <div class="insight-title">{title}</div>
                        <div class="insight-text">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
                <div class="insight">
                    <div class="insight-title">Hasil prediksi</div>
                    <div class="insight-text">
                        Prediksi berhasil dibuat. Tidak tersedia data pembanding
                        yang cukup untuk analisis relatif.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="insight">
                <div class="insight-title">Cara membaca hasil</div>
                <div class="insight-text">
                    Angka prediksi adalah estimasi konsumsi berdasarkan kondisi
                    yang Anda masukkan, bukan pembacaan meter listrik secara langsung.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="good">
            <b>Mode prediksi: Cold-start</b><br>
            Input manual Anda sudah diubah menjadi 38 feature yang sama
            dengan struktur model. Data test tidak dipakai untuk membentuk
            histori input prediksi.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Ringkasan Kondisi</div>',
        unsafe_allow_html=True,
    )

    a, b, c = st.columns(3)

    with a:
        kpi(
            "Luas bangunan",
            f"{last_input['luas_bangunan']:,.1f} m²",
            "Profil rumah",
        )

    with b:
        kpi(
            "Penghuni",
            str(int(last_input["jumlah_penghuni"])),
            "Orang",
        )

    with c:
        kpi(
            "Temperatur",
            f"{last_input['temperatur']:,.1f} °C",
            "Kondisi lingkungan",
        )

    a, b, c = st.columns(3)

    with a:
        kpi(
            "Konsumsi terakhir",
            f"{last_input['konsumsi_terakhir']:,.2f} kWh",
            "Input pengguna",
        )

    with b:
        delta = prediction - float(last_input["konsumsi_terakhir"])
        kpi(
            "Perubahan",
            (
                f"{delta:+,.2f} kWh"
                if float(last_input["konsumsi_terakhir"]) > 0
                else "—"
            ),
            "Prediksi − konsumsi terakhir",
        )

    with c:
        if float(last_input["konsumsi_terakhir"]) > 0:
            pct = (
                (prediction - float(last_input["konsumsi_terakhir"]))
                / float(last_input["konsumsi_terakhir"])
                * 100
            )
            pct_text = f"{pct:+.1f}%"
        else:
            pct_text = "—"

        kpi(
            "Perubahan (%)",
            pct_text,
            "Dibanding konsumsi terakhir",
        )

    st.markdown(
        '<div class="section-title">Prediksi vs Konsumsi Terakhir</div>',
        unsafe_allow_html=True,
    )

    compare_df = pd.DataFrame({
        "Kondisi": ["Konsumsi terakhir", "Prediksi"],
        "kWh per 15 menit": [
            float(last_input["konsumsi_terakhir"]),
            float(prediction),
        ],
    })

    fig_compare = go.Figure(
        go.Bar(
            x=compare_df["Kondisi"],
            y=compare_df["kWh per 15 menit"],
            text=[f"{v:.2f} kWh" for v in compare_df["kWh per 15 menit"]],
            textposition="auto",
            hovertemplate="%{x}<br>%{y:.4f} kWh per 15 menit<extra></extra>",
        )
    )
    fig_compare.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis_title="kWh per 15 menit",
        xaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(
        fig_compare,
        width='stretch',
        config={"displayModeBar": False},
    )

    st.markdown(
        """
        <div class="section-sub">
            Grafik membandingkan satu-satunya nilai historis yang dimasukkan
            pengguna dengan estimasi model. Semakin besar selisihnya,
            semakin besar perubahan yang diperkirakan model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Simulasi Akumulasi Energi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">'
        'Ini bukan prediksi multi-horizon. Nilai hanya merupakan simulasi '
        'akumulasi jika konsumsi sebesar estimasi per 15 menit dianggap tetap.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-sub">
            Contoh: nilai 1 hari = estimasi 15 menit × 96 interval.
            Ini membantu memperkirakan skala konsumsi, bukan meramalkan
            perubahan konsumsi di masa depan.
        </div>
        """,
        unsafe_allow_html=True,
    )

    projection = {
        "15 menit": prediction,
        "1 jam": prediction * 4,
        "1 hari": prediction * 96,
        "1 minggu": prediction * 96 * 7,
    }

    p1, p2, p3, p4 = st.columns(4)

    for col, (label, value) in zip(
        [p1, p2, p3, p4],
        projection.items(),
    ):
        with col:
            kpi(
                label,
                f"{value:,.2f} kWh",
                "Asumsi nilai per 15 menit tetap",
            )

    # Posisi prediksi terhadap data aktual yang tersedia.
    reference = reference_values()

    if len(reference) >= 10:

        st.markdown(
            '<div class="section-title">Posisi Prediksi</div>',
            unsafe_allow_html=True,
        )

        sorted_ref = np.sort(reference.to_numpy())

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                y=sorted_ref,
                mode="lines",
                name="Data pembanding",
                line=dict(color="#334155", width=2),
            )
        )

        fig.add_hline(
            y=prediction,
            line_color="#22d3ee",
            line_width=3,
            annotation_text=f"Prediksi: {prediction:.4f} kWh",
            annotation_position="top left",
        )

        fig.update_layout(
            title="Prediksi dibandingkan distribusi data",
            xaxis_title="Urutan nilai",
            yaxis_title="kWh",
            template="plotly_dark",
            height=350,
            margin=dict(l=25, r=20, t=55, b=25),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", size=10),
        )

        st.plotly_chart(
            fig,
            width='stretch',
        )

    with st.expander("Lihat detail teknis"):

        st.write("Model")
        st.code("LightGBM + Random Search")

        st.write("Jumlah feature model")
        st.code(str(len(feature_columns)))

        technical = st.session_state.get("last_model_input")

        if technical is not None:
            st.dataframe(
                technical,
                width='stretch',
                hide_index=True,
            )

    export = pd.DataFrame([last_input])
    export["prediksi_kWh"] = prediction

    st.download_button(
        "⬇ Unduh Hasil Prediksi",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_konsumsi.csv",
        mime="text/csv",
        width='stretch',
    )

    if st.button(
        "🔄 Prediksi Kondisi Lain",
        width='stretch',
    ):
        st.session_state["last_prediction"] = None
        st.session_state["last_input"] = None
        st.session_state["last_model_input"] = None
        st.rerun()


# ==========================================================
# RIWAYAT PREDIKSI
# ==========================================================

history = st.session_state.get("prediction_history", [])

if history:

    st.markdown(
        '<div class="section-title">Riwayat Sesi</div>',
        unsafe_allow_html=True,
    )

    history_df = pd.DataFrame(history)

    preferred = [
        "waktu_prediksi",
        "tanggal_lengkap",
        "jam",
        "jumlah_penghuni",
        "temperatur",
        "konsumsi_terakhir",
        "prediksi_kWh",
    ]

    cols = [c for c in preferred if c in history_df.columns]

    display = history_df[cols].iloc[::-1].copy()

    display = display.rename(
        columns={
            "waktu_prediksi": "Dibuat",
            "tanggal_lengkap": "Tanggal",
            "jam": "Jam",
            "jumlah_penghuni": "Penghuni",
            "temperatur": "Temperatur (°C)",
            "konsumsi_terakhir": "Konsumsi terakhir (kWh)",
            "prediksi_kWh": "Prediksi (kWh)",
        }
    )

    st.dataframe(
        display,
        width='stretch',
        hide_index=True,
    )

    st.download_button(
        "⬇ Unduh Riwayat CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="riwayat_prediksi.csv",
        mime="text/csv",
        width='stretch',
    )




# ==========================================================
# 11. FOOTER
# ==========================================================

st.markdown(
    """
    <div class="footer">
        ⚡ Smart Energy AI · Prediksi Konsumsi Energi Smart Grid ·
        LightGBM + Random Search
    </div>
    """,
    unsafe_allow_html=True,
)