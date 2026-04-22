import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_pipe_model_2026_04_22_1636.pkl")

FEATURE_ORDER = [
    "skor_kredit", "peminjam_baru", "kode_provinsi", "kondisi_motor",
    "persen_dp_awal", "dti_awal", "jumlah_pinjaman", "suku_bunga_awal",
    "kanal", "merek_motor", "tipe_motor", "tujuan_pinjaman",
    "tenor_pinjaman_awal", "cabang", "jenis_pekerjaan", "pendidikan_terakhir",
]

# ──────────────────────────────────────────────────────────────
# CATEGORICAL MAPS  (label → value)
# ──────────────────────────────────────────────────────────────
CAT_OPTIONS = {
    "peminjam_baru": {"Ya": "Y", "Tidak": "N"},
    "kondisi_motor": {"Baru (N)": "N", "Bekas (U)": "U"},
    "kanal": {"Dealer (D)": "D", "Broker (B)": "B", "Online (O)": "O"},
    "merek_motor": {"Honda": "Honda", "Kawasaki": "Kawasaki", "Suzuki": "Suzuki", "Yamaha": "Yamaha"},
    "tipe_motor": {"Matic (AT)": "AT", "Manual (MT)": "MT", "Sport (SP)": "SP"},
    "tujuan_pinjaman": {"Pembelian (P)": "P", "Refinancing (R)": "R"},
    "cabang": {
        "Cabang Jakarta Pusat": "Cabang Jakarta Pusat",
        "Cabang Jakarta Selatan": "Cabang Jakarta Selatan",
        "Cabang Bandung": "Cabang Bandung",
        "Cabang Semarang": "Cabang Semarang",
        "Cabang Surabaya": "Cabang Surabaya",
        "Cabang Makassar": "Cabang Makassar",
        "Cabang Medan": "Cabang Medan",
    },
    "jenis_pekerjaan": {"Karyawan (E)": "E", "Wiraswasta (I)": "I", "Profesional (S)": "S"},
    "pendidikan_terakhir": {"D3": "D3", "S1 ke atas": "S1 ke atas", "SMA/SMK": "SMA/SMK"},
}

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Default Pinjaman Motor",
    page_icon="🏍️",
    layout="centered",
)

# ──────────────────────────────────────────────────────────────
# CUSTOM CSS – BRUTALIST AESTHETIC
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Font: bold monospace + heavy sans ──────────── */
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Base overrides ─────────────────────────────── */
.stApp {
    background-color: #FFFDF0 !important;
}

/* ── Header banner ──────────────────────────────── */
.brut-header {
    background: #1a1a1a;
    border: 4px solid #000000;
    box-shadow: 8px 8px 0px #000000;
    padding: 2rem 1.8rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.brut-header::before {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 120px; height: 120px;
    background: #FFE500;
    clip-path: polygon(100% 0, 0 0, 100% 100%);
}
.brut-header h1 {
    font-family: 'Space Mono', monospace;
    color: #FFFFFF;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    text-transform: uppercase;
    letter-spacing: 2px;
    line-height: 1.2;
}
.brut-header p {
    color: #FFE500;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    margin: 0;
    letter-spacing: 0.5px;
}

/* ── Section blocks ─────────────────────────────── */
.brut-section {
    background: #FFFFFF;
    border: 3px solid #000000;
    box-shadow: 6px 6px 0px #000000;
    padding: 0;
    margin-bottom: 1.8rem;
    transition: box-shadow 0.15s ease, transform 0.15s ease;
}
.brut-section:hover {
    box-shadow: 8px 8px 0px #000000;
    transform: translate(-1px, -1px);
}
.brut-section-header {
    background: #000000;
    padding: 0.7rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.brut-section-header .tag {
    background: #FFE500;
    color: #000000;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    display: inline-block;
}
.brut-section-header .title {
    color: #FFFFFF;
    font-family: 'Space Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Labels ─────────────────────────────────────── */
.stSelectbox > label,
.stNumberInput > label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: #1a1a1a !important;
}

/* ── Inputs ─────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div,
.stNumberInput input {
    border: 2px solid #000000 !important;
    border-radius: 0px !important;
    background: #FFFDF0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    box-shadow: 3px 3px 0px #000000 !important;
    transition: box-shadow 0.1s ease, transform 0.1s ease !important;
}
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stNumberInput input:focus {
    box-shadow: 5px 5px 0px #FFE500 !important;
    border-color: #000000 !important;
    transform: translate(-1px, -1px) !important;
}

/* ── Result cards ───────────────────────────────── */
.brut-result {
    border: 4px solid #000000;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    position: relative;
}
.brut-result.is-default {
    background: #FF3B30;
    box-shadow: 8px 8px 0px #000000;
    color: #FFFFFF;
}
.brut-result.is-safe {
    background: #34C759;
    box-shadow: 8px 8px 0px #000000;
    color: #000000;
}
.brut-result .result-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}
.brut-result h2 {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin: 0 0 0.3rem 0;
}
.brut-result.is-default h2 { color: #FFFFFF; }
.brut-result.is-safe h2 { color: #000000; }
.brut-result p {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.95rem;
    margin: 0;
}
.brut-result.is-default p { color: rgba(255,255,255,0.85); }
.brut-result.is-safe p { color: rgba(0,0,0,0.7); }

/* ── Probability badges ─────────────────────────── */
.brut-prob-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1.2rem;
}
.brut-prob {
    background: #000000;
    border: 3px solid #000000;
    padding: 0.6rem 1.2rem;
    text-align: center;
    min-width: 140px;
}
.brut-prob .lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #FFE500;
    margin-bottom: 0.2rem;
}
.brut-prob .val {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #FFFFFF;
}

/* ── Summary table ──────────────────────────────── */
.brut-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}
.brut-table th, .brut-table td {
    padding: 0.6rem 0.8rem;
    text-align: left;
    border: 2px solid #000000;
}
.brut-table th {
    background: #1a1a1a;
    color: #FFE500;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    width: 50%;
}
.brut-table td {
    background: #FFFDF0;
    color: #1a1a1a;
    font-weight: 500;
}
.brut-table tr:hover td {
    background: #FFE500;
}

/* ── Button ─────────────────────────────────────── */
div.stButton > button {
    background: #FFE500 !important;
    color: #000000 !important;
    border: 3px solid #000000 !important;
    border-radius: 0px !important;
    padding: 0.85rem 2rem !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
    box-shadow: 6px 6px 0px #000000 !important;
    transition: all 0.1s ease !important;
    width: 100%;
}
div.stButton > button:hover {
    transform: translate(-2px, -2px) !important;
    box-shadow: 8px 8px 0px #000000 !important;
    background: #FFD000 !important;
}
div.stButton > button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0px #000000 !important;
}

/* ── Expander ───────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border: 2px solid #000000 !important;
    border-radius: 0 !important;
    background: #FFFFFF !important;
}

/* ── Footer divider ─────────────────────────────── */
hr {
    border-top: 3px solid #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="brut-header">
    <h1>🏍️ Prediksi Default<br>Pinjaman Motor</h1>
    <p>// masukkan data peminjam → prediksi gagal bayar</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LOAD MODEL (cached)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ══════════════════════════════════════════════════════════════
# INPUT FORM – grouped by logical domain
# ══════════════════════════════════════════════════════════════

# ── 1. Informasi Peminjam ──────────────────────────────────
st.markdown("""
<div class="brut-section">
    <div class="brut-section-header">
        <span class="tag">01</span>
        <span class="title">Informasi Peminjam</span>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    skor_kredit = st.number_input(
        "Skor Kredit",
        min_value=300, max_value=850, value=650, step=1,
        help="Skor kredit peminjam (300-850)"
    )
    peminjam_baru_label = st.selectbox(
        "Peminjam Baru?",
        options=list(CAT_OPTIONS["peminjam_baru"].keys()),
    )
with col2:
    jenis_pekerjaan_label = st.selectbox(
        "Jenis Pekerjaan",
        options=list(CAT_OPTIONS["jenis_pekerjaan"].keys()),
    )
    pendidikan_label = st.selectbox(
        "Pendidikan Terakhir",
        options=list(CAT_OPTIONS["pendidikan_terakhir"].keys()),
    )

kode_provinsi = st.number_input(
    "Kode Provinsi",
    min_value=11, max_value=94, value=33, step=1,
    help="Kode provinsi BPS (contoh: 33 = Jawa Tengah)"
)

# ── 2. Informasi Motor ────────────────────────────────────
st.markdown("""
<div class="brut-section">
    <div class="brut-section-header">
        <span class="tag">02</span>
        <span class="title">Informasi Motor</span>
    </div>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    merek_motor_label = st.selectbox(
        "Merek Motor",
        options=list(CAT_OPTIONS["merek_motor"].keys()),
    )
    tipe_motor_label = st.selectbox(
        "Tipe Motor",
        options=list(CAT_OPTIONS["tipe_motor"].keys()),
    )
with col4:
    kondisi_motor_label = st.selectbox(
        "Kondisi Motor",
        options=list(CAT_OPTIONS["kondisi_motor"].keys()),
    )
    tujuan_pinjaman_label = st.selectbox(
        "Tujuan Pinjaman",
        options=list(CAT_OPTIONS["tujuan_pinjaman"].keys()),
    )

# ── 3. Informasi Pinjaman ─────────────────────────────────
st.markdown("""
<div class="brut-section">
    <div class="brut-section-header">
        <span class="tag">03</span>
        <span class="title">Informasi Pinjaman</span>
    </div>
</div>
""", unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    jumlah_pinjaman = st.number_input(
        "Jumlah Pinjaman (Rp)",
        min_value=1_000_000, max_value=100_000_000,
        value=17_000_000, step=500_000,
        help="Jumlah pinjaman dalam Rupiah"
    )
    persen_dp_awal = st.number_input(
        "Persen DP Awal (%)",
        min_value=0.0, max_value=100.0, value=20.0, step=1.0,
        help="Persentase uang muka awal"
    )
    tenor_pinjaman_awal = st.selectbox(
        "Tenor Pinjaman (bulan)",
        options=[12, 18, 24, 36, 48],
        index=2,
    )
with col6:
    suku_bunga_awal = st.number_input(
        "Suku Bunga Awal (%)",
        min_value=0.0, max_value=50.0, value=17.0, step=0.5,
        help="Suku bunga pinjaman per tahun"
    )
    dti_awal = st.number_input(
        "DTI Awal (%)",
        min_value=0.0, max_value=100.0, value=25.0, step=0.1,
        help="Debt-to-Income ratio"
    )

# ── 4. Informasi Cabang & Kanal ────────────────────────────
st.markdown("""
<div class="brut-section">
    <div class="brut-section-header">
        <span class="tag">04</span>
        <span class="title">Cabang & Kanal</span>
    </div>
</div>
""", unsafe_allow_html=True)

col7, col8 = st.columns(2)
with col7:
    cabang_label = st.selectbox(
        "Cabang",
        options=list(CAT_OPTIONS["cabang"].keys()),
    )
with col8:
    kanal_label = st.selectbox(
        "Kanal",
        options=list(CAT_OPTIONS["kanal"].keys()),
    )

# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

if st.button("⚡ PREDIKSI SEKARANG"):

    # Map labels → values
    input_dict = {
        "skor_kredit": skor_kredit,
        "peminjam_baru": CAT_OPTIONS["peminjam_baru"][peminjam_baru_label],
        "kode_provinsi": kode_provinsi,
        "kondisi_motor": CAT_OPTIONS["kondisi_motor"][kondisi_motor_label],
        "persen_dp_awal": persen_dp_awal,
        "dti_awal": dti_awal,
        "jumlah_pinjaman": jumlah_pinjaman,
        "suku_bunga_awal": suku_bunga_awal,
        "kanal": CAT_OPTIONS["kanal"][kanal_label],
        "merek_motor": CAT_OPTIONS["merek_motor"][merek_motor_label],
        "tipe_motor": CAT_OPTIONS["tipe_motor"][tipe_motor_label],
        "tujuan_pinjaman": CAT_OPTIONS["tujuan_pinjaman"][tujuan_pinjaman_label],
        "tenor_pinjaman_awal": tenor_pinjaman_awal,
        "cabang": CAT_OPTIONS["cabang"][cabang_label],
        "jenis_pekerjaan": CAT_OPTIONS["jenis_pekerjaan"][jenis_pekerjaan_label],
        "pendidikan_terakhir": CAT_OPTIONS["pendidikan_terakhir"][pendidikan_label],
    }

    # Build DataFrame in correct column order
    input_df = pd.DataFrame([input_dict])[FEATURE_ORDER]

    # Predict
    prediction = model.predict(input_df)[0]

    # Try to get probability (underlying estimator is a pipeline)
    try:
        proba = model.estimator.predict_proba(input_df)[0]
        prob_no_default = proba[0]
        prob_default = proba[1]
        has_proba = True
    except Exception:
        has_proba = False

    # Display result
    if prediction == 1:
        icon = "🚨"
        card_class = "is-default"
        title = "DEFAULT"
        subtitle = "Peminjam berisiko tinggi gagal bayar"
    else:
        icon = "✅"
        card_class = "is-safe"
        title = "TIDAK DEFAULT"
        subtitle = "Peminjam diprediksi mampu membayar"

    result_html = f"""
    <div class="brut-result {card_class}">
        <div class="result-icon">{icon}</div>
        <h2>{title}</h2>
        <p>{subtitle}</p>
    """

    if has_proba:
        result_html += f"""
        <div class="brut-prob-row">
            <div class="brut-prob">
                <div class="lbl">Tidak Default</div>
                <div class="val">{prob_no_default:.1%}</div>
            </div>
            <div class="brut-prob">
                <div class="lbl">Default</div>
                <div class="val">{prob_default:.1%}</div>
            </div>
        </div>
        """

    result_html += "</div>"
    st.markdown(result_html, unsafe_allow_html=True)

    # ── Input summary table ────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 RINGKASAN INPUT", expanded=False):
        display_labels = {
            "skor_kredit": "Skor Kredit",
            "peminjam_baru": "Peminjam Baru",
            "kode_provinsi": "Kode Provinsi",
            "kondisi_motor": "Kondisi Motor",
            "persen_dp_awal": "Persen DP Awal (%)",
            "dti_awal": "DTI Awal (%)",
            "jumlah_pinjaman": "Jumlah Pinjaman (Rp)",
            "suku_bunga_awal": "Suku Bunga Awal (%)",
            "kanal": "Kanal",
            "merek_motor": "Merek Motor",
            "tipe_motor": "Tipe Motor",
            "tujuan_pinjaman": "Tujuan Pinjaman",
            "tenor_pinjaman_awal": "Tenor Pinjaman (bulan)",
            "cabang": "Cabang",
            "jenis_pekerjaan": "Jenis Pekerjaan",
            "pendidikan_terakhir": "Pendidikan Terakhir",
        }

        rows = ""
        for feat in FEATURE_ORDER:
            val = input_dict[feat]
            if feat == "jumlah_pinjaman":
                val = f"Rp {val:,.0f}"
            rows += f"<tr><th>{display_labels[feat]}</th><td>{val}</td></tr>"

        st.markdown(
            f'<table class="brut-table">{rows}</table>',
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.caption("MODEL → TunedThresholdClassifierCV → Pipeline (OneHotEncoder + BinaryEncoder + RobustScaler → RFE + LogisticRegression)")
