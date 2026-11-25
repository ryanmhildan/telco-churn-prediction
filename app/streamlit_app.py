"""Streamlit app: churn probability and its main drivers for one customer.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from src.config import ID_COL
from src.data_cleaning import clean_data, load_raw
from src.predict import ALLOWED_VALUES, INTERNET_ADDONS, load_artifacts, predict

st.set_page_config(page_title="Prediksi Churn Telco", layout="wide")

FIELD_LABELS = {
    "gender": "Jenis kelamin", "SeniorCitizen": "Senior citizen (65+)", "Partner": "Punya pasangan",
    "Dependents": "Punya tanggungan", "tenure": "Lama berlangganan (bulan)", "PhoneService": "Layanan telepon",
    "MultipleLines": "Lebih dari satu saluran", "InternetService": "Layanan internet",
    "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
    "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming film",
    "Contract": "Kontrak", "PaperlessBilling": "Tagihan elektronik (paperless)", "PaymentMethod": "Metode pembayaran",
    "MonthlyCharges": "Tagihan bulanan (USD)", "TotalCharges": "Total tagihan (USD)",
}
VALUE_LABELS = {
    "Yes": "Ya", "No": "Tidak", 1: "Ya", 0: "Tidak", "Female": "Perempuan", "Male": "Laki-laki",
    "No internet service": "Tidak ada internet", "No phone service": "Tidak ada telepon",
    "Month-to-month": "Bulanan", "One year": "1 tahun", "Two year": "2 tahun",
    "Electronic check": "Electronic check", "Mailed check": "Cek via pos",
    "Bank transfer (automatic)": "Transfer bank (otomatis)", "Credit card (automatic)": "Kartu kredit (otomatis)",
}
EXAMPLES = {
    "Isi manual": None,
    "Contoh: risiko tinggi (5178-LMXOP)": "5178-LMXOP",
    "Contoh: risiko menengah (9420-LOJKX)": "9420-LOJKX",
    "Contoh: risiko rendah (1166-PQLGG)": "1166-PQLGG",
}
DEFAULTS = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 840.0, "auto_total": True,
}
RAISE, LOWER = "#eb6834", "#2a78d6"


def label(value) -> str:
    return VALUE_LABELS.get(value, str(value))


def idr(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}Rp {abs(x):,.0f}".replace(",", ".")


@st.cache_resource
def artifacts():
    return load_artifacts()


@st.cache_data
def example_customers() -> dict:
    df = clean_data(load_raw())
    rows = df[df[ID_COL].isin([v for v in EXAMPLES.values() if v])].set_index(ID_COL)
    return {cid: rows.loc[cid].to_dict() for cid in rows.index}


def apply_example():
    cid = EXAMPLES[st.session_state["example"]]
    if cid is None:
        return
    row = example_customers()[cid]
    for field in DEFAULTS:
        if field == "auto_total":
            st.session_state[field] = False
            continue
        value = row[field]
        if field in INTERNET_ADDONS and value == "No internet service":
            value = "No"
        if field == "MultipleLines" and value == "No phone service":
            value = "No"
        if field in ("SeniorCitizen", "tenure"):
            value = int(value)
        if field in ("MonthlyCharges", "TotalCharges"):
            value = float(value)
        st.session_state[field] = value


for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("example", "Isi manual")


def select(field: str, options=None):
    options = options or ALLOWED_VALUES[field]
    return st.selectbox(FIELD_LABELS[field], options, key=field, format_func=label)


# ---------------------------------------------------------------- sidebar
_, _, decision = artifacts()
a = decision["assumptions"]
with st.sidebar:
    st.header("Contoh pelanggan")
    st.selectbox("Isi formulir dari data test", list(EXAMPLES), key="example", on_change=apply_example)
    st.header("Aturan keputusan")
    st.markdown(
        f"Hubungi pelanggan dengan penawaran retensi jika probabilitas churn **≥ {decision['threshold']:.3f}**.\n\n"
        f"Threshold ini dipilih di Fase 5 untuk memaksimalkan profit dengan asumsi:\n"
        f"- Biaya penawaran: **{idr(a['retention_cost_idr'])}**\n"
        f"- Kerugian per pelanggan churn: **{idr(a['churn_loss_idr'])}**\n"
        f"- Tingkat keberhasilan retensi: **{a['retention_success_rate']:.0%}**\n\n"
        "Ubah asumsi di `src/config.py`, lalu jalankan `python -m src.threshold`."
    )

# ---------------------------------------------------------------- form
st.title("Prediksi Churn Pelanggan Telco")
st.caption("Model LightGBM dilatih pada dataset sintetis IBM Telco Customer Churn (tagihan dalam USD). "
           "Hasilnya belum tentu mewakili pelanggan telco di Indonesia.")

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Profil")
    select("gender")
    select("SeniorCitizen")
    select("Partner")
    select("Dependents")
with col2:
    st.subheader("Layanan")
    phone = select("PhoneService")
    if phone == "Yes":
        select("MultipleLines", ["No", "Yes"])
    internet = select("InternetService")
    if internet != "No":
        for addon in INTERNET_ADDONS:
            select(addon, ["No", "Yes"])
    else:
        st.caption("Layanan tambahan internet tidak berlaku tanpa layanan internet.")
with col3:
    st.subheader("Akun dan tagihan")
    tenure = st.slider(FIELD_LABELS["tenure"], 0, 72, key="tenure")
    select("Contract")
    select("PaperlessBilling")
    select("PaymentMethod")
    monthly = st.number_input(FIELD_LABELS["MonthlyCharges"], 0.0, 200.0, step=0.05, key="MonthlyCharges")
    auto_total = st.checkbox("Hitung total tagihan otomatis (lama berlangganan × tagihan bulanan)", key="auto_total")
    if auto_total:
        total = round(tenure * monthly, 2)
        st.metric(FIELD_LABELS["TotalCharges"], f"{total:,.2f}")
    else:
        total = st.number_input(FIELD_LABELS["TotalCharges"], 0.0, 20000.0, step=1.0, key="TotalCharges")

customer = {f: st.session_state[f] for f in DEFAULTS if f not in ("auto_total", "TotalCharges")}
customer["TotalCharges"] = total
if phone == "No":
    customer["MultipleLines"] = "No phone service"
if internet == "No":
    customer.update({c: "No internet service" for c in INTERNET_ADDONS})

# ---------------------------------------------------------------- result
result = predict(customer, top_k=6)
st.divider()
st.subheader("Hasil prediksi")
m1, m2, m3 = st.columns(3)
m1.metric("Probabilitas churn", f"{result.proba:.1%}")
m2.metric("Hubungi?", "Ya" if result.contact else "Tidak",
          help=f"Hubungi dengan penawaran retensi jika probabilitas ≥ {result.threshold:.3f}")
m3.metric("Nilai harapan", idr(result.expected_value_idr),
          help="Nilai harapan jika pelanggan ini diberi penawaran retensi: "
               "probabilitas × tingkat keberhasilan × kerugian per churn − biaya penawaran")
if result.contact and result.expected_value_idr < 0:
    st.info("Pelanggan ini berada di zona batas: probabilitasnya di atas threshold empiris, tetapi nilai "
            "harapannya sedikit negatif. Di zona ini kurva profit hampir datar, jadi keputusan mana pun "
            "berdampak kecil.")

drivers = result.drivers.copy()
drivers["text"] = [f"{FIELD_LABELS[f]} = {label(v) if f not in ('tenure', 'MonthlyCharges', 'TotalCharges') else v}"
                   for f, v in zip(drivers["feature"], drivers["value"])]

st.markdown("#### Faktor utama di balik skor ini")
plot = drivers.iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 0.45 * len(plot) + 1.0))
fig.patch.set_alpha(0)
ax.set_facecolor("none")
bars = ax.barh(plot["text"], plot["shap"], color=[RAISE if s > 0 else LOWER for s in plot["shap"]], height=0.6)
ax.bar_label(bars, labels=[f"{s:+.2f}" for s in plot["shap"]], padding=3, fontsize=10, color="#0b0b0b")
ax.axvline(0, color="#52514e", lw=0.8)
lim = max(abs(plot["shap"]).max() * 1.25, 0.1)
ax.set_xlim(-lim, lim)
ax.set_xlabel("Kontribusi SHAP (log-odds)  ←  menurunkan risiko  |  menaikkan risiko  →", fontsize=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(labelsize=11)
ax.grid(axis="x", color="#e4e3df", lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

up = drivers[drivers["shap"] > 0]["text"].tolist()
down = drivers[drivers["shap"] < 0]["text"].tolist()
c_up, c_down = st.columns(2)
with c_up:
    st.markdown("**Menaikkan risiko:**\n" + ("\n".join(f"- {t}" for t in up) if up else "\n- (tidak ada)"))
with c_down:
    st.markdown("**Menurunkan risiko:**\n" + ("\n".join(f"- {t}" for t in down) if down else "\n- (tidak ada)"))
st.caption("Faktor ini menjelaskan cara model menghitung skor, bukan bukti sebab-akibat. "
           "Gunakan sebagai petunjuk awal untuk percakapan retensi.")
