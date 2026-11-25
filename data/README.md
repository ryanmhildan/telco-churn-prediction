# Data

## Sumber yang dipakai

| Item | Nilai |
|---|---|
| Dataset | IBM Telco Customer Churn (IBM Sample Data Sets) |
| Sumber unduhan | Hugging Face — [`scikit-learn/churn-prediction`](https://huggingface.co/datasets/scikit-learn/churn-prediction) |
| Revisi (commit) | `aa09900373d90780ee70d27571775aff0e51569c` |
| Tanggal unduh | 2026-11-25 (UTC 11:50:52) |
| Lisensi | CC BY 4.0 (sesuai tag lisensi di halaman dataset Hugging Face) |
| Ukuran | 7.043 baris × 21 kolom |
| File | `raw/telco_customer_churn.csv` |
| SHA-256 | `4cd901efab211ccb154051a93a1ab5e9298e37e8fd29bbf2c097da23cee36017` |

Metadata lengkap dicatat otomatis di `raw/download_metadata.json`.

**Hanya satu versi dataset yang dipakai**, yaitu file di atas. OpenML (`telco-customer-churn` versi 1) dan Kaggle (`blastchar/telco-customer-churn`) hanya cadangan di `src/download_data.py` kalau Hugging Face tidak bisa diakses. OpenML punya dua versi dataset dengan nama yang sama, jadi skrip mengunci versi 1 dan memvalidasi nama kolom serta jumlah baris sebelum menyimpan.

## Cara mengunduh ulang

```powershell
.\.venv\Scripts\python.exe -m src.download_data --force
```

Skrip akan berhenti dengan pesan error (bukan membuat data palsu) kalau semua sumber gagal. Kalau itu terjadi, unduh `WA_Fn-UseC_-Telco-Customer-Churn.csv` dari Kaggle secara manual dan simpan sebagai `data/raw/telco_customer_churn.csv`.

## Catatan penting

- Data ini **sintetis**, dibuat IBM sebagai contoh untuk produk analitik mereka. Pelanggan dan perusahaan telco-nya fiktif, tagihannya dalam USD, dan pola di dalamnya **belum tentu mewakili pelanggan telco di Indonesia** (struktur harga, metode pembayaran, jenis kontrak, dan perilaku pelanggan bisa sangat berbeda).
- Tidak ada informasi waktu (tanggal), jadi tidak bisa dilakukan validasi berbasis waktu.
- `raw/` tidak pernah diubah. Hasil cleaning disimpan di `processed/telco_clean.csv` oleh `src/data_cleaning.py`.

## Kamus kolom

| Kolom | Arti |
|---|---|
| `customerID` | ID pelanggan (dibuang sebelum modeling) |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demografi |
| `tenure` | Lama berlangganan (bulan) |
| `PhoneService`, `MultipleLines` | Layanan telepon |
| `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Layanan internet dan tambahan |
| `Contract`, `PaperlessBilling`, `PaymentMethod` | Akun dan pembayaran |
| `MonthlyCharges`, `TotalCharges` | Tagihan bulanan dan total (USD) |
| `Churn` | Target: pelanggan berhenti dalam bulan terakhir (Yes/No) |
