# Rekomendasi Bisnis — Prediksi Churn Pelanggan Telco

Dasar rekomendasi: nilai SHAP model LightGBM pada data test (`reports/shap_global_importance.csv`, grafik `reports/figures/09`–`12`) dan statistik segmen pada seluruh 7.043 pelanggan (`notebooks/03_interpretation.ipynb`).

> **Penting.** SHAP dan tabel segmen menunjukkan **hubungan**, bukan **sebab-akibat**. Data ini sintetis dari IBM dan belum tentu mewakili pelanggan telco di Indonesia. Setiap rekomendasi di bawah sebaiknya diuji dulu lewat eksperimen terkontrol (A/B test) sebelum diterapkan luas.

## Faktor risiko utama menurut model

| Peringkat | Fitur | Mean \|SHAP\| | Arah |
|---|---|---|---|
| 1 | Contract | 0,752 | Kontrak bulanan menaikkan risiko; kontrak 2 tahun menurunkannya paling kuat |
| 2 | tenure | 0,421 | Risiko tertinggi di bulan-bulan awal, turun tajam sampai sekitar bulan ke-20 |
| 3 | InternetService | 0,403 | Fiber optic menaikkan risiko; tanpa internet menurunkannya |
| 4 | MonthlyCharges | 0,195 | Risiko naik tajam saat tagihan di atas sekitar 95 (USD) |
| 5 | PaymentMethod | 0,169 | Electronic check menaikkan risiko |

`gender`, `Partner`, dan `DeviceProtection` hampir tidak berpengaruh (mean |SHAP| ≤ 0,011).

## Rekomendasi

### 1. Program onboarding 12 bulan pertama untuk pelanggan kontrak bulanan
- **Bukti:** pelanggan kontrak bulanan dengan tenure ≤ 12 bulan adalah 28,3% pelanggan, tetapi menyumbang **54,8% dari seluruh churn** (churn rate 51,4%). SHAP `tenure` paling tinggi di bulan 0–12.
- **Aksi:** check-in proaktif di bulan 1, 3, dan 6; panduan pemakaian layanan; penanganan keluhan pertama yang cepat.
- **Ukur:** churn rate kohort pelanggan baru (grup program vs grup kontrol).

### 2. Tawarkan migrasi ke kontrak 1–2 tahun kepada pelanggan bulanan berisiko tinggi
- **Bukti:** pada kelompok tenure yang sama (≤ 12 bulan), churn kontrak bulanan 51,4% vs kontrak 1 tahun 10,5% dan 2 tahun 0%. `Contract` adalah fitur terpenting model.
- **Aksi:** insentif (diskon atau bonus kuota) untuk pindah ke kontrak tahunan, ditargetkan ke pelanggan bulanan yang skor risikonya di atas threshold.
- **Catatan:** pelanggan yang memilih kontrak panjang kemungkinan memang sudah lebih loyal (*selection bias*). Memindahkan pelanggan ke kontrak panjang belum tentu menurunkan churn sebesar selisih di atas. Inilah alasan perlunya A/B test.

### 3. Evaluasi pengalaman pelanggan fiber optic dan bundel layanan dukungan
- **Bukti:** churn pelanggan fiber 41,9% vs DSL 19,0%. Selisih ini **bukan hanya soal harga**. Pada tagihan bulanan 70–90, churn fiber 50,9% vs DSL 7,1%. Pelanggan fiber tanpa TechSupport dan tanpa OnlineSecurity (25,1% pelanggan) churn 55,0% dan menyumbang 52,0% dari seluruh churn. Fiber dengan salah satu layanan itu churn 24,5%.
- **Aksi:** audit kualitas jaringan fiber, gangguan, dan harga dibanding kompetitor; uji coba gratis TechSupport/OnlineSecurity selama 3 bulan untuk pelanggan fiber baru.
- **Catatan:** pelanggan yang membeli TechSupport mungkin berbeda karakteristiknya, jadi efek bundel harus diuji, bukan diasumsikan.

### 4. Dorong pembayaran otomatis
- **Bukti:** churn pengguna electronic check 45,3% (57,3% dari seluruh churn) vs pembayaran otomatis (bank transfer/kartu kredit) 16,0%. SHAP electronic check konsisten positif.
- **Aksi:** insentif kecil (misalnya potongan tagihan satu kali) untuk beralih ke autodebit.
- **Catatan:** metode pembayaran kemungkinan proksi dari keterikatan pelanggan, bukan penyebab langsung.

### 5. Pakai skor model dan threshold berbasis biaya untuk memilih target kampanye, lalu ukur hasilnya
- **Bukti:** dengan asumsi default (biaya Rp 50.000, kerugian Rp 1.500.000, tingkat sukses 50%), threshold 0,045 menghasilkan profit Rp 224,75 juta di data test vs Rp 132,15 juta dengan threshold 0,5. Model makin bernilai saat biaya retensi lebih mahal (lihat `reports/threshold_sensitivity.csv`).
- **Aksi:** jalankan kampanye retensi dengan grup kontrol acak supaya **tingkat keberhasilan retensi yang sebenarnya** bisa diukur. Angka 50% saat ini hanya asumsi. Perbarui `src/config.py` dengan hasil pengukuran itu.
