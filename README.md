# Financial Diagnostics & Late Payment Risk Prediction (CRISP-DM)
### Kasus PT Maju Bersama: Analisis Piutang Tertahan & Prediksi Risiko Gagal Bayar

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Latar Belakang Masalah
**PT Maju Bersama** mengalami anomali finansial yang mengancam kelangsungan operasional bisnis:
* **Pendapatan (*Revenue*) meningkat pesat** seiring dengan perluasan volume penjualan barang dan penagihan faktur.
* Namun, **Laba Bersih (*Net Profit*) dan Arus Kas Operasional (*Cash Flow*) mengalami penurunan drastis**.

### Akar Masalah: Penumpukan Piutang (*Accounts Receivable*)
Setelah ditelaah lebih dalam, perusahaan menghadapi akumulasi piutang tidak tertagih yang melebihi batas toleransi jatuh tempo (*due date + 30 days*). Kondisi ini menyebabkan *Cash Conversion Cycle* membengkak, modal kerja tertahan di pihak pembeli, dan timbulnya beban pencadangan piutang ragu-ragu (*allowance for doubtful accounts*) yang mengikis laba bersih.

---

## 🎯 Tujuan Proyek
1. **Analisis Diagnostik Keuangan**:
   Mengidentifikasi sumber penumpukan piutang, distribusi kegagalan bayar berdasarkan metode pembayaran (*Giro, Kredit, Transfer*), dan pengaruh histori keterlambatan terhadap penundaan kas.
2. **Prediksi Risiko Keterlambatan Bayar (*Late Payment Risk*)**:
   Membangun model machine learning klasifikasi biner untuk memprediksi probabilitas faktur terlambat dibayar lebih dari 30 hari sejak tanggal jatuh tempo.
3. **Mitigasi Kebocoran Data (*Feature Leakage Prevention*)**:
   Menghindari penggunaan variabel yang hanya ada setelah tagihan bermasalah/ditagih intensif (seperti `collection_action_days`), sehingga model dapat diandalkan secara valid pada tahap penerbitan faktur (*pra-transaksi*).

---

## 🔬 Metodologi: CRISP-DM Framework

```mermaid
flowchart TD
    A[1. Business Understanding\nMasalah Likuiditas vs Revenue PT Maju Bersama] --> B[2. Data Understanding\nEksplorasi Transaksi, Nominal Faktur & Riwayat Delay]
    B --> C[3. Analytical Diagnostics\nRasio Piutang Macet: 51.93% dari Revenue]
    C --> D[4. Data Preparation\nEliminasi Feature Leakage & One-Hot Encoding]
    D --> E[5. Modeling\nGradient Boosting Classifier]
    E --> F[6. Evaluation\nROC-AUC: 0.9646 | Accuracy: 88.67%]
    F --> G[7. Financial Recommendations\nRestrukturisasi Giro, Early Discount 2/10 Net 30 & Early Warning System]
```

---

## 📊 Hasil Analisis Diagnostik Keuangan

Dari simulasi 1.200 faktur transaksi PT Maju Bersama:
* **Total Nilai Faktur Diterbitkan (*Revenue*)**: `Rp 6.684.535,41`
* **Total Piutang Tertahan (> 30 Hari)**: `Rp 3.470.980,03` (**51,93%** dari total revenue tertahan dalam piutang berisiko tinggi).
* **Frekuensi Faktur Macet**: **615 dari 1.200 faktur (51,25%)** terlambat lebih dari 30 hari.

### Breakdown per Metode Pembayaran:
| Metode Pembayaran | Jumlah Faktur | Faktur Macet | Rasio Macet (%) | Nilai Tertahan (Rp) | Kontribusi Tertahan |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Giro** | 441 | 341 | **77,3%** | Rp 1.969.473,68 | **74,8%** |
| **Kredit** | 395 | 143 | **36,2%** | Rp 768.014,80 | **35,1%** |
| **Transfer** | 364 | 131 | **36,0%** | Rp 733.491,55 | **39,4%** |

> **Temuan Kunci**: Metode pembayaran **Giro** merupakan kontributor utama penurunan arus kas dengan tingkat gagal bayar mencapai **77,3%**, mengindikasikan banyaknya giro mundur/kosong yang tidak tertagih tepat waktu.

### Pengaruh Histori Keterlambatan (*Historical Delay Count*):
* Pelanggan tanpa riwayat delay (`0`): Tingkat keterlambatan hanya **7,4%**.
* Pelanggan dengan riwayat delay (`1 - 2 kali`): Tingkat keterlambatan melonjak ke **53,7%**.
* Pelanggan dengan riwayat delay (`> 2 kali`): Tingkat keterlambatan mencapai **100,0%**.

---

## 🛡️ Mitigasi Feature Leakage
Variabel **`collection_action_days`** (jumlah hari tindakan penagihan) dihapus dari fitur pelatihan karena:
- Variabel ini diisi oleh tim penagihan (*debt collector*) **hanya setelah** invoice melewati batas toleransi jatuh tempo.
- Menyertakan variabel ini menciptakan korelasi artifisial semu yang tidak dapat dipakai saat inferensi penerbitan faktur pertama kali (*invoice creation time*).

---

## 🤖 Performa Model Prediktif

Model dilatih menggunakan **Gradient Boosting Classifier** dengan stratifikasi data 75:25.

### Metrik Evaluasi:
* **Akurasi Model**: `88,67%`
* **Area Under ROC Curve (ROC-AUC)**: `0.9646`

```
                         precision    recall  f1-score   support

        On-Time Payment       0.89      0.87      0.88       146
Late Payment (>30 Days)       0.88      0.90      0.89       154

               accuracy                           0.89       300
              macro avg       0.89      0.89      0.89       300
           weighted avg       0.89      0.89      0.89       300
```

### Feature Importance:
1. `historical_delay_count`: **59,64%**
2. `payment_method_Transfer`: **17,12%**
3. `payment_method_Kredit`: **14,88%**
4. `invoice_amount`: **6,08%**
5. `customer_tenure_months`: **2,28%**

---

## 💡 Rekomendasi Solusi untuk PT Maju Bersama

1. **Pengetatan Kebijakan Pembayaran Giro**:
   - Membatasi batas plafon transaksi giro dan mensyaratkan konfirmasi saldo bank sebelum persetujuan pesanan.
2. **Insentif Pembayaran Cepat (*Early Payment Discount*)**:
   - Menerapkan klausul termin pembayaran diskon tunai (contoh: *2/10 Net 30*) guna mempercepat realisasi kas masuk.
3. **Penerapan Early Warning System Berbasis AI**:
   - Setiap invoice yang diterbitkan di sistem ERP langsung dinilai tingkat risikonya. Faktur dengan skor risiko > 0.70 otomatis masuk ke radar *proactive reminder* (H-7 sebelum jatuh tempo).
4. **Pencegahan Data Leakage Berkelanjutan**:
   - Menjaga standardisasi data pipeline agar fitur yang digunakan hanya berasal dari data pra-penagihan.

---

## 📂 Struktur Proyek
```text
financial-diagnostics-late-payment-risk/
├── src/
│   └── financial_diagnostics.py   # Implementasi lengkap CRISP-DM
├── reports/
│   └── figures/
│       └── financial_diagnostics.png # Visualisasi analitik & feature importance
├── requirements.txt               # Daftar pustaka yang dibutuhkan
├── .gitignore                     # Filter file cache & lingkungan kerja
└── README.md                      # Dokumentasi komprehensif proyek
```

---

## 🚀 Panduan Menjalankan Kode

### 1. Kloning Repositori
```bash
git clone https://github.com/MuhammadFadliRahmansyahG1A023005/financial-diagnostics-late-payment-risk.git
cd financial-diagnostics-late-payment-risk
```

### 2. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 3. Eksekusi Analisis
```bash
python src/financial_diagnostics.py
```

---

## 👨‍💻 Penulis
**Muhammad Fadli Rahmansyah**  
GitHub: [@MuhammadFadliRahmansyahG1A023005](https://github.com/MuhammadFadliRahmansyahG1A023005)
