"""
Analisis Diagnostik Keuangan dan Prediksi Terlambat Bayar (Late Payment Risk)
Studi Kasus: PT Maju Bersama
Framework: CRISP-DM (Cross-Industry Standard Process for Data Mining)

Latar Belakang:
PT Maju Bersama menghadapi kondisi anomali finansial: pendapatan (revenue)
mengalami pertumbuhan, namun laba bersih (net profit) dan arus kas (cash flow)
mengalami penurunan tajam akibat penumpukan piutang usaha (accounts receivable).

Tujuan:
1. Menjalankan analisis diagnostik untuk mengidentifikasi akar penyebab penurunan arus kas.
2. Membangun model prediktif klasifikasi risiko tagihan piutang (> 30 hari dari due date).
3. Mencegah kebocoran data (feature leakage) dengan memvalidasi ketersediaan variabel saat pra-transaksi.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)


def generate_pt_maju_bersama_data(n_invoices: int = 1200, random_seed: int = 101) -> pd.DataFrame:
    """
    Fase 1 & 2 CRISP-DM: Business Understanding & Data Understanding.
    Menghasilkan data transaksi penagihan sintetis PT Maju Bersama.
    """
    np.random.seed(random_seed)

    data = {
        "invoice_id": [f"INV-{i:04d}" for i in range(n_invoices)],
        "invoice_amount": np.random.exponential(scale=5000, size=n_invoices) + 500,
        "customer_tenure_months": np.random.randint(1, 60, size=n_invoices),
        "payment_method": np.random.choice(["Transfer", "Kredit", "Giro"], size=n_invoices),
        "historical_delay_count": np.random.poisson(lam=1.5, size=n_invoices),
        # Fitur Leakage: Tindakan penagihan hanya ada SETELAH invoice terbit & jatuh tempo
        "collection_action_days": np.random.randint(0, 45, size=n_invoices),
    }

    df = pd.DataFrame(data)

    # Target Label: Terlambat Bayar > 30 Hari dari Due Date (Late Payment = 1)
    late_risk_score = (
        (df["historical_delay_count"] * 0.3)
        + (df["payment_method"] == "Giro").astype(int) * 0.4
    )
    df["is_late_payment"] = (
        late_risk_score + np.random.uniform(0, 0.4, n_invoices) > 0.75
    ).astype(int)

    return df


def run_financial_diagnostic(df: pd.DataFrame) -> dict:
    """
    Fase Analisis Diagnostik Keuangan:
    Menganalisis anomali penumpukan Accounts Receivable (AR) terhadap arus kas perusahaan.
    """
    print("\n" + "=" * 70)
    print(" 1. ANALISIS DIAGNOSTIK KEUANGAN (FINANCIAL DIAGNOSTICS) ")
    print("=" * 70)

    total_revenue = df["invoice_amount"].sum()
    late_mask = df["is_late_payment"] == 1
    uncollected_ar = df.loc[late_mask, "invoice_amount"].sum()
    late_invoice_count = late_mask.sum()
    total_invoices = len(df)
    ar_late_ratio = (uncollected_ar / total_revenue) * 100
    late_invoice_pct = (late_invoice_count / total_invoices) * 100

    print(f"[Deskriptif] Total Nilai Faktur (Revenue) : Rp {total_revenue:,.2f}")
    print(f"[Deskriptif] Total Jumlah Faktur Diterbitkan: {total_invoices} faktur")
    print(f"[Diagnostik] Piutang Tertahan (> 30 Hari)   : Rp {uncollected_ar:,.2f} ({ar_late_ratio:.2f}% dari total revenue)")
    print(f"[Diagnostik] Jumlah Faktur Berisiko Macet   : {late_invoice_count} faktur ({late_invoice_pct:.2f}%)")

    # Diagnostik per Metode Pembayaran
    print("\n[Diagnostik] Distribusi Piutang Macet per Metode Pembayaran:")
    method_diag = df.groupby("payment_method").agg(
        total_invoices=("invoice_id", "count"),
        late_invoices=("is_late_payment", "sum"),
        total_amount=("invoice_amount", "sum"),
        late_amount=("invoice_amount", lambda s: s[df.loc[s.index, "is_late_payment"] == 1].sum()),
    )
    method_diag["late_rate_pct"] = (method_diag["late_invoices"] / method_diag["total_invoices"]) * 100
    method_diag["ar_impact_pct"] = (method_diag["late_amount"] / method_diag["total_amount"]) * 100

    for method, row in method_diag.iterrows():
        print(f" - {method:<8}: {int(row['late_invoices']):>3}/{int(row['total_invoices']):>3} faktur macet ({row['late_rate_pct']:.1f}%) | "
              f"Nilai Tertahan: Rp {row['late_amount']:>12,.2f} ({row['ar_impact_pct']:.1f}% dari subtotal)")

    # Diagnostik per Histori Keterlambatan
    delay_diag = df.groupby(pd.cut(df["historical_delay_count"], bins=[-1, 0, 2, 5, 20])).agg(
        late_rate=("is_late_payment", "mean"),
        count=("invoice_id", "count")
    )
    print("\n[Diagnostik] Korelasi Histori Keterlambatan dengan Risiko Macet Saat Ini:")
    for interval, row in delay_diag.iterrows():
        print(f" - Riwayat Delay {str(interval):<10}: Tingkat Default = {row['late_rate']*100:.1f}% (Basis: {row['count']} pelanggan)")

    diagnostics_summary = {
        "total_revenue": total_revenue,
        "uncollected_ar": uncollected_ar,
        "ar_late_ratio": ar_late_ratio,
        "late_invoice_count": late_invoice_count,
        "method_diag": method_diag,
    }
    return diagnostics_summary


def prepare_data_and_prevent_leakage(df: pd.DataFrame):
    """
    Fase 3 CRISP-DM: Data Preparation & Pembersihan Feature Leakage.
    
    Penting:
    Variabel 'collection_action_days' (jumlah hari penagihan intensif) merupakan 
    FEATURE LEAKAGE karena nilainya baru tercatat SETELAH invoice mengalami gagal bayar.
    Jika diikutsertakan, model akan mengalami data leakage yang menghasilkan akurasi semu
    pada pengujian namun gagal saat inferensi nyata (pra-transaksi).
    """
    print("\n" + "=" * 70)
    print(" 2. DATA PREPARATION & MITIGASI FEATURE LEAKAGE ")
    print("=" * 70)

    # Membuang ID dan variabel yang terindikasi leakage
    leakage_columns = ["invoice_id", "collection_action_days", "is_late_payment"]
    print(f"[*] Menghapus variabel kebocoran data: {['collection_action_days']}")
    print(f"[*] Menghapus identifier non-prediktif: {['invoice_id']}")

    X = df.drop(columns=leakage_columns)
    y = df["is_late_payment"]

    # One-Hot Encoding fitur kategorikal (metode pembayaran)
    X_encoded = pd.get_dummies(X, columns=["payment_method"], drop_first=True)
    
    print(f"[*] Fitur yang digunakan untuk inferensi pra-jatuh tempo:")
    for col in X_encoded.columns:
        print(f"    - {col}")

    # Stratified Train-Test Split (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"[*] Distribusi Data Training: {len(X_train)} sampel | Data Testing: {len(X_test)} sampel")
    return X_train, X_test, y_train, y_test, X_encoded.columns


def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Fase 4 & 5 CRISP-DM: Modeling & Evaluation.
    Melatih GradientBoostingClassifier dan mengevaluasi kinerja klasifikasi.
    """
    print("\n" + "=" * 70)
    print(" 3. PREDICTIVE MODELING & EVALUATION ")
    print("=" * 70)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[Performa] Accuracy Score : {acc * 100:.2f}%")
    print(f"[Performa] ROC-AUC Score  : {roc_auc:.4f}\n")

    print("--- Classification Report (Late Payment Risk) ---")
    target_names = ["On-Time Payment", "Late Payment (>30 Days)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("--- Confusion Matrix ---")
    print(f"True Negative (Lancar diprediksi Lancar)  : {cm[0, 0]}")
    print(f"False Positive (Lancar diprediksi Telat)  : {cm[0, 1]}")
    print(f"False Negative (Telat diprediksi Lancar)  : {cm[1, 0]}")
    print(f"True Positive (Telat diprediksi Telat)   : {cm[1, 1]}")

    # Feature Importance Analysis
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\n--- Feature Importance Ranking ---")
    for rank, (feat, score) in enumerate(feat_imp.items(), start=1):
        print(f" {rank}. {feat:<25}: {score*100:.2f}%")

    return model, y_pred, y_prob, cm, feat_imp


def export_visualizations(df, feat_imp, output_dir: str = "reports/figures"):
    """
    Menyimpan grafik analisis diagnostik dan feature importance ke file visual.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "financial_diagnostics.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Piutang Macet per Metode Pembayaran
    method_data = df.groupby("payment_method")["is_late_payment"].mean() * 100
    axes[0].bar(method_data.index, method_data.values, color=["#3498db", "#e74c3c", "#2ecc71"])
    axes[0].set_title("Tingkat Keterlambatan (>30 Hari) per Metode Pembayaran (%)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Persentase Keterlambatan (%)")
    axes[0].set_xlabel("Metode Pembayaran")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for idx, val in enumerate(method_data.values):
        axes[0].text(idx, val + 1, f"{val:.1f}%", ha="center", fontweight="bold")

    # Plot 2: Feature Importance
    axes[1].barh(feat_imp.index[::-1], feat_imp.values[::-1] * 100, color="#8e44ad")
    axes[1].set_title("Feature Importance Prediksi Risiko Piutang", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Tingkat Kepentingan Relatif (%)")
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[Visualisasi] Grafik analisis diagnostik berhasil disimpan ke: {out_path}")


def print_financial_recommendations():
    """
    Fase 6 CRISP-DM: Deployment & Business Recommendations.
    Memberikan rekomendasi aksi konkret bagi PT Maju Bersama untuk pemulihan likuiditas.
    """
    print("\n" + "=" * 70)
    print(" 4. REKOMENDASI STRATEGIS & KESIMPULAN BISNIS (PT MAJU BERSAMA) ")
    print("=" * 70)
    print("""
1. Evaluasi Kebijakan Pembayaran Giro:
   - Diagnostik menunjukkan bahwa metode pembayaran 'Giro' memiliki rasio default tertinggi.
   - Manajemen disarankan memperketat verifikasi bank clearing dan mensyaratkan jaminan
     atau membatasi plafon kredit faktur berbasis giro.

2. Skema Insentif Diskon Pelunasan Awal (Early Payment Discount):
   - Menerapkan termin 2/10 net 30 (diskon 2% jika dilunasi dalam 10 hari) guna mempercepat
     perputaran arus kas dan memangkas Days Sales Outstanding (DSO).

3. Early Warning System (Model Prediktif):
   - Model Gradient Boosting dapat diintegrasikan langsung pada sistem ERP saat pembuatan invoice.
   - Tagihan yang diprediksi berisiko tinggi (High Risk) langsung mendapatkan perlakuan khusus:
     pengingat otomatis H-7, verifikasi berkala, dan eskalasi penagihan preventif.

4. Integritas Data & Anti-Leakage:
   - Penghapusan fitur seperti 'collection_action_days' memastikan model hanya menggunakan
     informasi yang valid tersedia pada saat transaksi disetujui, menjamin model dapat
     diandalkan dalam produksi nyata.
""")
    print("=" * 70)


def run_crispdm_financial_analysis():
    """
    Fungsi Utama: Menjalankan keseluruhan alur analisis keuangan CRISP-DM.
    """
    print("\n" + "#" * 70)
    print(" CRISP-DM FRAMEWORK: FINANCIAL DIAGNOSTICS & LATE PAYMENT RISK ")
    print(" Kasus PT Maju Bersama - Deteksi Piutang Macet & Mitigasi Leakage ")
    print("#" * 70)

    # 1 & 2. Data Synthesis & Understanding
    df = generate_pt_maju_bersama_data(n_invoices=1200, random_seed=101)

    # 3. Analytical Diagnostics
    run_financial_diagnostic(df)

    # 4. Data Preparation & Prevent Feature Leakage
    X_train, X_test, y_train, y_test, feature_names = prepare_data_and_prevent_leakage(df)

    # 5. Modeling & Evaluation
    model, y_pred, y_prob, cm, feat_imp = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, feature_names
    )

    # 6. Export Visuals
    export_visualizations(df, feat_imp)

    # 7. Strategic Recommendations
    print_financial_recommendations()


if __name__ == "__main__":
    run_crispdm_financial_analysis()
