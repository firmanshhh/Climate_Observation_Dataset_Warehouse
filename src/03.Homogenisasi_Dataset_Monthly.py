#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import ruptures as rpt
import os
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
# ==============================
# 1. KONFIGURASI
# ==============================
WORKING_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_ROOT     = os.path.join(WORKING_DIR, 'data')
QC_DIR        = os.path.join(DATA_ROOT, '01.QC_Level_01')
REGIONAL_DIR  = os.path.join(DATA_ROOT,'02.Regionalisasi')
HOMO_DIR      = os.path.join(DATA_ROOT,'03.QC_Level_02')

if os.path.exists(HOMO_DIR):
    shutil.rmtree(HOMO_DIR)
os.makedirs(HOMO_DIR, exist_ok=True)

# Parameter yang diproses (hanya suhu)
PARAM_CONFIG = {
    'TEMPERATURE_AVG_C': {'description': 'Suhu rata-rata harian'},
    'TEMP_24H_TN_C':     {'description': 'Suhu minimum harian'},
    'TEMP_24H_TX_C':     {'description': 'Suhu maksimum harian'}
}

BASELINES = ['1981', '1991']  # ← Proses kedua baseline
# Buat direktori utama
os.makedirs(HOMO_DIR, exist_ok=True)

# Opsi: proses hanya stasiun tertentu
SELECTED_STATIONS = None  # Contoh: [96207, 96208]

# Thresholds
OFFSET_LIMIT    = 1.0
THRESHOLD_FORCE = 0.5
PENALTY         = 5

# ==============================
# 2. FUNGSI HOMOGENISASI
# ==============================

def process_homogenization(param, description, baseline):
    print(f"\n{'='*60}")
    print(f"Memproses homogenisasi: {param} | Baseline: {baseline}")
    print(f"{'='*60}")

    avail_file = os.path.join(QC_DIR, param, '05.Summary', '00.Summary_80percent.csv')
    if not os.path.exists(avail_file):
        print(f"⚠️ File ketersediaan tidak ditemukan: {avail_file}")
        return False
    try:
        avail_df = pd.read_csv(avail_file)
    except Exception as e:
        print(f"❌ Gagal membaca file ketersediaan: {e}")
        return False
    # Gunakan nama kolom yang benar: 80PCT_ bukan 80%_
    flag_col = f'80PCT_{param}_{baseline}'
    if flag_col not in avail_df.columns:
        print(f"⚠️ Kolom '{flag_col}' tidak ditemukan di {avail_file}.")
        print(f"Kolom tersedia: {list(avail_df.columns)}")
        return False

    # Ambil daftar WMO_ID yang memenuhi ≥80% ketersediaan
    wmo_ids = avail_df[avail_df[flag_col] == True]['WMO_ID'].tolist()
    print(f"Jumlah stasiun dengan data ≥80% pada baseline {baseline}: {len(wmo_ids)}")


    # 🔑 PATH INPUT YANG SESUAI DENGAN PIPELINE BARU
    regional_dir  = os.path.join(REGIONAL_DIR, f"{param}_BASELINE_{baseline}")
    regional_file = os.path.join(regional_dir, 'regionalisasi_stasiun.csv')
    
    qc_file = os.path.join(QC_DIR, param, '06.Adjusted',f'06.Fklim_Qc_Level_1.csv')
    if not os.path.exists(regional_file):
        print(f"⚠️  File regionalisasi tidak ditemukan: {regional_file}. Lewati.")
        return False
    if not os.path.exists(qc_file):
        print(f"⚠️  File QC tidak ditemukan: {qc_file}. Lewati.")
        return False

    # Direktori output spesifik per baseline
    output_dir = os.path.join(HOMO_DIR, f"{param}_BASELINE_{baseline}")
    plot_dir   = os.path.join(output_dir, 'plots')
    csv_dir    = os.path.join(output_dir, 'data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Baca data
    regional_df = pd.read_csv(regional_file)
    df          = pd.read_csv(qc_file, parse_dates=['DATA_TIMESTAMP'])
    daily_df    = df[df['WMO_ID'].isin(wmo_ids)].copy()
    #data subset based on year
    if baseline == '1981':
        daily_df = daily_df[(daily_df['DATA_TIMESTAMP'].dt.year >= 1981)]
    elif baseline == '1991':
        daily_df = daily_df[(daily_df['DATA_TIMESTAMP'].dt.year >= 1991)]
        
    qc_col      = f'QC_{param}'
    if qc_col not in daily_df.columns:
        print(f"⚠️  Kolom '{qc_col}' tidak ditemukan. Lewati.")
        return False

    # Pivot data
    pivot_df = daily_df.pivot(index='DATA_TIMESTAMP', columns='WMO_ID', values=qc_col)

    all_stations = pivot_df.columns.tolist()
    stations_to_process = SELECTED_STATIONS if SELECTED_STATIONS is not None else all_stations

    valid_regional_ids = set(regional_df['WMO_ID'].values)
    stations_to_process = [sid for sid in stations_to_process if sid in valid_regional_ids and sid in pivot_df.columns]

    print(f"Jumlah stasiun yang akan diproses: {len(stations_to_process)}")

    for target_id in stations_to_process:
        print(f"▶ Proses WMO {target_id}")

        region_id = regional_df[regional_df['WMO_ID'] == target_id]['region'].iloc[0]
        region_stations = regional_df[regional_df['region'] == region_id]['WMO_ID'].tolist()
        neighbors = [s for s in region_stations if s != target_id and s in pivot_df.columns]

        if len(neighbors) < 2:
            print(f"  ⚠ Tidak cukup tetangga di region {region_id} untuk WMO {target_id}")
            continue

        corrs = pivot_df[neighbors].corrwith(pivot_df[target_id])
        best_neighbors = corrs.dropna().sort_values(ascending=False).head(5).index.tolist()

        if len(best_neighbors) < 2:
            print(f"  ⚠ Kurang dari 2 tetangga berkorelasi tinggi untuk WMO {target_id}")
            continue

        series_target = pivot_df[target_id]
        series_ref = pivot_df[best_neighbors].mean(axis=1)
        anomaly = series_target - series_ref
        anomaly = anomaly.dropna()

        used_fallback = False
        corrected_series = series_target.copy()

        if len(anomaly) >= 100:
            print("  ⏩ Gunakan fallback bulanan (lewati PHA harian)")
            anomaly_month = anomaly.resample('ME').mean()
            last_valid_date = series_target.dropna().index.max()
            last_valid_month = last_valid_date.to_period('M').to_timestamp('M')
            anomaly_month = anomaly_month[anomaly_month.index < last_valid_month].dropna()
            if len(anomaly_month) < 10:
                print("  ⚠ Data bulanan terlalu pendek. Lewati.")
                continue
            algo = rpt.Pelt(model="rbf").fit(anomaly_month.values)
            monthly_cps = algo.predict(pen=PENALTY)
            if len(monthly_cps) > 0:
                monthly_dates = anomaly_month.index
                if len(monthly_cps) > 0:
                    first_cp_idx = monthly_cps[0]
                    if first_cp_idx < len(monthly_dates):
                        first_cp_date = monthly_dates[first_cp_idx]
                        early_seg = anomaly_month.loc[:first_cp_date]
                        late_seg = anomaly_month.loc[first_cp_date:]
                        if len(early_seg) > 0 and len(late_seg) > 0:
                            offset = late_seg.mean() - early_seg.mean()
                            if abs(offset) >= THRESHOLD_FORCE:
                                corrected_series.loc[:first_cp_date] += offset
                                print(f"    [FORCE-EARLY] Offset awal: {offset:.2f}°C")

                if len(monthly_cps) > 1:
                    for i in range(len(monthly_cps) - 1):
                        start_idx = monthly_cps[i]
                        end_idx = monthly_cps[i+1]
                        if end_idx >= len(monthly_dates):
                            end_idx = len(monthly_dates) - 1
                        seg_start = monthly_dates[start_idx]
                        seg_end = monthly_dates[end_idx - 1]
                        early_mean = anomaly_month.loc[seg_start:seg_end].mean()
                        stable_mean = anomaly_month.loc[seg_end:].mean()
                        offset = stable_mean - early_mean
                        corrected_series.loc[seg_start:seg_end] += offset
                        print(f"    [SEGMENT] {seg_start.date()}–{seg_end.date()}: offset = {offset:.2f}°C")
                used_fallback = True
            else:
                print("  ⚠ Tidak ada change point terdeteksi. Lewati koreksi.")
        else:
            print("  ⚠ Data terlalu pendek (<100 titik). Lewati.")

        # Simpan hasil
        result = daily_df[daily_df['WMO_ID'] == target_id].copy()
        result = result.sort_values('DATA_TIMESTAMP')
        result[f'HOMO_{param}'] = corrected_series.reindex(result['DATA_TIMESTAMP']).values

        suffix = '_fallback' if used_fallback else ''
        csv_path = os.path.join(csv_dir, f'WMO_{target_id}_homogen{suffix}.csv')
        result.to_csv(csv_path, index=False)
        print(f"  ✔ CSV disimpan: {csv_path}")

        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(result['DATA_TIMESTAMP'], result[qc_col], label='QC Data', alpha=0.6)
        plt.plot(result['DATA_TIMESTAMP'], result[f'HOMO_{param}'], label='Homogenisasi', linestyle='--', color='orange')

        valid = result[['DATA_TIMESTAMP', f'HOMO_{param}']].dropna()
        if len(valid) > 10:
            x = valid['DATA_TIMESTAMP'].map(pd.Timestamp.toordinal)
            y = valid[f'HOMO_{param}']
            slope, intercept = np.polyfit(x, y, 1)
            trend = slope * x + intercept
            plt.plot(valid['DATA_TIMESTAMP'], trend, label='Trendline Homogen', color='red', linewidth=2)

        plt.title(f'Homogenisasi {param} – WMO {target_id} (Region {region_id})\nBaseline {baseline}')
        plt.xlabel('Tanggal')
        plt.ylabel('Suhu (°C)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plot_path = os.path.join(plot_dir, f'WMO_{target_id}_homogen_plot{suffix}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✔ Plot disimpan: {plot_path}")

    print(f"✅ Selesai memproses {param} (baseline {baseline})")
    return True

# ==============================
# 3. EKSEKUSI UTAMA
# ==============================
if __name__ == "__main__":
    print("Memulai pipeline homogenisasi untuk kedua baseline (1981 dan 1991)...")
    summary = {}
    for baseline in BASELINES:
        print(f"\n{'#'*70}")
        print(f"# HOMOGENISASI BASELINE: {baseline}")
        print(f"{'#'*70}")
        for param, config in PARAM_CONFIG.items():
            success = process_homogenization(param, config['description'], baseline)
            key = f"{param}_{baseline}"
            summary[key] = "Berhasil" if success else "Gagal"
    # Ringkasan
    print(f"\n{'='*60}")
    print("RINGKASAN HOMOGENISASI")
    print(f"{'='*60}")
    for key, status in summary.items():
        print(f"{key:<40} : {status}")
    
    print(f"\n✅ Pipeline homogenisasi selesai.")
    print(f"Hasil tersedia di: {HOMO_DIR}/")

