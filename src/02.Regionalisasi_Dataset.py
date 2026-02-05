#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shutil

# ==============================
# 1. KONFIGURASI
# ==============================
WORKING_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_ROOT     = os.path.join(WORKING_DIR, 'data')
QC_DIR        = os.path.join(DATA_ROOT,'01.QC_Level_01')
REGIONAL_DIR =  os.path.join(DATA_ROOT,'02.Regionalisasi_Dataset')

if os.path.exists(REGIONAL_DIR):
    shutil.rmtree(REGIONAL_DIR)
os.makedirs(REGIONAL_DIR, exist_ok=True)

# Daftar baseline yang akan diproses
BASELINES = ['1981', '1991']

# Buat direktori utama
os.makedirs(REGIONAL_DIR, exist_ok=True)

# Konfigurasi per parameter
PARAM_CONFIG = {
    'TEMPERATURE_AVG_C': {
        'n_clusters': 4,
        'random_state': 42,
        'description': 'Suhu rata-rata harian'
    },
    'TEMP_24H_TN_C': {
        'n_clusters': 4,
        'random_state': 42,
        'description': 'Suhu minimum harian'
    },
    'TEMP_24H_TX_C': {
        'n_clusters': 4,
        'random_state': 42,
        'description': 'Suhu maksimum harian'
    },
    'RAINFALL_24H_MM': {
        'n_clusters': 6,
        'random_state': 42,
        'description': 'Curah hujan harian'
    }
}

# ==============================
# 2. FUNGSI PROSES PER PARAMETER & BASELINE
# ==============================
def process_parameter(param, config, baseline):
    print(f"\n{'='*60}")
    print(f"Memproses: {param} | Baseline: {baseline}")
    print(f"Deskripsi: {config['description']}")
    print(f"Jumlah cluster: {config['n_clusters']}")
    print(f"{'='*60}")
    
    # Path file ketersediaan
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

    if len(wmo_ids) == 0:
        print("⚠️ Tidak ada stasiun yang memenuhi kriteria ketersediaan. Lewati.")
        return False

    # Path file data QC
    input_file = os.path.join(QC_DIR, param, '06.Adjusted', f'06.Fklim_Qc_Level_1.csv')
    if not os.path.exists(input_file):
        print(f"⚠️ File data tidak ditemukan: {input_file}")
        return False

    try:
        df_raw = pd.read_csv(input_file, parse_dates=['DATA_TIMESTAMP'])
        df_raw['DATA_TIMESTAMP'] = pd.to_datetime(df_raw['DATA_TIMESTAMP'], errors='coerce')
        df_raw = df_raw.dropna(subset=['DATA_TIMESTAMP']).copy()
        df_raw['YEAR'] = df_raw['DATA_TIMESTAMP'].dt.year
        df = df_raw[df_raw['YEAR'] >= int(baseline)].copy()
        df = df.drop(columns=['YEAR'])
        print(f"Jumlah data setelah filter tahun ≥{baseline}: {len(df)}")

        # Filter hanya stasiun yang memenuhi kriteria ketersediaan
        df = df[df['WMO_ID'].isin(wmo_ids)].copy()
        print(f"Jumlah data setelah filter stasiun valid: {len(df)}")

        qc_col = f'QC_{param}'
        if qc_col not in df.columns:
            print(f"⚠️ Kolom '{qc_col}' tidak ditemukan.")
            print(f"Kolom tersedia: {list(df.columns)}")
            return False

        # Agregasi rata-rata per stasiun
        agg_df = df.groupby('WMO_ID').agg({'CURRENT_LATITUDE': 'first','CURRENT_LONGITUDE': 'first',qc_col: 'mean'}).reset_index()
        agg_df = agg_df.dropna(subset=[qc_col]).copy()
        print(f"Jumlah stasiun valid setelah agregasi: {len(agg_df)}")
        if len(agg_df) == 0:
            print("⚠️ Tidak ada stasiun dengan data valid setelah agregasi. Lewati.")
            return False

        # Siapkan data untuk PCA
        X = agg_df[['CURRENT_LATITUDE', 'CURRENT_LONGITUDE', qc_col]].values

        # Analisis PCA penuh untuk variansi
        pca_full = PCA()
        pca_full.fit(X)
        explained_var  = pca_full.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        # Direktori output
        output_dir = os.path.join(REGIONAL_DIR, f"{param}_BASELINE_{baseline}")
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

        # Simpan konfigurasi dan hasil PCA
        with open(os.path.join(output_dir, 'pca_kmeans_config.txt'), 'w') as f:
            f.write(f"Parameter: {param}\n")
            f.write(f"Baseline: {baseline}\n")
            f.write(f"Deskripsi: {config['description']}\n")
            f.write(f"Jumlah stasiun: {len(agg_df)}\n")
            f.write(f"Jumlah cluster: {config['n_clusters']}\n")
            f.write(f"Random state: {config['random_state']}\n")
            f.write("\nExplained Variance per Komponen:\n")
            for i, (var, cum) in enumerate(zip(explained_var, cumulative_var)):
                line = f"Komponen {i+1}: Variansi = {var:.4f}, Kumulatif = {cum:.4f}\n"
                f.write(line)
                print(line.strip())

        # Plot variansi kumulatif
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(explained_var)+1), cumulative_var, marker='o', color='steelblue')
        plt.xlabel('Jumlah Komponen')
        plt.ylabel('Kumulatif Variansi')
        plt.title('PCA: Explained Variance Cumulative')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'Explained_Variance_Cumulative.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Lakukan PCA 2 komponen + KMeans
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        kmeans = KMeans(
            n_clusters=config['n_clusters'],
            random_state=config['random_state']
        )
        agg_df['region'] = kmeans.fit_predict(X_pca)
        agg_df['pca_comp1'] = X_pca[:, 0]
        agg_df['pca_comp2'] = X_pca[:, 1]

        # Simpan hasil regionalisasi
        output_csv = os.path.join(output_dir, 'regionalisasi_stasiun.csv')
        agg_df.to_csv(output_csv, index=False)
        print(f"✅ Hasil disimpan di: {output_csv}")

        # Visualisasi clustering
        plt.figure(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, config['n_clusters']))
        for i in range(config['n_clusters']):
            cluster_mask = agg_df['region'] == i
            plt.scatter(
                agg_df.loc[cluster_mask, 'pca_comp1'],
                agg_df.loc[cluster_mask, 'pca_comp2'],
                label=f'Region {i}',
                color=colors[i],
                alpha=0.8,
                s=50
            )
        plt.title(f'Regionalisasi Stasiun ({param})\nBaseline {baseline} | {config["description"]}')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(title='Region')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'plots', 'Station_Regions_PCA.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Selesai: {param} (baseline {baseline})")
        return True

    except Exception as e:
        print(f"❌ Gagal memproses {param} (baseline {baseline}): {e}")
        return False

# ==============================
# 3. EKSEKUSI UTAMA
# ==============================
if __name__ == "__main__":
    print("Memulai pipeline regionalisasi untuk kedua baseline (1981 dan 1991)...\n")
    
    summary = {}
    
    for baseline in BASELINES:
        print(f"\n{'#'*70}")
        print(f"# MEMPROSES BASELINE: {baseline}")
        print(f"{'#'*70}\n")
        
        for param, config in PARAM_CONFIG.items():
            success = process_parameter(param, config, baseline)
            key = f"{param}_{baseline}"
            summary[key] = "Berhasil" if success else "Gagal"
    
    # Ringkasan akhir
    print(f"\n{'='*60}")
    print("RINGKASAN EKSEKUSI")
    print(f"{'='*60}")
    for key, status in summary.items():
        print(f"{key:<40} : {status}")
    
    print(f"\n✅ Pipeline regionalisasi selesai.")
    print(f"Hasil tersedia di: {REGIONAL_DIR}/")