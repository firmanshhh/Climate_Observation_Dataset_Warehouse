#!/usr/bin/env python3
"""
Sinkronisasi data QC + Homogenisasi + Regionalisasi ke database TimescaleDB.
Versi final dengan integrasi lengkap dari semua pipeline Anda.
"""

import pandas as pd
import os
import logging
import psycopg2
import yaml
from pathlib import Path  # 🔑 IMPORT PATHLIB
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
import numpy as np

# 🔑 KONFIGURASI PATH - SEMUA MENJADI OBJEK PATH
WORKING_DIR   = Path("/mnt/dataset/02_REPO_GITHUB_FIRMAN/Developing_Climate_Observation_Dataset")
DATA_DIR      = WORKING_DIR / 'data'
QC_DIR        = DATA_DIR / '01.QC_Dataset_Level_01'
HOMO_FINAL_DIR = DATA_DIR / '04.Dataset_Final'
REGIONAL_DIR  = DATA_DIR / '02.Regionalisasi_Dataset'
CONFIG_PATH   = WORKING_DIR / 'database' / 'config.yaml'
METADIR       = DATA_DIR / '00.Metadata'

# Parameter
PARAMS = [
    'TEMPERATURE_AVG_C',
    'TEMP_24H_TN_C',
    'TEMP_24H_TX_C',
    'RAINFALL_24H_MM'
]

SUHU_PARAMS = ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C']
BASELINES = ['1981', '1991']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SYNC_DB - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync_db.log'),
        logging.StreamHandler()
    ]
)

# ====== FUNGSI DATABASE ======

def load_db_config():
    config_path = CONFIG_PATH
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['database']

def get_connection():
    config = load_db_config()
    return psycopg2.connect(
        host=config['host'],
        port=config['port'],
        database=config['name'],
        user=config['user'],
        password=config['password']
    )

def ingest_station_metadata(df):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO station_metadata (
                    wmo_id, name, latitude, longitude, elevation, province, regency
                ) VALUES %s
                ON CONFLICT (wmo_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    latitude = EXCLUDED.latitude,
                    longitude = EXCLUDED.longitude,
                    elevation = EXCLUDED.elevation,
                    province = EXCLUDED.province,
                    regency = EXCLUDED.regency;
                """,
                df.where(pd.notnull(df), None).values.tolist()
            )
        conn.commit()
        print(f"✅ {len(df)} stasiun dimasukkan ke station_metadata")
    finally:
        conn.close()

def ingest_station_availability(df):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO station_availability (
                    wmo_id, parameter, baseline, availability, meets_80pct
                ) VALUES %s
                ON CONFLICT (wmo_id, parameter, baseline) DO UPDATE SET
                    availability = EXCLUDED.availability,
                    meets_80pct = EXCLUDED.meets_80pct;
                """,
                df.where(pd.notnull(df), None).values.tolist()
            )
        conn.commit()
        print(f"✅ {len(df)} entri kelengkapan dimasukkan")
    finally:
        conn.close()

def ingest_observations(df, batch_size=1000000):
    conn = get_connection()
    total = len(df)
    # ✅ HANYA kolom yang ada di PRIMARY KEY tabel observations
    pk_cols = ['time', 'wmo_id', 'parameter', 'source']
    
    try:
        with conn.cursor() as cur:
            for i in range(0, total, batch_size):
                batch = df.iloc[i:i+batch_size].copy()
                
                # ✅ Hapus duplikat berdasarkan PRIMARY KEY yang sebenarnya
                before = len(batch)
                batch = batch.groupby(pk_cols, dropna=False).last().reset_index()
                after = len(batch)
                if before != after:
                    print(f"  ⚠️  Batch {i//batch_size + 1}: {before - after} duplikat dihapus")
                if batch.empty:
                    continue
                
                # Ganti missing values dengan None
                batch_clean = batch.replace([pd.NA, np.nan], None)
                records = batch_clean.values.tolist()
                
                execute_values(
                    cur,
                    """
                    INSERT INTO observations (
                        time, wmo_id, parameter, source, value, baseline, region
                    ) VALUES %s
                    ON CONFLICT (time, wmo_id, parameter, source) DO UPDATE SET
                        value = EXCLUDED.value,
                        baseline = EXCLUDED.baseline,
                        region = EXCLUDED.region,
                        updated_at = NOW();
                    """,
                    records
                )
                conn.commit()
                print(f"  → {min(i+batch_size, total)}/{total} observasi dimasukkan")
        
        print(f"✅ Total {total:,} observasi dimasukkan ke database")
    finally:
        conn.close()
        
# ====== FUNGSI MUAT DATA ======
def load_availability_data():
    all_records = []

    for param in PARAMS:
        summary_file = QC_DIR / param / '05.Summary' / '00.Summary_80percent.csv'
        if not summary_file.exists():
            logging.warning(f"File tidak ditemukan: {summary_file}")
            continue

        df = pd.read_csv(summary_file)
        logging.info(f"Membaca summary kelengkapan: {param} ({len(df)} stasiun)")

        def safe_subset(avail_col, meets_col, baseline):
            if avail_col in df.columns and meets_col in df.columns:
                subset = df[['WMO_ID', avail_col, meets_col]].copy()
                subset.columns = ['WMO_ID', 'availability', 'meets_80pct']
                subset['param'] = param
                subset['baseline'] = baseline
                
                # Konversi availability ke numerik
                subset['availability'] = pd.to_numeric(subset['availability'], errors='coerce')
                
                # Konversi meets_80pct dengan aman
                def to_bool(val):
                    if pd.isna(val):
                        return False
                    if isinstance(val, bool):
                        return val
                    if isinstance(val, (int, float)):
                        return bool(val)
                    if isinstance(val, str):
                        return val.strip().lower() in ('true', '1', 'yes')
                    return False
                
                subset['meets_80pct'] = subset['meets_80pct'].apply(to_bool)
                
                # Hapus baris dengan availability NaN
                subset = subset.dropna(subset=['availability'])
                return subset
            return None

        # Baseline 1981
        subset_1981 = safe_subset(
            f'AVAIL_{param}_1981_2026',
            f'80%_{param}_1981',
            '1981'
        )
        if subset_1981 is not None:
            print(f"{param} 1981: {subset_1981['meets_80pct'].sum()} stasiun memenuhi")
            all_records.append(subset_1981)

        # Baseline 1991
        subset_1991 = safe_subset(
            f'AVAIL_{param}_1991_2026',
            f'80%_{param}_1991',
            '1991'
        )
        if subset_1991 is not None:
            print(f"{param} 1991: {subset_1991['meets_80pct'].sum()} stasiun memenuhi")
            all_records.append(subset_1991)

    if all_records:
        return pd.concat(all_records, ignore_index=True)
    else:
        return pd.DataFrame(columns=['WMO_ID', 'availability', 'meets_80pct', 'param', 'baseline'])
    
def processing_data(df, param, baseline):
    """
    Ubah DataFrame QC ke format long dengan kolom 'source'.
    """
    # Pastikan WMO_ID bersih
    df = df.copy()
    df['WMO_ID'] = df['WMO_ID'].astype(str).str.strip()
    df = df[df['WMO_ID'] != 'nan']
    records = []
    # 1. Tambahkan nilai RAW
    if f'RAW_{param}' in df.columns:
        raw_vals = pd.to_numeric(df[f'RAW_{param}'], errors='coerce')
        raw_df = pd.DataFrame({
            'time': df['DATA_TIMESTAMP'],
            'wmo_id': df['WMO_ID'],
            'parameter': param,
            'source': 'raw',
            'value': raw_vals,
            'baseline': baseline
        })
        records.append(raw_df)
    # 2. Tambahkan nilai QC
    if f'QC_{param}' in df.columns:
        qc_vals = pd.to_numeric(df[f'QC_{param}'], errors='coerce')
        qc_df = pd.DataFrame({
            'time': df['DATA_TIMESTAMP'],
            'wmo_id': df['WMO_ID'],
            'parameter': param,
            'source': 'qc',
            'value': qc_vals,
            'baseline': baseline
        })
        records.append(qc_df)
    # 3. Tambahkan nilai HOMOGENISASI (jika ada)
    homo_col = f'HOMO_{param}'  # atau nama kolom sesuai konvensi Anda
    if homo_col in df.columns:
        homo_vals = pd.to_numeric(df[homo_col], errors='coerce')
        homo_df   = pd.DataFrame({
            'time': df['DATA_TIMESTAMP'],
            'wmo_id': df['WMO_ID'],
            'parameter': param,
            'source': 'homo',
            'value': homo_vals,
            'baseline': baseline,
        })
        records.append(homo_df)
    # 4. Tambahkan nilai EXTENDED (ROBI)
    extended_col = 'QC_RAINFALL_24H_MM_ROBI_EXTEND'  # sesuaikan untuk suhu jika perlu
    if extended_col in df.columns:
        ext_vals = pd.to_numeric(df[extended_col], errors='coerce')
        ext_df = pd.DataFrame({
            'time': df['DATA_TIMESTAMP'],
            'wmo_id': df['WMO_ID'],
            'parameter': param,
            'source': 'extended',
            'value': ext_vals,
            'baseline': baseline,
        })
        records.append(ext_df)
    if records:
        return pd.concat(records, ignore_index=True)
    else:
        return pd.DataFrame(columns=['time', 'wmo_id', 'parameter', 'source', 'value', 'baseline'])
    
def load_qc_data():
    all_obs          = []
    metadata_records = []
    for param in PARAMS:
        for baseline in BASELINES:
            file_path = QC_DIR / param / '06.Adjusted' / f'{param}_FINAL_QC_DATA_LEVEL1.csv'
            if not file_path.exists():
                logging.warning(f"File tidak ditemukan: {file_path}")
                continue
            try:
                df = pd.read_csv(
                    file_path,
                    parse_dates=['DATA_TIMESTAMP'],
                    low_memory=False,
                    na_values=['', ' ', 'NA', 'N/A', '-', 'null', 'NULL', 'None'],
                    dtype={'WMO_ID': 'str'}
                )
                # Proses ke format long
                long_df = processing_data(df, param, baseline)
                all_obs.append(long_df)
                # Ekstrak metadata (sekali per file/stasiun)
                meta_cols = {
                    'wmo_id': 'WMO_ID',
                    'name': 'NAME',
                    'latitude': 'CURRENT_LATITUDE',
                    'longitude': 'CURRENT_LONGITUDE',
                    'province': 'PROVINSI',
                    'regency': 'KABUPATEN',
                    'elevation': 'ELEVATION'
                }
                available_meta = {k: v for k, v in meta_cols.items() if v in df.columns}
                if available_meta:
                    meta_df = df[list(available_meta.values())].rename(columns={v: k for k, v in available_meta.items()})
                    meta_df['wmo_id'] = meta_df['wmo_id'].astype(str).str.strip()
                    meta_df = meta_df[meta_df['wmo_id'] != 'nan']
                    meta_df = meta_df.drop_duplicates(subset='wmo_id')
                    metadata_records.append(meta_df)
            except Exception as e:
                logging.error(f"Error memproses {file_path}: {e}")
    # Gabungkan
    obs_df  = pd.concat(all_obs, ignore_index=True) if all_obs else pd.DataFrame()
    meta_df = pd.concat(metadata_records, ignore_index=True).drop_duplicates(subset='wmo_id') if metadata_records else pd.DataFrame()
    return obs_df, meta_df

def load_homogenization_data():
    """
    Muat data homogenisasi (hanya untuk suhu) dalam format long.
    Format output konsisten dengan kolom 'HOMO_{param}' dari script merging terbaru.
    
    Mengembalikan tuple:
        - DataFrame observasi: time, wmo_id, parameter, source='homo', value, baseline
        - DataFrame flag: time, wmo_id, parameter, homo_corrected, baseline
    """
    all_homo = []
    all_meta = []
    
    # 🔑 Prioritas kolom nilai: format baru (HOMO_{param}) sebagai primary
    PARAM_TO_VALUE_COLS = {
        'TEMPERATURE_AVG_C': ['HOMO_TEMPERATURE_AVG_C', 'VALUE', 'temperature', 'temp_avg'],
        'TEMP_24H_TN_C':     ['HOMO_TEMP_24H_TN_C',     'VALUE', 'tn', 'min_temp'],
        'TEMP_24H_TX_C':     ['HOMO_TEMP_24H_TX_C',     'VALUE', 'tx', 'max_temp'],
    }
    
    for param in SUHU_PARAMS:
        for baseline in BASELINES:
            file_path = HOMO_FINAL_DIR / f'{param}_homogen_final_baseline_{baseline}.csv'
            if not file_path.exists():
                logging.warning(f"File homogenisasi tidak ditemukan: {file_path}")
                continue
            
            try:
                # Baca file dengan penanganan NaN komprehensif
                df = pd.read_csv(
                    file_path,
                    low_memory=False,
                    na_values=['', ' ', 'NA', 'N/A', '-', 'null', 'NULL', 'None', 'NaN', 'nan'],
                    keep_default_na=True,
                    dtype={'WMO_ID': 'str'},
                    encoding='utf-8'
                )
                
                # 🔍 Validasi kolom wajib: DATA_TIMESTAMP & WMO_ID
                if 'DATA_TIMESTAMP' not in df.columns:
                    # Fallback untuk kolom timestamp alternatif
                    ts_map = {'timestamp': 'DATA_TIMESTAMP', 'time': 'DATA_TIMESTAMP', 
                             'date': 'DATA_TIMESTAMP', 'datetime': 'DATA_TIMESTAMP'}
                    renamed = False
                    for old, new in ts_map.items():
                        if old in df.columns:
                            df = df.rename(columns={old: new})
                            logging.warning(f"Kolom timestamp direname: '{old}' → '{new}' di {file_path.name}")
                            renamed = True
                            break
                    if not renamed:
                        logging.error(f"File {file_path.name} tidak memiliki kolom timestamp valid. Kolom: {list(df.columns)}")
                        continue
                
                if 'WMO_ID' not in df.columns:
                    # Fallback untuk kolom stasiun alternatif
                    id_map = {'wmo_id': 'WMO_ID', 'station_id': 'WMO_ID', 'station': 'WMO_ID'}
                    renamed = False
                    for old, new in id_map.items():
                        if old in df.columns:
                            df = df.rename(columns={old: new})
                            logging.warning(f"Kolom WMO_ID direname: '{old}' → '{new}' di {file_path.name}")
                            renamed = True
                            break
                    if not renamed:
                        logging.error(f"File {file_path.name} tidak memiliki kolom WMO_ID valid. Kolom: {list(df.columns)}")
                        continue
                
                # Normalisasi timestamp dan WMO_ID
                df['DATA_TIMESTAMP'] = pd.to_datetime(df['DATA_TIMESTAMP'], errors='coerce')
                df = df[df['DATA_TIMESTAMP'].notna()]
                df['WMO_ID'] = df['WMO_ID'].astype(str).str.strip().str.upper()
                df = df[~df['WMO_ID'].isin(['', 'NAN', 'NONE', 'NULL', 'NA'])]
                
                if df.empty:
                    logging.warning(f"Tidak ada data valid setelah filtering di {file_path.name}")
                    continue
                
                # 🔑 Deteksi kolom nilai homogenisasi (prioritas: HOMO_{param})
                value_col = None
                for col in PARAM_TO_VALUE_COLS.get(param, ['HOMO_TEMPERATURE_AVG_C', 'VALUE']):
                    if col in df.columns:
                        numeric_vals = pd.to_numeric(df[col], errors='coerce')
                        if numeric_vals.notna().sum() > 0:  # Minimal ada 1 nilai valid
                            value_col = col
                            break
                
                if value_col is None:
                    logging.error(f"Tidak ditemukan kolom nilai valid di {file_path.name}. Kolom: {list(df.columns)}")
                    continue
                
                # Ekstrak nilai homogenisasi
                df['homo_value'] = pd.to_numeric(df[value_col], errors='coerce')
                df = df[df['homo_value'].notna()]
                
                if df.empty:
                    logging.warning(f"Tidak ada nilai homogenisasi valid di {file_path.name} (kolom: {value_col})")
                    continue
                
                # Tentukan parameter & baseline aktual (fallback ke nilai loop jika tidak tersedia di data)
                actual_param = df['parameter'].iloc[0] if 'parameter' in df.columns and df['parameter'].notna().any() else param
                actual_baseline = df['baseline'].iloc[0] if 'baseline' in df.columns and df['baseline'].notna().any() else baseline
                
                # Format long untuk observasi
                homo_df = pd.DataFrame({
                    'time': df['DATA_TIMESTAMP'].dt.normalize(),  # Normalisasi ke tengah malam (harian)
                    'wmo_id': df['WMO_ID'],
                    'parameter': str(actual_param),
                    'source': 'homo',
                    'value': df['homo_value'],
                    'baseline': str(actual_baseline)
                }).drop_duplicates(subset=['time', 'wmo_id', 'parameter', 'baseline'], keep='last')
                
                # Format long untuk flag
                homo_meta = pd.DataFrame({
                    'time': df['DATA_TIMESTAMP'].dt.normalize(),
                    'wmo_id': df['WMO_ID'],
                    'parameter': str(actual_param),
                    'homo_corrected': True,
                    'baseline': str(actual_baseline)
                }).drop_duplicates(subset=['time', 'wmo_id', 'parameter', 'baseline'], keep='last')
                
                all_homo.append(homo_df)
                all_meta.append(homo_meta)
                
                logging.info(
                    f"✓ Berhasil muat {len(homo_df):,} observasi dari {file_path.name} "
                    f"(kolom: '{value_col}', stasiun: {homo_df['wmo_id'].nunique()})"
                )
                
            except Exception as e:
                logging.error(f"Error membaca {file_path.name}: {e}", exc_info=True)
                continue
    
    # Gabungkan observasi
    if all_homo:
        obs_df = pd.concat(all_homo, ignore_index=True)
        obs_df = obs_df.astype({
            'time': 'datetime64[ns]',
            'wmo_id': 'str',
            'parameter': 'str',
            'source': 'str',
            'value': 'float32',
            'baseline': 'str'
        }).sort_values(['time', 'wmo_id', 'parameter']).reset_index(drop=True)
        logging.info(f"Total observasi homogenisasi: {len(obs_df):,} dari {obs_df['wmo_id'].nunique()} stasiun")
    else:
        obs_df = pd.DataFrame(columns=['time', 'wmo_id', 'parameter', 'source', 'value', 'baseline'])
        logging.warning("Tidak ada data homogenisasi yang berhasil dimuat")
    
    # Gabungkan flag
    if all_meta:
        flag_df = pd.concat(all_meta, ignore_index=True)
        flag_df = flag_df.astype({
            'time': 'datetime64[ns]',
            'wmo_id': 'str',
            'parameter': 'str',
            'homo_corrected': 'bool',
            'baseline': 'str'
        }).drop_duplicates(subset=['time', 'wmo_id', 'parameter', 'baseline'], keep='last')
    else:
        flag_df = pd.DataFrame(columns=['time', 'wmo_id', 'parameter', 'homo_corrected', 'baseline'])
    
    return obs_df, flag_df

def load_regionalization_data():
    """Muat data regionalisasi."""
    regional_data = []
    for param in PARAMS:  # Cukup satu parameter
        for baseline in BASELINES:
            regional_file = REGIONAL_DIR / f'{param}_BASELINE_{baseline}' / 'regionalisasi_stasiun.csv'
            if not regional_file.exists():
                continue
            if param == 'RAINFALL_24H_MM':
                df = pd.read_csv(regional_file).drop(columns=['pca_comp1', 'pca_comp2','CURRENT_LONGITUDE','CURRENT_LATITUDE', f'QC_RAINFALL_24H_MM_ROBI_EXTEND'])
            else:
                df = pd.read_csv(regional_file).drop(columns=['pca_comp1', 'pca_comp2','CURRENT_LONGITUDE','CURRENT_LATITUDE', f'QC_{param}'])
            df['baseline']  = baseline
            df['parameter'] = param
            regional_data.append(df)
    return pd.concat(regional_data, ignore_index=True) if regional_data else pd.DataFrame()

# ====== MAIN EXECUTION ======
if __name__ == "__main__":

    # 1. Tentukan jendela update
    from datetime import datetime, timedelta
    today      = datetime.now().date()
    start_date = today - timedelta(days=7)  # ubah angka jika perlu
    print(f"📅 Memperbarui data dari {start_date} hingga {today}")

    print("🚀 Memulai pemrosesan dan ingest data iklim...")
    # 1. Muat semua data
    print("\n📥 Memuat data...")
    regions              = load_regionalization_data().rename(columns={'WMO_ID': 'wmo_id'})
    available            = load_availability_data()
    meta_qc              = pd.read_csv(METADIR / '00.Final_Station_Metadata.csv', sep=',', index_col=False)
    meta_qc              = meta_qc.set_index('wmo_id').drop_duplicates(keep='last').reset_index()
    meta_qc              = meta_qc.reset_index(drop=True)
    meta_qc['wmo_id']    = meta_qc['wmo_id'].astype(str)
    qc_obs, meta_from_qc = load_qc_data()
    homo_obs, meta_homo  = load_homogenization_data()

    # 2. Gabungkan observasi
    print("\n🔄 Menggabungkan data observasi...")
    all_obs = pd.concat([qc_obs, homo_obs], ignore_index=True)
    all_obs = all_obs.sort_values('source')  # 'homo' > 'qc' secara alfabetis? Sesuaikan!
    all_obs = all_obs.sort_values(by=['time', 'wmo_id', 'parameter', 'source']).reset_index(drop=True)
    all_obs = all_obs.drop_duplicates(subset=['time', 'wmo_id', 'parameter', 'source'],keep='last')

    # 3. Gabungkan metadata
    obs_with_meta = pd.merge(all_obs, meta_qc, on='wmo_id', how='left')

    # 4. Gabungkan region
    obs_with_meta['wmo_id'] = obs_with_meta['wmo_id'].astype(str)
    regions['wmo_id']       = regions['wmo_id'].astype(str)

    #dibagian ini nantinya kita bisa tambahkan untuk pengaturan updating    
    final_data = pd.merge(
        obs_with_meta,
        regions[['wmo_id', 'region', 'parameter', 'baseline']],
        on=['wmo_id', 'parameter', 'baseline'],
        how='left'
    )

    # 5. Siapkan data untuk database
    print("\n📦 Menyiapkan data untuk database...")
    meta_cols        = ['wmo_id', 'name', 'latitude', 'longitude', 'elevation', 'province', 'regency']
    station_metadata = final_data[meta_cols].drop_duplicates(subset='wmo_id').reset_index(drop=True)

    station_availability = available.rename(columns={'WMO_ID': 'wmo_id','param': 'parameter'})[['wmo_id', 'parameter', 'baseline', 'availability', 'meets_80pct']]

    obs_cols                 = ['time', 'wmo_id', 'parameter', 'source', 'value', 'baseline', 'region']
    observations             = final_data[obs_cols].copy()
    observations['time']     = pd.to_datetime(observations['time']).dt.date
    observations['wmo_id']   = observations['wmo_id'].astype(str)
    observations['baseline'] = observations['baseline'].astype(str)
    observations['region']   = pd.to_numeric(observations['region'], errors='coerce').astype('Int64')

    # 6. Ingest ke database
    print("\n📤 Mengirim data ke database...")
    ingest_station_metadata(station_metadata)
    ingest_station_availability(station_availability)
    ingest_observations(observations)
    print("\n🎉 Semua data berhasil diimpor!")