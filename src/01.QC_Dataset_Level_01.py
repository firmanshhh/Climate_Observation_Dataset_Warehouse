#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import glob
import os
import logging
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import shutil
import numpy as np

# ==============================
# 1. KONFIGURASI
# ==============================
WORKING_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR    = os.path.join(WORKING_DIR, 'data')
RAW_DIR     = os.path.join(DATA_DIR,'00.Raw_Dataset')
QC_DIR      = os.path.join(DATA_DIR,'01.QC_Level_01')
FINAL_DIR   = os.path.join(DATA_DIR,'04.Dataset_Final')
PARAMS      = ['TEMPERATURE_AVG_C','TEMP_24H_TN_C','TEMP_24H_TX_C','RAINFALL_24H_MM']
AVG_COL     = 'TEMPERATURE_AVG_C'
MIN_COL     = 'TEMP_24H_TN_C'
MAX_COL     = 'TEMP_24H_TX_C'
TEMP_COLS   = [AVG_COL, MIN_COL, MAX_COL]
ID_COL      = 'WMO_ID'
TIME_COL    = 'DATA_TIMESTAMP'
START_DATE  = '1981-01-01'
END_DATE    = datetime.now().strftime('%Y-%m-%d')

# KONFIGURASI QC
PHYSICAL_BOUNDS = {
    'TEMPERATURE_AVG_C': (15, 40),
    'TEMP_24H_TN_C':     (12, 28),
    'TEMP_24H_TX_C':     (20, 42),
    'RAINFALL_24H_MM':   (0, 800)
}
ABRUPT_THRESHOLDS = {
    'TEMPERATURE_AVG_C': 2.5,
    'TEMP_24H_TN_C':     2.5,
    'TEMP_24H_TX_C':     2.5,
    'RAINFALL_24H_MM':   None,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# 2. FUNGSI BANTU
# ==============================

def get_latest_raw_file(raw_dir, pattern="02.BMKGSOFT_VIEW_FKLIM_DAILY_1991-2024_UPDATED_*.csv"):
    files = glob.glob(os.path.join(raw_dir, pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada file ditemukan di {raw_dir}")
    return sorted(files)[-1]

def create_qc_dir(qc_dir, param):
    main_output_dir = os.path.join(qc_dir, param)
    if os.path.exists(main_output_dir):
        shutil.rmtree(main_output_dir)
    os.makedirs(main_output_dir, exist_ok=True)
    steps = [
        '00.80percent',
        '01.DuplicatesRemoved',
        '02.ConsistencyCheck',
        '03.RangeCheck',
        '04.AbruptChangesAdjusted'
    ]
    step_dirs = {
        step: {
            'plot': os.path.join(main_output_dir, step, 'plots'),
            'netcdf': os.path.join(main_output_dir, step, 'netcdf')
        }
        for step in steps
    }
    summary_dir = os.path.join(main_output_dir, '05.Summary')
    adjusted_dir = os.path.join(main_output_dir, '06.Adjusted')
    for d in [main_output_dir, adjusted_dir, summary_dir]:
        os.makedirs(d, exist_ok=True)
    for step in step_dirs.values():
        os.makedirs(step['plot'], exist_ok=True)
        os.makedirs(step['netcdf'], exist_ok=True)
    return main_output_dir, adjusted_dir, summary_dir, step_dirs

def save_station_plot_and_netcdf(df, station_id, time_col, param, plot_dir, netcdf_dir, title_suffix, qc_col=None):
    sdf = df[df[ID_COL] == station_id].copy()
    if sdf.empty:
        return
    plt.figure(figsize=(10, 4))
    if 'RAINFALL' in param:
        color = 'c'
        ylabel = 'Curah Hujan (mm)'
    else:
        color  = 'r' if 'AVG' in param else ('b' if 'MIN' in param else 'g')
        ylabel = 'Temperature (°C)'
    
    col_to_plot = f'RAW_{param}'
    plt.plot(sdf[time_col], sdf[col_to_plot], color + '-', alpha=0.7, label='RAW')
    if qc_col and qc_col in sdf.columns:
        plt.plot(sdf[time_col], sdf[qc_col], 'k-', alpha=0.6, label='QC')
        plt.legend()
    plt.title(f'Station {station_id} – {title_suffix}')
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    safe_name = title_suffix.replace(" ", "_").replace("/", "_").lower()
    plot_subdir = os.path.join(plot_dir, title_suffix.split(' ')[-1])
    os.makedirs(plot_subdir, exist_ok=True)
    plt.savefig(os.path.join(plot_subdir, f'Station_{station_id}_{safe_name}.png'), dpi=300)
    plt.close()

    try:
        nc_subdir = os.path.join(netcdf_dir, title_suffix.split(' ')[-1])
        os.makedirs(nc_subdir, exist_ok=True)
        sdf.set_index(time_col).to_xarray().to_netcdf(
            os.path.join(nc_subdir, f'Station_{station_id}_{safe_name}.nc')
        )
    except Exception as e:
        logging.warning(f"Gagal simpan NetCDF stasiun {station_id}: {e}")

def check_duplicate_date(df, time_col, id_col, start_date, end_date):
    df_full = df[(df[time_col] >= start_date) & (df[time_col] <= end_date)].copy()
    initial_rows = len(df_full)
    df_full = df_full.sort_values(by=[id_col, time_col], na_position='last')
    df_full = df_full.drop_duplicates(subset=[id_col, time_col], keep='first').reset_index(drop=True)
    logging.info(f"Duplikat berdasarkan waktu dihapus: {initial_rows - len(df_full)} baris")
    return df_full

def check_parameter_data(df, param_list):
    missing_cols = [p for p in param_list if p not in df.columns]
    if missing_cols:
        logging.warning(f"Kolom berikut tidak ditemukan: {missing_cols}")
        for col in missing_cols:
            df[col] = pd.NA
    else:
        logging.info(f"Kolom berikut ditemukan: {param_list}")

def convert_metadata_columns(df, meta_cols):
    """Konversi tipe data kolom metadata secara aman"""
    config = {
        'NAME': str,
        'WMO_ID': str,
        'DATA_TIMESTAMP': 'datetime64[ns]',
        'CURRENT_LATITUDE': float,
        'CURRENT_LONGITUDE': float,
        'PROVINSI': str,
        'KABUPATEN': str,
        'ELEVATION': float
    }
    for col in meta_cols:
        if col in df.columns:
            try:
                if config[col] == str:
                    df[col] = df[col].astype(str).replace({'nan': '', 'None': '', 'NaT': ''})
                elif config[col] == float:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif config[col] == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                else:
                    df[col] = df[col].astype(config[col])
            except Exception as e:
                logging.warning(f"Gagal mengonversi kolom {col}: {e}")
    return df

def check_availability(df, param, time_col, id_col, baseline_key):
    """
    Hitung persentase ketersediaan data dalam periode baseline standar.
    Baseline:
      - '1981': periode 1981-01-01 hingga 2010-12-31 (30 tahun)
      - '1991': periode 1991-01-01 hingga 2020-12-31 (30 tahun)
    """
    if baseline_key == '1981':
        start_date = '1981-01-01'
        total_days = pd.to_datetime(END_DATE) - pd.to_datetime(start_date) + pd.Timedelta(days=1)  # 30 tahun termasuk 8 tahun kabisat
        total_days = total_days.days
        label      = '1981_2010'
    elif baseline_key == '1991':
        start_date = '1991-01-01'
        total_days = pd.to_datetime(END_DATE) - pd.to_datetime(start_date) + pd.Timedelta(days=1)  # 30 tahun termasuk 8 tahun kabisat
        total_days = total_days.days
        label      = '1991_2020'
    else:
        raise ValueError("Baseline harus '1981' atau '1991'")
    
    mask = (df[time_col] >= pd.to_datetime(start_date)) & (df[time_col] <= pd.to_datetime(END_DATE))
    df_baseline = df[mask].copy()
    
    availability = (
        df_baseline.groupby(id_col)[f'RAW_{param}']
        .agg(**{f'AVAIL_{param}_{label}': lambda x: (x.notna().sum() / total_days) * 100})
        .reset_index()
    )
    
    flag_col = f'80PCT_{param}_{baseline_key}'
    availability[flag_col] = availability[f'AVAIL_{param}_{label}'] >= 80
    valid_stations = availability[availability[flag_col]][id_col].tolist()
    print(f"jumlah stasiun dengan data ≥80% pada baseline {baseline_key}: {len(valid_stations)}")
    return availability, valid_stations, label

def save_qc_step(df, param, valid_1991, valid_1981, adj_dir, step_dir, prefix, title_suffix):
    qc_col = f'QC_{param}'
    for baseline, valid_list in [('1991', valid_1991), ('1981', valid_1981)]:
        start = '1981-01-01' if baseline == '1981' else '1991-01-01'
        subset = df[df[ID_COL].isin(valid_list)].copy()
        subset = subset[subset[TIME_COL] >= pd.to_datetime(start)]
        path = os.path.join(adj_dir, f'{prefix}_{baseline}.csv')
        subset.to_csv(path, index=False)
        logging.info(f"Subset ≥80% (baseline {baseline}) disimpan: {path}")
        for sid in subset[ID_COL].unique():
            save_station_plot_and_netcdf(
                subset, sid, TIME_COL, param,
                step_dir['plot'], step_dir['netcdf'],
                f"{title_suffix} {baseline}",
                qc_col=qc_col
            )


def check_consistency(df, param, avg_col, min_col, max_col, id_col, time_col, sum_dir):
    df = df.copy().sort_values([id_col, time_col])
    for base_col in [avg_col, min_col, max_col]:
        qc_name  = f'QC_{base_col}'
        raw_name = f'RAW_{base_col}'
        if raw_name not in df.columns:
            logging.warning(f"Kolom {raw_name} tidak ditemukan. Lewati consistency check.")
            continue
        if qc_name not in df.columns:
            df[qc_name] = df[raw_name].copy()
    if param not in [avg_col, min_col, max_col]:
        return df
    qc_avg = f'QC_{avg_col}'
    qc_min = f'QC_{min_col}'
    qc_max = f'QC_{max_col}'
    complete = df[qc_avg].notna() & df[qc_min].notna() & df[qc_max].notna()
    inconsistent = ((df[qc_avg] < df[qc_min]) | (df[qc_avg] > df[qc_max])) & complete
    df['TEMP_CONSISTENCY_FLAG'] = inconsistent.astype(int)
    df.loc[inconsistent, qc_avg] = pd.NA
    df[qc_avg] = df.groupby(id_col)[qc_avg].transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
    to_estimate = df[qc_avg].isna() & df[qc_min].notna() & df[qc_max].notna()
    df.loc[to_estimate, qc_avg] = (df.loc[to_estimate, qc_min] + df.loc[to_estimate, qc_max]) / 2.0
    df[f'QC_{param}'] = df[qc_avg] if param == avg_col else (df[qc_min] if param == min_col else df[qc_max])
    summary = df.groupby(id_col)['TEMP_CONSISTENCY_FLAG'].sum().reset_index()
    summary.to_csv(os.path.join(sum_dir, '02.Summary_betweenTnTx.csv'), index=False)
    return df

def check_range(df, param, bounds, id_col, sum_dir):
    df = df.copy()
    qc_col = f'QC_{param}'
    min_val, max_val = bounds[param]
    flag = (df[qc_col] < min_val) | (df[qc_col] > max_val)
    df[f'{param}_range_check_flag'] = flag.astype(int)
    if param == 'RAINFALL_24H_MM':
        invalid = (df[qc_col] < 0) | (df[qc_col] > max_val)
        df.loc[invalid, qc_col] = pd.NA
    else:
        df.loc[flag, qc_col] = pd.NA
        df[qc_col] = df.groupby(id_col)[qc_col].transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
    summary = df.groupby(id_col)[f'{param}_range_check_flag'].sum().reset_index()
    summary.to_csv(os.path.join(sum_dir, '03.Summary_range_check.csv'), index=False)
    return df

def adjust_for_abrupt_changes(df, param, threshold, id_col, time_col):
    if 'RAINFALL' in param:
        return df
    df = df.copy().sort_values([id_col, time_col])
    qc_col = f'QC_{param}'
    flag_col = f'abrupt_change_flag_{param}'
    df['diff'] = df.groupby(id_col)[qc_col].diff(-1).abs()
    is_abrupt = (df['diff'] > threshold) & df[qc_col].notna()
    df[flag_col] = is_abrupt.astype(int)
    df.loc[is_abrupt, qc_col] = pd.NA
    df[qc_col] = df.groupby(id_col)[qc_col].transform(lambda g: g.interpolate(method='linear', limit_direction='both'))
    df.drop(columns=['diff'], inplace=True)
    return df

# =================================================================
# MEMULAI PIPELINE QC
# =================================================================
logging.info("Memulai pipeline QC untuk parameter suhu...")
raw_file = get_latest_raw_file(RAW_DIR)
logging.info(f"Loading raw data from: {raw_file}")

na_values = ["", " ", "NA", "N/A", "-", "null", "NULL", "None"]
raw_df = pd.read_csv(raw_file, low_memory=False, na_values=na_values, keep_default_na=True)

# Validasi kolom wajib
META_COLS = ['NAME','WMO_ID','DATA_TIMESTAMP', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'PROVINSI', 'KABUPATEN', 'ELEVATION']
# HANYA gunakan kolom yang benar-benar ada (hapus referensi hujan)
used_cols = META_COLS + PARAMS

# Validasi keberadaan kolom
missing_cols = [col for col in used_cols if col not in raw_df.columns]
if missing_cols:
    logging.error(f"Kolom wajib tidak ditemukan: {missing_cols}")
    raise KeyError(f"Kolom berikut tidak tersedia di data mentah: {missing_cols}")

raw_df = raw_df[used_cols].copy()

# Konversi tipe data metadata
raw_df = convert_metadata_columns(raw_df, META_COLS)

check_parameter_data(raw_df, PARAMS)

# Inisialisasi kolom RAW_* dan QC_*
for p in PARAMS:
    raw_df[f'RAW_{p}'] = raw_df[p].copy()
    raw_df[f'QC_{p}'] = raw_df[p].copy()
    raw_df.drop(columns=[p], inplace=True)
    logging.info(f"Kolom RAW_{p} dan QC_{p} diinisialisasi.")

# Langkah 0: Hapus duplikat berdasarkan waktu
raw_df_clean00 = check_duplicate_date(raw_df, TIME_COL, ID_COL, START_DATE, END_DATE)

DF_FINAL_DICT = {}
for param in PARAMS:
    logging.info(f"\n{'='*60}")
    logging.info(f"Memproses parameter: {param}")
    logging.info(f"{'='*60}")
    main_out, adj_dir, sum_dir, step_dirs = create_qc_dir(QC_DIR, param)
    df_avail = raw_df_clean00.copy()

    # Langkah 1: Kelengkapan data (80%)
    logging.info(f"Langkah 1: Menghitung kelengkapan untuk baseline 1981 dan 1991...")
    avail_1991, valid_1991, label_1991 = check_availability(df_avail, param, TIME_COL, ID_COL, '1991')
    avail_1981, valid_1981, label_1981 = check_availability(df_avail, param, TIME_COL, ID_COL, '1981')
    avail_combined = avail_1991.merge(avail_1981, on=ID_COL, how='outer')
    avail_combined.to_csv(os.path.join(sum_dir, '00.Summary_80percent.csv'), index=False)
    df_avail[f'80PCT_1991_{param}'] = df_avail[ID_COL].isin(valid_1991).astype(int)
    df_avail[f'80PCT_1981_{param}'] = df_avail[ID_COL].isin(valid_1981).astype(int)
    
    for baseline, valid_list in [('1991', valid_1991), ('1981', valid_1981)]:
        df_subset00 = df_avail[df_avail[ID_COL].isin(valid_list)].copy()
        start_baseline = '1981-01-01' if baseline == '1981' else '1991-01-01'
        df_subset00 = df_subset00[df_subset00[TIME_COL] >= pd.to_datetime(start_baseline)]
        output_path = os.path.join(adj_dir, f'00.filtered_80percent_{baseline}.csv')
        df_subset00.to_csv(output_path, index=False)
        for sid in df_subset00[ID_COL].unique():
            save_station_plot_and_netcdf(
                df_subset00, sid, TIME_COL, param,
                step_dirs['00.80percent']['plot'],
                step_dirs['00.80percent']['netcdf'],
                f"Original Time Series Baseline {baseline}",
                qc_col=None
            )

    # Langkah 2: Handling Duplicates Across All Variables
    logging.info(f"Langkah 2: Removing duplicates for {param}...")
    meta_cols_set = {'NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'PROVINSI', 'KABUPATEN', 'ELEVATION', 'WIND_DIR_24H_CARDINAL'}
    value_cols = [c for c in df_avail.columns if c not in {ID_COL, TIME_COL} and c not in meta_cols_set]
    df_qc01 = df_avail.sort_values([ID_COL, TIME_COL]).copy()
    df_qc01[f'DVAT_{param}'] = df_qc01.duplicated(subset=[ID_COL] + value_cols, keep='first')
    before = len(df_qc01)
    df_dvat = df_qc01.drop_duplicates(subset=[ID_COL] + value_cols, keep='first').reset_index(drop=True)
    after = len(df_dvat)
    logging.info(f"Removed {before - after} duplicate rows.")
    df_dvat.groupby(ID_COL)[f'DVAT_{param}'].sum().reset_index().to_csv(os.path.join(sum_dir, '01.Summary_DVAT.csv'), index=False)
    df_dvat = df_dvat.drop_duplicates(subset=[ID_COL, TIME_COL], keep='first').reset_index(drop=True)
    save_qc_step(df_dvat, param, valid_1991, valid_1981, adj_dir, step_dirs['01.DuplicatesRemoved'], '01.DVAT_removed', 'After Duplicate Removal')

    df_qc02 = df_dvat.copy()
    if f'QC_{param}' not in df_qc02.columns:
        df_qc02[f'QC_{param}'] = df_qc02[f'RAW_{param}'].copy()
    if param in TEMP_COLS:
        for temp_col in TEMP_COLS:
            if f'QC_{temp_col}' not in df_qc02.columns:
                df_qc02[f'QC_{temp_col}'] = df_qc02[f'RAW_{temp_col}'].copy()

    # Langkah 3: Consistency Check
    logging.info(f"Langkah 3: Consistency check untuk {param}...")
    df_qc03 = check_consistency(df_qc02, param, AVG_COL, MIN_COL, MAX_COL, ID_COL, TIME_COL, sum_dir)
    save_qc_step(df_qc03, param, valid_1991, valid_1981, adj_dir, step_dirs['02.ConsistencyCheck'], '02.consistency_checked', 'After consistency check')

    # Langkah 4: Range Check
    logging.info(f"Langkah 4: Range check untuk {param}...")
    df_qc04 = check_range(df_qc03, param, PHYSICAL_BOUNDS, ID_COL, sum_dir)
    save_qc_step(df_qc04, param, valid_1991, valid_1981, adj_dir, step_dirs['03.RangeCheck'], '03.range_checked', 'After range check')

    # Langkah 5: Abrupt Change Adjustment
    logging.info(f"Langkah 5: Abrupt change adjustment untuk {param}...")
    threshold = ABRUPT_THRESHOLDS.get(param, 2.5)
    df_qc05 = adjust_for_abrupt_changes(df_qc04, param, threshold, ID_COL, TIME_COL)
    save_qc_step(df_qc05, param, valid_1991, valid_1981, adj_dir, step_dirs['04.AbruptChangesAdjusted'], '05.abrupt_adjusted', 'After abrupt change adjustment')
    DF_FINAL_DICT[param] = df_qc05.copy()

# Simpan file final gabungan
for param, df_final in DF_FINAL_DICT.items():
    final_dir = os.path.join(QC_DIR, param, '06.Adjusted')
    meta_cols = ['NAME', 'WMO_ID', 'DATA_TIMESTAMP', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'PROVINSI', 'KABUPATEN', 'ELEVATION']
    value_cols = [f'RAW_{param}', f'QC_{param}']
    
    # Pastikan semua kolom yang dibutuhkan ada
    missing_final = [col for col in meta_cols + value_cols if col not in df_final.columns]
    if missing_final:
        logging.warning(f"Kolom hilang di final untuk {param}: {missing_final}")
        continue
        
    df_final = df_final[meta_cols + value_cols].copy()
    final_path = os.path.join(final_dir, f'06.Fklim_Qc_Level_1.csv')
    #remove file if exists
    if os.path.exists(final_path):
        os.remove(final_path)
    df_final.to_csv(final_path, index=False)
    logging.info(f"File final disimpan: {final_path}")