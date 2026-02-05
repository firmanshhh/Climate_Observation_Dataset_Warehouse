#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[2]:


PARAMS    = ['RAINFALL_24H_MM', 'TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C']
BASELINES = ['1991', '1981']
meta_cols = ['NAME', 'WMO_ID', 'DATA_TIMESTAMP', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'PROVINSI', 'KABUPATEN', 'ELEVATION']


# In[3]:


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
    return df.copy()

def convert_meta_name(df):
    df = convert_metadata_columns(df, meta_cols)
    df = df.rename(columns={'WMO_ID':'wmo_id', 'NAME':'name', 'CURRENT_LATITUDE':'latitude', 'CURRENT_LONGITUDE':'longitude', 
            'PROVINSI':'provinsi', 'KABUPATEN':'kabupaten', 'ELEVATION':'elevasi', 'DATA_TIMESTAMP':'time'})
    return df


# **TAHAP 0: OPEN METADATA STASIUN**

# In[4]:


WORKING_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR      = os.path.join(WORKING_DIR, 'data')
LONG_DIR      = os.path.join(DATA_DIR, '05.Long_Format_Dataset')
os.makedirs(LONG_DIR, exist_ok=True)
METADATA_PATH = os.path.join(DATA_DIR, '00.Metadata', '00.Final_Station_Metadata.csv')
METADATA      = pd.read_csv(METADATA_PATH)
METADATA.columns = METADATA.columns.str.strip().str.upper()
METADATA      = METADATA.rename(columns={'LATITUDE': 'CURRENT_LATITUDE', 'LONGITUDE': 'CURRENT_LONGITUDE', 'PROVINCE': 'PROVINSI', 'REGENCY': 'KABUPATEN'})
METADATA      = convert_metadata_columns(METADATA, meta_cols)
METADATA      = convert_meta_name(METADATA)
print("Metadata loaded with shape:", METADATA.columns.tolist(), METADATA.shape)
METADATA.to_csv(os.path.join(LONG_DIR, '00.METADATA.csv'), index=False)


# **TAHAP 1: OPEN AVAIBILITY DATASET**

# In[5]:


import re
def get_availability_data(path: str, param: str) -> pd.DataFrame:
    # Path ke file summary
    summary_dir = os.path.join(path, '01.QC_Level_01', param, '05.Summary')
    summary_file = os.path.join(summary_dir, '00.Summary_80percent.csv')
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"File summary tidak ditemukan: {summary_file}")
    # Baca data
    df = pd.read_csv(summary_file)
    # Validasi kolom wajib
    if 'WMO_ID' not in df.columns:
        raise ValueError("Kolom 'WMO_ID' tidak ditemukan dalam file summary")
    # Filter kolom availability dan 80pct yang relevan dengan parameter
    avail_cols = [col for col in df.columns 
                  if col.startswith('AVAIL_') and param in col]
    pct_cols = [col for col in df.columns 
                if col.startswith('80PCT_') and param in col]
    if not avail_cols and not pct_cols:
        raise ValueError(
            f"Tidak ditemukan kolom availability/80pct untuk parameter '{param}'. "
            f"Kolom tersedia: {list(df.columns)}"
        )
    # Helper: Ekstrak baseline dan periode dari nama kolom
    def extract_baseline_period(col_name: str, col_type: str) -> tuple:
        # Pola untuk AVAIL: ..._BASELINE_ENDYEAR
        if col_type == 'avail':
            match = re.search(rf'{re.escape(param)}_(\d{{4}})_(\d{{4}})$', col_name)
            if match:
                baseline, end_year = match.groups()
                return baseline, f"{baseline}-{end_year}"
        # Pola untuk 80PCT: ..._BASELINE
        elif col_type == 'pct':
            match = re.search(rf'{re.escape(param)}_(\d{{4}})$', col_name)
            if match:
                baseline = match.group(1)
                # Infer periode berdasarkan baseline (konvensi klimatologi)
                end_year = '2010' if baseline == '1981' else '2020'
                return baseline, f"{baseline}-{end_year}"
        # Fallback: ekstrak tahun 4-digit terakhir sebagai baseline
        years = re.findall(r'\d{4}', col_name)
        if years:
            baseline = years[-1]
            end_year = '2010' if baseline == '1981' else '2020'
            return baseline, f"{baseline}-{end_year}"
        raise ValueError(f"Tidak dapat ekstrak baseline dari kolom: {col_name}")

    # Proses kolom availability
    records = []
    for col in avail_cols:
        try:
            baseline, period = extract_baseline_period(col, 'avail')
            for _, row in df.iterrows():
                records.append({
                    'WMO_ID': str(row['WMO_ID']),
                    'baseline': baseline,
                    'period': period,
                    'availability': float(row[col]) if pd.notna(row[col]) else None,
                    'meets_80pct': None  # Akan diisi dari pct_cols
                })
        except Exception as e:
            print(f"⚠️ Warning: Gagal proses kolom {col}: {e}")
            continue

    # Buat dataframe sementara
    if records:
        df_long = pd.DataFrame(records)
    else:
        df_long = pd.DataFrame(columns=['WMO_ID', 'baseline', 'period', 'availability', 'meets_80pct'])

    # Proses kolom 80pct dan merge
    for col in pct_cols:
        try:
            baseline, period = extract_baseline_period(col, 'pct')
            for _, row in df.iterrows():
                wmo_id = str(row['WMO_ID'])
                # Cari record yang sesuai untuk update meets_80pct
                mask = (df_long['WMO_ID'] == wmo_id) & (df_long['baseline'] == baseline)
                if mask.any():
                    # Konversi ke boolean yang robust
                    val = row[col]
                    if pd.isna(val):
                        bool_val = False
                    elif isinstance(val, bool):
                        bool_val = val
                    elif isinstance(val, (int, float)):
                        bool_val = bool(val)
                    elif isinstance(val, str):
                        bool_val = val.strip().lower() in ['true', '1', 'yes', 't']
                    else:
                        bool_val = False

                    df_long.loc[mask, 'meets_80pct'] = bool_val
                else:
                    # Tambah record baru jika belum ada
                    df_long = pd.concat([df_long, pd.DataFrame([{
                        'WMO_ID': wmo_id,
                        'baseline': baseline,
                        'period': period,
                        'availability': None,
                        'meets_80pct': bool_val
                    }])], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Warning: Gagal proses kolom 80pct {col}: {e}")
            continue
    # Validasi dan pembersihan akhir
    if df_long.empty:
        raise ValueError("Tidak ada data availability yang berhasil diekstrak")
    # Konversi tipe data
    df_long['WMO_ID'] = df_long['WMO_ID'].astype(str)
    df_long['baseline'] = df_long['baseline'].astype(str)
    df_long['period'] = df_long['period'].astype(str)
    df_long['availability'] = pd.to_numeric(df_long['availability'], errors='coerce')
    df_long['meets_80pct'] = df_long['meets_80pct'].astype(bool)
    # Urutkan dan reset index
    df_long = df_long.sort_values(['WMO_ID', 'baseline']).reset_index(drop=True)
    df_long['parameter'] = param
    # Pastikan hanya kolom yang diperlukan
    return df_long[['WMO_ID', 'baseline', 'availability', 'meets_80pct', 'parameter']].copy()

AVAILABILITY_DATA = pd.DataFrame()
for parameter in PARAMS:
    try:
        param_data = get_availability_data(DATA_DIR, parameter)
        AVAILABILITY_DATA = pd.concat([AVAILABILITY_DATA, param_data], ignore_index=True)
    except Exception as e:
        print(f"⚠️ Gagal proses data availability untuk parameter {parameter}: {e}")
AVAILABILITY_DATA = convert_meta_name(AVAILABILITY_DATA)
print("Availability data shape:", AVAILABILITY_DATA.columns.tolist(), AVAILABILITY_DATA.shape)
AVAILABILITY_DATA = AVAILABILITY_DATA.rename(columns={'meets_80pct': 'data_80%'})
AVAILABILITY_DATA.to_csv(os.path.join(LONG_DIR, '01.AVAILABILITY.csv'), index=False)


# **CREATE DB METADATA**

# In[6]:


DB_METADATA = METADATA.merge(AVAILABILITY_DATA,how='right',on='wmo_id').reset_index(drop=True)
DB_METADATA = DB_METADATA.rename(columns={'meets_80pct': 'data_80%'})
#Drop data wmoid 97500
#DB_METADATA = DB_METADATA.isna(subset=['value'])
#DB_METADATA.to_csv(os.path.join(LONG_DIR, 'METADATA_DB.csv'), index=False)


# **TAHAP 1: QC LEVEL 1 DATASET AND CONVERT TO LONG FORMAT**

# In[15]:


DataQc    = os.path.join(DATA_DIR, '01.QC_Level_01')
params    = ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C','RAINFALL_24H_MM']
def get_qclevel1_dataset(path,param):
    datapath     = os.path.join(path, param, '06.Adjusted')
    df = pd.read_csv(
        os.path.join(datapath, f'06.Fklim_Qc_Level_1.csv'),
        low_memory=False )
    qc_col = f'QC_{param}'
    Dataset = df[['WMO_ID', 'NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 
                        'PROVINSI', 'KABUPATEN', 'ELEVATION', 'DATA_TIMESTAMP', 
                        qc_col]].copy()
    Dataset = convert_meta_name(Dataset) # Fungsi custom Anda
    Dataset['parameter'] = param
    Dataset['baseline']  = None
    Dataset['source']    = 'qc'
    Dataset = Dataset.rename(columns={qc_col: 'value', 'DATA_TIMESTAMP': 'time'})
    return Dataset

def get_raw_dataset(path,param):
    datapath     = os.path.join(path, param, '06.Adjusted')
    df = pd.read_csv(
        os.path.join(datapath, f'06.Fklim_Qc_Level_1.csv'),
        low_memory=False )
    qc_col = f'RAW_{param}'
    Dataset = df[['WMO_ID', 'NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 
                        'PROVINSI', 'KABUPATEN', 'ELEVATION', 'DATA_TIMESTAMP', 
                        qc_col]].copy()
    Dataset = convert_meta_name(Dataset) # Fungsi custom Anda
    Dataset['parameter'] = param
    Dataset['baseline']  = None
    Dataset['source']    = 'raw'
    Dataset = Dataset.rename(columns={qc_col: 'value', 'DATA_TIMESTAMP': 'time'})
    return Dataset

#Initialize empty DataFrames
dataQc  = pd.DataFrame()
dataRaw = pd.DataFrame()
for param in params:
    data       = get_qclevel1_dataset(DataQc, param)
    data_raw   = get_raw_dataset(DataQc, param)
    dataQc     = pd.concat([dataQc, data], ignore_index=True)
    dataRaw    = pd.concat([dataRaw, data_raw], ignore_index=True)

#sorting data by parameter
dataQc  = dataQc.sort_values(['wmo_id', 'parameter', 'time']).reset_index(drop=True)
dataRaw = dataRaw.sort_values(['wmo_id', 'parameter', 'time']).reset_index(drop=True)

#gabungkan dataQc dan dataRaw ke dalam satu file CSV terpisah
dataRaw.to_csv(os.path.join(LONG_DIR, '02.DATA_RAW_DB.csv'), index=False)
dataQc.to_csv(os.path.join(LONG_DIR,  '03.DATA_QC_DB.csv'), index=False)


# **TAHAP 2: HOMOGENIZE DATASET AND CONVERT TO LONG FORMAT**

# In[8]:


DataHomo   = os.path.join(DATA_DIR, '04.Dataset_Final')
homoparams = ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C']
def get_homgenized_dataset(path,param, baseline):
    df       = pd.read_csv(os.path.join(path, f'{param}_BASELINE_{baseline}.csv'))
    Dataset  = df[['WMO_ID', 'NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 
                'PROVINSI', 'KABUPATEN', 'ELEVATION', 'DATA_TIMESTAMP', 
                f'HOMO_{param}']].copy()
    Dataset  = convert_meta_name(Dataset)
    Dataset['parameter'] = param
    Dataset['baseline']  = baseline
    Dataset              = Dataset.rename(columns={f'HOMO_{param}': 'value'})
    Dataset['source']    = 'homogenisasi'
    return Dataset

dataHomo = pd.DataFrame()
for param in homoparams:
    for baseline in BASELINES:
        data  = get_homgenized_dataset(DataHomo, param, baseline)
        dataHomo = pd.concat([dataHomo, data], ignore_index=True)
#save data
dataHomo.to_csv(os.path.join(LONG_DIR, '04.DATA_HOMO_DB.csv'), index=False)


# **TAHAP 3: OPEN ROBI DATASET AND CONVERT TO LONG FORMAT**

# In[47]:


import glob
def get_latest_raw_file(raw_dir, pattern="02.BMKGSOFT_VIEW_FKLIM_DAILY_1991-2024_UPDATED_*.csv"):
    files = glob.glob(os.path.join(raw_dir, pattern))
    if not files:
        raise FileNotFoundError(f"Tidak ada file ditemukan di {raw_dir}")
    return sorted(files)[-1]

dataRobi  = pd.read_csv(get_latest_raw_file(os.path.join(DATA_DIR, '00.Raw_Dataset')), low_memory=False)
dataRobi  = dataRobi[meta_cols + ['QC_RAINFALL_24H_MM_ROBI']]
dataRobi  = convert_meta_name(dataRobi)
dataRobi  = dataRobi[dataRobi['wmo_id'].astype(str) != '99992']
dataRobi['parameter'] = 'RAINFALL_24H_MM'
dataRobi              = dataRobi.rename(columns={'QC_RAINFALL_24H_MM_ROBI':'value'})
dataRobi['time']      = pd.to_datetime(dataRobi['time'], errors='coerce')
#drop data per stasiun yang datanya null semua
dataRobi              = dataRobi.groupby('wmo_id').filter(lambda x: x['value'].notna().any()).reset_index(drop=True)
dataRobi              = dataRobi[(dataRobi['time'] < '2024-01-01') & (dataRobi['time'] >= '1991-01-01') ]

# isi data kosong periode selain 1991-2020 dengan data dari dataQC
dataRain               = dataQc[(dataQc['parameter'] == 'RAINFALL_24H_MM') & (dataQc['source'] == 'qc')].copy()
dataRobi_extra         = dataRain[(dataRain['time'] < '1991-01-01') | (dataRain['time'] > '2023-12-31')].copy()
dataRobi_extra         = dataRobi_extra.drop(columns=['source'])
dataRobi_extra['time'] = pd.to_datetime(dataRobi_extra['time'], errors='coerce')
dataRobi_extra         = dataRobi_extra[dataRobi_extra['wmo_id'].isin(dataRobi['wmo_id'].unique())].drop(columns=['baseline'])

# ganti datarobi pada periode selain 1991-2023 dengan dari dataRobi_extra
dataRobi_extend = pd.concat([dataRobi, dataRobi_extra], ignore_index=True)
dataRobi_extend = dataRobi_extend.sort_values(['wmo_id', 'time']).reset_index(drop=True)
dataRobi_extend.to_csv(os.path.join(LONG_DIR, '05.DATA_ROBI_DB.csv'), index=False)


# In[ ]:


#DEBUGING CEPAT
import matplotlib.pyplot as plt
plot_dir = os.path.join(LONG_DIR, 'Rainfall_Extend')
for wmo_id in dataRobi['wmo_id'].unique():
    A = dataRain[dataRain['wmo_id']==wmo_id][['wmo_id','time','value']]
    B = dataRobi_extend[dataRobi_extend['wmo_id']==wmo_id][['wmo_id','time','value']]
    merged = A.merge(B, on=['wmo_id', 'time'], how='outer', suffixes=('_qc', '_extend'))
    merged['diff'] = merged['value_extend'] - merged['value_qc']
    #create plot function to compare rainfall data
    import matplotlib.pyplot as plt
    def plot_difference(merged, wmo_id, plot_dir):
        plt.figure(figsize=(12,6))
        plt.plot(merged['time'], merged['diff'], label='Difference', linestyle='--')
        plt
        plt.title(f'QC VS EXTEND: {wmo_id}')
        plt.xlabel('Time')
        plt.ylabel('Rainfall (mm)')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'Station_{wmo_id}_Rainfall_Comparison.png'))
        plt.close()
    plot_difference(merged, wmo_id, plot_dir)


# In[46]:


def plot_rainfall_comparison(dataRobi, dataRobi_extend, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    # Pastikan kolom 'time' bertipe datetime
    dataRobi['time'] = pd.to_datetime(dataRobi['time'])
    dataRobi_extend['time'] = pd.to_datetime(dataRobi_extend['time'])
    for sid in dataRobi_extend['wmo_id'].unique():
        # Ambil data per stasiun
        raw = dataRobi[dataRobi['wmo_id'] == sid].copy()
        extend = dataRobi_extend[dataRobi_extend['wmo_id'] == sid].copy()
        # Filter raw hanya untuk periode 1991–2023
        raw = raw[(raw['time'] >= '1991-01-01') & (raw['time'] <= '2023-12-31')]
        raw = raw.sort_values('time')
        extend = extend.sort_values('time')
        # Gabungkan data
        merged = raw.merge(extend, on=['wmo_id', 'time'], how='outer', suffixes=('_raw', '_extend'))
        # Buat plot
        plt.figure(figsize=(12, 5))
        # Plot data extended (QC + ROBI) di atas
        if 'value_extend' in merged.columns:
            plt.plot(merged['time'], merged['value_extend'], 
                     color='#0066CC', linewidth=1.2, label='QC + ROBI', zorder=3)
        # Plot data raw (ROBI) di bawah dengan transparansi
        if 'value_raw' in merged.columns:
            plt.plot(merged['time'], merged['value_raw'], 
                     color='#FFA500', alpha=0.4, label='ROBI', zorder=2)
        # Tambahkan area abu-abu untuk periode ekstensi
        # Sebelum 1991
        extend_before = extend[extend['time'] < '1991-01-01']
        if not extend_before.empty:
            plt.axvspan(extend_before['time'].min(), '1991-01-01',
                        color='lightgrey', alpha=0.3, zorder=1)
        # Setelah 2023
        extend_after = extend[extend['time'] > '2023-12-31']
        if not extend_after.empty:
            plt.axvspan('2023-12-31', extend_after['time'].max(),
                        color='lightgrey', alpha=0.3, zorder=1)
        # Garis batas periode utama
        plt.axvline(pd.to_datetime('1991-01-01'), color='gray', linestyle='--', alpha=0.7)
        plt.axvline(pd.to_datetime('2023-12-31'), color='gray', linestyle='--', alpha=0.7)
        # Judul dan label
        plt.title(f'Perbandingan Data Hujan – Stasiun {sid}', fontsize=14)
        plt.xlabel('Tanggal')
        plt.ylabel('Curah Hujan (mm)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        # Simpan
        plot_path = os.path.join(plot_dir, f'Station_{sid}_rainfall_extend_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot disimpan: {plot_path}")
        plt.close()
# Panggil fungsi
plot_dir = os.path.join(LONG_DIR, 'Rainfall_Extend')
plot_rainfall_comparison(dataRobi, dataRobi_extend, plot_dir)
