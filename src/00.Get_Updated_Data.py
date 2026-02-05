#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import os
import glob
import logging
import re
import datetime


# In[10]:


#FUNCTION QC LEVEL 1 BMKG
def qc_level1_bmkg(df):
    df = df.replace(9999, pd.NA)
    df = df.replace(9999.0, pd.NA)
    df = df.replace(8888, pd.NA)
    df = df.replace(8888.0, pd.NA)
    return df

def normalize_date_columns_for_melt(df):
    """
    Normalisasi kolom tanggal ke ['Thn', 'Bln', 'Tgl'] agar kompatibel dengan melt_ch_data.
    Mendukung variasi: THN/thn/tahun/year, BLN/bln/bulan/month, TGL/tgl/tanggal/day.
    """
    col_map = {}
    found = {'Thn': False, 'Bln': False, 'Tgl': False}
    
    # Pola pencarian (case-insensitive)
    patterns = {
        'Thn': r'^(thn|tahun|year|yr|taun)$',
        'Bln': r'^(bln|bulan|month|mon|bln)$',
        'Tgl': r'^(tgl|tanggal|day|dy|date)$'
    }
    
    for col in df.columns:
        col_clean = col.strip().lower()
        for target, pattern in patterns.items():
            if re.match(pattern, col_clean):
                if found[target]:
                    raise ValueError(f"Lebih dari satu kolom cocok untuk '{target}': {col} dan sebelumnya")
                col_map[col] = target
                found[target] = True
                break
    
    if not all(found.values()):
        missing = [k for k, v in found.items() if not v]
        raise ValueError(f"Kolom tanggal tidak lengkap. Tidak ditemukan: {missing}. Kolom tersedia: {list(df.columns)}")
    
    return df.rename(columns=col_map)

def melt_ch_data(ch_df, id_vars=['Thn', 'Bln', 'Tgl']):
    """Fungsi melting sesuai definisi Anda."""
    value_vars = [col for col in ch_df.columns if col not in id_vars]
    df_melted = pd.melt(
        ch_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='WMO_ID',
        value_name='VALUE'
    )
    return df_melted

def get_data_from_ftp(vars='RRR', list_year=['199101', '199102']):
    # === Pemetaan variabel ===
    var_mapping = {
        'RRR': 'RAINFALL_24H_MM',
        'TAVE': 'TEMPERATURE_AVG_C',
        'TMIN': 'TEMP_24H_TN_C',
        'TMAX': 'TEMP_24H_TX_C',
        'RH': 'REL_HUMIDITY_AVG_PC',
        'SUN': 'SUNSHINE_24H_H',
        'WDMAX': 'WIND_DIR_24H_MAX_DEG',
        'WSMAX': 'WIND_SPEED_24H_MAX_MS',
        'WSMEAN': 'WIND_SPEED_24H_MEAN_MS'
    }
    
    if vars not in var_mapping:
        raise ValueError(f'Variabel tidak dikenali: {vars}. Pilihan: {list(var_mapping.keys())}')
    
    vars_name = var_mapping[vars]
    df = pd.DataFrame()

    for yyyymm in list_year:
        year = int(str(yyyymm)[:4])
        try:
            # Tentukan lokasi FTP
            if vars == 'TMAX' and year < 2024:
                FTP_Loc = f'http://172.19.1.208/direktori-bidang-avi/DATA_FKLIM_UPDATE/{vars}/{year}'
            else:
                FTP_Loc = f'http://172.19.1.208/direktori-bidang-avi/DATA_FKLIM_UPDATE/{vars}'
                
            url = f'{FTP_Loc}/{vars}_fklim_{yyyymm}.csv'
            
            # Unduh data
            df_month = pd.read_csv(url)
            
            # === Normalisasi kolom tanggal SEBELUM melting ===
            df_month = normalize_date_columns_for_melt(df_month)
            
            # === Melting sesuai fungsi Anda ===
            melted_df = melt_ch_data(df_month, id_vars=['Thn', 'Bln', 'Tgl'])
            
            # Konversi tipe data tanggal
            melted_df['Thn'] = pd.to_numeric(melted_df['Thn'], errors='coerce')
            melted_df['Bln'] = pd.to_numeric(melted_df['Bln'], errors='coerce')
            melted_df['Tgl'] = pd.to_numeric(melted_df['Tgl'], errors='coerce')
            
            # Buat DATA_TIMESTAMP
            melted_df['DATA_TIMESTAMP'] = pd.to_datetime({
                'year': melted_df['Thn'],
                'month': melted_df['Bln'],
                'day': melted_df['Tgl']
            }, errors='coerce')
            
            # Hapus baris dengan tanggal invalid
            melted_df = melted_df.dropna(subset=['DATA_TIMESTAMP'])
            
            # Rename VALUE → nama variabel
            melted_df = melted_df.rename(columns={'VALUE': vars_name})
            
            # Simpan kolom penting
            cols_to_keep = ['WMO_ID', 'DATA_TIMESTAMP', vars_name]
            melted_df = melted_df[cols_to_keep]
            
            df = pd.concat([df, melted_df], ignore_index=True)
            logging.info(f"Sukses mengunduh dan memproses data {vars_name} untuk {yyyymm}")
            
        except Exception as e:
            logging.error(f"Gagal mengunduh atau memproses data {vars_name} untuk {yyyymm}: {e}")
    
    return df


# **SECTION A : UPDATING DATA FROM FTP BAVI AND JOIN WITH VIEWFKLIM DATASET** 

# In[11]:


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=== STARTING DATA PROCESSING PIPELINE ===")

# GET METADATA FROM RAW DATASET
logger.info("Setting data directories...")
WORKING_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_ROOT     = os.path.join(WORKING_DIR, 'data')
DATA_ARI_ROBI = os.path.join(DATA_ROOT, '00.Robi_Dataset')
RAW_DIR       = os.path.join(DATA_ROOT, '00.Raw_Dataset')
os.makedirs(RAW_DIR, exist_ok=True)
logger.info(f"Raw directory ensured: {RAW_DIR}")

logger.info("Loading station metadata from Excel...")
metadata = pd.read_excel(f'{DATA_ARI_ROBI}/02.NC4_STABMKG_PDB2024_CH_DAILY_1991-2024_QC_LEVEL2.xlsx', sheet_name='INFOSTA')
metadata = metadata[['NOSTA', 'STA','LAT', 'LON', 'PROV','KAB', 'ELEV']]
metadata = metadata.rename(columns={
    'NOSTA': 'WMO_ID', 
    'STA': 'NAME', 
    'LON': 'CURRENT_LONGITUDE',
    'LAT': 'CURRENT_LATITUDE', 
    'PROV': 'PROVINSI', 
    'KAB': 'KABUPATEN', 
    'ELEV': 'ELEVATION'
})
metadata['WMO_ID'] = metadata['WMO_ID'].astype(str)
logger.info(f"Metadata loaded with {len(metadata)} stations.")

# REALISASI DATASET
logger.info("Loading FKLIM_DAILY raw dataset...")
FKLIM_DAILY = pd.read_csv(f'{RAW_DIR}/01.BMKGSOFT_VIEW_FKLIM_DAILY_1981-01-01_2024-07-31.csv')
FKLIM_DAILY['DATA_TIMESTAMP'] = pd.to_datetime(FKLIM_DAILY['DATA_TIMESTAMP'], format='%Y-%m-%d')
FKLIM_DAILY = FKLIM_DAILY[
    (FKLIM_DAILY['DATA_TIMESTAMP'].dt.year >= 1981) & 
    (FKLIM_DAILY['DATA_TIMESTAMP'].dt.year <= 2024)
]
logger.info(f"FKLIM_DAILY loaded with {len(FKLIM_DAILY)} records (1981–2024).")

# Create Metadata for CH QC
logger.info("Processing CH QC dataset (Robi)...")
CH_QC = pd.read_excel(f'{DATA_ARI_ROBI}/02.NC4_STABMKG_PDB2024_CH_DAILY_1991-2024_QC_LEVEL2.xlsx', sheet_name='DATA')
CH_QC = melt_ch_data(CH_QC, id_vars=['Thn', 'Bln', 'Tgl'])
CH_QC['Thn'] = CH_QC['Thn'].astype(int)
CH_QC['Bln'] = CH_QC['Bln'].astype(int)
CH_QC['Tgl'] = CH_QC['Tgl'].astype(int)
CH_QC['WMO_ID'] = CH_QC['WMO_ID'].astype(str)
CH_QC['DATA_TIMESTAMP'] = pd.to_datetime(
    CH_QC.rename(columns={'Thn': 'year', 'Bln': 'month', 'Tgl': 'day'})[['year', 'month', 'day']], 
    format='%Y-%m-%d'
)
CH_QC = CH_QC.drop(columns=['Thn', 'Bln', 'Tgl'])
CH_QC = CH_QC.rename(columns={'VALUE': 'QC_RAINFALL_24H_MM_ROBI'})
logger.info(f"CH QC dataset processed with {len(CH_QC)} records.")

logger.info("Merging CH QC with station metadata...")
CH_QC_META = CH_QC.merge(metadata, on='WMO_ID', how='right')
CH_QC_META = CH_QC_META.drop(columns=['PROVINSI','KABUPATEN', 'ELEVATION'])
CH_QC_META['WMO_ID'] = CH_QC_META['WMO_ID'].astype(str)
logger.info(f"CH QC metadata merged; total records: {len(CH_QC_META)}")

# Align WMO_ID types and merge with FKLIM_DAILY
logger.info("Preparing to merge FKLIM_DAILY with CH QC metadata...")
FKLIM_DAILY['WMO_ID'] = FKLIM_DAILY['WMO_ID'].astype(str)
FKLIM_DAILY = FKLIM_DAILY.merge(
    CH_QC_META, 
    on=['WMO_ID', 'NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'DATA_TIMESTAMP'], 
    how='left'
)
logger.info(f"Merge completed. FKLIM_DAILY now has {len(FKLIM_DAILY)} records.")

# Generate time range for FTP update
now = datetime.datetime.now()
current_year = now.year
current_month = now.month
current_day = now.day
logger.info(f"Current date: {current_year}-{current_month:02d}-{current_day:02d}")

list_data = []
for year in range(2021, current_year + 1):
    start_month = 1
    end_month = 12 if year < current_year else current_month
    for month in range(start_month, end_month + 1):
        list_data.append(year * 100 + month)
logger.info(f"Generated {len(list_data)} time periods for FTP update (e.g., last 10: {list_data[-10:]})")

# Fetch updated data from FTP
logger.info("Fetching updated data from FTP for multiple variables...")
UPDATE_DATA_RRR    = get_data_from_ftp(vars='RRR', list_year=list_data)
UPDATE_DATA_TAVE   = get_data_from_ftp(vars='TAVE', list_year=list_data)
UPDATE_DATA_TMIN   = get_data_from_ftp(vars='TMIN', list_year=list_data)
UPDATE_DATA_TMAX   = get_data_from_ftp(vars='TMAX', list_year=list_data)
UPDATE_DATA_RH     = get_data_from_ftp(vars='RH',   list_year=list_data)
UPDATE_DATA_SUN    = get_data_from_ftp(vars='SUN',   list_year=list_data)
UPDATE_DATA_WDMAX  = get_data_from_ftp(vars='WDMAX', list_year=list_data)
UPDATE_DATA_WSMAX  = get_data_from_ftp(vars='WSMAX', list_year=list_data)
UPDATE_DATA_WSMEAN = get_data_from_ftp(vars='WSMEAN', list_year=list_data)

logger.info("Merging all updated variables...")
UPDATE_DATA = (
    UPDATE_DATA_RRR
    .merge(UPDATE_DATA_TAVE, on=['WMO_ID', 'DATA_TIMESTAMP'],   how='outer')
    .merge(UPDATE_DATA_TMIN, on=['WMO_ID', 'DATA_TIMESTAMP'],   how='outer')
    .merge(UPDATE_DATA_TMAX, on=['WMO_ID', 'DATA_TIMESTAMP'],   how='outer')
    .merge(UPDATE_DATA_RH, on=['WMO_ID', 'DATA_TIMESTAMP'],     how='outer')
    .merge(UPDATE_DATA_SUN, on=['WMO_ID', 'DATA_TIMESTAMP'],    how='outer')
    .merge(UPDATE_DATA_WDMAX, on=['WMO_ID', 'DATA_TIMESTAMP'],  how='outer')
    .merge(UPDATE_DATA_WSMAX, on=['WMO_ID', 'DATA_TIMESTAMP'],  how='outer')
    .merge(UPDATE_DATA_WSMEAN, on=['WMO_ID', 'DATA_TIMESTAMP'], how='outer')
)
logger.info(f"All updated data merged. Total records: {len(UPDATE_DATA)}")

# Column alignment
logger.info("Aligning columns between FKLIM_DAILY and UPDATE_DATA...")
for col in FKLIM_DAILY.columns:
    if col not in UPDATE_DATA.columns:
        UPDATE_DATA[col] = pd.NA
        logger.debug(f"Added missing column: {col}")

# Drop location columns for consistency
UPDATE_DATA = UPDATE_DATA.drop(columns=['NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE'])
FKLIM_DAILY = FKLIM_DAILY.drop(columns=['NAME', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE'])

# Ensure datetime types
UPDATE_DATA['DATA_TIMESTAMP'] = pd.to_datetime(UPDATE_DATA['DATA_TIMESTAMP'], format='%Y-%m-%d')
FKLIM_DAILY['DATA_TIMESTAMP'] = pd.to_datetime(FKLIM_DAILY['DATA_TIMESTAMP'], format='%Y-%m-%d')

# Set consistent data types
UPDATE_DATA['WMO_ID'] = UPDATE_DATA['WMO_ID'].astype(str)
UPDATE_DATA['DATA_TIMESTAMP'] = UPDATE_DATA['DATA_TIMESTAMP'].astype('datetime64[ns]')
FKLIM_DAILY['WMO_ID'] = FKLIM_DAILY['WMO_ID'].astype(str)
FKLIM_DAILY['DATA_TIMESTAMP'] = FKLIM_DAILY['DATA_TIMESTAMP'].astype('datetime64[ns]')

# Combine datasets
logger.info("Combining historical and updated data...")
UPDATE_DATA = UPDATE_DATA[FKLIM_DAILY.columns]
UPDATE_DB_CLIMATE = pd.concat([FKLIM_DAILY, UPDATE_DATA], ignore_index=True)
logger.info(f"Combined dataset size: {len(UPDATE_DB_CLIMATE)} records")

logger.info("Merging with full metadata...")
UPDATE_DB = metadata.merge(UPDATE_DB_CLIMATE, on='WMO_ID', how='right')
logger.info(f"Final dataset after metadata merge: {len(UPDATE_DB)} records")

# Clean up old CSV files
logger.info("Removing old updated CSV files...")
csv_files = glob.glob(f'{RAW_DIR}/02.BMKGSOFT_VIEW_FKLIM_DAILY_1991-2024_UPDATED_*.csv')
for file in csv_files:
    os.remove(file)
    logger.info(f"Removed existing file: {file}")

# Apply QC Level 1
logger.info("Applying QC Level 1 processing...")
UPDATE_DB = qc_level1_bmkg(UPDATE_DB)

# Save final output
output_filename = f'{RAW_DIR}/02.BMKGSOFT_VIEW_FKLIM_DAILY_1991-2024_UPDATED_{current_year}{current_month:02d}{current_day:02d}.csv'
UPDATE_DB.to_csv(output_filename, index=False)
logger.info(f"Final dataset saved to: {output_filename}")
logger.info("=== DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY ===")

