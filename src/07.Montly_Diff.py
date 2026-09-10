import pandas as pd
import os
import numpy as np
import calendar

PARAMS      = ['RAINFALL_24H_MM', 'TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C']
BASELINES   = ['1991', '1981']
METDAT_COLS = ['NAME', 'WMO_ID', 'DATA_TIMESTAMP', 'CURRENT_LATITUDE', 'CURRENT_LONGITUDE', 'PROVINSI', 'KABUPATEN', 'ELEVATION']

WORKING_DIR   = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_ROOT     = os.path.join(WORKING_DIR, 'data')
LONG_DIR      = os.path.join(DATA_ROOT, '05.Long_Format_Dataset')

# 1. Persiapan Data: Filter data harian suhu rata-rata dari hasil homogenisasi dengan baseline 1991
def get_monthlydiff_dataset(path):
    df_path    = os.path.join(path, '03.DATA_QC_DB.csv')
    df_homo    = pd.read_csv(df_path,low_memory=False)
    df_anomali = df_homo[(df_homo['parameter'] == 'TEMPERATURE_AVG_C') &(df_homo['source'] == 'qc')].copy()
    df_anomali['time'] = pd.to_datetime(df_anomali['time'])
    return df_anomali

def get_valid_agg_monthly(df):
    df_monthly = df.groupby(['wmo_id', pd.Grouper(key='time', freq='ME')]).agg(
        parameter=('parameter', 'first'),
        source=('source', 'first'),
        baseline=('baseline', 'first'),
        name=('name', 'first'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first'),
        province=('provinsi', 'first'),
        regency=('kabupaten', 'first'),
        elevation=('elevasi', 'first'),
        value=('value', 'mean'),                # Rata-rata suhu bulanan (hanya dari data non-NaN)
        days_present=('value', 'count')         # Jumlah hari dengan data non-NaN
    ).reset_index()
    # 3. Hitung kelengkapan data (%)
    df_monthly['days_in_month']     = df_monthly['time'].dt.daysinmonth
    df_monthly['data_completeness'] = (df_monthly['days_present'] / df_monthly['days_in_month']) * 100
    # 4. Masking nilai bulanan jika kelengkapan < 80%
    df_monthly.loc[df_monthly['data_completeness'] < 80, 'value'] = pd.NA
    df_monthly['is_valid_for_normal'] = df_monthly['data_completeness'] >= 80
    df_monthly['month'] = df_monthly['time'].dt.month
    df_monthly['year']  = df_monthly['time'].dt.year
    return df_monthly

df_monthly         = get_monthlydiff_dataset(LONG_DIR)
df_monthly         = get_valid_agg_monthly(df_monthly)
df_monthly['diff'] = df_monthly.groupby('wmo_id')['value'].diff()
# 9. Sortir dan tambahkan informasi tambahan
df_monthly['year']                     = df_monthly['time'].dt.year
df_monthly['rank_from_all_station']    = df_monthly.groupby(['year', 'month'])['diff'].rank(ascending=False, method='min')
df_monthly['rank_from_all_month']      = df_monthly.groupby(['wmo_id', 'month'])['diff'].rank(ascending=False, method='min')
df_monthly = df_monthly.dropna(subset=['diff'])
df_monthly = df_monthly.sort_values(['wmo_id', 'year', 'month']).reset_index(drop=True)
df_monthly['baseline'] = 1991

df_monthly=df_monthly[['wmo_id', 'time', 'parameter', 'source', 'baseline','value','days_present', 'days_in_month', 'data_completeness','is_valid_for_normal', 'month', 'year', 'diff', 'rank_from_all_station','rank_from_all_month']]
meta=pd.read_csv('/mnt/dataset/02_REPO_GITHUB_FIRMAN/Developing_Climate_Observation_Dataset/data/00.Metadata/00.Final_Station_Metadata.csv')
meta=meta[meta['wmo_id'].isin(df_monthly['wmo_id'].unique())]
monthlydiff_sel=pd.merge(meta,df_monthly, on='wmo_id')

# 10. Ringkasan monthly diff Indonesia
summary_list = []
for (year, month), group in monthlydiff_sel.groupby(['year', 'month']):
    valid_diffs = group['diff'].dropna()
    if not valid_diffs.empty:
        summary = {
            'year': year,
            'month': month,
            'mean_diff': valid_diffs.mean(),
            'median_diff': valid_diffs.median(),
            'max_diff': valid_diffs.max(),
            'min_diff': valid_diffs.min(),
            'std_diff': valid_diffs.std(),
            'count_stations': valid_diffs.count()
        }
        summary_list.append(summary)
summary_df = pd.DataFrame(summary_list)
summary_df = summary_df.sort_values(['year', 'month']).reset_index(drop=True)

# Simpan hasil ke CSV
key_cols = ['wmo_id', 'year', 'month']
monthlydiff_sel = (monthlydiff_sel.sort_values(key_cols).drop_duplicates(subset=key_cols, keep='last').reset_index(drop=True))
output_path = os.path.join(LONG_DIR, '07.TEMPERATURE_MONTHLY_DIFF_DB.csv')
monthlydiff_sel.to_csv(output_path, index=False)
summary_output_path = os.path.join(LONG_DIR, '07.TEMPERATURE_MONTHLY_DIFF_SUMMARY_DB.csv')
summary_df.to_csv(summary_output_path, index=False)

# Selesai
print(f'Monthly diff dataset saved to: {output_path}')
print(f'Monthly diff summary dataset saved to: {summary_output_path}')
print('Done.')