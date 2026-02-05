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
def get_anomali_dataset(path):
    df_path    = os.path.join(path, '04.DATA_HOMO_DB.csv')
    df_homo    = pd.read_csv(df_path)
    df_anomali = df_homo[(df_homo['parameter'] == 'TEMPERATURE_AVG_C') &(df_homo['source']    == 'homogenisasi') &(df_homo['baseline']  == 1991)].copy()
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
    df_monthly['days_in_month'] = df_monthly['time'].dt.daysinmonth
    df_monthly['data_completeness'] = (df_monthly['days_present'] / df_monthly['days_in_month']) * 100
    # 4. Masking nilai bulanan jika kelengkapan < 80%
    df_monthly.loc[df_monthly['data_completeness'] < 80, 'value'] = pd.NA
    df_monthly['is_valid_for_normal'] = df_monthly['data_completeness'] >= 80
    df_monthly['month'] = df_monthly['time'].dt.month
    df_monthly['year']  = df_monthly['time'].dt.year
    return df_monthly

def get_dataNormal(df_monthly, start_year=1991, end_year=2020):
    clim_period_data    = df_monthly[(df_monthly['year'] >= start_year) & (df_monthly['year'] <= end_year)].copy()
    data_normal         = clim_period_data[clim_period_data['is_valid_for_normal']]
    #definisikan menghitung jumlah data pada bulan jan secara terpisah sampai desember untuk setiap stasiun
    data_count_per_station = data_normal.groupby(['wmo_id', 'month'])['time'].nunique().reset_index()
    data_count_per_station['valid_for_normal'] = data_count_per_station['time'] >= 24  # Minimal 24 tahun data valid
    #filter stasiun yang valid untuk normal di setiap bulan
    valid_stations_per_month = data_count_per_station[data_count_per_station['valid_for_normal']==True]
    #ambil data berdasarkan stasiun yang valid untuk normal di setiap bulan
    valid_data_for_normal = pd.merge(data_normal, valid_stations_per_month[['wmo_id', 'month']], on=['wmo_id', 'month'], how='inner')
    #buat NORMAL berdasarkan data valid tersebut buat NaN untuk stasiun yang tidak valid
    dataNormal = valid_data_for_normal.groupby(['wmo_id', 'month'])['value'].mean().reset_index()
    dataNormal.rename(columns={'value': 'normal'}, inplace=True)
    return dataNormal

df_anomali         = get_anomali_dataset(LONG_DIR)
df_monthly         = get_valid_agg_monthly(df_anomali)

# 6. Hitung nilai normal (rata-rata klimatologi per stasiun per bulan)
dataNormal              = get_dataNormal(df_monthly, start_year=1991, end_year=2020)
dataAnaomali            = pd.merge(df_monthly, dataNormal, on=['wmo_id', 'month'], how='left')
dataAnaomali['anomali'] = dataAnaomali['value'] - dataAnaomali['normal']

# 9. Sortir dan tambahkan informasi tambahan
dataAnaomali                             = dataAnaomali.sort_values(['wmo_id', 'time']).reset_index(drop=True)
dataAnaomali['anomali_diff']             = dataAnaomali.groupby('wmo_id')['anomali'].diff()
dataAnaomali['year']                     = dataAnaomali['time'].dt.year
dataAnaomali['rank_from_all_station']    = dataAnaomali.groupby(['year', 'month'])['anomali'].rank(ascending=False, method='min')
dataAnaomali['rank_from_all_month']      = dataAnaomali.groupby(['wmo_id', 'month'])['anomali'].rank(ascending=False, method='min')

# 10. Ringkasan Data Normal
import calendar
summary_normal                           = dataNormal.groupby('month')['wmo_id'].count().reset_index()
summary_normal.columns                   = ['Bulan', 'Jumlah_Stasiun_Valid']
total_stasiun_unik                       = dataNormal['wmo_id'].nunique()
summary_normal['persentase_cakupan']     = (summary_normal['Jumlah_Stasiun_Valid'] / total_stasiun_unik) * 100
summary_normal['Nama_Bulan']             = summary_normal['Bulan'].apply(lambda x: calendar.month_name[x])
summary_normal                           = summary_normal[['Bulan', 'Nama_Bulan', 'Jumlah_Stasiun_Valid', 'persentase_cakupan']]
summary_normal['Total_Stasiun']          = total_stasiun_unik
summary_normal['Normal_Indonesia']       = dataNormal.groupby('month')['normal'].mean().values

# 11. Anomali Indonesia Bulanan
dataAnomaliIndo = dataAnaomali[dataAnaomali['is_valid_for_normal']].groupby(['year', 'month']).agg(anomali = ('anomali', 'mean'),trata=('value', 'mean'), anomali_diff=('anomali_diff', 'mean'),normal=('normal', 'mean'), total_stasiun=('is_valid_for_normal', 'count')).reset_index()
dataAnomaliIndo['rank_from_all_month'] = dataAnomaliIndo.groupby('month')['anomali'].rank(ascending=False, method='min')
dataAnomaliIndo = dataAnomaliIndo.rename(columns={'year':'Tahun', 'month':'Bulan'})
dataAnaomali.to_csv(os.path.join(LONG_DIR, '06.TEMPERATURE_ANOMALI_DB.csv'), index=False)
dataAnomaliIndo.to_csv(os.path.join(LONG_DIR, '06.TEMPERATURE_ANOMALI_INDONESIA_DB.csv'), index=False)
summary_normal.to_csv(os.path.join(LONG_DIR, '06.TEMPERATURE_NORMAL_SUMMARY_DB.csv'), index=False)