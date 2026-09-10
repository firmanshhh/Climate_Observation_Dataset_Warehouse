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
    df_anomali = df_homo[(df_homo['parameter'] == 'TEMP_24H_TX_C') &(df_homo['source']    == 'homogenisasi') &(df_homo['baseline']  == 1991)].copy()
    df_anomali['time'] = pd.to_datetime(df_anomali['time'])
    return df_anomali

def get_qc_dataset(path=LONG_DIR,wmoids=None):
    df_path    = os.path.join(path, '02.DATA_RAW_DB.csv')
    df_homo    = pd.read_csv(df_path)
    df_homo['time'] = pd.to_datetime(df_homo['time'])
    df_homo['year'] = df_homo['time'].dt.year
    df_anomali = df_homo[(df_homo['parameter'] == 'TEMP_24H_TX_C') & (df_homo['source'] == 'raw')& (df_homo['year'] >= 1991) ].copy()
    if wmoids is not None:
        df_anomali = df_anomali[df_anomali['wmo_id'].isin(wmoids)]
    else:
        df_anomali = df_anomali
    df_anomali = df_anomali.dropna(subset='value')
    df_monthly = df_anomali.groupby(['wmo_id', pd.Grouper(key='time', freq='ME')]).agg(
        parameter=('parameter', 'first'),
        source=('source', 'first'),
        baseline=('baseline', 'first'),
        name=('name', 'first'),
        latitude=('latitude', 'first'),
        longitude=('longitude', 'first'),
        province=('provinsi', 'first'),
        regency=('kabupaten', 'first'),
        elevation=('elevasi', 'first'),
        value=('value', 'mean'),
        days_present=('value', 'count')
    ).reset_index()
    df_monthly['days_in_month']     = df_monthly['time'].dt.daysinmonth
    df_monthly['data_completeness'] = (df_monthly['days_present'] / df_monthly['days_in_month']) * 100
    return df_monthly


def get_valid_agg_monthly(df):
    # Validasi input
    assert pd.api.types.is_datetime64_any_dtype(df['time']), "'time' must be datetime"
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
        value=('value', 'mean'),
        days_present=('value', 'count')
    ).reset_index()
    # Hitung kelengkapan
    df_monthly['days_in_month']     = df_monthly['time'].dt.daysinmonth
    df_monthly['data_completeness'] = (df_monthly['days_present'] / df_monthly['days_in_month']) * 100
    # Masking jika <80%
    mask_invalid = df_monthly['data_completeness'] < 80
    df_monthly.loc[mask_invalid, 'value'] = pd.NA  # atau np.nan
    df_monthly['is_valid_for_normal'] = ~mask_invalid
    # Tambahkan year/month untuk analisis
    df_monthly['month'] = df_monthly['time'].dt.month
    df_monthly['year']  = df_monthly['time'].dt.year
    return df_monthly

def get_dataNormal(df_monthly, start_year=1991, end_year=2020):
    clim_period_data   = df_monthly[(df_monthly['year'] >= start_year) & (df_monthly['year'] <= end_year)].copy()
    data_normal        = clim_period_data[clim_period_data['is_valid_for_normal']]
    data_normal        = data_normal[data_normal['is_valid_for_normal'] == True]
    #persetse data perbuanan untuk setiap stasiun
    data_count_per_station = data_normal.groupby(['wmo_id', 'month'])['time'].nunique().reset_index()
    data_count_per_station['valid_for_normal'] = data_count_per_station['time'] >= 24  # Minimal 24 tahun data valid
    valid_stations_per_month = data_count_per_station[data_count_per_station['valid_for_normal']==True]
    valid_data_for_normal = pd.merge(data_normal, valid_stations_per_month[['wmo_id', 'month']], on=['wmo_id', 'month'], how='inner')
    #buat NORMAL berdasarkan data valid tersebut buat NaN untuk stasiun yang tidak valid
    dataNormal = valid_data_for_normal.groupby(['wmo_id', 'month'])['value'].mean().reset_index()
    dataNormal.rename(columns={'value': 'normal'}, inplace=True)
    # print jumlah stasiun pada dataNormal per bulan
    print(dataNormal.groupby('month')['wmo_id'].nunique())
    return dataNormal


df                  = get_anomali_dataset(LONG_DIR).copy()
df_monthly          = get_valid_agg_monthly(df).copy()
monthly_qc = get_qc_dataset(LONG_DIR,wmoids=df['wmo_id'].unique())
monthly_qc = monthly_qc[['wmo_id','time','data_completeness']]
monthly_qc = monthly_qc.rename(columns={'data_completeness':'flag_anomali'})

data_for_anomali = df_monthly.merge(monthly_qc, on=['wmo_id','time'], how='left')
# Logika: Flag kurang dari 80 ATAU Flag-nya kosong (NaN)
mask_80 = (data_for_anomali['flag_anomali'] < 80) | (data_for_anomali['flag_anomali'].isna())
# Terapkan mask
data_for_anomali.loc[mask_80, 'value'] = np.nan
data_for_anomali['value'] = round(data_for_anomali['value'], 2)
data_for_anomali['month'] = data_for_anomali['time'].dt.month
data_for_anomali.to_csv(os.path.join(LONG_DIR, '07.TMAX_MONTHLY_DB.csv'))


# Dapatkan normal (pastikan kolomnya bernama 'normal')
# dataNormal           = get_dataNormal(data_for_anomali, start_year=1991, end_year=2020)
# dataNormal           = dataNormal.dropna(subset=['normal'])  # Hapus NaN di kolom normal
# dataNormal['normal'] = round(dataNormal['normal'], 2)
# dataNormal.to_csv(os.path.join(LONG_DIR, '07.TMAX_ANOMALI_NORMAL_DB.csv'), index=False)
dataNormal = pd.read_csv(os.path.join(LONG_DIR, '07.TMAX_ANOMALI_NORMAL_DB.csv'))

#Hitung anomali bulanan
dataAnomali = pd.DataFrame()
for month in range(1, 13):
    normal_bulanan      = dataNormal[dataNormal['month'] == month]
    wmoids              = normal_bulanan['wmo_id'].unique()
    df_bulanan          = data_for_anomali[(data_for_anomali['month'] == month) & (data_for_anomali['wmo_id'].isin(wmoids))].copy()
    df_merge            = pd.merge(df_bulanan, normal_bulanan[['wmo_id', 'normal']], on='wmo_id', how='left')
    df_merge['anomali'] = round(df_merge['value'] - df_merge['normal'],2)
    df_merge['suhu_bulan_sebelum']    = round(df_merge['value'].shift(1),2)
    df_merge['selisih_suhu']          = round((df_merge['value'] - df_merge['suhu_bulan_sebelum']),2)
    dataAnomali = pd.concat([dataAnomali, df_merge], ignore_index=True)

# 9. Sortir dan tambahkan informasi tambahan
dataAnomali                             = dataAnomali.sort_values(['wmo_id', 'time']).reset_index(drop=True)
dataAnomali['anomali_diff']             = round(dataAnomali.groupby('wmo_id')['anomali'].diff(),2)
dataAnomali['year']                     = dataAnomali['time'].dt.year
dataAnomali['rank_from_all_station']    = dataAnomali.groupby(['year', 'month'])['anomali'].rank(ascending=False, method='min')
dataAnomali['rank_from_all_month']      = dataAnomali.groupby(['wmo_id', 'month'])['anomali'].rank(ascending=False, method='min')
dataAnomali['rank_from_all_mont_abs']      = dataAnomali.groupby(['wmo_id', 'month'])['value'].rank(ascending=False, method='min')
# 10. Ringkasan Data Normal
dataNormal_Indo                         = dataNormal[['month','normal']].groupby('month').mean()
dataNormal_Indo['Jumlah_Stasiun_Valid']= dataNormal[['month','normal']].groupby('month').count()
dataNormal_Indo['Nama_Bulan']           = dataNormal['month'].apply(lambda x: calendar.month_name[x])
dataNormal_Indo['Total_Stasiun']        = dataNormal['wmo_id'].nunique()
dataNormal_Indo = dataNormal_Indo.reset_index(drop=False)
dataNormal_Indo['persentase_cakupan']     = (dataNormal_Indo['Jumlah_Stasiun_Valid'] / dataNormal_Indo['Total_Stasiun'] ) * 100


# 11. Anomali Indonesia Bulanan
dataAnomaliIndonesia                    = data_for_anomali[['time','value']].groupby('time').mean().reset_index()
dataAnomaliIndonesia['jumlah_stasiun_bulan_ini']  = data_for_anomali[['time','value']].groupby('time').count().reset_index(drop=True)
dataAnomaliIndonesia['year']            = dataAnomaliIndonesia['time'].dt.year
dataAnomaliIndonesia['month']           = dataAnomaliIndonesia['time'].dt.month
dataAnomaliIndonesia                    = dataAnomaliIndonesia.drop(columns='time')
dataAnomaliIND                          = pd.merge(dataAnomaliIndonesia, dataNormal_Indo, on='month').reset_index(drop=True)
dataAnomaliIND['value']                 = round(dataAnomaliIND['value'],2)
dataAnomaliIND['normal']                = round(dataAnomaliIND['normal'],2)
dataAnomaliIND['anomali']               = dataAnomaliIND['value'] - dataAnomaliIND['normal']
dataAnomaliIND['anomali']               = round(dataAnomaliIND['anomali'],2)

dataAnomaliIND                            = dataAnomaliIND[['year','month','value','normal','anomali','Jumlah_Stasiun_Valid','jumlah_stasiun_bulan_ini']]
dataAnomaliIND                            = dataAnomaliIND.dropna(axis=0)
dataAnomaliIND['suhu_bulan_sebelum']      = round(dataAnomaliIND['value'].shift(1),2)
dataAnomaliIND['anomali_diff']            = round(dataAnomaliIND['anomali'].shift(1),2)
dataAnomaliIND['selisih_suhu']            = round((dataAnomaliIND['value'] - dataAnomaliIND['suhu_bulan_sebelum']),2)
dataAnomaliIND['rank_from_all_month']     = dataAnomaliIND.groupby('month')['anomali'].rank(method='min', ascending=False).astype(int)
dataAnomaliIND['rank_from_all_month_abs'] = dataAnomaliIND.groupby('month')['value'].rank(method='min', ascending=False).astype(int)
dataAnomaliIND['coverage_percentage']     = (dataAnomaliIND['jumlah_stasiun_bulan_ini'] / dataAnomaliIND['Jumlah_Stasiun_Valid'] * 100).round(2)
dataAnomaliIND['coverage_flag']           = np.where(dataAnomaliIND['coverage_percentage'] >= 80, 'VALID', 'LOW_COVERAGE')
dataAnomaliIND = dataAnomaliIND.rename(columns={'value':'trata'})
dataAnomali.to_csv(os.path.join(LONG_DIR, '07.TMAX_ANOMALI_DB.csv'), index=False)
dataAnomaliIND.to_csv(os.path.join(LONG_DIR, '07.TMAX_ANOMALI_INDONESIA_DB.csv'), index=False)
dataNormal_Indo = dataNormal_Indo.rename(columns={'normal':'Normal_Indonesia'})
dataNormal_Indo.to_csv(os.path.join(LONG_DIR, '07.TMAX_NORMAL_SUMMARY_DB.csv'), index=False)
print("Proses perhitungan anomali suhu rata-rata selesai.")