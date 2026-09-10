#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import os


# In[ ]:


import pandas as pd
import os

WORKDIR  = os.getcwd() 
METADIR  = os.path.join(WORKDIR, 'data', '00.Metadata')

# --- 1. Baca dan siapkan BMKGSoft (prioritas utama) ---
metadata = pd.read_csv(
    os.path.join(METADIR, 'Metadata BMKGSoft 14052024 - NEW.csv'),
    sep=',', dtype={'WMO_ID': str}
)
metadata = metadata.rename(columns={
    'WMO_ID': 'wmo_id',
    'NAME': 'name',
    'CURRENT_LATITUDE': 'latitude',
    'CURRENT_LONGITUDE': 'longitude',
    'CURRENT_ELEVATION_M': 'elevation',          # atau 'CURRENT_ELEVATION_M'?
    'KABUPATEN_NAME': 'regency',
    'PROPINSI_NAME': 'province'
})
# Hapus duplikat & normalisasi
metadata = metadata.drop_duplicates(subset='wmo_id').reset_index(drop=True)
metadata['wmo_id'] = metadata['wmo_id'].astype(str).str.strip()

# Simpan hanya kolom yang relevan
meta_cols = ['wmo_id', 'name', 'latitude', 'longitude', 'elevation', 'regency', 'province']
metadata  = metadata[meta_cols]

# --- 2. Siapkan BMKGsatu (prioritas kedua untuk name & province) ---
metadata_bmkg = pd.read_csv(
    os.path.join(METADIR, 'BMKG_SATU_WMOID.csv'),
    sep=';', dtype={'WMO_ID': str}
)
metadata_bmkg = metadata_bmkg.rename(columns={
    'Station ID': 'wmo_id',
    'Station Name': 'name',
    'Province': 'province'
})
metadata_bmkg = metadata_bmkg.drop_duplicates(subset='wmo_id').reset_index(drop=True)
metadata_bmkg['wmo_id'] = metadata_bmkg['wmo_id'].astype(str).str.strip()
metadata_bmkg = metadata_bmkg[['wmo_id', 'name', 'province']]

# --- 3. Siapkan Veta (prioritas ketiga untuk name) ---
metadata_veta = pd.read_csv(
    os.path.join(METADIR, '01_WMO.csv'),
    sep=';', dtype={'WMO_ID': str}
)
# Normalisasi nama kolom (pastikan tidak bentrok)
metadata_veta = metadata_veta.rename(columns={
    'WMOid': 'wmo_id',      # asumsi: ini nama kolom asli
    'Stasiun': 'name',
    'Lat': 'latitude',
    'Lon': 'longitude'
})
metadata_veta = metadata_veta.drop_duplicates(subset='wmo_id').reset_index(drop=True)
metadata_veta['wmo_id'] = metadata_veta['wmo_id'].astype(str).str.strip()
metadata_veta = metadata_veta[['wmo_id', 'name']]

# --- 4. Gabungkan secara bertahap ---
# Mulai dari BMKGSoft sebagai basis
final_metadata = metadata.copy()

# Tambahkan BMKGsatu: isi name/province jika missing di BMKGSoft
final_metadata = final_metadata.merge(
    metadata_bmkg[['wmo_id', 'name', 'province']],
    on='wmo_id',
    how='left',
    suffixes=('', '_bmkg')
)

# Isi kolom yang null dari BMKGsatu
final_metadata['name']     = final_metadata['name'].fillna(final_metadata['name_bmkg'])
final_metadata['province'] = final_metadata['province'].fillna(final_metadata['province_bmkg'])

# Tambahkan Veta: hanya untuk name
final_metadata = final_metadata.merge(
    metadata_veta[['wmo_id', 'name']],
    on='wmo_id',
    how='left',
    suffixes=('', '_veta')
)
final_metadata['name'] = final_metadata['name'].fillna(final_metadata['name_veta'])
# Hapus kolom bantuan
final_metadata = final_metadata.drop(columns=['name_bmkg', 'province_bmkg', 'name_veta'], errors='ignore')
# Pastikan tidak ada duplikat akhir
final_metadata = final_metadata.drop_duplicates(subset='wmo_id').reset_index(drop=True)
# Opsional: drop baris tanpa wmo_id valid
final_metadata = final_metadata[final_metadata['wmo_id'].notna() & (final_metadata['wmo_id'] != '')]


# Add Metadata from robi
metadata_robi = pd.read_csv(os.path.join(METADIR,'01.WMO_ID_ROBI.csv'), sep=';', dtype={'WMO_ID': str})
metadata_robi = metadata_robi.rename(columns={'WMO_ID': 'wmo_id'})
metadata_robi = metadata_robi.rename(columns={'WMO_ID': 'wmo_id', 'Station_Name': 'name', 'Lat': 'latitude', 'Lon': 'longitude'})
metadata_robi['wmo_id'] = metadata_robi['wmo_id'].astype(str).str.strip()
final_metadata = pd.merge(
    final_metadata,
    metadata_robi,
    on='wmo_id',
    how='left',
    suffixes=('', '_new2')
)
for col in ['name', 'latitude', 'longitude', 'province', 'elevation', 'regency']:
    if col in final_metadata.columns and f"{col}_new2" in final_metadata.columns:
        final_metadata[col] = final_metadata[col].fillna(final_metadata[f"{col}_new2"])
        final_metadata.drop(columns=[f"{col}_new2"], inplace=True)

# Simpan hasil akhir
final_metadata.to_csv(os.path.join(METADIR, '00.Final_Station_Metadata.csv'), index=False)


# In[ ]:


# # 2. Pastikan station_metadata juga clean
# station_metadata = final_data[meta_cols].drop_duplicates(subset='wmo_id').reset_index(drop=True)
# station_metadata['wmo_id'] = station_metadata['wmo_id'].astype(str).str.strip()

# #tambah data dari sumber lain
# wmoidnew0 = pd.read_csv(
#     os.path.join(DATA_DIR, '00.Robi_Dataset', 'Metadata BMKGSoft 14052024 - NEW.csv'),
#     sep=',', dtype={'WMO_ID': str}
# )
# wmoidnew0 = wmoidnew0.rename(columns={'WMO_ID': 'wmo_id'})
# wmoidnew0 = wmoidnew0.rename(columns={'WMO_ID': 'wmo_id', 'NAME': 'name', 'CURRENT_LATITUDE': 'latitude', 'CURRENT_LONGITUDE': 'longitude', 'ELEVATION': 'elevation', 'KABUPATEN_NAME': 'regency', 'PROPINSI_NAME': 'province', 'REGION_DESC': 'balai', 'CURRENT_ELEVATION_M': 'elevation'})
# wmoidnew0 = wmoidnew0.drop(columns=['STATION_ID','TYPE_MKG','OPERATING_HOURS','TIME_ZONE','balai'])
# wmoidnew0['wmo_id'] = wmoidnew0['wmo_id'].astype(str).str.strip()
# metadata_inisiasi = pd.merge(
#     station_metadata,
#     wmoidnew0,
#     on='wmo_id',
#     how='left',
#     suffixes=('', '_new0')
# )
# for col in ['name', 'latitude', 'longitude', 'elevation', 'regency','province']:
#     if col in metadata_inisiasi.columns and f"{col}_new0" in metadata_inisiasi.columns:
#         metadata_inisiasi[col] = metadata_inisiasi[col].fillna(metadata_inisiasi[f"{col}_new0"])
#         metadata_inisiasi.drop(columns=[f"{col}_new0"], inplace=True)

# wmoidnew1 = pd.read_csv(os.path.join(DATA_DIR, '00.Robi_Dataset', '01_WMO.csv'), sep=';', dtype={'WMO_ID': str})
# wmoidnew1 = wmoidnew1.rename(columns={'WMO_ID': 'wmo_id'})
# wmoidnew1 = wmoidnew1.rename(columns={'WMOid': 'wmo_id', 'Stasiun': 'name', 'Lat': 'latitude', 'Lon': 'longitude', 'Kab/Kota/Prov': 'region'})
# wmoidnew1 = wmoidnew1.drop(columns=['region'])
# wmoidnew1['wmo_id'] = wmoidnew1['wmo_id'].astype(str).str.strip()
# metadata_inisiasi = pd.merge(
#     metadata_inisiasi,
#     wmoidnew1,
#     on='wmo_id',
#     how='left',
#     suffixes=('', '_new1')
# )
# for col in ['name', 'latitude', 'longitude']:
#     if col in metadata_inisiasi.columns and f"{col}_new1" in metadata_inisiasi.columns:
#         metadata_inisiasi[col] = metadata_inisiasi[col].fillna(metadata_inisiasi[f"{col}_new1"])
#         metadata_inisiasi.drop(columns=[f"{col}_new1"], inplace=True)

# #menambahkan metadata dari data lain
# wmoidnew2 = pd.read_csv(os.path.join(DATA_DIR, '00.Robi_Dataset', '01.WMO_ID_ROBI.csv'), sep=';', dtype={'WMO_ID': str})
# wmoidnew2 = wmoidnew2.rename(columns={'WMO_ID': 'wmo_id'})
# wmoidnew2 = wmoidnew2.rename(columns={'WMO_ID': 'wmo_id', 'Station_Name': 'name', 'Lat': 'latitude', 'Lon': 'longitude'})
# wmoidnew2['wmo_id'] = wmoidnew2['wmo_id'].astype(str).str.strip()
# metadata_inisiasi = pd.merge(
#     metadata_inisiasi,
#     wmoidnew2,
#     on='wmo_id',
#     how='left',
#     suffixes=('', '_new2')
# )
# for col in ['name', 'latitude', 'longitude', 'province', 'elevation', 'regency']:
#     if col in metadata_inisiasi.columns and f"{col}_new2" in metadata_inisiasi.columns:
#         metadata_inisiasi[col] = metadata_inisiasi[col].fillna(metadata_inisiasi[f"{col}_new2"])
#         metadata_inisiasi.drop(columns=[f"{col}_new2"], inplace=True)

# #menambahkan metadata dari data lain
# wmoidnew3 = pd.read_csv(os.path.join(DATA_DIR, '00.Robi_Dataset', 'WMO_ID_2025.csv'), sep=',', dtype={'WMO_ID': str})
# wmoidnew3 = wmoidnew3.rename(columns={'WMOid': 'wmo_id', 'Stasiun': 'name', 'Lat': 'latitude', 'Lon': 'longitude', 'Kab/Kota/Prov': 'regency'})
# wmoidnew3['wmo_id'] = wmoidnew3['wmo_id'].astype(str).str.strip()
# metadata_inisiasi = pd.merge(
#     metadata_inisiasi,
#     wmoidnew3,
#     on='wmo_id',
#     how='left',
#     suffixes=('', '_new3')
# )
# for col in ['name', 'latitude', 'longitude','regency']:
#     if col in metadata_inisiasi.columns and f"{col}_new3" in metadata_inisiasi.columns:
#         metadata_inisiasi[col] = metadata_inisiasi[col].fillna(metadata_inisiasi[f"{col}_new3"])
#         metadata_inisiasi.drop(columns=[f"{col}_new3"], inplace=True)

# metadata_inisiasi = metadata_inisiasi.drop_duplicates(subset='wmo_id').reset_index(drop=True)
# #save metadata stasiun yang sudah diinisiasi
# ingest_station_metadata(metadata_inisiasi)
# #save to csv sebagai backup
# metadata_inisiasi.to_csv(os.path.join(DATA_DIR, 'Metadata_Stasiun_Indonesia_2025.csv'), index=False)

