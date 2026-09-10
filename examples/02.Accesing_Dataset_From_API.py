#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pandas as pd

# Perbaiki URL: gunakan endpoint yang benar
url = 'http://172.19.0.201:8100/climate/anomaly/monthly'
params = {
    'stations': '96745', # Ganti dengan WMO ID yang valid (Apabila dikosongkan maka otomatis mengambil semua stasiun)
    'start_date': '2011-01-01',
    'end_date': '2024-12-31',
    'min_completeness': 0.8}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()  # akan raise error jika status != 200
    data = response.json()
    if data['count'] == 0:
        print("⚠️ Tidak ada data ditemukan.")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(data['data'])
        # Konversi kolom waktu ke datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        print(f"✅ Berhasil memuat {len(df)} baris data.")
except requests.exceptions.RequestException as e:
    print(f"❌ Error saat mengakses API: {e}")
except KeyError as e:
    print(f"❌ Struktur respons tidak sesuai: {e}")
except Exception as e:
    print(f"❌ Error tidak terduga: {e}")

