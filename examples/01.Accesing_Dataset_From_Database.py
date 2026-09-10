#!/usr/bin/env python
# coding: utf-8

# In[54]:


#!/usr/bin/env python3
"""
Script untuk mengambil data iklim dari database TimescaleDB (SKEMA BARU).
- Format observasi: long (source sebagai kolom)
- Metadata stasiun terpisah
- Tanpa qc_flags (karena dihapus sementara)
"""

import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path
import sys
# ==============================
# 1. KONEKSI DATABASE
# ==============================
# database:
#   host: 172.19.0.201    # ← IP server database
#   port: 5432            # ← Port PostgreSQL default
#   name: climate_db      # ← Nama database
#   user: api             # ← User database
#   password: climate2026 # ← Password (pastikan aman!)

# def load_db_config(config_path=None):
#     """Muat konfigurasi database dari file YAML."""
#     if config_path is None:
#         project_dir = Path.cwd()
#         config_path = project_dir / 'database' / 'config.yaml'
#         if not config_path.exists():
#             config_path = Path('database/config.yaml')
    
#     config_file = Path(config_path)
#     if not config_file.exists():
#         print(f"❌ File konfigurasi tidak ditemukan: {config_file}")
#         sys.exit(1)
    
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
#         return config['database']

# def get_db_engine(config_path=None):
#     """Buat koneksi database SQLAlchemy."""
#     config = load_db_config(config_path)
#     url = f"postgresql://{config['user']}:{config['password']}@" \
#           f"{config['host']}:{config['port']}/{config['name']}"
#     return create_engine(url)


config = {'database': {
        'host': '172.19.0.201',
        'port': 5432,
        'name': 'climate_db',
        'user': 'api',
        'password': 'climate2026'
    }}
def get_db_engine(config_path=None):
    """Buat koneksi database SQLAlchemy."""
    url = f"postgresql://{config['database']['user']}:{config['database']['password']}@" \
          f"{config['database']['host']}:{config['database']['port']}/{config['database']['name']}"
    return create_engine(url)
# ==============================
# 2. FUNGSI UTAMA QUERY (SKEMA BARU)
# ==============================

def query_climate_data(
    parameters=None,
    sources='qc',
    baseline=None,  # ← biarkan None untuk otomatisasi
    min_80pct_only=True,
    min_80pct_baseline='1991',  # default: gunakan 1991 untuk filter
    start_date=None,
    end_date=None,
    stations=None,
    provinces=None,
    regions=None,
    time_aggregation=None,
    spatial_aggregation=None,
    limit=None,
    config_path=None
):
    """
    Ambil data iklim dengan logika otomatis:
    - Jika source='qc', baseline otomatis='1981'
    - Jika min_80pct_only=True, filter berdasarkan kelengkapan di min_80pct_baseline ('1991')
    - Jika min_80pct_baseline='1991', otomatis batasi waktu >= '1991-01-01'
    """
    
    engine = get_db_engine(config_path)
    
    # Normalisasi sources
    if isinstance(sources, str):
        sources = [sources]
    valid_sources = {'raw', 'qc', 'homo', 'extended'}
    if not set(sources).issubset(valid_sources):
        raise ValueError(f"sources harus salah satu dari {valid_sources}")
    
    # 🔑 OTOMATISASI BASELINE
    if baseline is None:
        if 'qc' in sources or sources == ['qc']:
            baseline = '1981'
        else:
            # Untuk homo/extended, biarkan user tentukan atau default ke '1981'
            baseline = '1981'
    
    if baseline not in ['1981', '1991']:
        raise ValueError("baseline harus '1981' atau '1991'")
    
    # Normalisasi parameters
    if parameters is None:
        parameters = ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C', 'RAINFALL_24H_MM']
    elif isinstance(parameters, str):
        parameters = [parameters]

    # 🔑 OTOMATISASI START_DATE JIKA FILTER BERDASARKAN BASELINE 1991
    effective_start_date = start_date
    if min_80pct_only and min_80pct_baseline == '1991':
        # Pastikan waktu minimal 1991
        if effective_start_date is None or effective_start_date < '1991-01-01':
            effective_start_date = '1991-01-01'

    # Bangun query
    query = """
    SELECT 
        o.time,
        o.wmo_id,
        o.parameter,
        o.source,
        o.value,
        o.baseline,
        o.region,
        m.name,
        m.latitude,
        m.longitude,
        m.province,
        m.regency,
        m.elevation,
        a_obs.availability AS availability,
        a_obs.meets_80pct AS meets_80pct
    FROM observations o
    JOIN station_metadata m ON o.wmo_id = m.wmo_id
    JOIN station_availability a_obs 
        ON o.wmo_id = a_obs.wmo_id 
        AND o.parameter = a_obs.parameter 
        AND o.baseline = a_obs.baseline
    """
    
    # Tambahkan join untuk filter kelengkapan dari baseline lain
    if min_80pct_only:
        query += f"""
        JOIN station_availability a_filter 
            ON o.wmo_id = a_filter.wmo_id 
            AND o.parameter = a_filter.parameter 
            AND a_filter.baseline = '{min_80pct_baseline}'
        """
    
    query += """
    WHERE 
        o.baseline = :baseline
        AND o.parameter = ANY(:parameters)
        AND o.source = ANY(:sources)
    """
    
    params = {
        'baseline': baseline,
        'parameters': parameters,
        'sources': sources
    }
    
    # Filter kelengkapan
    if min_80pct_only:
        query += " AND a_filter.meets_80pct = TRUE"
    
    # Filter waktu (gunakan effective_start_date)
    if effective_start_date:
        query += " AND o.time >= :start_date"
        params['start_date'] = effective_start_date
    if end_date:
        query += " AND o.time <= :end_date"
        params['end_date'] = end_date
    
    # Filter lokasi
    if stations:
        query += " AND o.wmo_id = ANY(:stations)"
        params['stations'] = stations
    if provinces:
        query += " AND m.province = ANY(:provinces)"
        params['provinces'] = provinces
    if regions:
        query += " AND o.region = ANY(:regions)"
        params['regions'] = regions
    
    query += " ORDER BY o.time, o.wmo_id"
    if limit:
        query += " LIMIT :limit"
        params['limit'] = limit

    df = pd.read_sql(text(query), engine, params=params)
    
    # Proses agregasi (sama seperti sebelumnya)
    if time_aggregation and len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])
        if time_aggregation == 'monthly':
            df['time'] = df['time'].dt.to_period('M').dt.start_time
        elif time_aggregation == 'yearly':
            df['time'] = df['time'].dt.to_period('Y').dt.start_time
        
        group_cols = [
            'time', 'wmo_id', 'parameter', 'source', 'baseline', 'region',
            'name', 'latitude', 'longitude', 'province', 'regency', 'elevation'
        ]
        numeric_cols = ['value', 'availability']
        agg_dict = {col: 'mean' for col in numeric_cols}
        df = df.groupby(group_cols).agg(agg_dict).reset_index()
    
    if spatial_aggregation and len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])
        if spatial_aggregation == 'province':
            group_cols = ['time', 'parameter', 'source', 'province']
        elif spatial_aggregation == 'region':
            group_cols = ['time', 'parameter', 'source', 'region']
        elif spatial_aggregation == 'national':
            group_cols = ['time', 'parameter', 'source']
        else:
            group_cols = ['time', 'parameter', 'source']
        
        numeric_cols = ['value', 'availability']
        agg_dict = {col: ['mean', 'std', 'count'] for col in numeric_cols}
        df = df.groupby(group_cols).agg(agg_dict).round(4).reset_index()
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    
    print(f"✅ Mengambil {len(df)} baris data (baseline={baseline}, "
          f"filter kelengkapan dari baseline={min_80pct_baseline}, "
          f"waktu ≥ {effective_start_date})")
    return df

# ==============================
# 3. FUNGSI TAMBAHAN: INFO DATA
# ==============================
def get_data_summary(df):
    """Dapatkan ringkasan informasi tentang data yang diambil."""
    if df.empty:
        print("DataFrame kosong")
        return
    print("\n📊 RINGKASAN DATA:")
    print(f"Total baris: {len(df)}")
    print(f"Stasiun unik: {df['wmo_id'].nunique()}")
    print(f"Parameter: {df['parameter'].unique().tolist()}")
    print(f"Sumber: {df['source'].unique().tolist()}")
    print(f"Periode: {df['time'].min()} s.d. {df['time'].max()}")
    if 'province' in df.columns:
        print(f"Provinsi: {df['province'].nunique()}")
    if 'region' in df.columns:
        print(f"Region: {df['region'].nunique()}")
    if 'meets_80pct' in df.columns:
        good_pct = df['meets_80pct'].mean() * 100
        print(f"Persentase stasiun ≥80%: {good_pct:.1f}%")        


# **01.Sample : Load Dataset Untuk Analisa Trend Indeks Ekstrim**

# In[55]:


df = query_climate_data(
    parameters=['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C'],
    sources='homo',           # → otomatis baseline='1981'
    min_80pct_only=True,
    min_80pct_baseline='1981',  # → filter dari baseline 1991
    # start_date tidak perlu diisi — otomatis jadi '1991-01-01'
    end_date='2025-12-31'
)
print(df[df['parameter'] == 'TEMPERATURE_AVG_C']['wmo_id'].nunique())
print(df[df['parameter'] == 'TEMP_24H_TN_C']['wmo_id'].nunique())
print(df[df['parameter'] == 'TEMP_24H_TX_C']['wmo_id'].nunique())


# **02.Sample : Load Dataset Untuk Analisa Anomali Suhu Udara**

# In[56]:


df = query_climate_data(
    parameters=['TEMPERATURE_AVG_C'],
    sources='homo',
    baseline='1991',
    min_80pct_only=True,
    start_date='1991-01-01',
    end_date='2025-12-31'
)
df.head()
print(df[df['parameter'] == 'TEMPERATURE_AVG_C']['wmo_id'].nunique())


# **03.Sample : Load Dataset Untuk Analisa Monthly Diff**

# In[57]:


df = query_climate_data(
    parameters=['TEMPERATURE_AVG_C'],
    sources='qc',
    min_80pct_only=False,
    min_80pct_baseline='1991',
)
df.head()
print(df[df['parameter'] == 'TEMPERATURE_AVG_C']['wmo_id'].nunique())

