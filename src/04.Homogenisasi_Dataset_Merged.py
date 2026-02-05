#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from datetime import datetime
import glob

# ==============================
# 1. KONFIGURASI
# ==============================
WORKING_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_ROOT   = os.path.join(WORKING_DIR, 'data')
HOMO_DIR    = os.path.join(DATA_ROOT, '03.QC_Level_02')
MERGED_DIR  = os.path.join(DATA_ROOT, '04.Dataset_Final')
for param in ['TEMPERATURE_AVG_C', 'TEMP_24H_TN_C', 'TEMP_24H_TX_C']:
    for baseline in ['1981','1991']:
        #open multi station homogenized data
        files=glob.glob(os.path.join(HOMO_DIR,f"{param}_BASELINE_{baseline}", 'data', '*.csv'))
        print(f"Processing {param} with baseline {baseline}, found {len(files)} files.")
        dataall = pd.DataFrame()
        for file in files:
            data=pd.read_csv(file)
            dataall = pd.concat([dataall, data], ignore_index=True)
        dataall['baseline'] = baseline
        dataall['parameter'] = param
        dataall.to_csv(os.path.join(MERGED_DIR, f"{param}_BASELINE_{baseline}.csv"), index=False)