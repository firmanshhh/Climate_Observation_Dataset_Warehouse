# Developing_Climate_Observation_Dataset_Warehouse
📊 Dokumentasi Pipeline QC Data Iklim Harian
Versi: 1.0
Penulis: Firmansyah (Direktorat Perubahan Iklim)
Tanggal: 19 Januari 2026

Pipeline ini melakukan Quality Control (QC) Level 01 terhadap data iklim harian dari stasiun observasi, mencakup suhu (TEMPERATURE_AVG_C, TEMP_24H_TN_C, TEMP_24H_TX_C) dan curah hujan (RAINFALL_24H_MM). Output berupa data yang telah divalidasi, dibersihkan, dan dikoreksi untuk digunakan dalam analisis lanjutan (regionalisasi, homogenisasi, prediksi).
🗂️ Struktur Direktori
Proyek/
├── 00.Raw_Dataset/                     # Data mentah (CSV)
├── 02.QC_Dataset_Level_01/             # Output QC
│   ├── [PARAMETER]/
│   │   ├── 00.[PARAM]_Non_Homogen_[TAHAP]/
│   │   │   ├── plots/                  # Time series per stasiun (PNG)
│   │   │   └── netcdf/                 # Data per stasiun (NetCDF)
│   │   ├── 01.[PARAM]_Summary/         # Ringkasan statistik QC
│   │   └── 02.[PARAM]_Adjusted/        # File CSV hasil tiap tahap
│   └── ...                             # Satu folder per parameter
└── script_qc.py                        # File utama pipeline

⚙️ Konfigurasi Utama
Rentang Waktu
Mulai: 1991-01-01
Akhir: Otomatis sampai hari ini (datetime.now())
Batasan Fisik


