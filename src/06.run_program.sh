#!/bin/bash

# --- KONFIGURASI PATH ---
# Lokasi interpreter Python di env wrfpython Anda
PYTHON_BIN="/home/api/anaconda3/envs/wrfpython/bin/python"

# Ambil lokasi folder tempat script .sh ini berada
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/pipeline_execution.log"

# Daftar file yang akan dijalankan secara berurutan
SCRIPTS=(
    "00.Get_Updated_Data.py"
    "01.QC_Dataset_Level_01.py"
    "02.Regionalisasi_Dataset.py"
    "03.Homogenisasi_Dataset_Monthly.py"
    "04.Homogenisasi_Dataset_Merged.py"
    "05.Generate_Longform_Dataset.py"
    "06.Anomali.py"
)

# --- VALIDASI ---
if [ ! -f "$PYTHON_BIN" ]; then
    echo "❌ Error: Python tidak ditemukan di $PYTHON_BIN"
    exit 1
fi

# --- MULAI EKSEKUSI ---
echo "--- Pipeline Log ($(date)) ---" > "$LOG_FILE"
echo "📂 Direktori Project: $SCRIPT_DIR"
echo "🐍 Menggunakan Python: $PYTHON_BIN"
echo "------------------------------------------------"

for script in "${SCRIPTS[@]}"; do
    FULL_PATH="$SCRIPT_DIR/$script"
    
    # Cek apakah file script python-nya ada
    if [ -f "$FULL_PATH" ]; then
        echo -n "▶️ Menjalankan $script... "
        
        # Eksekusi langsung menggunakan path python env
        # Kita gunakan 'cd' agar script python bisa membaca file lokal di foldernya
        (cd "$SCRIPT_DIR" && "$PYTHON_BIN" "$FULL_PATH") >> "$LOG_FILE" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ SELESAI"
        else
            echo "❌ GAGAL"
            echo "⚠️ Cek error log di: $LOG_FILE"
            exit 1
        fi
    else
        echo "⏭️ DILEWATI (File $script tidak ditemukan)"
    fi
done

echo "------------------------------------------------"
echo "🎉 Seluruh proses 00-05 telah selesai!"