from docxtpl import DocxTemplate
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import os
import re
import pandas as pd


WORKING_DIR      = os.path.abspath(os.path.join(os.getcwd(), '..'))
DATA_DIR         = os.path.join(WORKING_DIR, 'data')
LONG_DIR         = os.path.join(DATA_DIR, '05.Long_Format_Dataset')

def generate_report(year, month):
    PLOT_ANOMALI_DIR = os.path.join(LONG_DIR, 'PLOT_ANOMALI')
    RESULT_DIR       = os.path.join(PLOT_ANOMALI_DIR, f'{year}/{year}_{month:02d}')
    TEMPLATE_PATH    = os.path.join(WORKING_DIR, 'src', '00.TEMPLATE_REPORT.docx')
    try:
        SUMMARY_INDO     = pd.read_csv(os.path.join(LONG_DIR, f'06.TEMPERATURE_ANOMALI_INDONESIA_DB.csv'))
        SUMMARY_INDO     = SUMMARY_INDO.rename(columns={'year':'Tahun','month':'Bulan'})
        SUMMARY_INDO     = SUMMARY_INDO[(SUMMARY_INDO['Tahun'] == year) & (SUMMARY_INDO['Bulan'] == month)].reset_index(drop=True)
        DF_ANOMALI       = pd.read_csv(os.path.join(LONG_DIR, f'06.TEMPERATURE_ANOMALI_DB.csv'))
        DF_ANOMALI       = DF_ANOMALI.rename(columns={'month':'Bulan', 'year':'Tahun', 'regency':'kabupaten', 'province':'provinsi'})
        DF_ANOMALI       = DF_ANOMALI[(DF_ANOMALI['Tahun'] == year) & (DF_ANOMALI['Bulan'] == month)].reset_index(drop=True)
        DF_MONTHDIFF       = pd.read_csv(os.path.join(LONG_DIR, f'07.TEMPERATURE_MONTHLY_DIFF_DB.csv'))
        DF_MONTHDIFF     = DF_MONTHDIFF.rename(columns={'month':'Bulan', 'year':'Tahun', 'regency':'kabupaten', 'province':'provinsi'})
        DF_MONTHDIFF     = DF_MONTHDIFF[(DF_MONTHDIFF['Tahun'] == year) & (DF_MONTHDIFF['Bulan'] == month)].reset_index(drop=True)
    except FileNotFoundError:
        print(f"Data tidak ditemukan untuk {year}-{month:02d}")
        return


    month_names      = {1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'}
    context = {
        'bulan': month_names[month],
        'tahun': year,
        'bulan_sebelum': month_names[month - 1] if month != 1 else month_names[12],
        'tahun_sebelum': year if month != 1 else year - 1,
        'jumlah_stasiun': SUMMARY_INDO.at[0, 'jumlah_stasiun_bulan_ini'],
        'suhu_rata': round(SUMMARY_INDO.at[0, 'trata'], 2),
        'normal_suhu': round(SUMMARY_INDO.at[0, 'normal'], 2),
        'kisaran_min': round(DF_ANOMALI['value'].min(), 2),
        'kisaran_max': round(DF_ANOMALI['value'].max(), 2),
        'anomali': round(SUMMARY_INDO.at[0, 'anomali'], 2),
        'jenis_anomali': 'positif' if SUMMARY_INDO.at[0, 'anomali'] >= 0 else 'negatif',
        'peringkat_anomali': int(SUMMARY_INDO.at[0, 'rank_from_all_month']),
        'peringkat_jenis': 'tertinggi',
        'tahun_awal_historis': 1991,
        'dominan_anomali': 'positif' if DF_ANOMALI[DF_ANOMALI['anomali'] >= 0].shape[0] >= DF_ANOMALI[DF_ANOMALI['anomali'] < 0].shape[0] else 'negatif',
        'jumlah_dominan_stasiun': DF_ANOMALI[DF_ANOMALI['anomali'] >= 0].shape[0],
        'stasiun_anomali_max': DF_ANOMALI.loc[DF_ANOMALI['anomali'].idxmax(), 'name'],
        'anomali_max': round(DF_ANOMALI.loc[DF_ANOMALI['anomali'].idxmax(), 'anomali'], 2),
        'stasiun_anomali_min': DF_ANOMALI.loc[DF_ANOMALI['anomali'].idxmin(), 'name'],
        'anomali_min': round(DF_ANOMALI.loc[DF_ANOMALI['anomali'].idxmin(), 'anomali'], 2),
        'jumlah_stasiun_mondiff': len(DF_MONTHDIFF['diff']),
        'tren_selisih': 'negatif (penurunan Suhu)' if DF_MONTHDIFF['diff'].mean() < 0 else 'positif (peningkatan Suhu)',
        'stasiun_peningkatan_max': DF_MONTHDIFF.loc[DF_MONTHDIFF['rank_from_all_station'].idxmin(), 'name'],
        'peningkatan_max': round(DF_MONTHDIFF.loc[DF_MONTHDIFF['rank_from_all_station'].idxmin(), 'diff'], 2),
        'stasiun_penurunan_max': DF_MONTHDIFF.loc[DF_MONTHDIFF['rank_from_all_station'].idxmax(), 'name'],
        'penurunan_max': round(DF_MONTHDIFF.loc[DF_MONTHDIFF['rank_from_all_station'].idxmax(), 'diff'], 2),
        'no_gambar_16': 16,
        'no_gambar_17': 17,
        'no_gambar_18': 18,
        'no_gambar_19': 19,
        'no_gambar_20': 20,

    # ==================== 5 STASIUN ANOMALI TERTINGGI ====================
        'ANPLUS1_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[0]['name'],
        'ANPLUS1_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[0]['kabupaten'],
        'ANPLUS1': round(DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[0]['anomali'], 2),                  # nilai anomali dalam °C (bisa tambah °C di template jika mau)

        'ANPLUS2_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[1]['name'],
        'ANPLUS2_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[1]['kabupaten'],
        'ANPLUS2': round(DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[1]['anomali'], 2),

        'ANPLUS3_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[2]['name'],
        'ANPLUS3_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[2]['kabupaten'],
        'ANPLUS3': round(DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[2]['anomali'], 2),

        'ANPLUS4_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[3]['name'],
        'ANPLUS4_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[3]['kabupaten'],
        'ANPLUS4': round(DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[3]['anomali'], 2),

        'ANPLUS5_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[4]['name'],
        'ANPLUS5_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[4]['kabupaten'],
        'ANPLUS5': round(DF_ANOMALI.sort_values(by='anomali', ascending=False).iloc[4]['anomali'], 2),

        # ==================== 5 STASIUN ANOMALI TERENDAH ====================
        'ANMIN1_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[0]['name'],
        'ANMIN1_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[0]['kabupaten'],
        'ANMIN1': round(DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[0]['anomali'], 2),                  # negatif otomatis terbaca sebagai terendah

        'ANMIN2_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[1]['name'],
        'ANMIN2_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[1]['kabupaten'],
        'ANMIN2': round(DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[1]['anomali'], 2),

        'ANMIN3_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[2]['name'],
        'ANMIN3_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[2]['kabupaten'],
        'ANMIN3': round(DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[2]['anomali'], 2),

        'ANMIN4_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[3]['name'],
        'ANMIN4_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[3]['kabupaten'],
        'ANMIN4': round(DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[3]['anomali'], 2),

        'ANMIN5_NAME': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[4]['name'],
        'ANMIN5_KAB': DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[4]['kabupaten'],
        'ANMIN5': round(DF_ANOMALI.sort_values(by='anomali', ascending=True).iloc[4]['anomali'], 2),
    # ==================== 5 STASIUN ANOMALI TERTINGGI ====================
        'TPLUS1_NAME': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[0]['name'],
        'TPLUS1_KAB': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[0]['kabupaten'],
        'TPLUS1': round(DF_ANOMALI.sort_values(by='value', ascending=False).iloc[0]['value'], 2),                  # nilai Tomali dalam °C (bisa tambah °C di template jika mau)

        'TPLUS2_NAME': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[1]['name'],
        'TPLUS2_KAB': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[1]['kabupaten'],
        'TPLUS2': round(DF_ANOMALI.sort_values(by='value', ascending=False).iloc[1]['value'], 2),

        'TPLUS3_NAME': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[2]['name'],
        'TPLUS3_KAB': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[2]['kabupaten'],
        'TPLUS3': round(DF_ANOMALI.sort_values(by='value', ascending=False).iloc[2]['value'], 2),

        'TPLUS4_NAME': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[3]['name'],
        'TPLUS4_KAB': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[3]['kabupaten'],
        'TPLUS4': round(DF_ANOMALI.sort_values(by='value', ascending=False).iloc[3]['value'], 2),

        'TPLUS5_NAME': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[4]['name'],
        'TPLUS5_KAB': DF_ANOMALI.sort_values(by='value', ascending=False).iloc[4]['kabupaten'],
        'TPLUS5': round(DF_ANOMALI.sort_values(by='value', ascending=False).iloc[4]['value'], 2),

        # ==================== 5 STASIUN TOMALI TERENDAH ====================
        'TMIN1_NAME': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[0]['name'],
        'TMIN1_KAB': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[0]['kabupaten'],
        'TMIN1': round(DF_ANOMALI.sort_values(by='value', ascending=True).iloc[0]['value'], 2),                  # negatif otomatis terbaca sebagai terendah

        'TMIN2_NAME': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[1]['name'],
        'TMIN2_KAB': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[1]['kabupaten'],
        'TMIN2': round(DF_ANOMALI.sort_values(by='value', ascending=True).iloc[1]['value'], 2),

        'TMIN3_NAME': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[2]['name'],
        'TMIN3_KAB': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[2]['kabupaten'],
        'TMIN3': round(DF_ANOMALI.sort_values(by='value', ascending=True).iloc[2]['value'], 2),

        'TMIN4_NAME': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[3]['name'],
        'TMIN4_KAB': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[3]['kabupaten'],
        'TMIN4': round(DF_ANOMALI.sort_values(by='value', ascending=True).iloc[3]['value'], 2),

        'TMIN5_NAME': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[4]['name'],
        'TMIN5_KAB': DF_ANOMALI.sort_values(by='value', ascending=True).iloc[4]['kabupaten'],
        'TMIN5': round(DF_ANOMALI.sort_values(by='value', ascending=True).iloc[4]['value'], 2),

    # ==================== 5 STASIUN SELISIH RATA-RATA TERTINGGI ====================
        'DPLUS1_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[0]['name'],
        'DPLUS1_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[0]['kabupaten'],
        'DPLUS1': round(DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[0]['diff'], 2),                  # nilai Tomali dalam °C (bisa tambah °C di template jika mau)

        'DPLUS2_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[1]['name'],
        'DPLUS2_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[1]['kabupaten'],
        'DPLUS2': round(DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[1]['diff'], 2),

        'DPLUS3_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[2]['name'],
        'DPLUS3_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[2]['kabupaten'],
        'DPLUS3': round(DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[2]['diff'], 2),

        'DPLUS4_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[3]['name'],
        'DPLUS4_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[3]['kabupaten'],
        'DPLUS4': round(DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[3]['diff'], 2),

        'DPLUS5_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[4]['name'],
        'DPLUS5_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[4]['kabupaten'],
        'DPLUS5': round(DF_MONTHDIFF.sort_values(by='diff', ascending=False).iloc[4]['diff'], 2),

        # ==================== 5 STASIUN SELISIH TERENDAH ====================
        'DMIN1_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[0]['name'],
        'DMIN1_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[0]['kabupaten'],
        'DMIN1': round(DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[0]['diff'], 2),                  # negatif otomatis terbaca sebagai terendah

        'DMIN2_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[1]['name'],
        'DMIN2_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[1]['kabupaten'],
        'DMIN2': round(DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[1]['diff'], 2),

        'DMIN3_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[2]['name'],
        'DMIN3_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[2]['kabupaten'],
        'DMIN3': round(DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[2]['diff'], 2),

        'DMIN4_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[3]['name'],
        'DMIN4_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[3]['kabupaten'],
        'DMIN4': round(DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[3]['diff'], 2),

        'DMIN5_NAME': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[4]['name'],
        'DMIN5_KAB': DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[4]['kabupaten'],
        'DMIN5': round(DF_MONTHDIFF.sort_values(by='diff', ascending=True).iloc[4]['diff'], 2),

    }

    #rangking peringkat anomali suhu 5 tertinggi dan terendah (1981-2024) ANPLUS1NAMA, ANPLUS1KAB, ANPLUSVALUE

    # Render template dengan docxtpl
    doc = DocxTemplate(TEMPLATE_PATH)
    doc.render(context)
    temp_file = "temp_anomali.docx"
    doc.save(temp_file)

    # Buka dengan python-docx untuk proses gambar
    doc_final = Document(temp_file)

    # Mapping marker → path file gambar spesifik (sesuaikan dengan struktur folder Anda)
    gambar_mapping = {
        'GAMBAR 16': os.path.join(RESULT_DIR, f'TRATA_{year}_vs_NORMAL_1991-2020.png'),
        'GAMBAR 17': os.path.join(RESULT_DIR, f'GRAFIK_HISTORIS_ANOMALI_{context["bulan"].upper()}.png'),
        'GAMBAR 18': os.path.join(RESULT_DIR, f'ANOMALI_TEMPERATURE_BULANAN_{year}_{context["bulan"].upper()}_POINT.png'),
        'GAMBAR 19': os.path.join(RESULT_DIR, f'TEMPERATURE_BULANAN_{year}_{context["bulan"].upper()}_POINT.png'),
        'GAMBAR 20': os.path.join(RESULT_DIR, f'MONTHLY_DIFF_TEMPERATURE_BULANAN_{year}_{context["bulan"].upper()}_POINT.png'),
    }
    # Kumpulkan paragraf yang mengandung marker
    markers_to_replace = []
    for i, para in enumerate(doc_final.paragraphs):
        text = para.text.strip()
        if text in gambar_mapping:
            markers_to_replace.append((i, text))
    # Proses dari belakang ke depan untuk menghindari shifting index
    for i, marker in reversed(markers_to_replace):
        para = doc_final.paragraphs[i]
        img_path = gambar_mapping[marker]
        if os.path.exists(img_path):
            # Clear existing paragraph dan tambahkan gambar
            para.clear()
            run = para.add_run()
            run.add_picture(img_path, width=Inches(6.5))
            para.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
            para.paragraph_format.space_after = Inches(0.3)
            print(f"✓ Gambar '{marker}' berhasil diinsert: {img_path}")
        else:
            # Fallback: beri warning tapi jangan hapus konten
            print(f"⚠ File gambar tidak ditemukan: {img_path}")
            para.text = f"[GAMBAR {marker} TIDAK DITEMUKAN]"
    # Simpan final
    nama_file = f"TECHNICAL_GUIDANCE_{context['bulan'].upper()}_{context['tahun']}.docx"
    out_file  = os.path.join(RESULT_DIR, nama_file)
    doc_final.save(out_file)
    os.remove(temp_file)
    print(f"\n✅ Laporan selesai: {os.path.abspath(nama_file)}")

import datetime
now     = datetime.datetime.now()
yearnow = now.year
for year in range(2026, yearnow + 1):
    for month in range(1, 13):
        try:
            generate_report(year, month)
        except Exception as e:
            print(f"Data tidak tersedia untuk {year}-{month:02d} - Melewati. Error: {e}")