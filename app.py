import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from scipy.stats import linregress

st.set_page_config(page_title="Analisis Pengangguran Jawa Barat", layout="wide")

# =========================
# 1. LOAD & CLEANING DATA
# =========================

from typing import Any

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and clean the unemployment data from CSV.
    Returns a DataFrame with an additional 'pendidikan_bersih' column.
    """
    try:
        df = pd.read_csv('disnakertrans-od_15807_jumlah_pengangguran_terbuka_berdasarkan_pendidikan_v1_data.csv', delimiter=';')
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty dataframe on error

    # Check if 'pendidikan' column exists
    if 'pendidikan' not in df.columns:
        st.error("Kolom 'pendidikan' tidak ditemukan dalam data.")
        return pd.DataFrame()

    # Define mapping for education categories
    education_map = {
        'SMA': 'SMA',
        'SD KE BAWAH': 'SD KE BAWAH',
        'TIDAK/BELUM PERNAH SEKOLAH': 'SD KE BAWAH',
        'TIDAK/BELUM TAMAT SD': 'SD KE BAWAH',
        'SD': 'SD',
        'SMP': 'SMP',
        'DIPLOMA': 'DIPLOMA/UNIV',
        'UNIVERSITAS': 'DIPLOMA/UNIV',
        'AKADEMI': 'DIPLOMA/UNIV'
    }

    def clean_education(x: Any) -> str:
        if not isinstance(x, str):
            return 'UNKNOWN'
        x_upper = x.upper()
        for key, val in education_map.items():
            if key in x_upper:
                return val
        return x_upper

    # Handle missing or NaN values in 'pendidikan'
    df['pendidikan'] = df['pendidikan'].fillna('UNKNOWN')

    # Apply cleaning function
    df['pendidikan_bersih'] = df['pendidikan'].apply(clean_education)

    return df

df = load_data()

# =========================
# 2. DATA EXPLORATION
# =========================

st.title("📊 Analisis Pengangguran Jawa Barat berdasarkan Pendidikan (2011-2023)")
st.caption("Sumber data: BPS Jawa Barat | Visualisasi: Streamlit")

# Sidebar filter
st.sidebar.header("Filter Data")
tahun_min, tahun_max = int(df['tahun'].min()), int(df['tahun'].max())
tahun_range = st.sidebar.slider("Pilih rentang tahun", tahun_min, tahun_max, (tahun_min, tahun_max), 1)
pendidikan_list = ['SD KE BAWAH', 'SD', 'SMP', 'SMA', 'DIPLOMA/UNIV']
pendidikan_pilih = st.sidebar.multiselect("Pilih pendidikan", pendidikan_list, pendidikan_list)

# Filter data
df_filtered = df[(df['tahun'] >= tahun_range[0]) & (df['tahun'] <= tahun_range[1])]
df_filtered = df_filtered[df_filtered['pendidikan_bersih'].isin(pendidikan_pilih)]

# =========================
# 3. STATISTIK DESKRIPTIF
# =========================

st.subheader("Statistik Deskriptif")
if df_filtered.empty:
    st.info("Tidak ada data untuk ditampilkan pada statistik deskriptif.")
else:
    st.dataframe(
        df_filtered.groupby(['pendidikan_bersih'])['jumlah_pengangguran_terbuka']
        .describe()[['mean', 'std', 'min', 'max']].round(0)
    )

missing = df.isnull().sum()
if missing.any():
    st.warning("Ada data hilang:\n" + str(missing[missing>0]))

# =========================
# 4. VISUALISASI TREN
# =========================

st.subheader("Tren Pengangguran Terbuka per Pendidikan")

# Pivot table dari hasil filter sidebar
pivot = df_filtered.pivot_table(
    index='tahun',
    columns='pendidikan_bersih',
    values='jumlah_pengangguran_terbuka',
    aggfunc='sum'
).fillna(0)
available_cols = [col for col in pendidikan_list if col in pivot.columns]
pivot = pivot[available_cols]

if pivot.empty or len(available_cols) == 0:
    st.info("Silakan pilih minimal satu pendidikan dan tahun untuk menampilkan grafik tren pengangguran.")
else:
    fig, ax = plt.subplots(figsize=(10,5))
    pivot.plot(ax=ax, marker='o')
    ax.set_ylabel("Jumlah Pengangguran")
    ax.set_xlabel("Tahun")
    ax.set_title("Tren Pengangguran Terbuka per Pendidikan")
    ax.legend(title="Pendidikan")
    st.pyplot(fig)

# =========================
# 5. STACKED BAR CHART
# =========================

st.subheader("Proporsi Pengangguran per Pendidikan (Stacked Bar)")
if pivot.empty or len(available_cols) == 0:
    st.info("Silakan pilih minimal satu pendidikan dan tahun untuk menampilkan grafik proporsi pengangguran.")
else:
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    fig2, ax2 = plt.subplots(figsize=(10,5))
    pivot_pct.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
    ax2.set_ylabel("Persentase (%)")
    ax2.set_xlabel("Tahun")
    ax2.set_title("Proporsi Pengangguran Terbuka per Pendidikan")
    st.pyplot(fig2)

# =========================
# 6. PIE CHART TAHUN TERTENTU
# =========================

st.subheader("Pie Chart Proporsi Pengangguran per Pendidikan pada Tahun Tertentu")
tahun_opsi = sorted(df_filtered['tahun'].unique())
if len(tahun_opsi) == 0:
    st.warning("Tidak ada data tahun untuk pie chart pada filter ini.")
else:
    tahun_pie = st.selectbox("Pilih tahun untuk pie chart", tahun_opsi, index=len(tahun_opsi)-1)
    pie_data = df_filtered[df_filtered['tahun']==tahun_pie].groupby('pendidikan_bersih')['jumlah_pengangguran_terbuka'].sum()
    pie_data = pie_data[pie_data > 0]  # hilangkan kategori 0
    if pie_data.empty:
        st.info("Tidak ada data pengangguran pada tahun terpilih.")
    else:
        fig3, ax3 = plt.subplots()
        ax3.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        ax3.axis('equal')
        st.pyplot(fig3)

# =========================
# 7. HEATMAP KORELASI
# =========================

st.subheader("Heatmap Korelasi Tahun vs Pengangguran per Pendidikan")
if pivot.empty or len(available_cols) == 0:
    st.info("Tidak ada data untuk membuat heatmap korelasi.")
else:
    corr_df = pivot.fillna(0)
    corr = corr_df.corr()
    fig4, ax4 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
    ax4.set_title("Korelasi Jumlah Pengangguran antar Pendidikan")
    st.pyplot(fig4)

# =========================
# 8. REGRESI LINEAR SEDERHANA
# =========================

st.subheader("Regresi Linear Sederhana (Tren Pengangguran per Pendidikan)")
from scipy.stats import linregress

if pivot.empty or len(available_cols) == 0:
    st.info("Tidak ada data untuk regresi linear.")
else:
    for p in available_cols:
        y = pivot[p].dropna()
        x = y.index.values
        if len(y) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            st.info(f"{p}: y = {slope:.0f}x + {intercept:.0f} | R²={r_value**2:.2f} | {'Naik' if slope>0 else 'Turun'}")

# =========================
# 9. GROUPED BAR CHART (BAR SAMPINGAN)
# =========================

st.subheader("Jumlah Pengangguran Terbuka per Pendidikan per Tahun (Grouped Bar Chart)")
if pivot.empty or len(available_cols) == 0:
    st.info("Silakan pilih minimal satu pendidikan dan tahun untuk menampilkan grouped bar chart.")
else:
    fig5, ax5 = plt.subplots(figsize=(12,6))
    bar_width = 0.15
    index = np.arange(len(pivot.index))
    for i, col in enumerate(pivot.columns):
        ax5.bar(index + i*bar_width, pivot[col], bar_width, label=col)
    ax5.set_xlabel('Tahun')
    ax5.set_ylabel('Jumlah Pengangguran')
    ax5.set_title('Jumlah Pengangguran Terbuka per Pendidikan per Tahun (Grouped Bar Chart)')
    ax5.set_xticks(index + bar_width * (len(pivot.columns)-1) / 2)
    ax5.set_xticklabels(pivot.index)
    ax5.legend()
    plt.xticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig5)

# =========================
# 10. INSIGHT OTOMATIS
# =========================

st.subheader("Insight Otomatis")
if df_filtered.empty:
    st.info("Tidak ada data untuk insight otomatis pada filter ini.")
else:
    avg_pengangguran = df_filtered.groupby('pendidikan_bersih')['jumlah_pengangguran_terbuka'].mean().idxmax()
    tahun_tertinggi = df_filtered.loc[df_filtered['jumlah_pengangguran_terbuka'].idxmax()]['tahun']
    st.success(f"""
    - Rata-rata pengangguran terbuka tertinggi berasal dari pendidikan *{avg_pengangguran}*
    - Tahun dengan pengangguran terbuka tertinggi: *{tahun_tertinggi}*
    """)


# =========================
# 11. DOWNLOAD DATA & CHART
# =========================

st.sidebar.header("Download")
csv = df_filtered.to_csv(index=False).encode()
st.sidebar.download_button("Download Data Filtered (CSV)", csv, "data_filtered.csv", "text/csv")

# Download chart (contoh: chart tren)
if not pivot.empty and len(available_cols) > 0:
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.sidebar.download_button("Download Chart Tren (PNG)", buf.getvalue(), "chart_tren.png", "image/png")

# =========================
# 12. TAMPILKAN DATA MENTAH
# =========================

with st.expander("Lihat Data Mentah"):
    st.write(df_filtered)

# =========================
# 13. README SINGKAT
# =========================

with st.expander("📄 Penjelasan Analisis"):
    st.markdown("""
    *Proyek Analisis Data Pengangguran Jawa Barat*
    
    - Data diambil dari BPS Jawa Barat (2011-2023)
    - Data dibersihkan dan distandarisasi kategori pendidikan
    - Analisis meliputi statistik deskriptif, tren, proporsi, korelasi, dan regresi linear
    - Insight otomatis dan fitur download tersedia
    - Visualisasi interaktif dengan Streamlit
    """)

st.markdown("""
---
## ✍ Interpretasi Hasil Visualisasi

1. *Tren Pengangguran Terbuka per Pendidikan*
   - Lulusan *SMA* (umum & kejuruan) secara konsisten menjadi penyumbang pengangguran terbuka tertinggi di Jawa Barat setiap tahun, bahkan sejak 2020 jumlahnya melonjak tajam.
   - Pengangguran lulusan *SMP* dan *SD* cenderung menurun, namun tetap signifikan dalam jumlah.
   - *Lulusan Diploma/Universitas* jumlahnya paling kecil, tetapi sejak 2020 juga mengalami lonjakan, meski proporsinya tetap jauh di bawah SMA.

2. *Proporsi Pengangguran*
   - Proporsi pengangguran lulusan *SMA* (umum & kejuruan) selalu dominan, bahkan setelah pandemi proporsinya lebih dari 50% dari total pengangguran terbuka.
   - Proporsi lulusan *SD ke bawah* dan *SMP* cenderung menurun, menandakan kelompok pendidikan rendah semakin sedikit yang aktif mencari kerja atau terserap sektor informal.
   - Proporsi lulusan *Diploma/Universitas* naik tipis, menandakan tantangan baru untuk lulusan pendidikan tinggi.

3. *Dampak Pandemi*
   - Tahun *2020* adalah puncak pengangguran terbuka di seluruh tingkat pendidikan, terutama pada lulusan SD dan SMA, akibat dampak ekonomi pandemi COVID-19.
   - Setelah 2020, jumlah pengangguran menurun namun belum kembali ke level sebelum pandemi, terutama pada lulusan SMA dan Diploma/Universitas.

4. *Insight Kebijakan*
   - *Lulusan SMA* perlu mendapat perhatian khusus dalam kebijakan penanggulangan pengangguran, misalnya pelatihan vokasi, magang, atau peningkatan soft skill.
   - *Lulusan Diploma/Universitas* juga mulai perlu difokuskan pada link & match pendidikan dengan kebutuhan industri.
   - *Pendidikan dasar* (SD/SMP) tetap penting, namun risiko pengangguran terbesar kini ada di lulusan menengah dan atas.

---
""")