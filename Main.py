import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from io import StringIO
import base64
from datetime import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set tema warna untuk visualisasi
COLORS = px.colors.qualitative.Prism
BG_COLOR = 'rgba(240, 250, 245, 0.8)'
PRIMARY_COLOR = '#0c6e42'
SECONDARY_COLOR = '#e0f7fa'

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Beras Indonesia", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #eefbf5;
        background-image: linear-gradient(120deg, #e0f7fa 0%, #f3fdee 100%);
    }
    .stApp {
        background-color: #eefbf5;
        background-image: linear-gradient(120deg, #e0f7fa 0%, #f3fdee 100%);
    }
    .css-18e3th9 {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #0c6e42;
    }
    .stButton>button {
        background-color: #0c8051;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0a6a44;
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e0f7fa;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #0c6e42;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0c6e42 !important;
        color: white !important;
    }
    .info-box {
        background-color: #e0f7fa; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #0c6e42;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .annotation {
        font-size: 12px;
        color: #666;
        font-style: italic;
    }
    .footer {
        background-color: #e0f7fa;
        padding: 10px; 
        border-radius: 10px; 
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk membuat sampel data default
def create_default_data():
    # Data yang diberikan
    data = "Data.csv"
    data = pd.read_csv(data)
    data = data.to_dict(orient="records")
    
    # Mengubah tipe data menjadi integer
    for d in data:
        d["tahun"] = int(d["tahun"])
        d["produksi"] = int(d["produksi"])
        d["harga"] = int(d["harga"])
        d["nomor"] = int(d["nomor"])
    
    # Membuat DataFrame
    df = pd.DataFrame(data, columns=["provinsi", "tahun", "produksi", "harga", "nomor"])
    return df

# Fungsi untuk load data CSV
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        required_columns = ["provinsi", "tahun", "produksi", "harga", "nomor"]
        
        # Periksa kolom yang diperlukan
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_columns)}")
            return None
            
        # Pastikan tipe data yang benar
        if df["tahun"].dtype != "int64":
            df["tahun"] = pd.to_numeric(df["tahun"], errors="coerce")
        if df["produksi"].dtype != "float64" and df["produksi"].dtype != "int64":
            df["produksi"] = pd.to_numeric(df["produksi"], errors="coerce")
        if df["harga"].dtype != "float64" and df["harga"].dtype != "int64":
            df["harga"] = pd.to_numeric(df["harga"], errors="coerce")
        if df["nomor"].dtype != "int64":
            df["nomor"] = pd.to_numeric(df["nomor"], errors="coerce")
            
        return df
    except Exception as e:
        st.error(f"Error saat memproses file: {e}")
        return None

# Fungsi untuk analisis statistik
def statistical_analysis(df):
    stats = {}
    
    # Statistik deskriptif
    stats["describe"] = df[["produksi", "harga"]].describe()
    
    # Korelasi
    stats["correlation"] = df[["produksi", "harga"]].corr()
    
    # Produksi dan harga rata-rata per provinsi
    stats["avg_by_province"] = df.groupby("provinsi")[["produksi", "harga"]].mean()
    
    # Tren tahunan
    stats["yearly_trend"] = df.groupby("tahun")[["produksi", "harga"]].mean()
    
    return stats

# Fungsi untuk membuat prediksi sederhana
def make_simple_prediction(df, province, target_col):
    province_data = df[df["provinsi"] == province].sort_values("tahun")
    
    if len(province_data) < 2:
        return None, None
    
    # OLS model sederhana berdasarkan tahun
    X = sm.add_constant(province_data["tahun"])
    y = province_data[target_col]
    
    model = sm.OLS(y, X).fit()
    
    # Prediksi untuk tahun berikutnya
    last_year = province_data["tahun"].max()
    next_year = last_year + 1
    prediction = model.predict([1, next_year])[0]
    
    return next_year, prediction

# Fungsi untuk clustering
def perform_kmeans_clustering(df, year):
    year_df = df[df["tahun"] == year].copy()
    
    if len(year_df) < 3:  # Minimal 3 data untuk clustering
        return None, None
    
    # Standarisasi
    features = ["produksi", "harga"]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(year_df[features])
    
    # Tentukan jumlah cluster optimal dengan metode elbow
    wcss = []
    max_clusters = min(5, len(year_df)) # Maksimal 5 cluster atau jumlah data
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)
    
    # Pilih jumlah cluster (simplified)
    n_clusters = 2 if len(wcss) > 1 else 1
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    year_df['cluster'] = kmeans.fit_predict(scaled_features)
    
    # PCA untuk visualisasi
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    year_df['pca1'] = pca_result[:, 0]
    year_df['pca2'] = pca_result[:, 1]
    
    return year_df, kmeans.cluster_centers_

# Fungsi untuk visualisasi
def create_provincial_comparison(df, year):
    year_df = df[df["tahun"] == year]
    
    # Buat visualisasi harga dan produksi
    fig_price = px.bar(
        year_df.sort_values("harga", ascending=False), 
        x="provinsi", 
        y="harga",
        color="provinsi",
        color_discrete_sequence=COLORS,
        title=f"Harga Beras per Provinsi - {year}",
        labels={"provinsi": "Provinsi", "harga": "Harga (Rp/Kg)"}
    )
    fig_price.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    fig_prod = px.bar(
        year_df.sort_values("produksi", ascending=False),
        x="provinsi",
        y="produksi",
        color="provinsi",
        color_discrete_sequence=COLORS,
        title=f"Produksi Beras per Provinsi - {year}",
        labels={"provinsi": "Provinsi", "produksi": "Produksi (Ton)"}
    )
    fig_prod.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    return fig_price, fig_prod

def create_trend_visualization(df, provinces):
    filtered_df = df[df["provinsi"].isin(provinces)]
    
    # Visualisasi tren harga
    fig_price_trend = px.line(
        filtered_df,
        x="tahun",
        y="harga",
        color="provinsi",
        markers=True,
        title="Tren Harga Beras per Provinsi",
        labels={"tahun": "Tahun", "harga": "Harga (Rp/Kg)", "provinsi": "Provinsi"},
        color_discrete_sequence=COLORS
    )
    fig_price_trend.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    # Visualisasi tren produksi
    fig_prod_trend = px.line(
        filtered_df,
        x="tahun",
        y="produksi",
        color="provinsi",
        markers=True,
        title="Tren Produksi Beras per Provinsi",
        labels={"tahun": "Tahun", "produksi": "Produksi (Ton)", "provinsi": "Provinsi"},
        color_discrete_sequence=COLORS
    )
    fig_prod_trend.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    return fig_price_trend, fig_prod_trend

def create_correlation_visualization(df, year):
    year_df = df[df["tahun"] == year]
    
    # Hitung korelasi
    correlation = year_df["harga"].corr(year_df["produksi"])
    
    # Visualisasi korelasi (scatter plot)
    fig_scatter = px.scatter(
        year_df,
        x="produksi",
        y="harga",
        color="provinsi",
        hover_name="provinsi",
        size=[30] * len(year_df),
        color_discrete_sequence=COLORS,
        title=f"Korelasi Harga vs Produksi Beras ({year})",
        labels={"produksi": "Produksi (Ton)", "harga": "Harga (Rp/Kg)", "provinsi": "Provinsi"},
        trendline="ols" if len(year_df) > 2 else None
    )
    
    fig_scatter.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        annotations=[
            dict(
                x=0.5,
                y=1.05,
                showarrow=False,
                text=f"Korelasi: {correlation:.2f}",
                xref="paper",
                yref="paper",
                font=dict(size=14, color=PRIMARY_COLOR)
            )
        ]
    )
    
    return fig_scatter, correlation

def create_heatmap(df):
    # Pivot table untuk heatmap (provinsi vs tahun)
    heatmap_data_price = df.pivot_table(index="provinsi", columns="tahun", values="harga", aggfunc="mean")
    heatmap_data_prod = df.pivot_table(index="provinsi", columns="tahun", values="produksi", aggfunc="mean")
    
    # Visualisasi heatmap harga
    fig_heatmap_price = px.imshow(
        heatmap_data_price,
        text_auto=True,
        color_continuous_scale="Viridis",
        title="Heatmap Harga Beras per Provinsi dan Tahun",
        labels=dict(x="Tahun", y="Provinsi", color="Harga (Rp/Kg)")
    )
    fig_heatmap_price.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    # Visualisasi heatmap produksi
    fig_heatmap_prod = px.imshow(
        heatmap_data_prod,
        text_auto='.2s',
        color_continuous_scale="Viridis",
        title="Heatmap Produksi Beras per Provinsi dan Tahun",
        labels=dict(x="Tahun", y="Provinsi", color="Produksi (Ton)")
    )
    fig_heatmap_prod.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    return fig_heatmap_price, fig_heatmap_prod

def create_clustering_visualization(clustered_df):
    if clustered_df is None:
        return None
    
    # Visualisasi cluster
    fig_cluster = px.scatter(
        clustered_df,
        x="pca1",
        y="pca2",
        color="cluster",
        hover_name="provinsi",
        text="provinsi",
        title="Clustering Provinsi berdasarkan Harga dan Produksi",
        color_continuous_scale=px.colors.qualitative.G10
    )
    
    # Tambahkan label teks
    fig_cluster.update_traces(
        textposition='top center',
        marker=dict(size=15, opacity=0.8)
    )
    
    fig_cluster.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    return fig_cluster

def create_bubble_chart(df, year):
    year_df = df[df["tahun"] == year]
    
    # Visualisasi bubble chart
    fig_bubble = px.scatter(
        year_df,
        x="produksi",
        y="harga",
        size="produksi",
        color="provinsi",
        hover_name="provinsi",
        text="provinsi",
        title=f"Bubble Chart Produksi vs Harga Beras ({year})",
        labels={"produksi": "Produksi (Ton)", "harga": "Harga (Rp/Kg)", "provinsi": "Provinsi"},
        size_max=50,
        color_discrete_sequence=COLORS
    )
    
    fig_bubble.update_traces(
        textposition='top center',
        marker=dict(opacity=0.7, line=dict(width=1, color='black'))
    )
    
    fig_bubble.update_layout(
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
    )
    
    return fig_bubble

def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.2f} Juta"
    elif num >= 1000:
        return f"{num/1000:.2f} Ribu"
    else:
        return f"{num:.2f}"

# UI utama aplikasi
def main():
    # Judul aplikasi
    st.markdown("<h1 style='text-align: center; color: #0c6e42;'>📊 Dashboard Analisis Beras Indonesia</h1>", unsafe_allow_html=True)
    
    # Sidebar dengan kontrol utama
    with st.sidebar:
        st.markdown("""
        <div style="background-color: #e0f7fa; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: #0c6e42; text-align: center;">Data dan Kontrol</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Unggah file CSV
        st.markdown("### 📂 Unggah Data")
        uploaded_file = st.file_uploader("Unggah file CSV dengan kolom: nomor, provinsi, tahun, produksi, harga", type="csv")
        
        use_sample_data = st.checkbox("Gunakan data contoh", value=True)
        
        # Pemisah visual
        st.markdown("---")

    # Inisialisasi atau muat data
    if uploaded_file is not None:
        data_df = load_data(uploaded_file)
        if data_df is None:
            st.error("Format file tidak valid. Gunakan data contoh atau unggah file dengan format yang benar.")
            if use_sample_data:
                data_df = create_default_data()
            else:
                st.stop()
    elif use_sample_data:
        data_df = create_default_data()
    else:
        st.warning("Silakan unggah file CSV atau gunakan data contoh untuk melanjutkan.")
        st.stop()
    
    # Tampilkan statistik cepat
    col1, col2, col3, col4 = st.columns(4)
    
    total_production = data_df["produksi"].sum()
    avg_price = data_df["harga"].mean()
    num_provinces = data_df["provinsi"].nunique()
    years_range = f"{data_df['tahun'].min()} - {data_df['tahun'].max()}"
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0c6e42;">Total Produksi</h3>
            <h2>{format_number(total_production)} Ton</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0c6e42;">Rata-rata Harga</h3>
            <h2>Rp {format_number(avg_price)}/Kg</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0c6e42;">Jumlah Provinsi</h3>
            <h2>{num_provinces} Provinsi</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #0c6e42;">Rentang Tahun</h3>
            <h2>{years_range}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tampilkan data mentah
    with st.expander("📋 Lihat Data Mentah"):
        st.dataframe(data_df, use_container_width=True)
        
        # Opsi download data
        csv = data_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data CSV",
            data=csv,
            file_name=f"data_beras_indonesia_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Tabs untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Perbandingan Daerah", 
        "📈 Tren Waktu", 
        "🔄 Korelasi", 
        "🌡️ Heatmap & Pola",
        "📚 Analisis Lanjutan"
    ])
    
    # Tab 1: Perbandingan Antar Daerah
    with tab1:
        st.markdown("<h2 style='color: #0c6e42;'>Perbandingan Antar Provinsi</h2>", unsafe_allow_html=True)
        
        # Filter tahun
        selected_year = st.selectbox(
            "Pilih Tahun untuk Perbandingan:", 
            sorted(data_df["tahun"].unique(), reverse=True)
        )
        
        # Visualisasi perbandingan provinsi
        fig_price, fig_prod = create_provincial_comparison(data_df, selected_year)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_prod, use_container_width=True)
        
        # Bubble chart
        st.markdown("<h3 style='color: #0c6e42;'>Bubble Chart - Produksi vs Harga</h3>", unsafe_allow_html=True)
        fig_bubble = create_bubble_chart(data_df, selected_year)
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Insight
        st.markdown("""
        <div class="info-box">
            <h4 style="color: #0c6e42;">💡 Insight Perbandingan Provinsi</h4>
            <ul>
                <li>Provinsi dengan produksi tertinggi bisa dibandingkan dengan tingkat harganya.</li>
                <li>Bubble chart menunjukkan hubungan antara produksi (ukuran bubble) dan harga beras.</li>
                <li>Perhatikan provinsi dengan produksi tinggi tapi harga rendah yang bisa menjadi contoh efisiensi.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Tab 2: Tren Waktu
    with tab2:
        st.markdown("<h2 style='color: #0c6e42;'>Tren Harga dan Produksi Beras dari Waktu ke Waktu</h2>", unsafe_allow_html=True)
        
        # Filter provinsi
        all_provinces = sorted(data_df["provinsi"].unique())
        
        # Pilih provinsi
        selected_provinces = st.multiselect(
            "Pilih Provinsi untuk Analisis Tren:",
            options=all_provinces,
            default=all_provinces[:3] if len(all_provinces) >= 3 else all_provinces
        )
        
        if selected_provinces:
            # Visualisasi tren
            fig_price_trend, fig_prod_trend = create_trend_visualization(data_df, selected_provinces)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_price_trend, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_prod_trend, use_container_width=True)
            
            # Prediksi sederhana
            st.markdown("<h3 style='color: #0c6e42;'>Prediksi Sederhana Tahun Depan</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Pilih provinsi untuk prediksi
                province_for_prediction = st.selectbox(
                    "Pilih Provinsi untuk Prediksi:",
                    options=selected_provinces
                )
            
            # Buat prediksi
            next_year_price, price_prediction = make_simple_prediction(data_df, province_for_prediction, "harga")
            next_year_prod, prod_prediction = make_simple_prediction(data_df, province_for_prediction, "produksi")
            
            if next_year_price and price_prediction and next_year_prod and prod_prediction:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #0c6e42;">
                        <h4 style="color: #0c6e42;">Prediksi Harga {next_year_price}</h4>
                        <h2>Rp {price_prediction:.2f}/Kg</h2>
                        <p class="annotation">Berdasarkan tren historis</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #0c6e42;">
                        <h4 style="color: #0c6e42;">Prediksi Produksi {next_year_prod}</h4>
                        <h2>{format_number(prod_prediction)} Ton</h2>
                        <p class="annotation">Berdasarkan tren historis</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <h4 style="color: #0c6e42;">⚠️ Catatan Prediksi</h4>
                    <p>Prediksi ini menggunakan model regresi linear sederhana berdasarkan data historis. 
                    Akurasi prediksi tergantung pada jumlah dan kualitas data historis serta 
                    asumsi bahwa tren akan berlanjut secara linear.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Data tidak cukup untuk membuat prediksi yang akurat. Minimal diperlukan data 2 tahun.")
        else:
            st.info("Silakan pilih minimal satu provinsi untuk menampilkan visualisasi tren.")
    
    # Tab 3: Korelasi
    with tab3:
        st.markdown("<h2 style='color: #0c6e42;'>Korelasi antara Harga dan Produksi Beras</h2>", unsafe_allow_html=True)
        
        # Pilih tahun untuk analisis korelasi
        selected_year_corr = st.selectbox(
            "Pilih Tahun untuk Analisis Korelasi:", 
            sorted(data_df["tahun"].unique(), reverse=True),
            key="corr_year"
        )
        
        # Visualisasi korelasi
        fig_scatter, correlation = create_correlation_visualization(data_df, selected_year_corr)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Interpretasi korelasi
        st.markdown("<h3 style='color: #0c6e42;'>Interpretasi Korelasi</h3>", unsafe_allow_html=True)
        
        interpretation = ""
        if correlation > 0.7:
            interpretation = """
            <div class="info-box" style="border-left: 5px solid #0c6e42;">
                <strong>Korelasi positif kuat:</strong> Saat produksi meningkat, harga cenderung naik signifikan. Ini mungkin mengindikasikan adanya faktor-faktor lain yang mempengaruhi, seperti kualitas beras yang lebih tinggi di daerah dengan produksi besar, atau tingginya permintaanyang melampaui pasokan.
            </div>
            """
        elif correlation > 0.3:
            interpretation = """
            <div class="info-box" style="border-left: 5px solid #1976D2;">
                <strong>Korelasi positif moderat:</strong> Ada hubungan positif antara produksi dan harga, namun tidak terlalu kuat. Faktor-faktor lain juga berperan dalam penentuan harga.
            </div>
            """
        elif correlation > -0.3:
            interpretation = """
            <div class="info-box" style="border-left: 5px solid #FFA726;">
                <strong>Korelasi lemah atau tidak ada:</strong> Produksi dan harga cenderung tidak saling mempengaruhi secara signifikan. Ini mengindikasikan bahwa faktor-faktor lain seperti kebijakan pemerintah, impor, atau distribusi mungkin lebih berpengaruh terhadap harga.
            </div>
            """
        elif correlation > -0.7:
            interpretation = """
            <div class="info-box" style="border-left: 5px solid #EF5350;">
                <strong>Korelasi negatif moderat:</strong> Saat produksi meningkat, harga cenderung menurun. Ini sesuai dengan teori ekonomi dasar tentang penawaran dan permintaan.
            </div>
            """
        else:
            interpretation = """
            <div class="info-box" style="border-left: 5px solid #D32F2F;">
                <strong>Korelasi negatif kuat:</strong> Saat produksi meningkat, harga menurun secara signifikan. Ini menunjukkan pasar yang efisien dimana peningkatan pasokan langsung mempengaruhi penurunan harga.
            </div>
            """
        
        st.markdown(interpretation, unsafe_allow_html=True)
        
        # Statistik deskriptif
        with st.expander("Statistik Deskriptif"):
            stats = statistical_analysis(data_df)
            st.write("### Statistik Deskriptif:")
            st.dataframe(stats["describe"], use_container_width=True)
            
            st.write("### Korelasi Keseluruhan:")
            st.dataframe(stats["correlation"], use_container_width=True)
    
    # Tab 4: Heatmap & Pola
    with tab4:
        st.markdown("<h2 style='color: #0c6e42;'>Heatmap dan Pola Data</h2>", unsafe_allow_html=True)
        
        # Visualisasi heatmap
        fig_heatmap_price, fig_heatmap_prod = create_heatmap(data_df)
        
        st.markdown("<h3 style='color: #0c6e42;'>Heatmap Harga Beras</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_heatmap_price, use_container_width=True)
        
        st.markdown("<h3 style='color: #0c6e42;'>Heatmap Produksi Beras</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig_heatmap_prod, use_container_width=True)
        
        # Clustering
        st.markdown("<h3 style='color: #0c6e42;'>Clustering Provinsi</h3>", unsafe_allow_html=True)
        
        # Pilih tahun untuk clustering
        selected_year_cluster = st.selectbox(
            "Pilih Tahun untuk Clustering:", 
            sorted(data_df["tahun"].unique(), reverse=True),
            key="cluster_year"
        )
        
        # Visualisasi clustering
        clustered_df, _ = perform_kmeans_clustering(data_df, selected_year_cluster)
        
        if clustered_df is not None:
            fig_cluster = create_clustering_visualization(clustered_df)
            if fig_cluster:
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                    <h4 style="color: #0c6e42;">💡 Interpretasi Cluster</h4>
                    <p>Clustering membantu mengidentifikasi provinsi-provinsi dengan karakteristik serupa berdasarkan produksi dan harga beras. 
                    Provinsi-provinsi dalam cluster yang sama memiliki pola produksi dan harga yang mirip.</p>
                    <p>Ini bisa membantu dalam:</p>
                    <ul>
                        <li>Identifikasi kelompok provinsi untuk kebijakan harga yang serupa</li>
                        <li>Memahami pola distribusi produksi dan harga secara geografis</li>
                        <li>Mendeteksi provinsi dengan karakteristik unik yang memerlukan perhatian khusus</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Tidak dapat membuat visualisasi cluster dengan data yang tersedia.")
        else:
            st.warning("Data tidak cukup untuk melakukan clustering. Minimal diperlukan 3 provinsi dengan data lengkap untuk tahun yang dipilih.")
    
    # Tab 5: Analisis Lanjutan
    with tab5:
        st.markdown("<h2 style='color: #0c6e42;'>Analisis Lanjutan</h2>", unsafe_allow_html=True)
        
        # Pilih provinsi untuk analisis lanjutan
        province_for_analysis = st.selectbox(
            "Pilih Provinsi untuk Analisis Lanjutan:",
            options=sorted(data_df["provinsi"].unique())
        )
        
        # Filter data untuk provinsi yang dipilih
        province_data = data_df[data_df["provinsi"] == province_for_analysis].sort_values("tahun")
        
        if len(province_data) >= 2:
            # Analisis tren dengan regresi
            X = sm.add_constant(province_data["tahun"])
            y_price = province_data["harga"]
            y_prod = province_data["produksi"]
            
            model_price = sm.OLS(y_price, X).fit()
            model_prod = sm.OLS(y_prod, X).fit()
            
            # Create a range of years for prediction
            min_year = province_data["tahun"].min()
            max_year = province_data["tahun"].max()
            year_range = range(min_year, max_year + 2)
            X_pred = sm.add_constant(pd.Series(year_range))
            
            # Prediksi
            price_pred = model_price.predict(X_pred)
            prod_pred = model_prod.predict(X_pred)
            
            # Dataframe untuk visualisasi
            pred_df = pd.DataFrame({
                "tahun": year_range,
                "harga_pred": price_pred,
                "produksi_pred": prod_pred
            })
            
            # Visualisasi tren dan prediksi
            st.markdown("<h3 style='color: #0c6e42;'>Tren dan Prediksi</h3>", unsafe_allow_html=True)
            
            # Plot untuk harga
            fig_price_pred = go.Figure()
            
            # Data aktual
            fig_price_pred.add_trace(go.Scatter(
                x=province_data["tahun"],
                y=province_data["harga"],
                mode='markers+lines',
                name='Data Aktual',
                line=dict(color=PRIMARY_COLOR, width=2),
                marker=dict(size=10)
            ))
            
            # Garis prediksi
            fig_price_pred.add_trace(go.Scatter(
                x=pred_df["tahun"],
                y=pred_df["harga_pred"],
                mode='lines',
                name='Tren & Prediksi',
                line=dict(color='rgba(12, 110, 66, 0.5)', width=2, dash='dash')
            ))
            
            # Highlight prediksi tahun depan
            next_year = max_year + 1
            next_year_price = pred_df[pred_df["tahun"] == next_year]["harga_pred"].values[0]
            
            fig_price_pred.add_trace(go.Scatter(
                x=[next_year],
                y=[next_year_price],
                mode='markers',
                name=f'Prediksi {next_year}',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig_price_pred.update_layout(
                title=f"Analisis Trend Harga Beras di {province_for_analysis}",
                xaxis_title="Tahun",
                yaxis_title="Harga (Rp/Kg)",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=BG_COLOR,
                hovermode="x unified"
            )
            
            # Plot untuk produksi
            fig_prod_pred = go.Figure()
            
            # Data aktual
            fig_prod_pred.add_trace(go.Scatter(
                x=province_data["tahun"],
                y=province_data["produksi"],
                mode='markers+lines',
                name='Data Aktual',
                line=dict(color=PRIMARY_COLOR, width=2),
                marker=dict(size=10)
            ))
            
            # Garis prediksi
            fig_prod_pred.add_trace(go.Scatter(
                x=pred_df["tahun"],
                y=pred_df["produksi_pred"],
                mode='lines',
                name='Tren & Prediksi',
                line=dict(color='rgba(12, 110, 66, 0.5)', width=2, dash='dash')
            ))
            
            # Highlight prediksi tahun depan
            next_year_prod = pred_df[pred_df["tahun"] == next_year]["produksi_pred"].values[0]
            
            fig_prod_pred.add_trace(go.Scatter(
                x=[next_year],
                y=[next_year_prod],
                mode='markers',
                name=f'Prediksi {next_year}',
                marker=dict(color='red', size=12, symbol='star')
            ))
            
            fig_prod_pred.update_layout(
                title=f"Analisis Trend Produksi Beras di {province_for_analysis}",
                xaxis_title="Tahun",
                yaxis_title="Produksi (Ton)",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=BG_COLOR,
                hovermode="x unified"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_price_pred, use_container_width=True)
            
            with col2:
                st.plotly_chart(fig_prod_pred, use_container_width=True)
            
            # Tampilkan informasi model regresi
            with st.expander("Detail Model Regresi"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"#### Model Regresi Harga")
                    st.write(f"R-squared: {model_price.rsquared:.4f}")
                    st.write(f"P-value: {model_price.f_pvalue:.4f}")
                    st.write(f"Formula: Harga = {model_price.params[0]:.2f} + {model_price.params[1]:.2f} × Tahun")
                
                with col2:
                    st.markdown(f"#### Model Regresi Produksi")
                    st.write(f"R-squared: {model_prod.rsquared:.4f}")
                    st.write(f"P-value: {model_prod.f_pvalue:.4f}")
                    st.write(f"Formula: Produksi = {model_prod.params[0]:.2f} + {model_prod.params[1]:.2f} × Tahun")
            
            # Analisis rasio harga-produksi
            st.markdown("<h3 style='color: #0c6e42;'>Analisis Indeks Efisiensi Pasar</h3>", unsafe_allow_html=True)
            
            # Menghitung rasio harga per satuan produksi
            province_data["price_per_prod"] = province_data["harga"] / province_data["produksi"] * 1000000  # Konversi ke harga per juta ton
            
            # Visualisasi rasio
            fig_ratio = go.Figure()
            
            fig_ratio.add_trace(go.Bar(
                x=province_data["tahun"],
                y=province_data["price_per_prod"],
                marker_color=PRIMARY_COLOR,
                name="Indeks Efisiensi"
            ))
            
            fig_ratio.update_layout(
                title=f"Indeks Efisiensi Pasar Beras di {province_for_analysis}",
                xaxis_title="Tahun",
                yaxis_title="Harga (Rp) per Juta Ton Produksi",
                plot_bgcolor=BG_COLOR,
                paper_bgcolor=BG_COLOR,
            )
            
            st.plotly_chart(fig_ratio, use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
                <h4 style="color: #0c6e42;">💡 Interpretasi Indeks Efisiensi Pasar</h4>
                <p>Indeks ini menunjukkan harga relatif terhadap volume produksi. Semakin rendah nilai indeks, semakin efisien
                pasar beras di provinsi tersebut. Turunnya nilai indeks dari waktu ke waktu menandakan perbaikan efisiensi pasar,
                sementara nilai yang meningkat bisa mengindikasikan masalah dalam rantai pasok atau kebijakan harga.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Rekomendasi kebijakan
            st.markdown("<h3 style='color: #0c6e42;'>Rekomendasi Kebijakan</h3>", unsafe_allow_html=True)
            
            # Menentukan tren produksi dan harga
            prod_trend = "naik" if model_prod.params[1] > 0 else "turun"
            price_trend = "naik" if model_price.params[1] > 0 else "turun"
            
            # Logic untuk rekomendasi
            if prod_trend == "turun" and price_trend == "naik":
                recommendation = """
                <div class="info-box" style="border-left: 5px solid #D32F2F;">
                    <h4 style="color: #D32F2F;">⚠️ Peringatan: Tren Negatif</h4>
                    <p>Produksi beras cenderung <strong>menurun</strong> sedangkan harga <strong>meningkat</strong>. 
                    Ini mengindikasikan masalah dalam produktivitas pertanian dan pasokan yang tidak mencukupi permintaan.</p>
                    <p><strong>Rekomendasi:</strong></p>
                    <ul>
                        <li>Implementasi program intensifikasi pertanian untuk meningkatkan produktivitas</li>
                        <li>Evaluasi dan tingkatkan sistem irigasi</li>
                        <li>Berikan subsidi pupuk dan bibit unggul kepada petani</li>
                        <li>Tingkatkan akses petani terhadap teknologi pertanian modern</li>
                        <li>Pertimbangkan untuk mengontrol harga beras jika terjadi lonjakan</li>
                    </ul>
                </div>
                """
            elif prod_trend == "naik" and price_trend == "turun":
                recommendation = """
                <div class="info-box" style="border-left: 5px solid #2E7D32;">
                    <h4 style="color: #2E7D32;">✅ Tren Positif</h4>
                    <p>Produksi beras cenderung <strong>meningkat</strong> sedangkan harga <strong>menurun</strong>. 
                    Ini menunjukkan perbaikan produktivitas dan efisiensi pasar yang baik.</p>
                    <p><strong>Rekomendasi:</strong></p>
                    <ul>
                        <li>Pertahankan kebijakan yang sudah berjalan dengan baik</li>
                        <li>Pastikan harga beras tidak turun terlalu drastis yang dapat merugikan petani</li>
                        <li>Pertimbangkan untuk menetapkan harga pembelian pemerintah yang menguntungkan bagi petani</li>
                        <li>Kembangkan jaringan distribusi dan penyimpanan untuk menstabilkan harga</li>
                        <li>Tingkatkan ekspor beras jika produksi berlebih</li>
                    </ul>
                </div>
                """
            elif prod_trend == "naik" and price_trend == "naik":
                recommendation = """
                <div class="info-box" style="border-left: 5px solid #FFA726;">
                    <h4 style="color: #FFA726;">⚠️ Tren Campuran</h4>
                    <p>Produksi beras cenderung <strong>meningkat</strong> namun harga juga <strong>meningkat</strong>. 
                    Ini mungkin mengindikasikan peningkatan permintaan yang lebih cepat dibandingkan peningkatan produksi,
                    atau adanya faktor eksternal seperti inflasi.</p>
                    <p><strong>Rekomendasi:</strong></p>
                    <ul>
                        <li>Analisis faktor-faktor yang menyebabkan kenaikan harga meskipun produksi meningkat</li>
                        <li>Evaluasi rantai distribusi dan pasok untuk mengurangi biaya perantara</li>
                        <li>Tingkatkan efisiensi transportasi dan logistik beras</li>
                        <li>Pertimbangkan intervensi pasar jika diperlukan untuk menstabilkan harga</li>
                        <li>Kembangkan sistem informasi pasar beras yang lebih transparan</li>
                    </ul>
                </div>
                """
            else:  # prod_trend == "turun" and price_trend == "turun"
                recommendation = """
                <div class="info-box" style="border-left: 5px solid #1976D2;">
                    <h4 style="color: #1976D2;">🔍 Tren yang Perlu Perhatian</h4>
                    <p>Produksi beras cenderung <strong>menurun</strong> dan harga juga <strong>menurun</strong>. 
                    Ini mungkin mengindikasikan penurunan kualitas beras, berkurangnya permintaan, atau peningkatan impor beras.</p>
                    <p><strong>Rekomendasi:</strong></p>
                    <ul>
                        <li>Evaluasi penyebab penurunan produksi (apakah alih fungsi lahan, migrasi petani, dsb)</li>
                        <li>Tingkatkan kualitas beras untuk mempertahankan nilai jual</li>
                        <li>Kembangkan varietas beras premium dengan nilai tambah lebih tinggi</li>
                        <li>Diversifikasi produk turunan beras untuk meningkatkan nilai ekonomi</li>
                        <li>Evaluasi kebijakan impor beras dan dampaknya terhadap petani lokal</li>
                    </ul>
                </div>
                """
            
            st.markdown(recommendation, unsafe_allow_html=True)
            
        else:
            st.warning(f"Data tidak cukup untuk analisis lanjutan pada provinsi {province_for_analysis}. Minimal diperlukan data 2 tahun.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="color: #0c6e42;">Dashboard Analisis Beras Indonesia | Data disajikan untuk tujuan visualisasi dan analisis.</p>
            <p style="color: #0c6e42;"> Walaupun masih ada mistake dalam data maupun kodingan kami minta maaf karena kami adalah ultramen</p>
        <p>Dibuat dengan Streamlit dan Python by Kelompok 2</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()