# Pembahasan 
# Judul
Dashboard Analisis Beras Indonesia
# Fitur 
* Dashboard
* Tren Waktu
* Korelasi
* Heatmap dan pola
* Analisis Lanjutan

# Metode
* Tren Waktu = time series analysis sederhana
* Korelasi = Korelasi Pearson dan OLS regression.



# Penjelasan fungsi create_default_data()
Fungsi ini membersihkan data mentah dari file CSV dan memastikannya dalam format yang siap digunakan untuk analisis atau visualisasi, terutama dalam aplikasi berbasis data seperti Streamlit.

Fungsi: create_default_data()
1. data = "Data.csv"
Menentukan nama file sumber data, yaitu file bernama Data.csv.
2. data = pd.read_csv(data)
Menggunakan pandas (pd) untuk membaca file CSV ke dalam bentuk DataFrame.
3. data = data.to_dict(orient="records")
Mengubah DataFrame menjadi list of dictionaries (setiap baris jadi dictionary), agar lebih fleksibel untuk diproses di Streamlit atau aplikasi lain.
4. Perulangan for d in data
Iterasi setiap entri (baris) dalam data dan mengubah tipe data pada kolom:
tahun, produksi, harga, nomor diubah menjadi integer.
Ini memastikan data numerik benar-benar bertipe numerik untuk analisis lebih lanjut (misalnya visualisasi atau perhitungan statistik).

# Fungsi load_data(file)
Fungsi load_data(file) digunakan dalam aplikasi Streamlit untuk membaca, memverifikasi, dan membersihkan data CSV yang diunggah.
1. Dekorator @st.cache_data: Menyimpan hasil fungsi agar tidak perlu dimuat ulang tiap kali aplikasi dijalankan ulang (efisiensi).
2. Membaca CSV: Menggunakan pandas.read_csv untuk memuat data ke dalam DataFrame df.
3. Validasi kolom: Memastikan kolom penting seperti provinsi, tahun, produksi, harga, dan nomor tersedia. Jika tidak, ditampilkan pesan kesalahan.
4. Konversi tipe data: Mengubah kolom numerik (tahun, produksi, harga, nomor) ke format angka dengan pd.to_numeric() jika belum bertipe int64 atau float64, agar bisa dianalisis dengan benar.
5. Penanganan error: Jika terjadi kesalahan saat membaca atau memproses file, akan ditampilkan pesan kesalahan menggunakan st.error().

Fungsi ini penting untuk memastikan data bersih dan sesuai sebelum dilakukan analisis atau visualisasi di aplikasi.

# Fungsi statistical_analysis(df)
 melakukan analisis statistik deskriptif dan eksploratif terhadap DataFrame df
 1. stats = {}
Inisialisasi dictionary kosong untuk menyimpan hasil-hasil analisis.
2. stats["describe"] = df[["produksi", "harga"]].describe()
Menghasilkan statistik deskriptif untuk kolom produksi dan harga, seperti:
Mean (rata-rata)
Std (standar deviasi)
Min, Max
Kuartil (25%, 50%, 75%)
3. stats["correlation"] = df[["produksi", "harga"]].corr()
Menghitung korelasi antara produksi dan harga menggunakan koefisien Pearson.
4. stats["avg_by_province"] = df.groupby("provinsi")[["produksi", "harga"]].mean()
Mengelompokkan data berdasarkan provinsi, lalu menghitung rata-rata produksi dan harga per provinsi.
5. stats["yearly_trend"] = df.groupby("tahun")[["produksi", "harga"]].mean()
Mengelompokkan data berdasarkan tahun, lalu menghitung rata-rata produksi dan harga per tahun, untuk melihat tren tahunan.
6. return stats
Mengembalikan seluruh hasil analisis sebagai dictionary.

Fungsi ini menyajikan statistik deskriptif, korelasi, rata-rata per provinsi, dan tren tahunan, yang berguna sebagai ringkasan awal sebelum analisis lebih lanjut atau visualisasi data dilakukan.

# Fungsi make_simple_prediction
memprediksi nilai masa depan (produksi atau harga) berdasarkan data historis suatu provinsi menggunakan regresi linier sederhana.

1. province_data = df[df["provinsi"] == province].sort_values("tahun")
Menyaring data hanya untuk provinsi tertentu dan mengurutkannya berdasarkan tahun.

2. if len(province_data) < 2:
Jika data kurang dari 2 baris (tidak cukup untuk regresi), kembalikan None.

3. X = sm.add_constant(province_data["tahun"])
Menambahkan konstanta (intersep) ke variabel independen tahun agar regresi linier dapat menghitung intercept dan slope.

4. y = province_data[target_col]
Menentukan variabel dependen (produksi atau harga), tergantung input target_col.

5. model = sm.OLS(y, X).fit()
Membuat dan melatih model regresi linier menggunakan metode OLS (Ordinary Least Squares).

6. Prediksi tahun berikutnya:
last_year = ...: Ambil tahun terakhir dalam data.
next_year = last_year + 1: Tentukan tahun selanjutnya.
prediction = model.predict([1, next_year])[0]: Prediksi nilai (produksi/harga) untuk next_year.

7. return next_year, prediction
Mengembalikan tahun prediksi dan hasil prediksinya.

# Fungsi correlation = year_df["harga"].corr(year_df["produksi"])
Untuk melihat hubungan antara harga dan produksi beras di berbagai provinsi pada tahun tertentu.
1. Menghitung Korelasi
correlation = year_df["harga"].corr(year_df["produksi"])
Mengukur kekuatan hubungan antara harga dan produksi beras di satu tahun.
Nilainya berkisar dari -1 (berbanding terbalik sempurna) hingga +1 (sebanding sempurna).

2. Membuat Grafik Sebar (Scatter Plot)
fig_scatter = px.scatter(
    year_df,
    x="produksi",
    y="harga",
    color="provinsi",
    hover_name="provinsi",
    size=[30] * len(year_df),
    color_discrete_sequence=COLORS,
    title=...,
    labels=...,
    trendline="ols" if len(year_df) > 2 else None
)
Membuat grafik hubungan antara produksi dan harga.
Titik diberi warna berdasarkan provinsi.
Menambahkan garis tren (OLS) jika data cukup.

3.  Menambahkan Korelasi ke Tampilan Grafik
fig_scatter.update_layout(
    plot_bgcolor=BG_COLOR,
    paper_bgcolor=BG_COLOR,
    annotations=[
        dict(
            x=0.5,
            y=1.05,
            showarrow=False,
            text=f"Korelasi: {correlation:.2f}",
            ...
        )
    ]
)
Mengatur warna latar belakang grafik.
Menampilkan nilai korelasi di atas grafik sebagai keterangan tambahan.

4. Mengembalikan Hasil
return fig_scatter, correlation
Mengembalikan:
Grafik scatter yang sudah jadi.
Angka korelasi sebagai nilai statistik.


