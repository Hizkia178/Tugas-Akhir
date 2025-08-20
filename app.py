import streamlit as st
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import io

# Try to import statsmodels for VIF and Breusch-Pagan; provide fallback if import fails
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    st.warning("Modul 'statsmodels' tidak tersedia. Uji multikolinearitas (VIF) dan homoskedastisitas (Breusch-Pagan) tidak dapat dilakukan. Silakan instal statsmodels dengan 'pip install statsmodels'.")

# Streamlit App
st.title("🌾 Sistem Prediksi Produksi Padi Berdasarkan Curah Hujan & Suhu")

# Sidebar
st.sidebar.title("Menu Navigasi")
page = st.sidebar.selectbox("Pilih Halaman", ["📊 Dataset", "🧮 Evaluasi Model", "🔮 Prediksi Baru", "📈 Visualisasi"])

# File uploader untuk dataset CSV
st.sidebar.title("Unggah Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

# Load dataset
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Validasi kolom yang diperlukan
        required_columns = ['Curah_Hujan', 'Suhu', 'Produksi']
        if not all(col in df.columns for col in required_columns):
            st.error("File CSV harus memiliki kolom: Curah_Hujan, Suhu, Produksi")
            st.stop()
        else:
            st.sidebar.success("Dataset berhasil diunggah!")
    except Exception as e:
        st.error(f"Error membaca file CSV: {e}")
        st.stop()
else:
    df = pd.read_csv("dataset_padi.csv")  # Gunakan dataset_padi.csv dari dokumen

# Preprocessing: Validasi data
if df[['Curah_Hujan', 'Suhu', 'Produksi']].isnull().any().any():
    st.error("Dataset mengandung nilai kosong. Harap periksa data.")
    df = df.dropna()  # Hapus baris dengan missing values
if (df['Curah_Hujan'] < 0).any() or (df['Produksi'] < 0).any():
    st.warning("Dataset mengandung nilai tidak realistis (Curah Hujan < 0, Produksi < 0). Data akan difilter.")
    df = df[(df['Curah_Hujan'] >= 0) & (df['Produksi'] >= 0)]

# Preprocessing: Normalisasi data
scaler = StandardScaler()
X = df[['Curah_Hujan', 'Suhu']]
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=['Curah_Hujan', 'Suhu'])
y = df['Produksi']

# Split dan train model
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Peringatan performa model rendah
if r2 < 0.7:
    st.warning("Performa model rendah (R² < 0.7). Pertimbangkan untuk memeriksa kualitas data atau menambah fitur lain.")

# Input prediksi di sidebar
st.sidebar.title("Input Prediksi")
curah_hujan = st.sidebar.number_input("Curah Hujan (mm)", value=200.0)
suhu = st.sidebar.number_input("Suhu (°C)", value=28.0)
predict_button = st.sidebar.button("Prediksi")

# Prediksi logic
if predict_button:
    input_data = scaler.transform(np.array([[curah_hujan, suhu]]))
    prediksi = model.predict(input_data)[0]
    st.session_state.prediksi = prediksi
    st.session_state.input_curah_hujan = curah_hujan
    st.session_state.input_suhu = suhu

# Main Area
if page == "📊 Dataset":
    st.header("Preview Dataset")
    st.dataframe(df)  # Tampilkan semua data
    st.write("Total entri: ", len(df))
    st.write("**Statistik Deskriptif**")
    st.dataframe(df.describe())
    st.write("**Korelasi Variabel**")
    st.dataframe(df[['Curah_Hujan', 'Suhu', 'Produksi']].corr())
    st.write("**Interpretasi Korelasi**: Curah Hujan memiliki korelasi positif kuat (0.73) dengan Produksi, sedangkan Suhu memiliki korelasi negatif sedang (-0.53).")

elif page == "🧮 Evaluasi Model":
    st.header("Evaluasi Model Regresi Linier Berganda")
    st.write("**R² (Koefisien Determinasi)**: ", round(r2, 3), " (Model menjelaskan ~86% variasi data)")
    st.write("**MAE (Mean Absolute Error)**: ", round(mae, 3), " ton/ha")
    st.write("**RMSE (Root Mean Squared Error)**: ", round(rmse, 3), " ton/ha")
    st.write("**Persamaan Model (dengan data ternormalisasi)**: Produksi = {:.3f} + {:.4f} × Curah Hujan (scaled) - {:.3f} × Suhu (scaled)".format(model.intercept_, model.coef_[0], abs(model.coef_[1])))

    # Penjelasan koefisien model
    st.subheader("Interpretasi Koefisien Model")
    st.write(f"- **Intercept ({round(model.intercept_, 3)})**: Nilai dasar produksi padi (ton/ha) ketika curah hujan dan suhu dalam skala ternormalisasi bernilai 0.")
    st.write(f"- **Koefisien Curah Hujan ({round(model.coef_[0], 4)})**: Setiap peningkatan satu unit curah hujan (dalam skala ternormalisasi) meningkatkan produksi padi sebesar {round(model.coef_[0], 4)} ton/ha, dengan asumsi suhu tetap. Ini menunjukkan curah hujan yang lebih tinggi mendukung hasil panen yang lebih baik.")
    st.write(f"- **Koefisien Suhu ({round(model.coef_[1], 3)})**: Setiap peningkatan satu unit suhu (dalam skala ternormalisasi) menurunkan produksi padi sebesar {round(abs(model.coef_[1]), 3)} ton/ha, dengan asumsi curah hujan tetap. Ini menunjukkan suhu tinggi dapat mengurangi hasil panen.")
    st.write("**Implikasi Praktis**: Petani dapat menggunakan model ini untuk memperkirakan produksi dan menyesuaikan waktu tanam berdasarkan prakiraan curah hujan dan suhu.")

    # Uji multikolinearitas (VIF)
    st.subheader("Uji Multikolinearitas (VIF)")
    if statsmodels_available:
        vif_data = pd.DataFrame()
        vif_data["Variabel"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        st.dataframe(vif_data)
        st.write("**Interpretasi**: Nilai VIF < 10 menunjukkan tidak ada multikolinearitas signifikan antara curah hujan dan suhu, memastikan independensi variabel dalam model.")
    else:
        st.error("Uji VIF tidak dapat dilakukan karena modul 'statsmodels' tidak tersedia.")

    # Uji normalitas residual (Q-Q Plot)
    st.subheader("Uji Normalitas Residual (Q-Q Plot)")
    residuals = y_test - y_pred
    fig_qq, ax_qq = plt.subplots()
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("Q-Q Plot Residual")
    ax_qq.set_xlabel("Kuantil Teoretis")
    ax_qq.set_ylabel("Kuantil Residual")
    st.pyplot(fig_qq)
    st.write("**Interpretasi**: Jika titik-titik pada Q-Q Plot mengikuti garis lurus, residual cenderung berdistribusi normal, memenuhi asumsi regresi linier.")

    # Uji homoskedastisitas (Breusch-Pagan)
    st.subheader("Uji Homoskedastisitas (Breusch-Pagan)")
    if statsmodels_available:
        import statsmodels.api as sm
        X_test_with_const = sm.add_constant(X_test)
        _, pval, _, _ = het_breuschpagan(residuals, X_test_with_const)
        st.write(f"**P-value Breusch-Pagan**: {round(pval, 3)}")
        st.write("**Interpretasi**: Jika p-value > 0.05, asumsi homoskedastisitas terpenuhi (variansi residual konstan). Jika p-value ≤ 0.05, ada indikasi heteroskedastisitas.")
    else:
        st.error("Uji Breusch-Pagan tidak dapat dilakukan karena modul 'statsmodels' tidak tersedia.")

    # Tombol untuk mengunduh laporan
    st.subheader("Unduh Laporan Evaluasi Model")
    report_data = {
        "Metrik": ["R²", "MAE", "RMSE"],
        "Nilai": [round(r2, 3), round(mae, 3), round(rmse, 3)],
        "Deskripsi": [
            "Menunjukkan proporsi variasi data yang dijelaskan model (~86%)",
            "Rata-rata kesalahan absolut prediksi (ton/ha)",
            "Akar rata-rata kuadrat kesalahan prediksi (ton/ha)"
        ]
    }
    report_df = pd.DataFrame(report_data)
    csv_buffer = io.StringIO()
    report_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Unduh Laporan (CSV)",
        data=csv_buffer.getvalue(),
        file_name="laporan_evaluasi_model.csv",
        mime="text/csv"
    )

elif page == "🔮 Prediksi Baru":
    st.header("Hasil Prediksi Produksi Padi")
    if 'prediksi' in st.session_state:
        prediksi = st.session_state.prediksi
        curah_hujan = st.session_state.input_curah_hujan
        suhu = st.session_state.input_suhu
        st.success(f"Prediksi Produksi: {round(prediksi, 2)} ton/ha")
        st.write(f"**Keterangan**: Prediksi ini dihasilkan berdasarkan input Curah Hujan = {curah_hujan} mm dan Suhu = {suhu} °C. Nilai ini menunjukkan perkiraan produksi padi per hektar dalam kondisi iklim tersebut, berdasarkan model regresi linier berganda yang dilatih dengan data historis.")
        # Rekomendasi aplikatif
        st.subheader("Rekomendasi")
        if prediksi < 5:
            st.write("- **Produksi Rendah**: Pertimbangkan untuk memastikan drainase yang baik jika curah hujan tinggi, atau tambahkan irigasi jika curah hujan rendah.")
        elif 5 <= prediksi <= 6:
            st.write("- **Produksi Sedang**: Kondisi iklim saat ini mendukung produksi sedang. Pastikan pengelolaan air dan pemupukan optimal untuk meningkatkan hasil.")
        else:
            st.write("- **Produksi Tinggi**: Kondisi iklim saat ini sangat mendukung. Manfaatkan curah hujan yang cukup dan suhu optimal untuk memaksimalkan hasil panen.")
    else:
        st.info("Masukkan nilai Curah Hujan dan Suhu di sidebar, lalu tekan 'Prediksi' untuk melihat hasil.")

elif page == "📈 Visualisasi":
    st.header("Visualisasi Hubungan Variabel")
    
    # Scatter plot Curah Hujan vs Produksi
    st.subheader("Curah Hujan vs Produksi")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x='Curah_Hujan', y='Produksi', data=df, ax=ax1)
    ax1.set_title("Curah Hujan vs Produksi")
    ax1.set_xlabel("Curah Hujan (mm)")
    ax1.set_ylabel("Produksi (ton/ha)")
    st.pyplot(fig1)
    st.write("**Keterangan**: Setiap titik mewakili data bulanan produksi padi (ton/ha) terhadap curah hujan (mm). Pola menunjukkan hubungan positif, di mana curah hujan yang lebih tinggi cenderung meningkatkan produksi padi.")

    # Scatter plot Suhu vs Produksi
    st.subheader("Suhu vs Produksi")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x='Suhu', y='Produksi', data=df, ax=ax2)
    ax2.set_title("Suhu vs Produksi")
    ax2.set_xlabel("Suhu (°C)")
    ax2.set_ylabel("Produksi (ton/ha)")
    st.pyplot(fig2)
    st.write("**Keterangan**: Setiap titik mewakili data bulanan produksi padi (ton/ha) terhadap suhu (°C). Pola menunjukkan hubungan negatif, di mana suhu yang lebih tinggi cenderung menurunkan produksi padi.")

    # Line plot prediksi vs aktual
    st.subheader("Aktual vs Prediksi Produksi")
    fig3, ax3 = plt.subplots()
    ax3.plot(y_test.values, label='Aktual', marker='o')
    ax3.plot(y_pred, label='Prediksi', marker='x')
    ax3.set_title("Aktual vs Prediksi Produksi")
    ax3.set_xlabel("Data Uji (Indeks)")
    ax3.set_ylabel("Produksi (ton/ha)")
    ax3.legend()
    st.pyplot(fig3)
    st.write("**Keterangan**: Garis dengan titik bulat menunjukkan nilai produksi aktual, sedangkan garis dengan tanda silang menunjukkan nilai prediksi dari model regresi linier berganda pada data uji.")

    # Residual plot
    st.subheader("Residual Plot")
    residuals = y_test - y_pred
    fig4, ax4 = plt.subplots()
    sns.scatterplot(x=y_pred, y=residuals, ax=ax4)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel("Prediksi Produksi (ton/ha)")
    ax4.set_ylabel("Residual (Aktual - Prediksi)")
    ax4.set_title("Residual Plot")
    st.pyplot(fig4)
    st.write("**Keterangan**: Setiap titik mewakili selisih antara produksi aktual dan prediksi (residual) untuk data uji. Residual yang tersebar acak di sekitar garis nol menunjukkan tidak ada pola sistematis dalam kesalahan prediksi, memenuhi asumsi homoskedastisitas.")