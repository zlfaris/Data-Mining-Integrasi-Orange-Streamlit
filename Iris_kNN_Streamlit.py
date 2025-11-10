import streamlit as st
import Orange
import numpy as np

# Judul Aplikasi
st.title("Prediksi Kategori Bunga Iris (Model Orange)")

# Load model dari Orange (.pkcls)
model = Orange.classification.Model.load("Iris_kNN_ClassificationModel.pkcls")

# Input fitur
st.header("Masukkan Nilai Fitur:")
sepal_length = st.number_input("sepal length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("sepal width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("petal length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("petal width", min_value=0.0, max_value=10.0, step=0.1)

# Tombol Prediksi
if st.button("Prediksi"):
    # Buat data input sesuai format Orange
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    domain = model.domain
    orange_data = Orange.data.Table(domain, data)

    # Prediksi
    prediction = model(orange_data)[0]
    st.success(f"Hasil Prediksi: **{prediction}**")

# Footer
st.markdown("---")
st.caption("Model .pkcls dari Orange3 digunakan untuk prediksi jenis bunga Iris.")
