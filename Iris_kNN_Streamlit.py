import streamlit as st
import pickle
import numpy as np

st.title("Prediksi Spesies Bunga Iris")
st.write("Model ini menggunakan algoritma k-Nearest Neighbors (kNN).")

# Path file model
model_path = "Iris-kNN-Classification-Model.pkcls"

# Load model menggunakan pickle
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.success("Model berhasil dimuat: Iris-kNN-Classification-Model.pkcls")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Input data 
col1, col2 = st.columns([1, 1])
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model(input_data) 
        if hasattr(model, "domain") and hasattr(model.domain.class_var, "values"):
            label = model.domain.class_var.values[int(prediction[0])]
        else:
            label = str(prediction[0])

        # Tampilkan hasil horizontal
        st.markdown(
            f"<h4 style='text-align:center;'>Hasil Prediksi: <span style='color:#006400;'>{label}</span></h4>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")