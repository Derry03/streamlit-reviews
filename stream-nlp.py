import pickle
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer

model_fraud = pickle.load(open('model_fraud.sav', 'rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(
    pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))

add_title = st.sidebar.header("Selamat Datang")
with st.sidebar:
    subheader = st.caption(
        "Aplikasi ini dibangun untuk mengalisis sentimen ulasan produk yang dikumpulkan dari salah satu marketplace yang banyak digunakan oleh masyarakat Indonesia."
    )

# Judul
st.title('DETEKSI REVIEWS')
st.markdown("Tugas UAS Business intelligence")

tab1, tab2 = st.tabs(["Dashboard", "About"])

with tab1:
    st.header("COBA DETEKSI")
    clean_reviews = st.text_input('MASUKAN KATA :')

    fraud_detection = ''
    if st.button('HASIL DETEKSI'):
        predict_fraud = model_fraud.predict(
            loaded_vec.fit_transform([clean_reviews]))

        if (predict_fraud == 1):
            fraud_detection = 'INI REVIEWS POSITIF'
        else:
            fraud_detection = 'INI REVIEWS NEGATIF'
        st.success(fraud_detection)

with tab2:
    st.header("About")

    image = Image.open("dery.jpg")
    st.image(image, caption='Derry Asari Nuryadi', width=200)
    st.markdown("NIM    : 191351018")
    st.markdown("Kelas  : Malam B")
