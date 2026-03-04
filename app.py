import streamlit as st
import pickle
from sklearn.datasets import load_iris
import numpy as np

# Page config
st.set_page_config(page_title="Iris Classifier 🌸", page_icon="🌸", layout="centered")

# Load dataset để lấy tên loài
iris = load_iris()

# Load model
clf = pickle.load(open("iris_classifier.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🌸 Iris Flower Classifier</h1>", unsafe_allow_html=True)
st.write("Demo dự đoán loài hoa Iris bằng Machine Learning")

st.divider()

# Chia làm 2 cột cho gọn hơn
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
    sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
    petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

st.divider()

# Predict button ở giữa
if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = clf.predict(input_data)
    result = iris.target_names[prediction[0]]

    st.success(f"🌼 Predicted Species: **{result.upper()}**")