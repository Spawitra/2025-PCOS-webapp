import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
model = joblib.load("PcosApp.joblib")

st.title("🔮 PCOS Prediction App")
st.write("ทำนายความเสี่ยงภาวะถุงน้ำรังไข่หลายใบ (PCOS)")

# ฟอร์มกรอกข้อมูล
age = st.number_input("อายุ", min_value=10, max_value=60, value=25)
bmi = st.number_input("ค่า BMI", min_value=10.0, max_value=50.0, value=22.0)
cycle = st.number_input("รอบเดือน (วัน)", min_value=15, max_value=60, value=28)
weight_gain = st.radio("มีภาวะน้ำหนักเพิ่มผิดปกติหรือไม่?", ["ไม่มี", "มี"])

# แปลงข้อมูลเป็น input vector
X = np.array([[
    age,
    bmi,
    cycle,
    1 if weight_gain == "มี" else 0
]])

if st.button("🔍 ทำนาย"):
    prediction = model.predict(X)
    if prediction[0] == 1:
        st.error("⚠️ มีความเสี่ยงเป็น PCOS")
    else:
        st.success("✅ ความเสี่ยงต่ำ")

st.markdown("---")
st.markdown("📋 [รบกวนทำแบบสอบถามประสิทธิภาพของระบบ](https://forms.gle/u7GK9hvWkpWjJjaD9)")
