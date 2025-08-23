import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# โหลดรูปภาพแบบปลอดภัย
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ภาพ: {path}")
        return None

HairG = load_image("hairgrowP.jpg")
Skindarken = load_image("skin darkenP.jpg")

# ส่วนหัว
st.header("แอปพลิเคชัน\n## ประเมินความเสี่ยงโรคถุงน้ำรังไข่หลายใบ")
st.write("<<< หากไม่พบแบบประเมิน คลิกลูกศรมุมซ้ายบนเพื่อเปิดเมนูกรอกข้อมูล")

st.error(
    """แบบประเมินความเสี่ยงนี้เป็นเพียงการประเมินเบื้องต้น
    ❌ ไม่ใช่การวินิจฉัยจากแพทย์

    หากผลการประเมินพบว่ามีความเสี่ยง ควรพบแพทย์ผู้เชี่ยวชาญ  
    เช่น แผนกนรีเวช หรือสูติ-นรีเวช ของโรงพยาบาลใกล้บ้าน  

    👉 อ่านข้อมูลเพิ่มเติม:  
    [บทความจาก Bangkok Hospital](https://www.bangkokhospital.com/content/overweight-women-are-more-likely-to-face-polycystic-ovary-syndrome)
    """
)

# Sidebar
st.sidebar.header("📋 แบบประเมินความเสี่ยง")
st.sidebar.subheader("กรอกข้อมูลด้านล่าง")

def user_input_features():
    Age = st.sidebar.slider("อายุ (ปี)", 0, 100, 22)
    Weight = st.sidebar.slider("น้ำหนัก (Kg)", 0, 150, 60)
    Cycle = st.sidebar.slider("ระยะที่มีประจำเดือน (วัน)", 0, 31, 7)
    CycleLength = st.sidebar.slider("รอบเดือนห่างกันกี่วัน", 0, 60, 28)

    st.sidebar.write("### อาการ (0 = ไม่ใช่, 1 = ใช่)")

    hairGrowth = st.sidebar.slider("ขนเพิ่มขึ้นผิดปกติหรือไม่", 0, 1, 0)
    if HairG:
        st.sidebar.image(HairG, caption="Ferriman Hair Growth Chart", use_container_width=True)

    SkinDarkening = st.sidebar.slider("ผิวดำคล้ำตามข้อหรือไม่", 0, 1, 0)
    if Skindarken:
        st.sidebar.image(Skindarken, caption="จุดสังเกตผิวคล้ำ", use_container_width=True)

    Pimples = st.sidebar.slider("สิวเพิ่มขึ้นผิดปกติหรือไม่", 0, 1, 0)
    Fastfood = st.sidebar.slider("ทานอาหารไขมันสูงบ่อยหรือไม่", 0, 1, 0)
    FollicleL = st.sidebar.slider("รูขุมขนกว้างด้านซ้าย", 0, 1, 0)
    FollicleR = st.sidebar.slider("รูขุมขนกว้างด้านขวา", 0, 1, 0)
    WeightGain = st.sidebar.slider("น้ำหนักเพิ่มขึ้นเร็วหรือไม่", 0, 1, 0)

    data = {
        "Age (yrs)": Age,
        "Weight (Kg)": Weight,
        "Cycle(R/I)": Cycle,
        "Cycle length(days)": CycleLength,
        "hair growth(Y/N)": hairGrowth,
        "Skin darkening (Y/N)": SkinDarkening,
        "Pimples(Y/N)": Pimples,
        "Fast food (Y/N)": Fastfood,
        "Follicle No. (L)": FollicleL,
        "Follicle No. (R)": FollicleR,
        "Weight gain(Y/N)": WeightGain,
    }

    return pd.DataFrame(data, index=[0])

# โหลดโมเดล
model_path = "PcosApp.joblib"
if not os.path.exists(model_path):
    st.error(f"❌ ไม่พบไฟล์โมเดล: {model_path}")
    st.stop()
model = joblib.load(model_path)

# ทำการทำนาย
df = user_input_features()
st.subheader("🧾 ข้อมูลที่คุณกรอก")
st.write(df)

if st.button("🔎 ทำการประเมินความเสี่ยง"):
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader("📊 ผลการประเมิน")
    if prediction[0] == 1:
        st.error("⚠️ ท่านมีความเสี่ยงสูง ควรพบแพทย์เพื่อตรวจเพิ่มเติม")
    else:
        st.success("✅ ความเสี่ยงต่ำ")

    st.subheader("เปอร์เซ็นต์ความเสี่ยง")
    st.write({
        "โอกาสเสี่ยงต่ำ": f"{prediction_proba[0][0]*100:.2f}%",
        "โอกาสเสี่ยงสูง": f"{prediction_proba[0][1]*100:.2f}%"
    })

# Section แบบสอบถามท้ายหน้า
with st.expander("📝 รบกวนทำแบบสอบถามการใช้งานเว็บไซต์"):
    st.write("เพื่อปรับปรุงคุณภาพและประสิทธิภาพของแบบประเมิน กรุณาช่วยตอบแบบสอบถามค่ะ 🙏")
    st.markdown("[👉 กดที่นี่เพื่อตอบแบบสอบถาม](https://forms.gle/u7GK9hvWkpWjJjaD9)")
