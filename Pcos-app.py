import streamlit as st
import joblib
import numpy as np

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("PcosApp.joblib")

# ฟีเจอร์ที่ใช้
features = [
    'Age (yrs)', 'Weight (Kg)', 'Cycle(R/I)', 'Cycle length(days)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Follicle No. (L)', 'Follicle No. (R)', 'Weight gain(Y/N)'
]

# โหลดรูปภาพแบบปลอดภัย
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ภาพ: {path}")
        return None

HairG = load_image("hairgrowP.jpg")
Skindarken = load_image("skin darkenP.jpg")

# ฟังก์ชันแปลงค่าก่อนทำนาย
def preprocess_input(values):
    processed = []
    for f, v in zip(features, values):
        if isinstance(v, str):
            v = v.strip().upper()
            if v in ["Y", "YES", "1"]: 
                processed.append(1)
            elif v in ["N", "NO", "0"]: 
                processed.append(0)
            elif v in ["R"]: 
                processed.append(0)
            elif v in ["I"]: 
                processed.append(1)
            else:
                try:
                    processed.append(float(v))
                except:
                    processed.append(0)
        else:
            processed.append(v)
    return np.array(processed).reshape(1, -1)

# ฟังก์ชันทำนาย
def predict_risk(age, weight, cycle_ri, cycle_length, hair_growth, skin_dark, pimples,
                 fast_food, foll_l, foll_r, weight_gain):
    values = [age, weight, cycle_ri, cycle_length, hair_growth, skin_dark, pimples,
              fast_food, foll_l, foll_r, weight_gain]
    X = preprocess_input(values)
    prob = model.predict_proba(X)[0][1] * 100
    if prob < 33:
        risk = "ต่ำ"
    elif prob < 66:
        risk = "ปานกลาง"
    else:
        risk = "สูง"
    return risk, prob

# ================= Streamlit UI =================
st.title("🧬 PCOS Risk Self-Assessment")
st.write("ประเมินความเสี่ยงเบื้องต้น (ไม่ใช่การวินิจฉัยจากแพทย์)")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (yrs)", min_value=10, max_value=60, value=25)
    weight = st.number_input("Weight (Kg)", min_value=30, max_value=200, value=60)
    cycle_ri = st.radio("Cycle (R/I)", ["R", "I"])
    cycle_length = st.number_input("Cycle length (days)", min_value=15, max_value=60, value=28)
    hair_growth = st.radio("Hair growth", ["Y", "N"])
    if HairG:
        st.sidebar.image(HairG, caption="Ferriman Hair Growth Chart", use_container_width=True)
     skin_dark = st.radio("Skin darkening", ["Y", "N"])
    if Skindarken:
        st.sidebar.image(Skindarken, caption="จุดสังเกตผิวคล้ำ", use_container_width=True)
   

with col2:
    pimples = st.radio("Pimples", ["Y", "N"])
    fast_food = st.radio("Fast food", ["Y", "N"])
    foll_l = st.number_input("Follicle No. (L)", min_value=0, max_value=30, value=10)
    foll_r = st.number_input("Follicle No. (R)", min_value=0, max_value=30, value=10)
    weight_gain = st.radio("Weight gain", ["Y", "N"])

if st.button("🔍 ประเมินความเสี่ยง"):
    risk, prob = predict_risk(age, weight, cycle_ri, cycle_length, hair_growth, skin_dark,
                              pimples, fast_food, foll_l, foll_r, weight_gain)
    st.success(f"ระดับความเสี่ยง: {risk} ({prob:.2f}%)")


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




