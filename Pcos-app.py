import streamlit as st 
import pandas as pd
import joblib
import numpy as np
import os
import plotly.graph_objects as go
from PIL import Image

# ========== CSS Custom ==========
st.markdown("""
<style>
    .main {background-color: #f9f9fb;}
    h1, h2, h3 {color: #4B0082;}
    .stButton>button {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(to right, #2575fc, #6a11cb);
        transform: scale(1.05);
    }
    .risk-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load("PcosApp.joblib")

def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    else:
        st.warning(f"⚠️ ไม่พบไฟล์ภาพ: {path}")
        return None

HairG = load_image("hairgrowP.jpg")
Skindarken = load_image("skin darkenP.jpg")

# ฟีเจอร์ที่ใช้
features = [
    'Age (yrs)', 'Weight (Kg)', 'Cycle(R/I)', 'Cycle length(days)',
    'hair growth(Y/N)', 'Skin darkening (Y/N)', 'Pimples(Y/N)',
    'Fast food (Y/N)', 'Follicle No. (L)', 'Follicle No. (R)', 'Weight gain(Y/N)'
]
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
                processed.append(1)
            elif v in ["I"]: 
                processed.append(0)
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



# โหลดโมเดล
model = joblib.load("PcosApp.joblib")

# Sidebar
st.sidebar.title("ℹ️ ข้อมูลเพิ่มเติม")
st.sidebar.warning("⚠️ แบบประเมินนี้เป็นเพียงการประเมินเบื้องต้น ❌ ไม่ใช่การวินิจฉัยทางการแพทย์")
st.sidebar.warning("หากผลการประเมินพบว่ามีความเสี่ยง ควรพบแพทย์ผู้เชี่ยวชาญ เช่น แผนกนรีเวช หรือสูติ-นรีเวช ของโรงพยาบาลใกล้บ้าน ")
st.sidebar.markdown("[อ่านเพิ่มเติม: ภาวะ PCOS](https://www.bangkokhospital.com/content/overweight)")

# ========== Main UI ==========
st.title("🧬 PCOS Risk Self-Assessment")
st.markdown("### ประเมินความเสี่ยงโรคภาวะถุงน้ำในรังไข่หลายใบ ")

with st.container():
    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
    col1, col2, col3= st.columns(3)
    with col1:
        age = st.number_input("Age (yrs) อายุ (ปี)", min_value=10, max_value=60, value=25)
        weight = st.number_input("Weight (Kg) น้ำหนัก (กิโลกรัม) ", min_value=30, max_value=200, value=60)
        cycle_ri = st.radio("🔄 Cycle ประจำเดือนมาปกติทุกเดือนหรือไม่ (R ปกติ /Iไม่ปกติ )", ["R", "I"])
        cycle_length = st.number_input("🗓️ Cycle length (days) ต่อรอบเดือนประจำเดือนมากี่วัน", min_value=1, max_value=31, value=7)
        #st.error("หากผู้ประเมินมีประจำเดือนมามากกว่าเกณฑ์การประเมินที่ 1-15 วัน ควรปรึกษาแพทย์ผู้เชี่ยวชาญ")
        hair_growth = st.radio("Hair growth มีขนดกมากกว่าปกติหรือไม่?", ["Y", "N"]) 
        if HairG:
            st.image(HairG, caption="Ferriman Hair Growth Chart", use_container_width=True)
            
    with col2:       
        skin_dark = st.radio("🌑 Skin darkening มีรอยคล้ำเข้มขึ้นตามผิวหนังหรือไม่? ", ["Y", "N"])
        if Skindarken:
            st.image(Skindarken, caption="ตำแหน่งรอยคล้ำที่พบบ่อย", use_container_width=True)
              
    with col3:
        pimples = st.radio("Pimples มีสิวเพิ่มมากขึ้นหรือไม่ ", ["Y", "N"])          
        fast_food = st.radio("🍔 Fast food รับประทานอาหารที่มีไขมันเยอะบ่อยหรือไม่ ", ["Y", "N"])
        weight_gain = st.radio("📈 Weight gain น้ำหนักเพิ่มขึ้นผิดปกติหรือไม่?", ["Y", "N"])
        # กำหนดค่า foll เป็น 0 โดยอัตโนมัติ
        foll_l = 0
        foll_r = 0
        
    st.markdown('</div>', unsafe_allow_html=True)  
def user_input_features():
    return {
        "age": age,
        "weight": weight,
        "cycle_ri": cycle_ri,
        "cycle_length": cycle_length,
        "hair_growth": hair_growth,
        "skin_dark": skin_dark,
        "pimples": pimples,
        "fast_food": fast_food,
        "foll_l": foll_l,
        "foll_r": foll_r,
        "weight_gain": weight_gain
    }
user_data = user_input_features()


st.set_page_config(page_title="PCOS Risk Assessment", page_icon="🧬", layout="wide")
#st.write(user_data)
if st.button("🔍 ประเมินความเสี่ยง"):
    risk, prob = predict_risk(
        user_data["age"],
        user_data["weight"],
        user_data["cycle_ri"],
        user_data["cycle_length"],
        user_data["hair_growth"],
        user_data["skin_dark"],
        user_data["pimples"],
        user_data["fast_food"],
        user_data["foll_l"],
        user_data["foll_r"],
        user_data["weight_gain"]
    )
    st.markdown('<div class="risk-card">', unsafe_allow_html=True)
    st.subheader(f"🧾 ผลการประเมิน")
    st.success(f"ระดับความเสี่ยง: **{risk}** ({prob:.2f}%)")
    if cycle_length >= 15:
        st.error("⚠️ หากคุณมีประจำเดือนมามากกว่าเกณฑ์ (1–15 วัน) **ควรปรึกษาแพทย์ผู้เชี่ยวชาญ**")

    # แถบ Progress bar (gradient style)
    progress_html = f"""
    <div style="background-color:#e0e0e0;border-radius:20px;height:25px;">
        <div style="width:{prob}%;background:linear-gradient(90deg,#6a11cb,#2575fc);
        height:25px;border-radius:20px;text-align:center;color:white;font-weight:bold;">
        {prob:.1f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Risk Probability (%)", 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "#90EE90"},
                {'range': [33, 66], 'color': "#FFD700"},
                {'range': [66, 100], 'color': "#FF6347"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob
            }
        }
    ))
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("🟢 ต่ำ < 33%   |   🟡 ปานกลาง 33-66%   |   🔴 สูง > 66%")

# Section แบบสอบถามท้ายหน้า
with st.expander("📝 รบกวนทำแบบสอบถามการใช้งานเว็บไซต์"):
    st.write("เพื่อปรับปรุงคุณภาพและประสิทธิภาพของแบบประเมิน กรุณาช่วยตอบแบบสอบถามค่ะ 🙏")
    st.markdown("[👉 กดที่นี่เพื่อตอบแบบสอบถาม](https://forms.gle/4Np3VBaY4aeN5Ws27)")



























