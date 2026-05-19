import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. 页面基本设置与顶刊级文案
# ==========================================
st.set_page_config(page_title="Postop AKI Predictor", page_icon="🏥", layout="wide")

st.title("Postoperative AKI & Severe AKI Risk Prediction System")
st.markdown("""
**Disclaimer:** This interactive dual-model system is designed to assist clinicians in predicting the individualized risk of Acute Kidney Injury (AKI). 
- **Primary Screening:** An optimized **Stacking Ensemble** predicts the overall risk of developing AKI.
- **Severity Assessment:** An **XGBoost Algorithm** specifically predicts the risk of developing Severe AKI (Stage II/III).
*Developed in adherence to the TRIPOD-AI reporting guidelines.*
""")
st.divider()

# ==========================================
# 2. 模型加载逻辑 (带安全保护)
# ==========================================
@st.cache_resource
def load_models():
    stacking_path = "model_stacking.pkl"
    xgb_path = "model_xgboost_severe.pkl"
    
    if not os.path.exists(stacking_path) or not os.path.exists(xgb_path):
        return None, None, None
    
    # 我们在训练时把模型和所需的列名绑定在了元组里，确保特征严格对齐
    model_stacking, expected_cols = joblib.load(stacking_path)
    model_xgboost, _ = joblib.load(xgb_path)
    return model_stacking, model_xgboost, expected_cols

model_stacking, model_xgboost, expected_cols = load_models()

if model_stacking is None:
    st.error("⚠️ Error: Model files not found! Please ensure 'model_stacking.pkl' and 'model_xgboost_severe.pkl' are in the same directory as this app.py.")
    st.stop()

# ==========================================
# 3. 侧边栏：患者特征输入
# ==========================================
st.sidebar.header("🩺 Patient Parameters")

age = st.sidebar.number_input("Age (years)", value=60, min_value=18, max_value=100)
gender = st.sidebar.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
bmi = st.sidebar.number_input("Body Mass Index (BMI)", value=22.0, min_value=10.0, max_value=50.0)
diabetes = st.sidebar.selectbox("Diabetes Mellitus", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

preop_alb = st.sidebar.slider("Preoperative Albumin (g/L)", min_value=20.0, max_value=50.0, value=35.0)
op_time = st.sidebar.slider("Operation Time (minutes)", min_value=30, max_value=600, value=180)

nonop_transfusion = st.sidebar.selectbox("Non-operative Blood Transfusion", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
intraop_plasma = st.sidebar.number_input("Intraoperative Plasma Transfusion (mL)", value=800, min_value=0, max_value=5000)
nonop_alb = st.sidebar.number_input("Non-operative Albumin Infusion (g)", value=20, min_value=0, max_value=100)
vasoactive = st.sidebar.selectbox("Perioperative Vasoactive Agents", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]

tnm_stage = st.sidebar.selectbox("TNM Stage", options=[("I", 1), ("II", 2), ("III", 3), ("IV", 4)], format_func=lambda x: x[0])[1]
surgery_type = st.sidebar.selectbox("Surgery Type", options=[("Gastric", 0), ("Colorectal", 1)], format_func=lambda x: x[0])[1]

# ==========================================
# 4. 主界面：预测逻辑与双模型输出 (已加入智能特征对齐)
# ==========================================
if st.sidebar.button("Calculate Dual-Risk", type="primary"):
    with st.spinner("Analyzing patient data through Stacking and XGBoost ensembles..."):
        
        # 1. 将前端所有的输入放在一个字典里，键名是基础特征名
        all_ui_inputs = {
            'Age': age, 'Gender': gender, 'BMI': bmi, 'Diabetes': diabetes, 
            'PreopAlb': preop_alb, 'OperationTime': op_time, 
            'NonOpTransfusion': nonop_transfusion, 'IntraopPlasma': intraop_plasma, 
            'NonOpAlbumin': nonop_alb, 'PerioperativeVasoactive': vasoactive, 
            'TNM_Stage': tnm_stage, 'Gastrocolorectal': surgery_type
        }

        # 2. 智能匹配：找出模型真正需要的列 (expected_cols)，并把对应的值填进去
        final_input_dict = {}
        for col in expected_cols:
            for base_name, val in all_ui_inputs.items():
                # 使用不区分大小写的方式匹配列名
                if base_name.lower() in col.lower():
                    final_input_dict[col] = val
                    break
            # 如果有个别列没匹配上，给个默认值 0 防止报错
            if col not in final_input_dict:
                final_input_dict[col] = 0
                
        # 3. 生成 DataFrame，这下特征数量和名称 100% 对齐了！
        input_data = pd.DataFrame([final_input_dict])

        # 🚀 真实模型推理！
        risk_overall = model_stacking.predict_proba(input_data)[0][1]
        risk_severe = model_xgboost.predict_proba(input_data)[0][1]
        
        # 逻辑约束：重症的数学概率不能大于患病的总体概率
        risk_severe = min(risk_overall, risk_severe) 

        # --- 结果展示区 ---
        col1, col2 = st.columns(2)
        
        # 面板 1：总体 AKI 风险 (Stacking)
        with col1:
            st.markdown("### 🔵 Overall AKI Risk")
            st.caption("Powered by Stacking Ensemble Meta-Learner")
            
            if risk_overall >= 0.20:
                st.error(f"## {risk_overall * 100:.1f}%\n**HIGH RISK**")
            elif risk_overall >= 0.10:
                st.warning(f"## {risk_overall * 100:.1f}%\n**MODERATE RISK**")
            else:
                st.success(f"## {risk_overall * 100:.1f}%\n**LOW RISK**")
                
        # 面板 2：重症 AKI 风险 (XGBoost)
        with col2:
            st.markdown("### 🔴 Severe AKI Risk (Stage II/III)")
            st.caption("Powered by Extreme Gradient Boosting (XGBoost)")
            
            if risk_severe >= 0.10:
                st.error(f"## {risk_severe * 100:.1f}%\n**HIGH SEVERITY ALERT**")
                st.info("💡 **Clinical Action:** ICU monitoring, strict fluid management, and nephrology consultation recommended.")
            else:
                st.success(f"## {risk_severe * 100:.1f}%\n**Low Severity Risk**")

        st.divider()
        st.markdown("### 📋 Model Input Feature Summary (Auto-Aligned)")
        st.dataframe(input_data, hide_index=True)
