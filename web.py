import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
from PIL import Image
import base64
from io import BytesIO
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# 修复NumPy bool弃用问题
if not hasattr(np, 'bool'):
    np.bool = bool

# 设置页面标题和布局
st.set_page_config(
    page_title="Postoperative Delirium Risk Prediction Model in Patients Undergoing Cardiopulmonary Bypass Cardiac Surgery",
    page_icon="🏥",
    layout="wide"
)

# 定义全局变量
global feature_names, feature_dict, variable_descriptions

# 特征名称和描述
feature_names = [
    'Mvt', 'Largetransfus', 'Age', 'PoLAC', 'PoRBC', 
    'Coronaryhd', 'Intratransfus', 'EFT'
]

feature_names_en = [
    'Mechanical Ventilation Time', 'Large Postoperative Transfusion (>1L)', 'Age', 
    'Postoperative Lactate Value', 'Postoperative Red Blood Cell Count', 
    'Coronary Heart Disease', 'Intraoperative Transfusion Volume', 'Essential Frailty Toolset Score'
]

feature_dict = dict(zip(feature_names, feature_names_en))

# 变量说明字典
variable_descriptions = {
    'Mvt': 'Duration of mechanical ventilation (hours)',
    'Largetransfus': 'Whether the patient received large postoperative blood transfusion (>1L) (0=No, 1=Yes)',
    'Age': 'Patient age in years',
    'PoLAC': 'Postoperative blood lactate value (mmol/L)',
    'PoRBC': 'Postoperative red blood cell count (×10^12/L)',
    'Coronaryhd': 'Whether the patient has coronary heart disease (0=No, 1=Yes)',
    'Intratransfus': 'Intraoperative transfusion volume (mL)',
    'EFT': 'Essential Frailty Toolset score, assessing patient frailty (0-5 points)'
}

# 加载模型和预处理器
@st.cache_resource
def load_model():
    predictor = joblib.load('pod_predictor_ascii.pkl')
    return predictor

# 主应用
def main():
    global feature_names, feature_dict, variable_descriptions
    
    # 侧边栏标题
    st.sidebar.title("Postoperative Delirium Risk Prediction Model in Patients Undergoing Cardiopulmonary Bypass Cardiac Surgery")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200)
    
    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # System Description

    ## About This System
    This is a postoperative delirium (POD) risk prediction system for cardiopulmonary bypass surgery patients based on XGBoost algorithm, which predicts POD risk by analyzing patient clinical indicators.

    ## Prediction Results
    The system predicts:
    - POD probability
    - No POD probability
    - Risk assessment (low, medium, high risk)

    ## How to Use
    1. Fill in patient clinical indicators in the main interface
    2. Click the prediction button to generate prediction results
    3. View prediction results and feature importance analysis

    ## Important Notes
    - Please ensure accurate patient information input
    - All fields need to be filled
    - Numeric fields require number input
    - Selection fields require choosing from options
    """)
    
    # 添加变量说明到侧边栏
    with st.sidebar.expander("Variable Descriptions"):
        for feature in feature_names:
            st.markdown(f"**{feature_dict[feature]}**: {variable_descriptions[feature]}")
    
    # 主页面标题
    st.title("Postoperative Delirium Risk Prediction Model in Patients Undergoing Cardiopulmonary Bypass Cardiac Surgery")
    st.markdown("### Based on XGBoost Model")
    
    # 加载模型
    try:
        predictor = load_model()
        model = predictor['model']
        preprocessor = predictor['preprocessor']
        threshold = predictor['threshold']
        feature_importance_df = predictor['feature_importance']
        
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Model loading failed: {e}")
        return
    
    # 创建输入表单
    st.sidebar.header("Patient Information Input")
    
    # 创建两列布局用于输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Patient Information")
        age = st.number_input(f"{feature_dict['Age']} (years)", min_value=18, max_value=100, value=65)
        mvt = st.number_input(f"{feature_dict['Mvt']} (hours)", min_value=0.0, max_value=200.0, value=24.0, step=0.5)
        polac = st.number_input(f"{feature_dict['PoLAC']} (mmol/L)", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
        porbc = st.number_input(f"{feature_dict['PoRBC']} (×10^12/L)", min_value=2.0, max_value=7.0, value=4.0, step=0.1)
    
    with col2:
        st.subheader("Clinical Information")
        largetransfus = st.selectbox(f"{feature_dict['Largetransfus']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        coronaryhd = st.selectbox(f"{feature_dict['Coronaryhd']}", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        intratransfus = st.number_input(f"{feature_dict['Intratransfus']} (mL)", min_value=0, max_value=5000, value=800)
        eft = st.number_input(f"{feature_dict['EFT']} (score)", min_value=0, max_value=5, value=1)
    
    # 创建预测按钮
    predict_button = st.button("Predict POD Risk")
    
    if predict_button:
        # 收集所有输入特征
        features = [mvt, largetransfus, age, polac, porbc, coronaryhd, intratransfus, eft]
        
        # 转换为DataFrame
        input_df = pd.DataFrame([features], columns=feature_names)
        
        # 确保分类变量的类型与训练时一致
        categorical_cols = ['Largetransfus', 'Coronaryhd']
        for col in categorical_cols:
            input_df[col] = input_df[col].astype('category')
        
        # 应用预处理
        processed_data = preprocessor.transform(input_df)
        
        # 进行预测
        prediction = model.predict_proba(processed_data)[0]
        no_pod_prob = prediction[0]
        pod_prob = prediction[1]
        
        # 显示预测结果
        st.header("Prediction Results")
        
        # 使用进度条显示概率
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("No POD Probability")
            st.progress(float(no_pod_prob))
            st.write(f"{no_pod_prob:.2%}")
        
        with col2:
            st.subheader("POD Probability")
            st.progress(float(pod_prob))
            st.write(f"{pod_prob:.2%}")
        
        # 风险评估
        risk_level = "Low Risk" if pod_prob < 0.3 else "Medium Risk" if pod_prob < 0.6 else "High Risk"
        risk_color = "green" if pod_prob < 0.3 else "orange" if pod_prob < 0.6 else "red"
        
        st.markdown(f"### Risk Assessment: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
        
        # 临床建议
        st.header("Clinical Recommendations")
        st.write("Based on the model prediction, the following clinical recommendations are provided:")
        
        if pod_prob > threshold:
            st.warning("""
            This patient has a high risk of postoperative delirium. Consider:
            - Implementing delirium prevention protocols
            - Minimizing sedative medications
            - Early mobilization when possible
            - Regular cognitive assessment
            - Ensuring adequate pain control
            - Maintaining normal sleep-wake cycles
            """)
        else:
            st.success("""
            This patient has a relatively low risk of postoperative delirium. Standard care protocols are recommended:
            - Regular neurological assessments
            - Standard postoperative monitoring
            - Early mobilization as appropriate
            """)
        
        # 添加模型解释
        st.write("---")
        st.subheader("Model Interpretation")

        try:
            # 创建SHAP解释器
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_data)
            
            # 兼容XGBoost输出格式
            if isinstance(shap_values, list):
                # 二分类问题，取正类的SHAP值
                shap_value = shap_values[1][0]
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                shap_value = shap_values[0]
                expected_value = explainer.expected_value
            
            # 特征贡献分析表格
            st.subheader("Feature Contribution Analysis")
            
            # 创建贡献表格
            feature_values = []
            feature_impacts = []
            
            # 获取SHAP值
            for i, feature in enumerate(feature_names):
                feature_values.append(input_df[feature].iloc[0])
                feature_impacts.append(shap_value[i])
            
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': feature_values,
                'Impact': feature_impacts
            })
            
            # 按绝对影响排序
            shap_df['Absolute Impact'] = shap_df['Impact'].abs()
            shap_df = shap_df.sort_values('Absolute Impact', ascending=False)
            
            # 显示表格
            st.table(shap_df[['Feature', 'Value', 'Impact']])
            
            # SHAP瀑布图
            st.subheader("SHAP Waterfall Plot")
            
            # 创建SHAP瀑布图
            fig_waterfall = plt.figure(figsize=(10, 6))
            shap.plots._waterfall.waterfall_legacy(
                expected_value,
                shap_value,
                feature_names=feature_names,
                max_display=8,
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig_waterfall)
            plt.close(fig_waterfall)
            
            # SHAP力图
            st.subheader("SHAP Force Plot")
            
            # 使用与app.py相同的方式显示力图
            plt.figure(figsize=(15, 4))
            shap.force_plot(
                expected_value,
                shap_value,
                input_df.iloc[0],
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            st.pyplot(plt)
            plt.close()
            
        except Exception as e:
            st.error(f"无法生成SHAP解释: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.info("使用模型特征重要性作为替代")
            
            # 显示模型特征重要性
            st.write("---")
            st.subheader("Feature Importance")
            
            # 使用英文特征名称
            top_features = feature_importance_df.head(8)
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
            plt.yticks(range(len(top_features)), [feature_dict.get(f, f) for f in top_features['Feature']])
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

if __name__ == "__main__":
    main()
