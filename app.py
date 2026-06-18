import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Chennai Flood Risk Predictor", page_icon="🌊")
st.title("🌊 Chennai Flood Risk Predictor 2015")
st.write("Data Source: data.gov.in Statistical Handbook 2018")

# Data Load
df = pd.read_csv("rainfall.csv") # உன் CSV file name
X = df[["Actual_Rainfall", "Normal_Rainfall"]] 
y = df["Flood"] # 1=Yes, 0=No னு இருக்கணும்

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy Calculate பண்ணு - Real Value
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
st.write(f"**Model Accuracy: {acc:.1f}%**") # ← இப்போ 100% வராது

# UI
col1, col2 = st.columns(2)
with col1:
    st.subheader("📊 Input Rainfall Data")
    actual = st.number_input("Actual Rainfall (mm)", 0.0, 2000.0, 1126.80)
    normal = st.number_input("Normal Rainfall (mm)", 0.0, 2000.0, 971.80)
    excess = actual - normal
    st.info(f"Excess Rainfall: {excess:.1f} mm")
    
    if st.button("🔮 Predict Flood Risk"):
        result = model.predict([[actual, normal]])
        prob = model.predict_proba([[actual, normal]])
        
        with col2:
            st.subheader("📈 Prediction Result")
            if result[0] == 1:
                st.error(f"⚠️ High Flood Risk! Probability: {prob[0][1]*100:.1f}%")
            else:
                st.success(f"✅ Low Flood Risk. Probability: {prob[0][0]*100:.1f}%")
