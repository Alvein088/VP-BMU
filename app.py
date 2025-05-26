
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("🧬 AI Chẩn đoán Tác nhân & Gợi ý Kháng sinh")

# Huấn luyện mô hình trực tiếp
@st.cache_data
def train_model():
    import joblib

    def convert_binary(val):
        if isinstance(val, str):
            val = val.strip().lower()
            if val in ["x", "có", "yes"]:
                return 1
            elif val in ["/", "khong", "không", "no", "ko"]:
                return 0
        return np.nan

    def convert_age(val):
        if isinstance(val, str):
            val = val.strip()
            if "Thg" in val:
                return float(val.replace("Thg", "")) / 10
            try:
                return float(val)
            except:
                return np.nan
        return val

    # Đọc dữ liệu
    df = pd.read_csv("Mô hình AI.csv")

    df["Tuoi"] = df["Tuoi"].apply(convert_age)
    df = df[df["Tac nhan"].notna()]
    binary_cols = df.columns[df.dtypes == object]
    for col in binary_cols:
        if col not in ["ID", "Tac nhan", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"]:
            df[col] = df[col].apply(convert_binary)

    X = df.drop(columns=["ID", "Tac nhan", "Gioi Tinh", "Dân tộc", "Nơi ở", "Tình trạng xuất viện"])
    y = df["Tac nhan"]
    X = X.apply(lambda col: col.fillna(col.mean()) if col.dtype != object else col.fillna(0))

    pathogen_model = RandomForestClassifier(n_estimators=100, random_state=42)
    pathogen_model.fit(X, y)

    abx_models = {}
    target_abx = [
        "Ceftriaxone", "Vancomycin", "Meropenem",
        "Amoxicilin clavulanic", "Levofloxacin"
    ]
    for abx in target_abx:
        if abx in df.columns:
            y_abx = df[abx].apply(convert_binary).fillna(0)
            abx_model = RandomForestClassifier(n_estimators=100, random_state=42)
            abx_model.fit(X, y_abx)
            abx_models[abx] = abx_model

    return pathogen_model, abx_models, X.columns.tolist()

model, abx_models, feature_cols = train_model()

st.markdown("### Nhập dữ liệu bệnh nhân")

user_input = {}
for col in feature_cols:
    if col == "Tuoi":
        user_input[col] = st.number_input("Tuổi (năm)", min_value=0.0, max_value=120.0, step=1.0)
    elif col in ["Nhiet do", "Bach cau", "CRP", "Nhip tho", "Mach"]:
        user_input[col] = st.number_input(col, value=0.0)
    else:
        user_input[col] = st.selectbox(f"{col}", ["Không", "Có"]) == "Có"

if st.button("Dự đoán"):
    input_df = pd.DataFrame([user_input])
    for col in input_df.columns:
        if isinstance(input_df[col].iloc[0], bool):
            input_df[col] = input_df[col].astype(int)

    prediction = model.predict(input_df)[0]
    st.success(f"✅ Tác nhân gây bệnh dự đoán: **{prediction}**")

    st.markdown("### 💊 Kháng sinh gợi ý nên sử dụng:")
    any_suggested = False
    for abx, clf in abx_models.items():
        if clf.predict(input_df)[0] == 1:
            st.write(f"- **{abx}**")
            any_suggested = True
    if not any_suggested:
        st.info("Không có kháng sinh nào được gợi ý.")
