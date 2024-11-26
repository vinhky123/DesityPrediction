import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.pkl') 
scaler = joblib.load('scaler.pkl')


label_encoders = joblib.load('label_encoders.pkl')  # Path to saved encoders


# Feature columns
X_columns = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE',
    'CALC', 'MTRANS'
]

# Categorical columns
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']

# Mapping dictionaries for translation
translation_mapping = {
    'Gender': {'Nam': 'Male', 'Nữ': 'Female'},
    'family_history_with_overweight': {'Có': 'yes', 'Không': 'no'},
    'FAVC': {'Có': 'yes', 'Không': 'no'},
    'CAEC': {'Không': 'no', 'Thỉnh thoảng': 'Sometimes', 'Thường xuyên': 'Frequently', 'Luôn luôn': 'Always'},
    'SMOKE': {'Có': 'yes', 'Không': 'no'},
    'SCC': {'Có': 'yes', 'Không': 'no'},
    'CALC': {'Không': 'no', 'Thỉnh thoảng': 'Sometimes', 'Thường xuyên': 'Frequently', 'Luôn luôn': 'Always'},
    'MTRANS': {'Ô tô': 'Automobile', 'Xe máy': 'Motorbike', 'Xe đạp': 'Bike', 
               'Phương tiện công cộng': 'Public_Transportation', 'Đi bộ': 'Walking'}
}

# Define the prediction function
def predict_obesity(input_list):
    if len(input_list) != len(X_columns):
        raise ValueError(f"Expected {len(X_columns)} features, but got {len(input_list)}")

    input_dict = {col: [val] for col, val in zip(X_columns, input_list)}
    input_df = pd.DataFrame(input_dict)

    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = label_encoders[col].transform(input_df[col])

    input_normalized = scaler.transform(input_df)

    prediction_proba = model.predict_proba(input_normalized)
    obesity_indices = [2, 3, 4] 
    obesity_probability = sum(prediction_proba[0][index] for index in obesity_indices) * 100  

    return obesity_probability

st.title("Ứng dụng Dự Đoán Béo Phì 🌟")
st.write("Dự đoán tỷ lệ béo phì dựa trên các thông số cá nhân và thói quen hằng ngày.")

# Sidebar for input
st.sidebar.header("Thông tin cá nhân")
gender = st.sidebar.radio("Giới tính", list(translation_mapping['Gender'].keys()))
age = st.sidebar.slider("Tuổi", min_value=1, max_value=100, value=25)
height = st.sidebar.slider("Chiều cao (m)", min_value=0.5, max_value=2.5, value=1.7, step=0.01)
weight = st.sidebar.slider("Cân nặng (kg)", min_value=10, max_value=300, value=70)
family_history = st.sidebar.radio("Gia đình có người bị béo phì", list(translation_mapping['family_history_with_overweight'].keys()))

# Main input
st.header("Thói quen hằng ngày")
favc = st.selectbox("Thường xuyên sử dụng thực phẩm có Calo cao", list(translation_mapping['FAVC'].keys()))
fcvc = st.slider("Tần suất ăn trái cây (1-3)", min_value=1, max_value=3, value=2)
ncp = st.slider("Số bữa ăn mỗi ngày (1-4)", min_value=1, max_value=4, value=3)
caec = st.selectbox("Tiêu thụ thực phẩm giữa các bữa ăn", list(translation_mapping['CAEC'].keys()))
smoke = st.radio("Hút thuốc", list(translation_mapping['SMOKE'].keys()))
ch2o = st.slider("Lượng nước uống mỗi ngày (1-3)", min_value=1, max_value=3, value=2)
scc = st.radio("Theo dõi Calo", list(translation_mapping['SCC'].keys()))
faf = st.slider("Tần suất chơi thể thao (0-3)", min_value=0, max_value=3, value=1)
tue = st.slider("Thời gian sử dụng thiết bị công nghệ (0-2)", min_value=0, max_value=2, value=1)
calc = st.selectbox("Uống rượu bia", list(translation_mapping['CALC'].keys()))
mtrans = st.selectbox("Phương tiện di chuyển", list(translation_mapping['MTRANS'].keys()))

input_features = [
    translation_mapping['Gender'][gender],
    age,
    height,
    weight,
    translation_mapping['family_history_with_overweight'][family_history],
    translation_mapping['FAVC'][favc],
    fcvc,
    ncp,
    translation_mapping['CAEC'][caec],
    translation_mapping['SMOKE'][smoke],
    ch2o,
    translation_mapping['SCC'][scc],
    faf,
    tue,
    translation_mapping['CALC'][calc],
    translation_mapping['MTRANS'][mtrans]
]

# Predict button
if st.button("Dự đoán"):
    try:
        result = predict_obesity(input_features)
        st.success(f"🔍 Tỉ lệ bị béo phì: **{result:.2f}%**")
        
        # Visualize result
        st.progress(min(result / 100, 1.0))
        st.info("Hãy duy trì lối sống lành mạnh để cải thiện sức khỏe!")
    except ValueError as e:
        st.error(f"Lỗi: {e}")
