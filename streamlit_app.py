
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def load_data():
    df = pd.read_csv("breathesafe_sample_global_aqi.csv")
    return df

df = load_data()

st.title("🌿 BreatheSafe: AI-Powered Air Quality Predictor")
st.write("Predict the Air Quality Index (AQI) and get health-based alerts!")


st.sidebar.header("Enter pollutant & weather values:")

pm25 = st.sidebar.slider("PM2.5 (µg/m³)", 0, 250, 80)
pm10 = st.sidebar.slider("PM10 (µg/m³)", 0, 350, 120)
no2 = st.sidebar.slider("NO2 (ppb)", 0, 100, 40)
o3 = st.sidebar.slider("O3 (ppb)", 0, 150, 60)
co = st.sidebar.slider("CO (ppm)", 0.0, 5.0, 1.2)
temperature = st.sidebar.slider("Temperature (°C)", -10, 50, 28)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 2.0)

user_input = pd.DataFrame([{
    'pm25': pm25,
    'pm10': pm10,
    'no2': no2,
    'o3': o3,
    'co': co,
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed
}])

#  Train Model
features = ['pm25','pm10','no2','o3','co','temperature','humidity','wind_speed']
X = df[features]
y = df['aqi']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict AQI
predicted_aqi = model.predict(user_input)[0]

# AQI Category & Warning
if predicted_aqi <= 50:
    category = "Good"
    warning = "Air quality is healthy. Enjoy outdoor activities!"
elif predicted_aqi <= 100:
    category = "Moderate"
    warning = "Air quality is acceptable. Sensitive groups should consider precautions."
elif predicted_aqi <= 200:
    category = "Unhealthy for Sensitive Groups"
    warning = "Sensitive groups (elderly, lung patients) should limit outdoor exposure."
elif predicted_aqi <= 300:
    category = "Unhealthy"
    warning = "Air quality is unhealthy. Avoid outdoor activities."
else:
    category = "Very Unhealthy / Hazardous"
    warning = "Air is hazardous! Stay indoors and use masks or air purifiers."

# # Display Results
# st.subheader("📈 Predicted AQI")
# st.metric("AQI Value", f"{predicted_aqi:.1f}", delta=None)
# st.write(f"**Category:** {category}")
# st.write(f"**Health Advice:** {warning}")


# Display Results with animation & big visuals
st.subheader("📈 Predicted AQI")

# Animate the AQI value using a growing number effect
import time

placeholder = st.empty()
for i in range(int(predicted_aqi)+1):
    placeholder.markdown(f"<h1 style='text-align:center; color:#FF5733;'>{i}</h1>", unsafe_allow_html=True)
    time.sleep(0.01)  # adjust speed of animation

# Display category in huge font with color based on AQI
if predicted_aqi <= 50:
    color = "#2ECC71"  # green
elif predicted_aqi <= 100:
    color = "#F1C40F"  # yellow
elif predicted_aqi <= 200:
    color = "#E67E22"  # orange
elif predicted_aqi <= 300:
    color = "#E74C3C"  # red
else:
    color = "#8E44AD"  # purple

st.markdown(f"<h2 style='text-align:center; color:{color}; font-size:48px;'>{category}</h2>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align:center; font-size:24px;'>{warning}</p>", unsafe_allow_html=True)

# Optional: Celebrate with balloons for good AQI
if predicted_aqi <= 50:
    st.balloons()











# #Feature Importance Chart
# # -----------------------------
# importance = pd.DataFrame({
#     'Feature': features,
#     'Importance': model.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# st.subheader("Feature Importance")
# fig, ax = plt.subplots(figsize=(8,4))
# sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax)
# st.pyplot(fig)

# # -----------------------------
# #  Dataset Overview
# # -----------------------------
# st.subheader(" Dataset Sample")
# st.dataframe(df.head(10))

# st.subheader("Correlation Heatmap")
# fig2, ax2 = plt.subplots(figsize=(8,5))
# sns.heatmap(df[features + ['aqi']].corr(), annot=True, cmap='coolwarm', ax=ax2)
# st.pyplot(fig2)
