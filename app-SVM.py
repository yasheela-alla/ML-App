import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# ====== Streamlit Page Config ======
st.set_page_config(page_title="COVID-19 UK Prediction", layout="centered")

# ====== Styling with Markdown ======
st.markdown(
    """
    <style>
    .big-font { font-size:25px !important; font-weight: bold; }
    .stButton>button { background-color: #ff4b4b; color: white; font-size: 16px; border-radius: 10px; }
    .stMarkdown { font-size:18px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ====== Title & Introduction ======
st.markdown('<p class="big-font">📊 COVID-19 Cases Prediction (UK)</p>', unsafe_allow_html=True)
st.markdown("Using **Machine Learning (SVM)** to predict future COVID-19 cases based on historical data.")

# ====== Fetch COVID-19 Data ======
url = "https://disease.sh/v3/covid-19/countries/uk"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "Total Cases": data["cases"],
    "Cases Today": data["todayCases"],
    "Total Deaths": data["deaths"],
    "Deaths Today": data["todayDeaths"],
    "Recovered": data["recovered"],
    "Active Cases": data["active"],
    "Critical Cases": data["critical"],
    "Cases Per Million": data["casesPerOneMillion"],
    "Deaths Per Million": data["deathsPerOneMillion"],
}

# ====== Display COVID-19 Data ======
st.subheader("📌 Current COVID-19 Statistics in the UK")
st.dataframe(pd.DataFrame([covid_data]).T, use_container_width=True)

# ====== Generate Historical Data ======
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Simulated last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"Cases": historical_cases, "Deaths": historical_deaths})
df_historical["Day"] = range(1, 31)

# ====== Train SVM Model ======
X = df_historical[["Day"]]
y = df_historical["Cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1.0)
model.fit(X_train, y_train)

# ====== Prediction UI ======
st.subheader("🔮 Predict Future Cases")
day_input = st.number_input("Enter the future day number to predict cases (e.g., 31, 32, ...)", min_value=1, max_value=100)

if st.button("🔍 Predict Cases"):
    prediction = model.predict([[day_input]])
    st.markdown(f"<p class='big-font' style='color: #ff4b4b;'>📈 Predicted Cases for Day {day_input}: {int(prediction[0])}</p>", unsafe_allow_html=True)

    # ====== Visualization ======
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=df_historical["Day"], y=df_historical["Cases"], marker='o', label="Historical Cases", color="blue")
    
    # Highlight the prediction
    ax.scatter(day_input, prediction[0], color='red', s=150, label=f"Prediction (Day {day_input})", edgecolors='black')

    plt.xlabel("Days", fontsize=14)
    plt.ylabel("Number of Cases", fontsize=14)
    plt.title("📊 COVID-19 Cases Trend & Prediction", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

# ====== Footer ======
st.markdown("---")
st.markdown("🚀 Built with **Python, Streamlit, Seaborn, and Machine Learning (SVM)**")
st.markdown("📌 Data Source: [Disease.sh API](https://disease.sh/)")
