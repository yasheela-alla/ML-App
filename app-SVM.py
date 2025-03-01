import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# Fetch COVID-19 data
url = "https://disease.sh/v3/covid-19/countries/india"
r = requests.get(url)
data = r.json()

# Extract relevant fields
covid_data = {
    "cases": data["cases"],
    "todayCases": data["todayCases"],
    "deaths": data["deaths"],
    "todayDeaths": data["todayDeaths"],
    "recovered": data["recovered"],
    "active": data["active"],
    "critical": data["critical"],
    "casesPerMillion": data["casesPerOneMillion"],
    "deathsPerMillion": data["deathsPerOneMillion"],
}

# Convert to Pandas DataFrame
df = pd.DataFrame([covid_data])

# Streamlit UI
st.title("COVID-19 Cases Prediction in INDIA")
st.write("Predicting COVID-19 cases for the next day using **SVM (Support Vector Machine)**.")

st.subheader("Current COVID-19 Data for INDIA")
st.write(df)

# Generate random historical data
np.random.seed(42)
historical_cases = np.random.randint(30000, 70000, size=30)  # Last 30 days cases
historical_deaths = np.random.randint(500, 2000, size=30)

df_historical = pd.DataFrame({"cases": historical_cases, "deaths": historical_deaths})
df_historical["day"] = range(1, 31)

# Train-Test Split
X = df_historical[["day"]]
y = df_historical["cases"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1.0)
model.fit(X_train, y_train)

# User Input for Prediction
st.subheader("Predict Future Cases")
day_input = st.number_input("Enter day number (e.g., 31 for prediction)", min_value=1, max_value=100)

if st.button("Predict"):
    prediction = model.predict([[day_input]])
    st.write(f"Predicted cases for day {day_input}: **{int(prediction[0])}**")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot historical cases
    ax.plot(df_historical["day"], df_historical["cases"], marker='o', linestyle='-', color='blue', label="Historical Cases")
    
    # Plot prediction
    ax.scatter(day_input, prediction[0], color='red', marker='o', s=100, label=f"Prediction (Day {day_input})")
    
    ax.set_xlabel("Day")
    ax.set_ylabel("Number of Cases")
    ax.set_title("COVID-19 Cases Trend & Prediction")
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)
