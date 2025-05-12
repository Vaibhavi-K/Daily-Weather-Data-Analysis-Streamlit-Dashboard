
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the datasets
temperature = pd.read_csv("temperature.csv", index_col=0, parse_dates=True)
humidity = pd.read_csv("humidity.csv", index_col=0, parse_dates=True)

# Convert index to datetime
temperature.index = pd.to_datetime(temperature.index)
humidity.index = pd.to_datetime(humidity.index)

st.title(" Daily Weather Data Analysis Dashboard")

st.subheader("Temperature Trend Over Time")

city = st.selectbox("Select City for Temperature Trend", temperature.columns)
date_range = st.date_input("Select Date Range", [temperature.index.min(), temperature.index.max()])

# Filter based on selected dates
filtered_temp = temperature[city].loc[date_range[0]:date_range[1]]
fig1, ax1 = plt.subplots()
ax1.plot(filtered_temp.index, filtered_temp.values, label=city, color='orange')
ax1.set_xlabel("Date")
ax1.set_ylabel("Temperature")
ax1.set_title(f"Temperature Trend")
ax1.legend()
st.pyplot(fig1)

st.subheader("Humidity vs Temperature")

city2 = st.selectbox("Select City for Scatter Plot", temperature.columns, key="scatter")
merged = pd.DataFrame({
    'Temperature': temperature[city2],
    'Humidity': humidity[city2]
}).dropna()

fig2, ax2 = plt.subplots()
ax2.scatter(merged['Humidity'], merged['Temperature'], alpha=0.6, color='blue')
ax2.set_xlabel("Humidity")
ax2.set_ylabel("Temperature")
ax2.set_title(f"Humidity vs Temperature")
st.pyplot(fig2)

st.subheader("Rainfall Distribution")

weather_desc = pd.read_csv("weather_description.csv", index_col=0, parse_dates=True)

# Example: Count of 'rain' mentions per day
rain_counts = weather_desc.apply(lambda x: x.str.contains("rain", case=False).sum(), axis=1)

fig3, ax3 = plt.subplots()
rain_counts.plot(kind="hist", bins=30, ax=ax3, color='skyblue')
ax3.set_title("Histogram of Rain Mentions Per Day ")
ax3.set_xlabel("Rain Mentions")
ax3.set_ylabel("Frequency")
st.pyplot(fig3)


st.subheader("Predict Temperature from Humidity")

# Drop missing values
X = merged[['Humidity']].dropna()
y = merged.loc[X.index, 'Temperature']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plot
fig4, ax4 = plt.subplots()
ax4.scatter(y_test, y_pred, alpha=0.7, color='green')
ax4.set_xlabel("Actual Temperature")
ax4.set_ylabel("Predicted Temperature")
ax4.set_title("Actual vs Predicted Temperature")
st.pyplot(fig4)
