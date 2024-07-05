import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Specify the ticker symbol and date range
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2019-12-31'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter stock Ticker', 'AAPL')

# Fetch historical data
df = yf.download(user_input, start=start_date, end=end_date)

# Display the first few rows of the DataFrame
st.subheader('Data from 2010-2019')
st.write(df.describe())

# Plot Closing Price vs Time
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.title('Closing Price vs Time')
st.pyplot(fig)

# Plot Closing Price vs Time with 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100-Day MA', color='orange')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Closing Price vs Time with 100MA')
st.pyplot(fig)

# Plot Closing Price vs Time with 100MA and 200MA
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100-Day MA', color='orange')
plt.plot(ma200, label='200-Day MA', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Closing Price vs Time with 100MA & 200MA')
st.pyplot(fig)

# Prepare training and testing data
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

# Scale the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Load the pre-trained model
model = load_model('keras_model.h5')

# Prepare the final dataset for prediction
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# Prepare test data
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict the prices
y_predicted = model.predict(x_test)

# Rescale the predicted data
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot the predictions vs original
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Prediction vs Original')
st.pyplot(fig2)
