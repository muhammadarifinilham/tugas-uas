#!/usr/bin/env python
# coding: utf-8

# Pengumpulan Data: Mengambil data harga Ethereum dari Yahoo Finance.
# 
# Praproses Data: Menormalisasi data dan membagi menjadi set pelatihan dan pengujian.
# 
# Membuat dan Melatih Model LSTM: Menggunakan Keras untuk membangun dan melatih model LSTM.
# 
# Evaluasi Model: Membuat prediksi dan menghitung RMSE untuk mengukur kinerja model.
# 
# Visualisasi Hasil: Membuat plot untuk membandingkan harga aktual dan harga yang diprediksi.

# # LANGKAH 1 Pengumpulan Data

# In[96]:


import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


# In[97]:


# Mengambil data historis Ethereum
data = yf.download('ETH-USD', start='2019-01-04', end='2025-01-01')


# In[98]:


# Menampilkan data
print(data.head())


# # LANGKAH 2 PRAPROSES DATA

# In[99]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Menggunakan harga penutupan saja
data = data[['Close']]

# Normalisasi data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Membagi data menjadi set pelatihan dan pengujian
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Membuat dataset untuk LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Mengubah bentuk input menjadi [samples, time steps, features] sesuai LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# # Langkah 3: Membuat dan Melatih Model LSTM

# In[100]:


from keras.models import Sequential
from keras.layers import LSTM, Dense

# Membuat model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Kompilasi model
model.compile(optimizer='adam', loss='mean_squared_error')

# Melatih model
model.fit(X_train, y_train, batch_size=1, epochs=1)



# # Langkah 4: Evaluasi Model

# In[101]:


# Membuat prediksi
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Mengembalikan skala data ke bentuk aslinya
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Menghitung RMSE
import math
from sklearn.metrics import mean_squared_error

train_rmse = math.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
test_rmse = math.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')


# # Langkah 5: Visualisasi Hasil

# In[102]:


train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

plt.figure(figsize=(14, 8))
plt.title('Harga ETH Prediksi vs Aktual')
plt.xlabel('Tanggal')
plt.ylabel('Harga Penutupan USD')
plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Data Asli')
plt.plot(data.index, train_predict_plot, label='Prediksi Data Latihan')
plt.plot(data.index, test_predict_plot, label='Prediksi Data Uji')
plt.legend()
plt.show()


# In[ ]:




