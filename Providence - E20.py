#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# If you don't have the required packages installed, uncomment the lines below:
# !pip install numpy pandas matplotlib scikit-learn


# In[5]:


# Example dataset with dates from 8/1/2023 to 7/1/2024 and corresponding prices
data = {
    'Date': pd.date_range(start='2023-08-01', end='2024-07-01', freq='MS'),
    'Price': [369109, 373016, 376641, 380081, 383743, 386138, 388309, 390858, 394667, 398732, 401940, 405059]
}

df = pd.DataFrame(data)


# In[6]:


# Add a column representing the month number since the start of 2023
df['Month_Number'] = np.arange(1, len(df) + 1)

# Features (X) and target (y)
X = df[['Month_Number']]
y = df['Price']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict prices using the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[9]:


# Predict the price for 9/1/2024 (14th month)
price_2024 = model.predict([[14]])
print(f'Predicted price for 9/1/2024: ${price_2024[0]:,.2f}')


# In[11]:


# Predict the price for 3/1/2025 (20th month)
price_2025 = model.predict([[20]])
print(f'Predicted price for 3/1/2025: ${price_2025[0]:,.2f}')


# In[12]:


# Average price over the past 12 months
average_price = df['Price'].mean()
print(f'Average price over the past 12 months: ${average_price:,.2f}')

# Highest price over the past 12 months
highest_price = df['Price'].max()
print(f'Highest price over the past 12 months: ${highest_price:,.2f}')

# Lowest price over the past 12 months
lowest_price = df['Price'].min()
print(f'Lowest price over the past 12 months: ${lowest_price:,.2f}')


# In[18]:


# Plotting the historical prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Historical Prices', marker='o')

# Plotting the regression line (extrapolated to include September 2024)
extended_dates = pd.date_range(start='2023-08-01', end='2024-07-01', freq='MS')
extended_months = np.arange(1, len(extended_dates) + 1)
plt.plot(extended_dates, model.predict(extended_months.reshape(-1, 1)), label='Predicted Prices (Linear Regression)', linestyle='--')

# Marking the predicted price for 9/1/2024
plt.scatter(pd.Timestamp('2024-09-01'), price_2024, color='red', label=f'Predicted Price for 9/1/2024: ${price_2024[0]:,.2f}')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('2023-2024 Providence Single Family Monthly Prices with 9/1/2024 Prediction')
plt.legend()
plt.grid(True)
plt.show()


# In[14]:


# Plotting the historical prices
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Price'], label='Historical Prices', marker='o')

# Plotting the regression line (extrapolated to include March 2025)
extended_dates = pd.date_range(start='2023-08-01', end='2024-07-01', freq='MS')
extended_months = np.arange(1, len(extended_dates) + 1)
plt.plot(extended_dates, model.predict(extended_months.reshape(-1, 1)), label='Predicted Prices (Linear Regression)', linestyle='--')

# Marking the predicted price for 3/1/2025
plt.scatter(pd.Timestamp('2025-03-01'), price_2024, color='red', label=f'Predicted Price for 3/1/2025: ${price_2025[0]:,.2f}')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('2023-2024 Monthly Prices with 3/1/2025 Prediction')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




