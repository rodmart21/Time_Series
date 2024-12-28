import pandas as pd
from prophet import Prophet
from sklearn.model_selection import train_test_split

# Assume 'combined_df' is your dataframe
# Step 1: Filter out relevant columns
features = ['Store_Size', 'Num_Employees', 'Num_Customers', 'Pct_On_Sale', 
            'Marketing_Cleaned', 'Near_Xmas', 'Near_BlackFriday', 'Holiday', 
            'DestinationEvent']

# Creating a new DataFrame with 'Sales' and selected features
combined_df_model = combined_df[['Sales'] + features].copy()

# Step 2: Prepare data for Prophet
# Prophet requires two columns: 'ds' for date and 'y' for the target variable
combined_df_model['ds'] = combined_df.index  # Assuming date is in the index
combined_df_model['y'] = combined_df_model['Sales']
combined_df_model = combined_df_model.drop(['Sales'], axis=1)

# Step 3: Split the data into train and test
train_size = int(len(combined_df_model) * 0.8)
train_combined_df = combined_df_model[:train_size]
test_combined_df = combined_df_model[train_size:]

# Step 4: Initialize and fit the Prophet model
model = Prophet()
for feature in features:
    model.add_regressor(feature)

model.fit(train_combined_df)

# Step 5: Make predictions
future = model.make_future_dataframe(periods=len(test_combined_df), freq='D')  # Adjust `periods` based on your frequency
# Ensure all columns in the future dataframe match the model's training columns
future = future.merge(combined_df_model[features], how='left', left_on='ds', right_index=True)

forecast = model.predict(future)

# Step 6: Evaluate the model
from sklearn.metrics import mean_squared_error

y_true = test_combined_df['y']
y_pred = forecast['yhat'][-len(test_combined_df):]  # Predictions for the test period

rmse = mean_squared_error(y_true, y_pred, squared=False)
print(f'RMSE: {rmse}')

# Plot the forecast
model.plot(forecast)
