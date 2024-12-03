import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
import matplotlib.pyplot as plt

# Load the data from the .xlsx file
file_path = 'dian.xlsx'  # Update with your actual file path for dian.xlsx
data = pd.read_excel(file_path)  # No need for the 'engine' parameter for .xlsx files

# Display the first few rows to verify data
print(data.head())

# Select features and target variable
# Assuming 'AT' -> Temperature, 'V' -> Wind speed, 'AP' -> Atmospheric pressure, 'RH' -> Humidity, 'PE' -> Power output
features = data[['AT', 'V', 'AP', 'RH']]  # Feature columns
target = data['PE']  # Target column for prediction

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Convert y_test to DataFrame to retain index
y_test_df = y_test.reset_index(drop=True)

# Define functions to calculate evaluation metrics
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def relative_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rae = relative_absolute_error(y_true, y_pred)

    return mse, mae, mape, r2, rae

# Initialize a dictionary to store results
results = {}

# ------------------------- Linear Regression -------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
results['Linear Regression'] = evaluate_model(y_test, lr_predictions)

# Save Linear Regression predictions
lr_results = pd.DataFrame({'Actual': y_test, 'Prediction': lr_predictions})
lr_results.to_csv('lr_predictions.csv', index=False)

# ------------------------- Decision Tree -------------------------
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
results['Decision Tree'] = evaluate_model(y_test, dt_predictions)

# Save Decision Tree predictions
dt_results = pd.DataFrame({'Actual': y_test, 'Prediction': dt_predictions})
dt_results.to_csv('dt_predictions.csv', index=False)

# ------------------------- Random Forest -------------------------
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
results['Random Forest'] = evaluate_model(y_test, rf_predictions)

# Save Random Forest predictions
rf_results = pd.DataFrame({'Actual': y_test, 'Prediction': rf_predictions})
rf_results.to_csv('rf_predictions.csv', index=False)

# ------------------------- RNN -------------------------
# Reshape input for RNN
X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define and compile the RNN model
rnn_model = Sequential([
    SimpleRNN(20, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')

# Train the RNN model
rnn_model.fit(X_train_rnn, y_train, epochs=500, batch_size=1, validation_split=0.3)
rnn_predictions = rnn_model.predict(X_test_rnn).flatten()
results['RNN'] = evaluate_model(y_test, rnn_predictions)

# Save RNN predictions
rnn_results = pd.DataFrame({'Actual': y_test, 'Prediction': rnn_predictions})
rnn_results.to_csv('rnn_predictions.csv', index=False)

# ------------------------- Print results -------------------------
for model, metrics in results.items():
    print(f"{model} - MSE: {metrics[0]:.4f}, MAE: {metrics[1]:.4f}, MAPE: {metrics[2]:.4f}, R²: {metrics[3]:.4f}, RAE: {metrics[4]:.4f}")

# Create a 2x2 subplot layout
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot for Linear Regression
axs[0, 0].scatter(y_test_df, lr_predictions, color='steelblue', alpha=0.3, s=6)
axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='tomato', lw=2)  # Reference line
axs[0, 0].set_title('Linear Regression')
axs[0, 0].set_xlabel('Actual Value')
axs[0, 0].set_ylabel('Predicted Value')
axs[0, 0].grid(True)

# Plot for Decision Tree
axs[0, 1].scatter(y_test_df, dt_predictions, color='steelblue', alpha=0.3, s=6)
axs[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='tomato', lw=2)  # Reference line
axs[0, 1].set_title('Decision Tree')
axs[0, 1].set_xlabel('Actual Value')
axs[0, 1].set_ylabel('Predicted Value')
axs[0, 1].grid(True)

# Plot for Random Forest
axs[1, 0].scatter(y_test_df, rf_predictions, color='steelblue', alpha=0.3, s=6)
axs[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='tomato', lw=2)  # Reference line
axs[1, 0].set_title('Random Forest')
axs[1, 0].set_xlabel('Actual Value')
axs[1, 0].set_ylabel('Predicted Value')
axs[1, 0].grid(True)

# Plot for RNN
axs[1, 1].scatter(y_test_df, rnn_predictions, color='steelblue', alpha=0.3, s=6)
axs[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='tomato', lw=2)  # Reference line
axs[1, 1].set_title('RNN')
axs[1, 1].set_xlabel('Actual Value')
axs[1, 1].set_ylabel('Predicted Value')
axs[1, 1].grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()

# Show the plot
plt.show()
