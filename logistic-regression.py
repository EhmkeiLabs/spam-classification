import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate Sample Data
# Assuming we have a simple dataset with a linear relationship
np.random.seed(0)
X = np.random.rand(100, 1) * 10 # Feature: Random values between 0 and 10
y = 2.5 * X + np.random.randn(100, 1) * 2 # Target: Linear relationship with noise

# Split the data into training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use the Linear Regression method
model = LinearRegression()
model.fit(X_train, y_train)

# Plot the sample chart with the training and test data and the fitted model
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression Model')
plt.legend()
plt.show()

# Retrieve model fit statistics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Interpretation of the results
# A lower MSE and an R-squared close to 1 indicate a good fit. You can tweak the model by adding more features, transforming existing features, or using regularization teechniques.