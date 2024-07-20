import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting data
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for data normalization
from sklearn.linear_model import LinearRegression  # Importing LinearRegression for model building
import numpy as np  # Importing numpy for numerical operations

import seaborn as sns  # Importing seaborn for data visualization
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Reading the dataset into a DataFrame
df = pd.read_csv('Ecommerce_Customers')

# Setting the style and palette for seaborn plots
sns.set_palette("GnBu_d")
sns.set_style("whitegrid")

# Creating joint plots to visualize relationships between features and the target variable
sns.jointplot(data=df, x='Avg. Session Length', y='Yearly Amount Spent')
sns.jointplot(data=df, x='Time on App', y='Yearly Amount Spent')

# Creating a pairplot to visualize relationships between all pairs of features
sns.pairplot(data=df)

# Displaying the plots
plt.show()

# Preparing the features and target variable
df_x = df.drop(columns=['Email', 'Address', 'Avatar', 'Yearly Amount Spent'])  # Dropping non-feature columns
df_y = df[['Yearly Amount Spent']]  # Selecting the target variable

# Saving features and labels to CSV files (for later use if needed)
df_x.to_csv("train.csv")
df_y.to_csv("lable.csv")

# Reshaping the target variable to a 1-dimensional array
df_y = df_y.values.reshape(-1, )

# Normalizing the features using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(df_x)

# Splitting the normalized data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, df_y, test_size=0.3, random_state=101)

# Creating and training the linear regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Making predictions on the test data
pred = linear_model.predict(X_test)

# Plotting the actual vs predicted values
w = linear_model.coef_  # Coefficients of the features
b = linear_model.intercept_  # Intercept of the model
plt.scatter(pred, y_test, marker='x', c='r', label="Actual Value")
plt.plot(pred, pred, label="Predicted Value")

# Adding legend and displaying the plot
plt.legend()
plt.show()

# Calculating error metrics
test_error = y_test - pred
mean_abs_error = np.mean(np.abs(test_error))  # Mean Absolute Error
mean_squared_error = np.mean(test_error ** 2)  # Mean Squared Error
root_mean_squared_error = np.sqrt(mean_squared_error)  # Root Mean Squared Error

# Printing the error metrics
print(f'Mean ABS Error = {mean_abs_error}\nMean Squared Error = {mean_squared_error}\nRoot Mean Squared Error = '
      f'{root_mean_squared_error}\n')

# Extracting and displaying the coefficients of the features
coefficients = linear_model.coef_
features = df_x.columns

# Creating a DataFrame to display feature names and their corresponding coefficients
coef_table = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})

# Printing the coefficients table
print("Coefficients Table:")
print(coef_table)
