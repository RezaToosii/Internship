import pandas as pd  # Importing pandas for data manipulation
from sklearn.model_selection import train_test_split  # Importing train_test_split for splitting the dataset
from sklearn.preprocessing import StandardScaler  # Importing StandardScaler for feature scaling
from sklearn.linear_model import LogisticRegression  # Importing LogisticRegression for building the model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Importing metrics for evaluation
import seaborn as sns  # Importing seaborn for data visualization
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Reading the dataset from the CSV file
df = pd.read_csv('Q8_advertising.csv')

# Plotting a normalized histogram of the 'Age' column
plt.figure()
sns.histplot(df['Age'], kde=True, stat='density', bins=20, color='red')
plt.title('Normalized Histogram of Age')

# Setting the palette and style for seaborn plots
sns.set_palette("GnBu_d")
sns.set_style("whitegrid")

# Creating a joint plot for 'Age' vs 'Area Income'
sns.jointplot(data=df, x='Age', y='Area Income')

# Plotting a KDE plot for 'Daily Time Spent on Site' vs 'Age'
plt.figure()

df_temp = df[['Daily Time Spent on Site', 'Age']]
sns.kdeplot(data=df_temp)

# Creating a pair plot for the dataset with hue set to 'Clicked on Ad'
sns.pairplot(df, hue='Clicked on Ad')

# Displaying the plots
plt.show()

# Dropping unnecessary columns and separating features and labels
df_x = df.drop(columns=['Ad Topic Line', 'City', 'Country', 'Timestamp', 'Clicked on Ad'])
df_y = df[['Clicked on Ad']]

# Saving features and labels to CSV files
df_x.to_csv("train.csv")
df_y.to_csv("lable.csv")

# Reshaping the labels to a 1-dimensional array
df_y = df_y.values.reshape(-1, )

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_norm = scaler.fit_transform(df_x)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_norm, df_y, test_size=0.3, random_state=101)

# Creating and training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Printing the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Printing the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Printing the accuracy score
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

# Creating a DataFrame for the coefficients of the features and printing it
coefficients = pd.DataFrame(model.coef_[0], df_x.columns, columns=['Coefficient'])
print("\nCoefficients:")
print(coefficients)
