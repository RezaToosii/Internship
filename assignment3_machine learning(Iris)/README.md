
# SVM Classification with Iris Dataset

This project demonstrates the use of the Support Vector Machine (SVM) algorithm for classification using the famous Iris dataset. The objective is to classify different species of the Iris flower based on given features.

## Dataset

The Iris dataset contains the following columns:
- sepal_length
- sepal_width
- petal_length
- petal_width
- species

## Project Workflow

1. **Importing Libraries**: Import necessary libraries such as `sklearn`, `seaborn`, `matplotlib`, and `pandas`.
2. **Loading Data**: Load the Iris dataset using `seaborn`.
3. **Data Exploration**: Explore the dataset using `head()`, `info()`, and `describe()` methods.
4. **Data Visualization**:
   - Visualize the data using `seaborn`'s `pairplot`.
   - Generate KDE plots for different Iris species.
5. **Data Preprocessing**:
   - Split the data into training and testing sets using `train_test_split`.
6. **Model Training and Evaluation**:
   - Train an SVM model using `SVC`.
   - Evaluate the model using accuracy score, confusion matrix, and classification report.
7. **Hyperparameter Tuning**: Use `GridSearchCV` to find the best hyperparameters for the SVM model.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- seaborn
- matplotlib
- scikit-learn

You can install the required libraries using `pip`:

```bash
pip install pandas seaborn matplotlib scikit-learn
```

## Usage

1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/iris-svm-classification.git
```

2. **Navigate to the Project Directory**:

```bash
cd iris-svm-classification
```

3. **Run the Script**:

Make sure you have the dataset accessible. Then, run the script:

```bash
python svm_classification.py
```

## Code Explanation

### Importing Libraries

```python
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
```

### Loading and Exploring Data

```python
iris = sns.load_dataset('iris')
print(iris.loc[50])
print(iris.head())
print(iris.info())
print(iris.describe())
sns.pairplot(iris, hue='species', palette='Set1')
```

### Visualizing Data

```python
# KDE plots for different Iris species
setosa = iris[iris['species'] == 'setosa']
plt.figure(figsize=(10, 6))
sns.kdeplot(x=setosa['sepal_width'], y=setosa['sepal_length'], cmap='plasma', fill=True, bw_adjust=1)
plt.xlabel('sepal_width')
plt.ylabel('sepal_length')
plt.show()

virginica = iris[iris['species'] == 'virginica']
plt.figure(figsize=(10, 6))
sns.kdeplot(x=virginica['sepal_width'], y=virginica['sepal_length'], cmap='plasma', fill=True, bw_adjust=1)
plt.xlabel('sepal_width')
plt.ylabel('sepal_length')
plt.show()

versicolor = iris[iris['species'] == 'versicolor']
plt.figure(figsize=(10, 6))
sns.kdeplot(x=versicolor['sepal_width'], y=versicolor['sepal_length'], cmap='plasma', fill=True, bw_adjust=1)
plt.xlabel('sepal_width')
plt.ylabel('sepal_length')
plt.show()
```

### Splitting Data

```python
x = iris.drop('species', axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
```

### Training and Evaluating the Model

```python
svm = SVC(kernel='rbf')
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

df = pd.DataFrame({
    'Label': y_test,
    'Prediction': y_pred
})

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
```

### Hyperparameter Tuning

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}
model = SVC()

grid_search = GridSearchCV(model, param_grid, refit=True, verbose=2, cv=5)
grid_search.fit(x_train, y_train)

print(f"Best parameters found:\n{grid_search.best_params_}")
print(f"Best estimator:\n{grid_search.best_estimator_}")

y_pred = grid_search.predict(x_test)

df = pd.DataFrame({
    'Label': y_test,
    'Prediction': y_pred
})

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy after tuning: {accuracy}')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
```