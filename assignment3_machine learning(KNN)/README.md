
# KNN Classification Project

This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm for classification. The dataset used contains various features and a target class. The objective is to predict the target class based on the given features.

## Dataset

The dataset contains the following columns:
- XVPM
- GWYH
- TRAT
- TLLZ
- IGGA
- HYKR
- EDFS
- GUUB
- MGJM
- JHZC
- TARGET CLASS

## Project Workflow

1. **Importing Libraries**: Import necessary libraries such as `pandas`, `numpy`, `matplotlib.pyplot`, and `seaborn`.
2. **Loading Data**: Load the dataset using `pandas`.
3. **Data Exploration**: Explore the dataset using `head()`, `info()`, and `describe()` methods.
4. **Data Visualization**: Visualize the data using `seaborn`'s `pairplot`.
5. **Data Preprocessing**:
   - Standardize the data using `StandardScaler`.
   - Split the data into training and testing sets using `train_test_split`.
6. **Model Training and Evaluation**:
   - Train a KNN model using `KNeighborsClassifier`.
   - Evaluate the model using accuracy score, confusion matrix, and classification report.
7. **Error Rate Analysis**: Analyze the error rate for different values of K to find the optimal K value.

## Installation

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Clone the Repository**:

```bash
git clone https://github.com/your-username/knn-classification-project.git
```

2. **Navigate to the Project Directory**:

```bash
cd knn-classification-project
```

3. **Run the Script**:

Make sure you have the dataset (`KNN_Project_Data.csv`) in the project directory. Then, run the script:

```bash
python knn_classification.py
```

## Code Explanation

### Importing Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Loading and Exploring Data

```python
data = pd.read_csv('KNN_Project_Data')
print(data.head())
print(data.info())
print(data.describe())
```

### Data Visualization

```python
sns.pairplot(data, hue='TARGET CLASS', palette='bwr')
```

### Data Preprocessing

```python
scaler = StandardScaler()
scaler.fit(data)
data_scale = scaler.transform(data)
dataFrame = pd.DataFrame(data_scale, columns=['XVPM', 'GWYH', 'TRAT','TLLZ','IGGA','HYKR','EDFS','GUUB','MGJM','JHZC','TARGET CLASS'])
print(dataFrame.head())
```

### Splitting Data

```python
x = dataFrame.drop('TARGET CLASS', axis=1)
y = dataFrame['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
```

### Training and Evaluating the Model

```python
model = KNeighborsClassifier(n_neighbors=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
```

### Error Rate Analysis

```python
error = []
for k in range(1, 41):
    k_model = KNeighborsClassifier(k)
    k_model.fit(x_train, y_train)
    y_pred = k_model.predict(x_test)
    wrong_pred_avg = np.mean(y_test != y_pred)
    error.append(wrong_pred_avg)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 41), error, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=8)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```

### Final Model Evaluation

```python
print(f'WITH K=36\n\n-------\n\nScore:{accuracy}\n\n-------\n\nconfusion_matrix:\n\n{conf_matrix}\n\n-------\n\nclassification_report:\n\n{class_report}')
```