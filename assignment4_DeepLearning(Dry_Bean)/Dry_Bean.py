import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
data = pd.read_csv('Dry_Bean_Dataset.csv')

# Fix data for model
class_mapping = {'SEKER': 0, 'BARBUNYA': 1, 'BOMBAY': 2, 'CALI': 3, 'HOROZ': 4, 'SIRA': 5, 'DERMASON': 6}
data['Class'] = data.iloc[:, -1].map(class_mapping)
data.to_csv('transformed_Dataset.csv', index=False)

# Load correct data
data = pd.read_csv('transformed_Dataset.csv')
x = data.drop('Class', axis=1)
y = data['Class']

# split data for train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

# Normalize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Convert data to tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Create model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc1.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 7)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# define model
model = NeuralNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_history = []
accuracy_history = []

# Train the model
for epoch in range(100):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        result = model(inputs)
        loss = criterion(result, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        value, predicted = torch.max(result.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # validation model
    model.eval()
    model_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            model_loss += loss.item()
            value, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    model_loss /= len(test_loader)
    accuracy = val_correct / val_total
    loss_history.append(model_loss)
    accuracy_history.append(accuracy)

    print(f'Epoch [{epoch + 1}/{100}], Loss: {model_loss:.4f}, Accuracy: {accuracy:.4f}')

# in next update 'add plots' , 'save weight ' , 'user can test with new data' , 'text different stddev'
