import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('TkAgg')

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
def create_dataloader(batch_size):
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# Create model
class NeuralNet(nn.Module):
    def __init__(self, stddev):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=stddev)
        nn.init.zeros_(self.fc1.bias)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 7)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Training and validation function
def train_and_validate(stddev, batch_size):
    model = NeuralNet(stddev)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_loader, test_loader = create_dataloader(batch_size)

    loss_history = []
    accuracy_history = []

    best_accuracy = 0
    best_weights = None

    for epoch in range(100):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            result = model(inputs)
            loss = criterion(result, labels)
            loss.backward()
            optimizer.step()

        # validation model
        model.eval()
        val_correct = 0
        val_total = 0
        model_loss = 0
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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = model.state_dict()

        print(f'Epoch [{epoch + 1:03d}/100], Loss: {model_loss:.4f}, Accuracy: {accuracy:.4f}')

    return loss_history, accuracy_history, best_weights


# Experiment with different stddev and batch sizes
stddev_values = [0.01, 0.05, 0.1]
batch_sizes = [16, 32, 64]

results = {}
for stddev in stddev_values:
    for batch_size in batch_sizes:
        print(f"-----------Training with stddev={stddev} and batch_size={batch_size}-----------")
        loss_history, accuracy_history, best_weights = train_and_validate(stddev, batch_size)
        results[(stddev, batch_size)] = (loss_history, accuracy_history, best_weights)


best_key = max(results, key=lambda k: max(results[k][1]))
best_loss, best_accuracy, best_weights = results[best_key]


plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(best_loss, label='Loss')
plt.title(f'Loss vs. Epochs (stddev={best_key[0]}, batch_size={best_key[1]})')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(best_accuracy, label='Accuracy')
plt.title(f'Accuracy vs. Epochs (stddev={best_key[0]}, batch_size={best_key[1]})')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot batch size effect
plt.figure(figsize=(10, 5))
for stddev in stddev_values:
    accuracies = [max(results[(stddev, batch_size)][1]) for batch_size in batch_sizes]
    plt.plot(batch_sizes, accuracies, label=f'stddev={stddev}')
plt.title('Batch Size Effect on Accuracy')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot effect of stddev
plt.figure(figsize=(10, 5))
for batch_size in batch_sizes:
    accuracies = [max(results[(stddev, batch_size)][1]) for stddev in stddev_values]
    plt.plot(stddev_values, accuracies, label=f'batch_size={batch_size}')
plt.title('Effect of stddev on Accuracy')
plt.xlabel('stddev')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the best weights
if not os.path.exists('weights'):
    os.makedirs('weights')

torch.save(best_weights, f'weights/best_model_stddev_{best_key[0]}_batch_{best_key[1]}.pth')
print(f"Best model saved to 'weights/best_model_stddev_{best_key[0]}_batch_{best_key[1]}.pth'")
