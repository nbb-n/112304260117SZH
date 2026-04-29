import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MNISTDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.tensor(self.data[idx], dtype=torch.float32)

class ImprovedDNN(nn.Module):
    def __init__(self):
        super(ImprovedDNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50):
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        
        train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, LR: {scheduler.get_last_lr()[0]:.6f}')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Best Val Acc: {best_val_acc:.2f}%')
            print()
    
    return best_val_acc

def generate_submission(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    
    submission = pd.DataFrame({'ImageId': range(1, len(predictions)+1), 'Label': predictions})
    submission.to_csv('submission.csv', index=False)
    print('Submission file generated: submission.csv')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    print('Loading data...')
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    X = train_data.drop('label', axis=1).values / 255.0
    y = train_data['label'].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    train_dataset = MNISTDataset(X_train, y_train)
    val_dataset = MNISTDataset(X_val, y_val)
    test_dataset = MNISTDataset(test_data.values / 255.0)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    model = ImprovedDNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    
    print('Training model...')
    best_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=50)
    
    print(f'\nBest Validation Accuracy: {best_acc:.2f}%')
    
    print('Loading best model...')
    model.load_state_dict(torch.load('best_model.pth'))
    
    print('Generating submission...')
    generate_submission(model, test_loader, device)

if __name__ == '__main__':
    main()