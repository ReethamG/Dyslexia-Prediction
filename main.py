import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Set the path to your dataset
train_data_dir = '/Users/reethamgubba/Programming Projects/How To Read + Dyslexia Detector Game/Handwriting Dataset/Train'
test_data_dir = '/Users/reethamgubba/Programming Projects/How To Read + Dyslexia Detector Game/Handwriting Dataset/Test'

# HyperParameters
input_shape = (1, 29, 29) 
num_classes = 3
batch_size = 32
epochs = 5

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((29, 29)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageFolder(train_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(test_data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # Update the input size for the first fully connected layer
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor using the correct size
        x = self.fc(x)
        return x

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005) #The learning rate is a hyperparamter

# Train the model
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on the test set: {100 * correct / total}%")

# Save the trained model
# torch.save(model.state_dict(), 'dyslexia_detection_model.pth')
