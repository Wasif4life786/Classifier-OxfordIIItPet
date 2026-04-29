import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Unaltered Transform for testset, fitted to work in neural network
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 32

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Each convolution Layer scans the image and tries to detect more and more complex features
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        # Pooling settings
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4), # Slightly lower dropout
            nn.Linear(512, 37)
        )

    def forward(self, x):
        # Pool Image multiple times
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 112x112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 56x56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 28x28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # 14x14
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # 7x7

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # Changes GPU depending on device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TODO: consider changing num workers when on PC
    trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', transform=train_transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    net = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    
    # Using AdamW with weight decay for better regularization
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    
    epochs = 30
    # Scheduler with a floor (eta_min) to prevent learning rate from hitting zero
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)
    
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} done. LR: {scheduler.get_last_lr()[0]:.6f}")

    # puts network in evaluation mode, i.e. turns off dropout
    net.eval()
    test_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            test_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    final_acc = 100 * correct / total
    test_accuracies.append(final_acc)
    test_losses.append(test_loss / len(testloader))

    print("\nAccuracy against test:", f"{test_accuracies[-1]:.2f}%")