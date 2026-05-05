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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, 7,stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(64, 128,stride=2)
        self.layer2 = ResidualBlock(128, 256,stride=2)
        self.layer3 = ResidualBlock(256, 512,stride=2)
        self.layer4 = ResidualBlock(512, 1024,stride=2)

        # Pooling settings
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 37)
        )

    def forward(self, x):
        # Pool Image multiple times
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#TODO: Check if mixup with low dropout or label_smoothing with high dropout is superior
def mixup_data(x, y, alpha=0.1):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == '__main__':
    # Changes GPU depending on device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # TODO: consider changing num workers when on PC
    trainset = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', transform=train_transform, download=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.OxfordIIITPet(root='./data', split='test', transform=test_transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

    net = NeuralNetwork().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    #TODO: Consider changing weight_decay
    optimizer = optim.AdamW(net.parameters(), lr=0.0003, weight_decay=0.1)
    
    epochs = 30
    # Scheduler with a floor (eta_min) to prevent learning rate from hitting zero
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.005,steps_per_epoch=len(trainloader), epochs=epochs)
    
    train_losses, test_losses, test_accuracies = [], [], []

    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_correct, total = 0.0, 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = mixup_criterion(criterion,outputs, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            scheduler.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = train_correct / total * 100
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Training Accuracy: {epoch_acc:.2f}%")

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