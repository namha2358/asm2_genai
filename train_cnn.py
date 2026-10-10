# train_cnn.py
import torch                                   
import torch.nn as nn                          
import torch.optim as optim                    
import torchvision                             
import torchvision.transforms as transforms    
from torch.utils.data import DataLoader        

from app.cnn_model import CNN                 

torch.manual_seed(42)                          

# Pick the fastest available device 
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
print(f"Using device: {device}")

# Resize 32x32 CIFAR10 images to 64x64 to match the model's input size
# convert to tensors (shape [3, 64, 64], values in [0, 1])
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
])

# Download CIFAR10 into ./data (train and test splits)
train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# DataLoaders feed the model batches of 32 images
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

EPOCHS = 5  

model = CNN().to(device)                                    
criterion = nn.CrossEntropyLoss()                          
optimizer = optim.Adam(model.parameters(), lr=0.0005)       

for epoch in range(EPOCHS):
    model.train()                                           
    running_loss, running_correct, running_total = 0.0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()                               
        outputs = model(inputs)                             
        loss = criterion(outputs, labels)                   
        loss.backward()                                    
        optimizer.step()                                    

        _, predicted = torch.max(outputs, 1)               
        running_correct += (predicted == labels).sum().item()
        running_total += labels.size(0)
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}: "
          f"loss={running_loss/len(train_loader):.4f}, "
          f"train acc={100*running_correct/running_total:.2f}%")

# Evaluate on the 10,000 held-out test images
model.eval()                                               
test_correct, test_total = 0, 0
with torch.no_grad():                                       
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
print(f"Test accuracy: {100*test_correct/test_total:.2f}%")

# Save only the learned weights (state_dict)
# moved to CPU so API and Docker container can load them on any machine
torch.save(model.cpu().state_dict(), "app/cnn_cifar10.pth")
print("Saved weights to app/cnn_cifar10.pth")