import torch                       
import torch.nn as nn              
import torch.nn.functional as F    


class CNN(nn.Module):
    """CNN matching the assignment spec: input 3x64x64 -> 10 class scores."""

    def __init__(self):
        super(CNN, self).__init__()

        # Conv 1
        # 3 input channels (RGB) -> 16 filters, 3x3 kernel, stride 1, padding 1
        # With padding=1 and a 3x3 kernel, height and width stay the same (64x64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=1, padding=1)

        # Max pool
        # 2x2 window, stride 2 -> halves height and width
        # The same pooling layer is reused after both convs 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv 2: 16 input channels -> 32 filters, 3x3 kernel, stride 1, padding 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, stride=1, padding=1)

        # Fully connected 1
        # flattened feature map (32 x 16 x 16 = 8192) -> 100 units
        self.fc1 = nn.Linear(32 * 16 * 16, 100)

        # Fully connected 2
        # 100 units -> 10 output classes
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # x shape: [B, 3, 64, 64]
        x = self.pool(F.relu(self.conv1(x)))   
        x = self.pool(F.relu(self.conv2(x)))   
        x = x.view(-1, 32 * 16 * 16)           
        x = F.relu(self.fc1(x))               
        x = self.fc2(x)                        
        return x                               