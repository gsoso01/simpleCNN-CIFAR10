from torch import nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, in_height, in_width, out_size) -> None:
        super().__init__()
        self.conv1 = self.conv_block(in_channels, out_channels * 2, kernel_size)
        self.conv2 = self.conv_block(out_channels * 2, out_channels * 4, kernel_size)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 10, kernel_size, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels * 10) # No pooling
        )
        self.dropout = nn.Dropout(p=0.3)

        conv_out_size = self.calculate_conv_output(in_height, in_width, kernel_size)
        flattened_size = out_channels * 10 * conv_out_size[0] * conv_out_size[1]
        
        self.fc1 = nn.Linear(flattened_size, 128)
        self.fc2 = nn.Linear(128, out_size)
    
    def conv_block(self, in_channels, out_channels, kernel_size):
        """Helper function to create a conv -> relu -> batch_norm -> pool block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.MaxPool2d(kernel_size=2)
        )
    
    def calculate_conv_output(self, height, width, kernel_size):
        """Helper function to calculate the output dimensions after conv and pooling layers."""
        def conv_out_size(size, kernel_size):
            return (size - (kernel_size - 1) - 1) // 1 + 1
        
        def pool_out_size(size):
            return size // 2
        
        conv1_out_h = pool_out_size(conv_out_size(height, kernel_size))
        conv1_out_w = pool_out_size(conv_out_size(width, kernel_size))
        
        conv2_out_h = pool_out_size(conv_out_size(conv1_out_h, kernel_size))
        conv2_out_w = pool_out_size(conv_out_size(conv1_out_w, kernel_size))
        
        conv3_out_h = conv_out_size(conv2_out_h, kernel_size)
        conv3_out_w = conv_out_size(conv2_out_w, kernel_size)
        
        return conv3_out_h, conv3_out_w
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits