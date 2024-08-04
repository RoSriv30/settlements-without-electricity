import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn.functional import relu, pad

class DoubleConvHelper(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Module that implements 
             a convolution
            - a batch norm
            - relu
            - another convolution
            - another batch norm
        
        Input:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            mid_channels (int): number of channels to use in the intermediate layer    
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)



    def forward(self, x):
        """Forward pass through the layers of the helper block"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x


class UNETPlusPlusTransfer(nn.Module):
    def __init__(self, input_channels, output_channels, embedding_size: int = 64, scale_factor=50, **kwargs):
        """
        Initializes a UNet++ model for transfer learning.

        This model is based on the UNet++ architecture. It loads an encoder from Torch Hub,
        replaces the first and last layers, and adds a final pooling layer to adapt it for
        a specific problem. The first convolutional layer of the encoder is replaced by an
        input_channels -> 64 Conv2d with no bias. The last layer is replaced with a
        512 -> output_channels Conv2d layer. Additionally, a max pooling layer is added
        to reduce the resolution of the output to match the desired scale factor.

        Input:
            input_channels (int): Number of input channels of the image, e.g., for RGB images.
            output_channels (int): Number of output channels of the prediction.
                The prediction shape is (batch, output_channels, width//scale_factor, height//scale_factor).
            embedding_size (int): Size of the embedding space.
            scale_factor (int): Number of input pixels that map to 1 output pixel.
                For example, if the input is 800x800 and the output is 16x6,
                then the scale factor is 800/16 = 50.
            **kwargs: Additional keyword arguments.
        """

        super().__init__()
        

        self.conv = DoubleConvHelper(input_channels, embedding_size)
        self.model = smp.UnetPlusPlus(  
            encoder_name="efficientnet-b7",     # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=embedding_size,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=output_channels,
            encoder_depth = 2, 
            decoder_channels=(256, 128)                      # model output channels (number of classes in your dataset)
        )
        

        self.maxpool = nn.MaxPool2d(kernel_size=scale_factor)


      
    def forward(self, x):
        """
        Runs predictions on the modified FCN resnet
        followed by pooling

        Runs predictions on the modified UNet++ followed by pooling.

        Input:
            x (torch.Tensor): Input image tensor of shape
                (batch, self.input_channels, width, height),
                with width and height divisible by self.scale_factor.

        Output:
            torch.Tensor: Predicted labels tensor of size
                (batch, self.output_channels, width//self.scale_factor, height//self.scale_factor).
        """
        # Perform forward pass through modified UNet++
        x = self.conv.forward(x)
        x = self.model(x)
        
        # Apply final pooling
        x = self.maxpool(x)

        return x