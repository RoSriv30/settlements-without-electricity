"""
This code is adapted from the U-Net paper. See details in the paper:
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. 
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50
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


class Encoder(nn.Module):
    """ Downscale using the maxpool then call double conv helper. """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        print("Encoder", in_channels, out_channels)
        self.encode = nn.Sequential(
        nn.MaxPool2d(kernel_size=2),
        DoubleConvHelper(in_channels, out_channels))


    def forward(self, x):
        x = self.encode(x)
        
        
        return x


class Decoder(nn.Module):
    """ Upscale using ConvTranspose2d then call double conv helper. """
    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        print("Decoder", in_channels, out_channels)

        if mid_channels == None:
            mid_channels = out_channels
        
        self.convT = nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvHelper(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """ 
        1) x1 is passed through either the upsample or the convtranspose2d
        2) The difference between x1 and x2 is calculated to account for differences in padding
        3) x1 is padded (or not padded) accordingly
        4) x2 represents the skip connection
        5) Concatenate x1 and x2 together with torch.cat
        6) Pass the concatenated tensor through a doubleconvhelper
        7) Return output
        """
        # step 1: replace x1 with the upsampled version of x1
        x1 = self.convT(x1)

        # input is Channel Height Width, step 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # step 3
        x1 = pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        # step 4 & 5: Concatenate x1 and x2
        x = torch.cat([x1, x2], dim=1)  # Assuming the channels are along dim=1

        # step 6: Pass the concatenated tensor through a doubleconvhelper
        x = self.conv(x)
    

        # step 7: return output
        return x
    
class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_encoders: int = 2,
                 embedding_size: int = 64, scale_factor: int = 50, **kwargs):
        """
        Implements a unet, a network where the input is downscaled
        down to a lower resolution with a higher amount of channels,
        but the residual images between encoders are saved
        to be concatednated to later stages, creatin the
        nominal "U" shape.

        In order to do this, we will need n_encoders-1 encoders. 
        The first layer will be a doubleconvhelper that
        projects the in_channels image to an embedding_size
        image of the same size.

        After that, n_encoders-1 encoders are used which halve
        the size of the image, but double the amount of channels
        available to them (i.e, the first layer is 
        embedding_size -> 2*embedding size, the second layer is
        2*embedding_size -> 4*embedding_size, etc)

        The decoders then upscale the image and halve the amount of
        embedding layers, i.e., they go from 4*embedding_size->2*embedding_size.

        We then have a maxpool2d that scales down the output to by scale_factor,
        as the input for this architecture must be the same size as the output,
        but our input images are 800x800 and our output images are 16x16.

        Input:
            in_channels: number of input channels of the image
            of shape (batch, in_channels, width, height)
            out_channels: number of output channels of prediction,
            prediction is shape (batch, out_channels, width//scale_factor, height//scale_factor)
            n_encoders: number of encoders to use in the network (implementing this parameter is
            optional, but it is a good idea to have it as a parameter in case you want to experiment,
            if you do not implement it, you can assume n_encoders=2)
            embedding_size: number of channels to use in the first layer
            scale_factor: number of input pixels that map to 1 output pixel,
            for example, if the input is 800x800 and the output is 16x6
            then the scale factor is 800/16 = 50.
        """
        super(UNet, self).__init__()
        self.n_encoders = n_encoders
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        self.conv = DoubleConvHelper(self.in_channels, self.embedding_size)
        
        # Encoder layers
        self.encoders = nn.ModuleList([Encoder(self.embedding_size, 2*self.embedding_size)])
        for i in range(self.n_encoders - 1):
            new_in_channel_size = self.embedding_size * 2
            new_out_channel_size = self.embedding_size * 4
            self.encoders.append(Encoder(new_in_channel_size, new_out_channel_size))

        # Decoder layers
        self.decoders = nn.ModuleList()
        for i in range(n_encoders, 0, -1):
            if i == 1:
                self.decoders.append(Decoder(embedding_size * 2**i, out_channels, embedding_size * 2**(i - 1)))
            else:
                self.decoders.append(Decoder(embedding_size * 2**i, embedding_size * 2**(i - 1)))
        
        
        # Maxpool to scale down the output
        self.maxpool = nn.MaxPool2d(kernel_size=scale_factor)


    def forward(self, x):
        """
            The image is passed through the encoder layers,
            making sure to save the residuals in a list.

            Following this, the residuals are passed to the
            decoder in reverse, excluding the last residual
            (as this is used as the input to the first decoder).

            The ith decoder should have an input of shape
            (batch, some_embedding_size, some_width, some_height)
            as the input image and
            (batch, some_embedding_size//2, 2*some_width, 2*some_height)
            as the residual.
        """
        residuals = []
        x = self.conv.forward(x)
        residuals.append(x)
        
        
        # Encoder pass
        for encoder in self.encoders:
            x = encoder.forward(x)
            residuals.append(x)

        # Decoder pass
        for i, decoder in enumerate(self.decoders):
            x = decoder.forward(x, residuals[-(i + 2)])

        # Scale down the output
        x = self.maxpool(x)
        
        return x

