import os
import torch
import torch.utils.data as data
import torch.nn as nn
from PIL import Image

LIVER_FOLDER_DATA_NAMES = {
    'images': 'images',
    'masks': 'masks'
} 



class SegmentDataset(data.Dataset):
    def __init__(self, path, transform_img=None, transform_mask=None):
        self.path = path
        self.transform_img = transform_img
        self.transform_mask = transform_mask

        path = os.path.join(self.path, LIVER_FOLDER_DATA_NAMES['images'])
        list_files = os.listdir(path)
        self.length = len(list_files)
        self.images = list(map(lambda _x: os.path.join(path, _x), list_files))
        
        path = os.path.join(self.path, LIVER_FOLDER_DATA_NAMES['masks'])
        list_files = os.listdir(path)
        self.masks = list(map(lambda _x: os.path.join(path, _x), list_files))

    def __getitem__(self, item):
        path_img, path_mask = self.images[item], self.masks[item]
        img = Image.open(path_img).convert('RGB')
        mask = Image.open(path_mask).convert('L') #grayscale

        if self.transform_img:
            img = self.transform_img(img)
        
        if self.transform_mask:
            mask = self.transform_mask(mask)
            mask[mask < 250] = 1
            mask[mask >= 250] = 0

        return img, mask

    def __len__(self):
        return self.length
    
    
class UNetModel(nn.Module):
    class TwoConvLayers(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels)
            )
        
        def forward(self, x):
            return self.model(x)
        
    class EncoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.block = UNetModel.TwoConvLayers(in_channels, out_channels)
            self.max_pool = nn.MaxPool2d(2)
        
        def forward(self, x):
            x = self.block(x)
            y = self.max_pool(x)
            return y, x
    
    class DecoderBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.tranpose_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=True)
            self.block = UNetModel.TwoConvLayers(in_channels, out_channels)

        def forward(self, x, y):
            x = self.tranpose_conv(x)
            output = torch.cat([x, y], dim=1)
            output = self.block(output)
            return output
            
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.encoder1 = self.EncoderBlock(in_channels, 64)
        self.encoder2 = self.EncoderBlock(64, 128)
        self.encoder3 = self.EncoderBlock(128, 256)
        self.encoder4 = self.EncoderBlock(256, 512)

        self.conv = self.TwoConvLayers(512, 1024)

        self.decoder1 = self.DecoderBlock(1024, 512)
        self.decoder2 = self.DecoderBlock(512, 256)
        self.decoder3 = self.DecoderBlock(256, 128)
        self.decoder4 = self.DecoderBlock(128, 64)

        self.last_layer = nn.Conv2d(64, num_classes, 1, stride=1, padding=0)
    
    def forward(self, inp):
        y1, x = self.encoder1(inp)
        y2, x = self.encoder2(x)
        y3, x = self.encoder3(x)
        y4, x = self.encoder4(x) 

        x = self.conv(x)

        x = self.decoder1(x, y4)
        x = self.decoder2(x, y3)
        x = self.decoder3(x, y2)
        x = self.decoder4(x, y1)

        x = self.last_layer(x)

        return x
    