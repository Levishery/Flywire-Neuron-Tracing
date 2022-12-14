from .unet import UNet3D, UNet2D, UNetPlus3D
from .fpn import FPN3D
from .deeplab import DeepLabV3
from .my_zoo import UNet3D_MALA, FC3DDiscriminator, UNet3D_MALA_encoder, EdgeNetwork
from .byol_pytorch import BYOL, SSL
from .swin_unetr import SwinUNETR, SwinTransformer

__all__ = [
    'UNet3D',
    'UNetPlus3D',
    'UNet2D',
    'FPN3D',
    'DeepLabV3',
    'UNet3D_MALA',
    'FC3DDiscriminator',
    'UNet3D_MALA_encoder',
    'BYOL',
    'SSL',
    'SwinUNETR',
    'SwinTransformer',
    'EdgeNetwork'
]
