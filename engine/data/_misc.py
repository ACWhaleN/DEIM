"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
import torch
from torch import Tensor
from torchvision.tv_tensors import TVTensor

if '0.15.2' in importlib.metadata.version('torchvision'):
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']

elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')

from torchvision.tv_tensors import TVTensor
import torch

class Polygons(TVTensor):
    def __new__(
        cls,
        data,
        *,
        format: str,
        spatial_size: tuple = None,
        **kwargs,
    ):
        # 1. 先调用 TVTensor._to_tensor 或直接构造 Tensor
        tensor = cls._to_tensor(data,**kwargs)
        
        # 2. 转换为 Polygons 类型
        tensor = tensor.as_subclass(cls)
        
        # 3. 设置额外属性
        tensor.format = format
        tensor.spatial_size = spatial_size
        
        return tensor

    # 可选：覆盖 __init__ 进行额外初始化
    def __init__(
        self,
        data,
        *,
        format: str,
        spatial_size: tuple = None,
        **kwargs,
    ):
        # 如果已经在 __new__ 中设置了属性，这里可以省略
        pass




def convert_to_tv_tensor(
    tensor: torch.Tensor, 
    key: str, 
    box_format: str = 'xyxy', 
    poly_format: str = 'xy',
    spatial_size: tuple = None
) -> TVTensor:
    """
    Args:
        tensor: 输入张量
        key: 数据类型 ('boxes', 'masks', 'polys')
        box_format: 边界框格式 (仅当 key='boxes' 时有效)
        poly_format: 多边形格式 (仅当 key='polys' 时有效)
        spatial_size: 图像尺寸 (h, w)
    """
    assert key in ('boxes', 'masks', 'polys'), f"Invalid key: {key}"

    if key == 'boxes':
        return BoundingBoxes(
            tensor, 
            format=BoundingBoxFormat(box_format.upper()), 
            canvas_size=spatial_size
        )
    elif key == 'masks':
        return Mask(tensor)
    elif key == 'polys':
        # 必须传递为关键字参数
        return Polygons(
            tensor, 
            format=poly_format, 
            spatial_size=spatial_size
        )