"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import random
from PIL import Image

from .._misc import convert_to_tv_tensor
from ...core import register


@register()
class Mosaic(T.Transform):
    """
    Applies Mosaic augmentation to a batch of images. Combines four randomly selected images
    into a single composite image with randomized transformations.
    """

    def __init__(self, output_size=320, max_size=None, rotation_range=0, translation_range=(0.1, 0.1),
                 scaling_range=(0.5, 1.5), probability=1.0, fill_value=114, use_cache=True, max_cached_images=50,
                 random_pop=True) -> None:
        """
        Args:
            output_size (int): Target size for resizing individual images.
            rotation_range (float): Range of rotation in degrees for affine transformation.
            translation_range (tuple): Range of translation for affine transformation.
            scaling_range (tuple): Range of scaling factors for affine transformation.
            probability (float): Probability of applying the Mosaic augmentation.
            fill_value (int): Fill value for padding or affine transformations.
            use_cache (bool): Whether to use cache. Defaults to True.
            max_cached_images (int): The maximum length of the cache.
            random_pop (bool): Whether to randomly pop a result from the cache.
        """
        super().__init__()
        self.resize = T.Resize(size=output_size, max_size=max_size)
        self.probability = probability
        self.affine_transform = CustomRandomAffine(degrees=rotation_range, translate=translation_range,
                                                scale=scaling_range, fill=fill_value)
        # self.affine_transform = T.RandomAffine(degrees=rotation_range, translate=translation_range,
        #                                        scale=scaling_range, fill=fill_value)
        self.use_cache = use_cache
        self.mosaic_cache = []
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop

    def load_samples_from_dataset(self, image, target, dataset):
        """Loads and resizes a set of images and their corresponding targets."""
        # Append the main image
        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size  # torchvision >=0.17 is get_size
        image, target = self.resize(image, target)
        resized_images, resized_targets = [image], [target]
        max_height, max_width = get_size_func(resized_images[0])

        # randomly select 3 images
        sample_indices = random.choices(range(len(dataset)), k=3)
        for idx in sample_indices:
            # image, target = dataset.load_item(idx)
            image, target = self.resize(dataset.load_item(idx))
            height, width = get_size_func(image)
            max_height, max_width = max(max_height, height), max(max_width, width)
            resized_images.append(image)
            resized_targets.append(target)

        return resized_images, resized_targets, max_height, max_width

    def load_samples_from_cache(self, image, target, cache):
        image, target = self.resize(image, target)
        cache.append(dict(img=image, labels=target))

        if len(cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(cache) - 2)  # do not remove last image
            else:
                index = 0
            cache.pop(index)
        sample_indices = random.choices(range(len(cache)), k=3)
        mosaic_samples = [dict(img=cache[idx]["img"].copy(), labels=self._clone(cache[idx]["labels"])) for idx in
                          sample_indices]  # sample 3 images
        mosaic_samples = [dict(img=image.copy(), labels=self._clone(target))] + mosaic_samples

        get_size_func = F.get_size if hasattr(F, "get_size") else F.get_spatial_size
        sizes = [get_size_func(mosaic_samples[idx]["img"]) for idx in range(4)]
        max_height = max(size[0] for size in sizes)
        max_width = max(size[1] for size in sizes)

        return mosaic_samples, max_height, max_width

    def create_mosaic_from_cache(self, mosaic_samples, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=mosaic_samples[0]["img"].mode, size=(max_width * 2, max_height * 2), color=0)
        mosaic_target = []
        
        for i, sample in enumerate(mosaic_samples):
            img = sample["img"]
            target = sample["labels"]
            merged_image.paste(img, placement_offsets[i])
            
            # 处理boxes的偏移
            box_offset = torch.tensor([placement_offsets[i][0], placement_offsets[i][1], 
                                    placement_offsets[i][0], placement_offsets[i][1]], dtype=torch.float32)
            target['boxes'] = target['boxes'] + box_offset
            
            # 处理polys的偏移
            if 'polys' in target:
                off_x, off_y = placement_offsets[i]
                polys = target['polys']
                if polys.numel() > 0:
                    polys_reshaped = polys.view(-1, 4, 2)
                    polys_reshaped[..., 0] += off_x
                    polys_reshaped[..., 1] += off_y
                    target['polys'] = polys_reshaped.view(-1, 8)
            
            mosaic_target.append(target)
        
        # 合并目标
        merged_target = {}
        for key in mosaic_target[0]:
            if key == 'polys':
                merged_target[key] = torch.cat([t[key] for t in mosaic_target], dim=0)
            else:
                merged_target[key] = torch.cat([t[key] for t in mosaic_target], dim=0)
        
        return merged_image, merged_target
    
    def create_mosaic_from_dataset(self, images, targets, max_height, max_width):
        placement_offsets = [[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]]
        merged_image = Image.new(mode=images[0].mode, size=(max_width * 2, max_height * 2), color=0)
        
        # 构造每个图片的二维偏移量和四维box偏移量
        box_offsets = [torch.tensor([off_x, off_y, off_x, off_y], dtype=torch.float32) for (off_x, off_y) in placement_offsets]
        
        merged_target = {}
        for key in targets[0]:
            if key == 'boxes':
                values = [target[key] + box_offsets[i] for i, target in enumerate(targets)]
            elif key == 'polys':
                values = []
                for i, target in enumerate(targets):
                    off_x, off_y = placement_offsets[i]
                    polys = target[key]
                    if polys.numel() > 0:
                        polys_reshaped = polys.view(-1, 4, 2)
                        polys_reshaped[..., 0] += off_x
                        polys_reshaped[..., 1] += off_y
                        polys = polys_reshaped.view(-1, 8)
                    values.append(polys)
            else:
                values = [target[key] for target in targets]
            merged_target[key] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values
        
        return merged_image, merged_target

    @staticmethod
    def _clone(tensor_dict):
        return {key: value.clone() for (key, value) in tensor_dict.items()}

    def forward(self, *inputs):
        """
        Args:
            inputs (tuple): Input tuple containing (image, target, dataset).

        Returns:
            tuple: Augmented (image, target, dataset).
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # Skip mosaic augmentation with probability 1 - self.probability
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # Prepare mosaic components
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target,dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # Clamp boxes and convert target formats
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(mosaic_target['boxes'], 'boxes', box_format='xyxy',
                                                          spatial_size=mosaic_image.size[::-1])
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(mosaic_target['masks'], 'masks')
        if 'polys' in mosaic_target:
            mosaic_target['polys'] = convert_to_tv_tensor(mosaic_target['polys'], key='polys',spatial_size=image.size[::-1])
        # Apply affine transformations
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        return mosaic_image, mosaic_target, dataset



    def transform(self, inputs):
        """
        Args:
            inputs (tuple): 输入元组 (image, target, dataset)

        Returns:
            tuple: 增强后的 (image, target)
        """
        if len(inputs) == 1:
            inputs = inputs[0]
        image, target, dataset = inputs

        # 按概率跳过Mosaic增强
        if self.probability < 1.0 and random.random() > self.probability:
            return image, target, dataset

        # 准备Mosaic组件
        if self.use_cache:
            mosaic_samples, max_height, max_width = self.load_samples_from_cache(image, target, self.mosaic_cache)
            mosaic_image, mosaic_target = self.create_mosaic_from_cache(mosaic_samples, max_height, max_width)
        else:
            resized_images, resized_targets, max_height, max_width = self.load_samples_from_dataset(image, target, dataset)
            mosaic_image, mosaic_target = self.create_mosaic_from_dataset(resized_images, resized_targets, max_height, max_width)

        # 转换并约束坐标格式
        spatial_size = mosaic_image.size[::-1]  # (height, width)
        
        # 处理边界框
        if 'boxes' in mosaic_target:
            mosaic_target['boxes'] = convert_to_tv_tensor(
                mosaic_target['boxes'], 
                'boxes', 
                box_format='xyxy',
                spatial_size=spatial_size
            )
        
        # 处理掩码
        if 'masks' in mosaic_target:
            mosaic_target['masks'] = convert_to_tv_tensor(
                mosaic_target['masks'], 
                'masks'
            )
        
        # 处理多边形点集
        if 'polys' in mosaic_target:
            # 转换为tensor并约束坐标范围
            h, w = spatial_size
            polys = mosaic_target['polys']
            
            # 确保形状为[N, 8]
            if polys.dim() == 1:
                polys = polys.unsqueeze(0)
                
            # 约束坐标到图像范围内
            reshaped_polys = polys.view(-1, 4, 2)
            reshaped_polys[..., 0] = reshaped_polys[..., 0].clamp(0, w)
            reshaped_polys[..., 1] = reshaped_polys[..., 1].clamp(0, h)
            mosaic_target['polys'] = reshaped_polys.view(-1, 8)

        # 应用仿射变换
        mosaic_image, mosaic_target = self.affine_transform(mosaic_image, mosaic_target)

        # 变换后再次约束多边形坐标
        if 'polys' in mosaic_target:
            new_h, new_w = F.get_size(mosaic_image)
            polys = mosaic_target['polys']
            
            # 处理空多边形情况
            if polys.numel() > 0:
                reshaped_polys = polys.view(-1, 4, 2)
                reshaped_polys[..., 0] = reshaped_polys[..., 0].clamp(0, new_w)
                reshaped_polys[..., 1] = reshaped_polys[..., 1].clamp(0, new_h)
                mosaic_target['polys'] = reshaped_polys.view(-1, 8)
            else:
                # 保持空tensor的原始形状
                mosaic_target['polys'] = polys.reshape(-1, 8)

        return mosaic_image, mosaic_target




class CustomRandomAffine(T.RandomAffine):
    def _transform(self, inpt, params):
        if isinstance(inpt, dict) and 'polys' in inpt:
            polys = inpt['polys']
            if polys.numel() > 0:
                # 构造仿射矩阵
                angle = params['angle']
                translate = params['translate']
                scale = params['scale']
                shear = params['shear']
                center = (0, 0)  # 以图像中心为原点进行变换
                
                # 获取逆仿射矩阵参数
                matrix = F._get_inverse_affine_matrix(center, -angle, translate, 1.0/scale, (-shear[0], -shear[1]))
                
                # 将polys转换为点并应用变换
                polys_reshaped = polys.view(-1, 4, 2)
                ones = torch.ones((polys_reshaped.size(0), 4, 1), device=polys_reshaped.device)
                points_homo = torch.cat([polys_reshaped, ones], dim=2)
                
                # 构造仿射矩阵并应用
                affine_matrix = torch.tensor([
                    [matrix[0], matrix[1], matrix[2]],
                    [matrix[3], matrix[4], matrix[5]],
                    [0, 0, 1]
                ], dtype=torch.float32, device=polys_reshaped.device)
                
                transformed_points = torch.einsum('ab,npb->npa', affine_matrix, points_homo)
                transformed_polys = transformed_points[..., :2].contiguous().view(-1, 8)
                inpt['polys'] = transformed_polys
            
            # 调用父类处理其他键
            return super()._transform(inpt, params)
        else:
            return super()._transform(inpt, params)