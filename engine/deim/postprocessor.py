"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor']


def mod(a, b):
    out = a - a // b * b
    return out

# 修改后的 PostProcessor 类
@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False,
        num_poly_vertices=4  # 新增参数：多边形顶点数（例如4个顶点）
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.num_poly_vertices = num_poly_vertices  # 新增
        self.deploy_mode = False

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        # 1. 提取原始输出
        logits = outputs['pred_logits']
        boxes = outputs['pred_boxes']
        polys = outputs['pred_polygons']  # 新增：形状 (B, num_queries, num_vertices*2)
        
        # 2. 边界框坐标转换（原有逻辑）
        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)  # 还原为像素坐标
        
        # 3. 多边形坐标转换（新增）
        # 假设 polys 是归一化的 (cx, cy) 坐标，形状 (B, num_queries, num_vertices*2)
        B, NQ = polys.shape[:2]
        poly_pred = polys.view(B, NQ, self.num_poly_vertices, 2)  # (B, NQ, V, 2)
        
        # 将归一化坐标转换为像素坐标
        orig_sizes = orig_target_sizes.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, 2)
        poly_pred = poly_pred * orig_sizes  # (B, NQ, V, 2)
        
        # 4. 分类得分处理（原有逻辑）
        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
        
        # 5. 根据索引筛选结果（新增多边形处理）
        # 筛选边界框
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))
        # 筛选多边形
        poly_pred = poly_pred.gather(dim=1, index=index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, poly_pred.shape[2], poly_pred.shape[3]))
        
        # 6. 部署模式直接返回张量（新增多边形）
        if self.deploy_mode:
            return labels, boxes, scores, poly_pred
        
        # 7. 类别标签重映射（可选）
        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)
        
        # 8. 封装结果（新增多边形）
        results = []
        for lab, box, sco, poly in zip(labels, boxes, scores, poly_pred):
            result = dict(
                labels=lab, 
                boxes=box, 
                scores=sco,
                polygons=poly  # 新增：形状 (num_top_queries, V, 2)
            )
            results.append(result)
        
        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
