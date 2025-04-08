"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

import sys
import os
import cv2  # Added for video processing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig


def draw(images, labels, boxes, scores, polygons, thrh=0.04):
    """可视化函数，支持边界框和多边形绘制"""
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        
        # 当前样本数据解包 (所有张量形状均为 [num_queries, ...])
        scr = scores[i]    # 形状 [300]
        lab = labels[i]    # 形状 [300]
        box = boxes[i]     # 形状 [300, 4]
        poly = polygons[i] # 形状 [300, 4, 2]
        
        # 生成置信度掩码
        mask = scr > thrh
        
        # 过滤低置信度结果
        valid_labels = lab[mask]    # [num_valid]
        valid_boxes = box[mask]     # [num_valid, 4]
        valid_scores = scr[mask]    # [num_valid]
        valid_polys = poly[mask]    # [num_valid, 4, 2]

        # 绘制每个有效检测结果
        for j in range(len(valid_labels)):
            # 绘制边界框和标签
            b = valid_boxes[j].tolist()
            draw.rectangle(b, outline='red', width=2)
            draw.text((b[0], b[1]), 
                     text=f"{valid_labels[j].item()}:{valid_scores[j]:.2f}",
                     fill='blue')
            
            # 绘制多边形
            p = valid_polys[j]  # 形状 [4, 2]
            
            # 绘制四个顶点（蓝色点）
            for point in p:
                draw.ellipse([point[0]-3, point[1]-3, point[0]+3, point[1]+3],
                            fill='blue')
            
            # 绘制连接线（绿色）
            connections = [
                (p[0], p[2]),  # 0-2
                (p[1], p[2]),  # 1-2
                (p[1], p[3]),  # 1-3
                (p[0], p[3])   # 0-3
            ]
            for (start, end) in connections:
                draw.line([start[0], start[1], end[0], end[1]], 
                         fill='green', 
                         width=2)

        im.save('torch_results.jpg')

def process_image(model, device, file_path):
    im_pil = Image.open(file_path).convert('RGB')
    orig_size = torch.tensor([[im_pil.width, im_pil.height]]).to(device)
    
    # 数据预处理
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_tensor = transforms(im_pil).unsqueeze(0).to(device)
    
    # 模型推理
    labels, boxes, scores, polygons = model(im_tensor, orig_size)
    
    # 解包批次维度（假设batch_size=1）
    draw([im_pil], 
         labels.squeeze(0), 
         boxes.squeeze(0), 
         scores.squeeze(0),
         polygons.squeeze(0))


def process_video(model, device, file_path):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)

        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # Check if the input file is an image or a video
    file_path = args.input
    if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Process as image
        process_image(model, device, file_path)
        print("Image processing complete.")
    else:
        # Process as video
        process_video(model, device, file_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
