from ultralytics import YOLO
import torch
import os

def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model_path = 'runs/detect/traffic_signs_detector/weights/best.pt'
    if os.path.exists(model_path):
        print(f"从已有模型继续训练: {model_path}")
        model = YOLO(model_path)
    else:
        print("从头训练")
        model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data.yaml',
        epochs=50,           # 减少训练轮数
        batch=32,            # 增大batch size
        imgsz=416,           # 减小图像尺寸
        device=device,
        workers=4,
        optimizer='AdamW',   # AdamW通常更快收敛
        lr0=0.001,           # 稍大的学习率
        momentum=0.9,
        weight_decay=0.0005,
        verbose=True,
        name='traffic_signs_detector',
        exist_ok=True,
        plots=False,
        augment=False,       # 关闭数据增强
        mosaic=0.0,          # 关闭mosaic增强
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        perspective=0.0,
        degrees=0.0,
        translate=0.0
    )
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
