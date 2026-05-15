from ultralytics import YOLO
import openvino as ov
import os
import time

def export_to_openvino():
    print("正在导出模型为 OpenVINO 格式...")
    model = YOLO('runs/detect/traffic_signs_detector/weights/best.pt')
    model.export(format='openvino')
    print("导出完成!")

def run_inference_openvino():
    print("正在加载 OpenVINO 模型...")
    core = ov.Core()

    model_path = 'runs/detect/traffic_signs_detector/weights/best_openvino_model/'
    ir_xml = os.path.join(model_path, 'best.xml')

    if not os.path.exists(ir_xml):
        print("OpenVINO 模型未找到，正在导出...")
        export_to_openvino()

    print("加载 OpenVINO 模型...")
    ov_model = core.read_model(model=ir_xml)

    available_devices = core.available_devices
    print(f"可用设备: {available_devices}")

    GPU_DEVICE = "GPU"
    if GPU_DEVICE in available_devices:
        print(f"使用 Intel GPU ({GPU_DEVICE}) 进行推理...")
        compiled_model = core.compile_model(ov_model, device_name=GPU_DEVICE)
    else:
        print(f"GPU 不可用，使用 CPU 进行推理...")
        compiled_model = core.compile_model(ov_model, device_name="CPU")

    from ultralytics import YOLO
    yolo_model = YOLO(model_path)

    test_images_dir = 'test/images'
    image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]

    print(f"找到 {len(image_files)} 张测试图像")

    results_list = []
    for i, img_file in enumerate(image_files):
        img_path = os.path.join(test_images_dir, img_file)
        start_time = time.time()
        results = yolo_model.predict(source=img_path, verbose=False)
        elapsed = time.time() - start_time

        for r in results:
            boxes = r.boxes
            if len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xywhn = box.xywhn[0]
                    x_center, y_center, width, height = float(xywhn[0]), float(xywhn[1]), float(xywhn[2]), float(xywhn[3])
                    results_list.append({
                        'image_id': img_file,
                        'class_id': cls_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': conf
                    })

        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{len(image_files)} 张图像")

    print(f"推理完成! 共检测到 {len(results_list)} 个目标")
    return results_list

import pandas as pd

def main():
    try:
        results = run_inference_openvino()
        print(f"\n成功完成 OpenVINO 加速推理!")
        print(f"检测结果数量: {len(results)}")

        if results:
            df = pd.DataFrame(results)
            output_path = 'submission.csv'
            df.to_csv(output_path, index=False)
            print(f"提交文件已保存到: {output_path}")
            print(f"\n前5行数据:")
            print(df.head())
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
