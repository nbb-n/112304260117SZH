import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import os

class ImprovedDNN(nn.Module):
    def __init__(self):
        super(ImprovedDNN, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        x = self.dropout(self.relu(self.bn3(self.fc3(x))))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x

device = torch.device('cpu')
model = ImprovedDNN().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

def predict_digit(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.convert('L')
    image = image.resize((28, 28))
    image_np = np.array(image) / 255.0
    image_np = image_np.reshape(1, 784)
    tensor = torch.tensor(image_np, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100
    
    return f"预测数字: {prediction}\n置信度: {confidence:.2f}%"

with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# 📝 手写数字识别")
    gr.Markdown("上传一张手写数字图片（0-9），模型将预测数字")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="输入图片", scale=2)
            submit_btn = gr.Button("识别", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(label="预测结果", lines=2)
    
    submit_btn.click(fn=predict_digit, inputs=input_image, outputs=output_text)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)