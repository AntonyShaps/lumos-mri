import torch
import requests
from PIL import Image
from torchvision import transforms

model = torch.load('/home/anton/repos/lumos-mri/model.pkl')
model.eval()

labels = ["disease", "no_disease"]


def predict(inp):
  inp = transforms.ToTensor()(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    confidences = {labels[i]: float(prediction[i]) for i in range(1000)}
  return confidences

import gradio as gr

gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=3)).launch()