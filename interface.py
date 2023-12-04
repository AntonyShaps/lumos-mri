import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr

# Load the model and set to evaluation mode
model = torch.load('/home/anton/repos/lumos-mri/model.pkl')
model = model.to('cpu')  # Move the model to CPU
model.eval()

labels = ["disease", "no_disease"]

def predict(img):
    # Transform the image to the format expected by your model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # Process the image and predict
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        disease_probability = probabilities[0][labels.index("disease")].item()
        return disease_probability

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),  # Ensure the input is a PIL image
    outputs="number"
)

iface.launch(share=True)