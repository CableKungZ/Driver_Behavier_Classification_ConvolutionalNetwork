import streamlit as st
from PIL import Image
import torchvision.transforms as transforms
import torch
import torchvision.models as models

# Define the model
def get_mobilenet_v3_model(pretrained=True):
    model = models.mobilenet_v3_large(pretrained=pretrained) if pretrained else models.mobilenet_v3_large(pretrained=False)
    return model

# Define the transformation for input images
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Load the model
model = get_mobilenet_v3_model(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Streamlit file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform_image(image)

    # Make a prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Process the output
    def get_class_names():
        url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
        import requests
        response = requests.get(url)
        return response.json()

    class_names = get_class_names()
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]

    st.write(f'Predicted class: {predicted_label}')
