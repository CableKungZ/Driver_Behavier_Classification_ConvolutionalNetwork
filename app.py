import torch
from torchvision import transforms
from PIL import Image
import streamlit as st

# Check if a GPU is available and map the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained PyTorch model, mapping to the appropriate device
model = torch.load('models/MobilenetV3_Large0.pt', map_location=device)
model.eval()  # Set the model to evaluation mode

# Define the image preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size expected by your model
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

# Define the prediction function
def predict(image):
    image = preprocess_image(image)
    with torch.no_grad():
        prediction = model(image)
    return prediction

# Streamlit UI
st.title("PyTorch CNN Model Deployment")
st.write("Upload an image to make a prediction.")

# Upload File
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    prediction = predict(image)
    predicted_class = prediction.argmax(dim=1).item()  # Get the index of the max log-probability
    st.write(f"Prediction: {predicted_class}")
