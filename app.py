import streamlit as st
import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms
import timm
import torch.nn.functional as F  # Import for softmax
import io

# Define the class names
class_names = ['other_activities', 'safe_driving', 'texting_phone', 'talking_phone', 'turning']

# Define the model class
class MobileNetV3Large(torch.nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Large, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

# Function to load the model
def load_model(model_path):
    model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=len(class_names))  # Initialize model with number of classes
    try:
        # Load the state dictionary with weights_only=True
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
    except RuntimeError as e:
        st.error(f"RuntimeError: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()
    return model

# Function to transform the image for the model
def transform_image(image):
    try:
        image = Image.open(image).convert('RGB')
    except Exception as e:
        st.error(f"Error loading image: {e}")
        st.stop()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Function to make predictions
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        probabilities = probabilities.squeeze().cpu().numpy()  # Convert to numpy array
        class_probabilities = {class_names[i]: prob * 100 for i, prob in enumerate(probabilities)}
        return class_probabilities

# Streamlit app interface
def main():
    st.title("Driver Behavior Classification")
    model_path = 'MobilenetV3_Large0.pt'  # Update with actual model path

    model = load_model(model_path)

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image_tensor = transform_image(uploaded_file)
        class_probabilities = predict(model, image_tensor)
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Display probabilities for each class
        st.write("Probabilities for each class:")
        for class_name, probability in class_probabilities.items():
            st.write(f'{class_name}: {probability:.2f}%')

if __name__ == "__main__":
    main()
