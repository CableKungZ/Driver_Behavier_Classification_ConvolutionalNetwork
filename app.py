import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import os

# Set title
st.title('Driving Behavior Classification')

# Set Header
st.header('Upload Image')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ensure the model file exists
model_path = 'models/MobilenetV3_Large0.pt'
assert os.path.exists(model_path), f"Model file not found at {model_path}"

try:
    # Replace with your model architecture
    from torchvision.models import mobilenet_v3_large  # Example, replace with yours
    model = mobilenet_v3_large(num_classes=5)  # Replace with actual model definition
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_names = ['other_activities', 'safe_driving', 'texting_phone', 'talking_phone', 'turning']

    if st.button('Prediction'):
        try:
            # Prediction class
            classname, prob = pred_class(model, image, class_names)
            
            st.write("## Prediction Result")
            
            # Iterate over the class_names and probs list
            for i, name in enumerate(class_names):
                color = "blue" if name == classname else None
                st.write(f"## <span style='color:{color}'>{name} : {prob[i]*100:.2f}%</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
