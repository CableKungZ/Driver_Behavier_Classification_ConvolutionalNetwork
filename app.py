import streamlit as st
import torch
from PIL import Image
from prediction import pred_class
import numpy as np

# Set title
st.title('Driving Behavior Classification')

# Set Header
st.header('Upload Image')

# Load Model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Ensure that the model architecture is defined before loading the state_dict
# This example assumes MobilenetV3_Large0 architecture is defined
# Replace 'MyModel' with the actual model class used during training
# Example: 
# model = MobileNetV3_Large(num_classes=5)  # Example, replace with your model class and parameters
model = torch.load('models/MobilenetV3_Large0.pt', map_location=device)
model=model.half()
model.eval()  # Set the model to evaluation mode

# Display image & Prediction
uploaded_image = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    class_names = ['other_activities', 'safe_driving', 'texting_phone', 'talking_phone', 'turning']

    if st.button('Prediction'):
        # Prediction class
        classname, prob = pred_class(model, image, class_names)
        
        st.write("## Prediction Result")
        
        # Iterate over the class_names and probs list
        for i, name in enumerate(class_names):
            # Set the color to blue if it's the maximum value, otherwise use the default color
            color = "blue" if name == classname else None
            st.write(f"## <span style='color:{color}'>{name} : {prob*100:.2f}%</span>", unsafe_allow_html=True)
