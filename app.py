import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# โหลดโมเดลที่ฝึกไว้ล่วงหน้า
def load_model(model_path):
    model = models.mobilenet_v3_large(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# การแปลงภาพสำหรับโมเดล
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)  # เพิ่มมิติ batch

# โหลดโมเดลจากไฟล์
model_path = 'MobilenetV3_Large0.pt'  # แทนที่ด้วยที่อยู่ไฟล์ของคุณ
model = load_model(model_path)

# รายการชื่อคลาส
class_names = ['other_activities', 'safe_driving', 'texting_phone', 'talking_phone', 'turning']

# อัปโหลดไฟล์ภาพใน Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform_image(image)

    # ทำนายภาพ
    with torch.no_grad():
        output = model(input_tensor)

    # ดึงชื่อคลาสที่ทำนาย
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_names[predicted_idx.item()]

    st.write(f'Predicted class: {predicted_label}')
