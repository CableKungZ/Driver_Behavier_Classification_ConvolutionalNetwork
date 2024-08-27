import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define the model
def get_mobilenet_v3_model(pretrained=True):
    """
    Returns a MobileNetV3 model.
    
    Parameters:
    - pretrained (bool): If True, returns a model pre-trained on ImageNet.
    
    Returns:
    - model (torch.nn.Module): The MobileNetV3 model.
    """
    model = models.mobilenet_v3_large(pretrained=pretrained) if pretrained else models.mobilenet_v3_large(pretrained=False)
    return model

# Define the transformation for input images
def transform_image(image_path):
    """
    Transforms an image to the format expected by the model.
    
    Parameters:
    - image_path (str): Path to the image file.
    
    Returns:
    - tensor (torch.Tensor): Transformed image tensor.
    """
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Load the model
model = get_mobilenet_v3_model(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Load and preprocess an image
image_path = 'path/to/your/image.jpg'
input_tensor = transform_image(image_path)

# Make a prediction
with torch.no_grad():
    output = model(input_tensor)

# Process the output
def get_class_names():
    """
    Returns class names for ImageNet.
    
    Returns:
    - class_names (list of str): The class names.
    """
    # Class names for ImageNet
    url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
    import requests
    response = requests.get(url)
    return response.json()

class_names = get_class_names()
_, predicted_idx = torch.max(output, 1)
predicted_label = class_names[predicted_idx.item()]

print(f'Predicted class: {predicted_label}')
