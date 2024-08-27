from typing import List, Tuple
import torch
import torchvision.transforms as T
from PIL import Image

def pred_class(model: torch.nn.Module,
               image,
               class_names: List[str],
               image_size: Tuple[int, int] = (224, 224),
               ) -> Tuple[str, float]:
    
    # 2. Open image
    img = image

    # 3. Create transformation for image
    image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    
    # 4. Send model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # 6. Transform and add an extra dimension to the image
        transformed_image = image_transform(img).unsqueeze(dim=0).float()

        # 7. Make a prediction on the image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    # Get the class name and probability
    classname = class_names[target_image_pred_label]
    prob = target_image_pred_probs[0, target_image_pred_label].item()

    return classname, prob

# Example usage
# Assuming `model` is your defined model architecture and `state_dict` is the saved weights
# model = YourModelArchitecture()
# model.load_state_dict(torch.load("path_to_your_model_weights.pth"))
# class_name, probability = pred_class(model, image, class_names)
