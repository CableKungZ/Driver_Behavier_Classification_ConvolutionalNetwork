## Making Pridcition return class & prob
from typing import List, Tuple
import torch
import torchvision.transforms as T

from PIL import Image
def pred_class(model: torch.nn.Module,
                        image,
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        ):
    
    
    # 2. Open image
    img = image

    # 3. Create transformation for image (if one doesn't exist)
    image_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    

    ### Predict on image ### 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4. Make sure the model is on the target device
    model.to(device)
    

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0).float()

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    classname =  class_names[target_image_pred_label]
    prob = target_image_pred_probs.cpu().numpy()

    return prob
