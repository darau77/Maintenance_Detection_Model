from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps


def build_model():
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        nn.Dropout(0.3), nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 7)
    )


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])(image)
