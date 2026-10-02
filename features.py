"""Fixed, unlearned color, texture and shape measurements for a PIL image."""
import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern


def image_features(image):
    image = ImageOps.exif_transpose(image).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    image = Image.alpha_composite(background, image).convert("RGB")
    rgb = np.asarray(image.resize((128, 128), Image.Resampling.BILINEAR)) / 255.0
    hsv = rgb2hsv(rgb)
    features = []
    for colors in (rgb, hsv):
        for channel in range(3):
            counts, _ = np.histogram(colors[:, :, channel], bins=16, range=(0, 1))
            features.extend(counts / counts.sum())
    gray = (rgb2gray(rgb) * 255).astype(np.uint8)
    for points, radius in [(8, 1), (16, 2)]:
        texture = local_binary_pattern(gray, points, radius, method="uniform")
        counts, _ = np.histogram(texture, bins=np.arange(points + 3))
        features.extend(counts / counts.sum())
    small = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR)) / 255.0
    features.extend(hog(rgb2gray(small), orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), block_norm="L2-Hys"))
    features.extend(rgb.reshape(8, 16, 8, 16, 3).mean(axis=(1, 3)).ravel())
    features.extend(rgb.mean(axis=(0, 1)))
    features.extend(rgb.std(axis=(0, 1)))
    return np.asarray(features, dtype=np.float64)


# Public entry point used by the demonstration website.
extract_features = image_features
