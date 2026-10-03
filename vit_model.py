import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps


class SmallViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(3, 96, kernel_size=12, stride=12)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 96))
        self.position = nn.Parameter(torch.zeros(1, 65, 96))
        block = nn.TransformerEncoderLayer(96, 4, dim_feedforward=192, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=3, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, 7)
        # Reinitialize cloned encoder blocks with independent random draws.
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.MultiheadAttention):
                nn.init.xavier_uniform_(layer.in_proj_weight)
                nn.init.zeros_(layer.in_proj_bias)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, images):
        tokens = self.patches(images).flatten(2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.size(0), -1, -1)
        tokens = torch.cat([class_tokens, tokens], dim=1) + self.position
        tokens = self.encoder(tokens)
        return self.head(self.norm(tokens[:, 0]))


def build_model():
    return SmallViT()


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGBA")
    image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])(image)
