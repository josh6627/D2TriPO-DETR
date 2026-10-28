import sys
import torch
from PIL import Image
from torchvision import transforms

from models import build_dab_deformable_detr

image_path = sys.argv[1] if len(sys.argv) > 1 else "picture/1.png"
checkpoint_path = "checkpoints/checkpoint0110.pth"
device = torch.device("cuda")

# Load trained model
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

args = checkpoint["args"]
args.device = "cuda"

model, _, _ = build_dab_deformable_detr(args)
model.load_state_dict(checkpoint["model"], strict=True)
model = model.to(device).eval()

del checkpoint

# Load RGB image
image = Image.open(image_path).convert("RGB")
print("Image:", image_path)
print("Original size:", image.size)

# Match the validation preprocessing
preprocess = transforms.Compose([
    transforms.Resize(800, max_size=1333),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

input_tensor = preprocess(image).unsqueeze(0).to(device)

print("Network input shape:", tuple(input_tensor.shape))

# Run inference
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    outputs = model(input_tensor)

print("\nInference successful!")

for name, value in outputs.items():
    if torch.is_tensor(value):
        print(f"{name}: {tuple(value.shape)}")

print(
    f"\nPeak GPU memory: "
    f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
)
