import sys
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from models import build_dab_deformable_detr

image_path = sys.argv[1] if len(sys.argv) > 1 else "picture/1.png"
checkpoint_path = "checkpoints/checkpoint0110.pth"
device = torch.device("cuda")

# Load trained model
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
preprocess = transforms.Compose(
    [
        transforms.Resize(800, max_size=1333),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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

print(f"\nPeak GPU memory: " f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

# Examine the object detection predictions
class_names = [
    "banana",
    "watch",
    "pliers",
    "toothbrush",
    "wrench",
    "towel",
    "knife",
    "badminton",
    "tape",
    "charger",
    "pen",
    "box",
    "umbrella",
    "apple",
    "wallet",
    "shaver",
    "paper",
    "cup",
    "mobile phone",
    "wrist developer",
    "bottle",
    "headset",
    "toothpaste",
    "glasses",
    "remote controller",
    "card",
    "mouse",
    "cans",
    "screwdriver",
    "notebook",
    "stapler",
]

probabilities = outputs["pred_logits"][0].sigmoid()
num_classes = probabilities.shape[1]

scores, indices = torch.topk(probabilities.flatten(), 100)
keep = scores > 0.4

print("\n=== Object predictions above 0.4 ===")
print("Number of query/class predictions:", int(keep.sum()))

for score, index in zip(scores[keep][:15], indices[keep][:15]):
    query_id = int(index // num_classes)
    class_id = int(index % num_classes)

    name = (
        class_names[class_id]
        if class_id < len(class_names)
        else f"unnamed class {class_id}"
    )

    print(f"Query {query_id:3d} | " f"{name:20s} | " f"Score: {score.item():.3f}")


print("\n=== Predicted vs Ground Truth Boxes ===")

# Obtain the original image dimensions
W, H = image.size

# Determine the annotation path from the input image path
image_file = Path(image_path).resolve()
gt_objects = []

# Expected VMRD structure: images/<split>/<image_id>.jpg
if image_file.parent.parent.name == "images":
    split = image_file.parent.name
    image_id = image_file.stem
    dataset_root = image_file.parent.parent.parent

    label_path = dataset_root / "labels" / split / f"{image_id}.txt"

    if label_path.exists():
        print("Ground-truth file:", label_path)

        with open(label_path) as f:
            for line in f:
                parts = line.split()

                gt_objects.append(
                    {"class_id": int(parts[0]), "box": [float(v) for v in parts[1:5]]}
                )
    else:
        print("No ground-truth annotation found:", label_path)
else:
    print("Image is outside the expected VMRD folder structure.")

# Display predictions and corresponding ground-truth boxes
for score, index in zip(scores[keep], indices[keep]):
    query_id = int(index // num_classes)
    class_id = int(index % num_classes)

    cx, cy, w, h = outputs["pred_boxes"][0, query_id].tolist()

    predicted_box = [
        (cx - w / 2) * W,
        (cy - h / 2) * H,
        (cx + w / 2) * W,
        (cy + h / 2) * H,
    ]

    name = class_names[class_id] if class_id < len(class_names) else str(class_id)

    matching_gt = [obj["box"] for obj in gt_objects if obj["class_id"] == class_id]

    print(f"\n{name} (query {query_id})")
    print("  Confidence:", round(score.item(), 3))
    print("  Predicted box:", [round(v, 1) for v in predicted_box])
    print("  Ground-truth boxes:", matching_gt)


# Load the ground-truth manipulation relationships
print("\n=== Ground Truth Manipulation Relationships ===")

adj_path = dataset_root / "adj" / split / f"{image_id}.txt"

with open(adj_path) as f:
    gt_adj = torch.tensor(
        [[int(value) for value in line.split()] for line in f if line.strip()],
        dtype=torch.int64,
    )

print("Ground-truth adjacency file:", adj_path)

for row in gt_adj:
    print(row.tolist())

print("\n=== Predicted Manipulation Relationships ===")

# Ground-truth object classes, in annotation order
gt_classes = [obj["class_id"] for obj in gt_objects]

# Map each detected class to its query ID
class_to_query = {}

for index in indices[keep]:
    class_id = int(index % num_classes)
    query_id = int(index // num_classes)

    if class_id in class_to_query:
        raise ValueError("Repeated class detected; object matching required")

    class_to_query[class_id] = query_id

# This simple method only supports one instance per class
assert len(gt_classes) == len(set(gt_classes))
assert set(class_to_query) == set(gt_classes)

# Order query IDs to match the ground-truth object order
query_ids = [class_to_query[c] for c in gt_classes]

print("Object order:", [class_names[c] for c in gt_classes])
print("Query order:", query_ids)

# Extract the smaller adjacency matrix and apply sigmoid
pred_adj = outputs["pred_adj"][0]
pred_adj = pred_adj[query_ids][:, query_ids].sigmoid()

print("\nPredicted relationship scores:")
for row in pred_adj:
    print([round(x, 10) for x in row.tolist()])

print("\nPredicted binary adjacency (threshold = 0.5):")
for row in (pred_adj > 0.5).int():
    print(row.tolist())
