from instantlearn.data import Sample
from instantlearn.models import Matcher

# Initialize Matcher (device: "xpu", "cuda", or "cpu")
model = Matcher(device="cuda")

# Create reference sample (auto-loads image and mask from paths)
# Paths below are relative to the `library` directory in the repo; adjust if running from elsewhere.
ref_sample = Sample(
    image_path="library/examples/assets/coco/000000286874.jpg",
    mask_paths="library/examples/assets/coco/000000286874_mask.png",
)

# Fit once on reference
model.fit(ref_sample)

# Predict on multiple target images — no prompts needed
predictions = model.predict([
    "library/examples/assets/coco/000000390341.jpg",
    "library/examples/assets/coco/000000173279.jpg",
    "library/examples/assets/coco/000000267704.jpg",
])

# Access results for each image
for pred in predictions:
    masks = pred["pred_masks"]
