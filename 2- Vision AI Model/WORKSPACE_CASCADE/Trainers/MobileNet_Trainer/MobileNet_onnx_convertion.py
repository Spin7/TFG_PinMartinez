import torch
import torchvision.models as models

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_model_fixed.pth"   
ONNX_PATH = "best_CLmodel_fixed.onnx"
IMAGE_SIZE = 160
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 2

# =========================
# LOAD MODEL (MATCH TRAINING!)
# =========================
def build_model():
    model = models.mobilenet_v3_large(weights=None)  # <-- FIXED

    in_features = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_features, NUM_CLASSES)

    return model

model = build_model()

# Load weights
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

print("Model loaded correctly.")

# =========================
# DUMMY INPUT
# =========================
dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)

# =========================
# EXPORT ONNX
# =========================
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    dynamo=False
)

print(f"ONNX model exported to: {ONNX_PATH}")