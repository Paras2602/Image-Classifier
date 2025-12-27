import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import streamlit as st
from pathlib import Path


def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = (model_name or "").lower()
    if name == "resnet18":
        m = resnet18(weights=None)  # app should NOT download weights
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")


@st.cache_resource
def load_artifacts(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model_name = ckpt.get("model_name", "resnet18")
    num_classes = int(ckpt["num_classes"])
    class_names = ckpt["class_names"]

    input_size = int(ckpt.get("input_size", 224))
    norm_mean = tuple(ckpt.get("norm_mean", (0.485, 0.456, 0.406)))
    norm_std = tuple(ckpt.get("norm_std", (0.229, 0.224, 0.225)))

    transform = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_std),
    ])

    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, transform, class_names, device, model_name


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image to see Top‑K predictions with confidence.")

BASE_DIR = Path(__file__).resolve().parent
default_ckpt = BASE_DIR / "models" / "cifar10_resnet18_best.pt"

ckpt_path = st.text_input("Checkpoint path", value=str(default_ckpt))

if not Path(ckpt_path).exists():
    st.error(
        "Checkpoint not found.\n\n"
        f"Expected: {ckpt_path}\n\n"
        "Train your model in the notebook first and save it to models/cifar10_resnet18_best.pt"
    )
    st.stop()

try:
    model, transform, class_names, device, model_name = load_artifacts(ckpt_path)
    st.caption(f"Loaded model: {model_name} | Device: {device}")
except Exception as e:
    st.error("Failed to load checkpoint.")
    st.code(str(e))
    st.stop()

top_k = st.slider("Top‑K", 1, min(10, len(class_names)), 5)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    x = transform(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    values, indices = torch.topk(probs, k=top_k)

    st.subheader("Top Predictions")
    for rank, (v, i) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        st.write(f"{rank}. **{class_names[i]}** — {v:.4f} ({v*100:.2f}%)")

    st.subheader("Confidence chart")
    chart_data = {class_names[i]: float(v) for v, i in zip(values, indices)}
    st.bar_chart(chart_data)