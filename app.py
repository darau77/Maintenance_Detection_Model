from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from features import extract_features

ROOT = Path(__file__).parent
registry = json.loads((ROOT / "models/registry.json").read_text())
st.set_page_config(page_title="Wall Image Classifier", page_icon="🧱")
st.title("Wall Image Classifier")
st.write("Choose a model and upload a close-up wall photo.")
name = st.selectbox("Model", list(registry))
info = registry[name]
st.caption(f"Test accuracy: {info['test_accuracy']:.1%} · No pretrained models")

@st.cache_resource
def load_model(name):
    info = registry[name]
    if info["type"] in ("cnn", "vit"):
        import torch
        if info["type"] == "vit":
            from vit_model import build_model
        else:
            from cnn_model import build_model
        torch.set_num_threads(4)
        model = build_model()
        model.load_state_dict(torch.load(ROOT / info["file"], map_location="cpu", weights_only=True))
        return model.eval()
    return joblib.load(ROOT / info["file"])

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
if file is not None:
    try:
        image = ImageOps.exif_transpose(Image.open(file)).convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
        st.image(image, caption=file.name, width=350)
        if st.button("Classify", type="primary"):
            model = load_model(name)
            if info["type"] in ("cnn", "vit"):
                import torch
                if info["type"] == "vit":
                    from vit_model import prepare_image
                else:
                    from cnn_model import prepare_image
                with torch.inference_mode():
                    index = model(prepare_image(image).unsqueeze(0)).argmax(1).item()
                prediction = info["classes"][index]
            else:
                prediction = model.predict([extract_features(image)])[0]
            st.success("Prediction: " + prediction.replace("_", " ").title())
    except (OSError, ValueError, Image.DecompressionBombError):
        st.error("Please choose a valid image.")

with st.expander("Compare model results"):
    st.write("Test results (scores from 0 to 1)")
    st.dataframe(pd.read_csv(ROOT / "results/model_comparison.csv").round(4), hide_index=True)
    st.write("Five-fold cross-validation")
    cv = pd.read_csv(ROOT / "results/cv_summary.csv")
    cv = cv[["model", "test_accuracy_mean", "test_accuracy_std", "test_f1_macro_mean", "test_f1_macro_std"]]
    cv.columns = ["Model", "Accuracy mean", "Accuracy SD", "Macro F1 mean", "Macro F1 SD"]
    st.dataframe(cv.round(4), hide_index=True)
    st.caption("CV uses five development-data folds. Test metrics use the same 584 held-out images. CNN and ViT are additional experiments selected using a separate validation set.")
    st.caption("Balanced accuracy equals macro recall here. Training and prediction times vary between runs; CNN and ViT training times also include validation.")
st.caption("Classes: algae, major crack, minor crack, peeling, plain, spalling and stain.")
st.caption("Classroom demo. Images are processed locally and are not saved. Every image receives one of the seven labels.")
