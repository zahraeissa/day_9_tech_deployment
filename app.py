import streamlit as st
from PIL import Image

from model import CHECKPOINT, DEVICE, load_model_for_inference
from utils import predict, preprocess_image

st.set_page_config(page_title="Image Upload", page_icon="🖼️")
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image to see the model's top predictions.")

uploaded_file = st.file_uploader(
	"Choose an image",
	type=["png", "jpg", "jpeg", "webp"],
)


@st.cache_resource
def load_predictor():
	return load_model_for_inference(CHECKPOINT, DEVICE)


if uploaded_file is not None:
	image = Image.open(uploaded_file).convert("RGB")
	st.image(image, caption=uploaded_file.name, width="stretch")

	with st.spinner("Classifying image..."):
		model, class_names, transform = load_predictor()
		image_tensor = preprocess_image(image, transform)
		predictions = predict(model, image_tensor, class_names, DEVICE)

	predicted_class, confidence = predictions[0]
	st.subheader("Image classification result")
	st.metric("Predicted class", predicted_class, f"{confidence:.1%} confidence")

	with st.expander("See all predictions"):
		for class_name, confidence in predictions:
			st.write(f"**{class_name}** — {confidence:.1%}")
			st.progress(confidence)
