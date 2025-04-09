import streamlit as st
import torch
from diffusers import StableDiffusionPipeline

st.set_page_config(page_title="Text to Image Generator", layout="centered")

@st.cache_resource
def load_pipeline():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32,
        use_safetensors=True,
        safety_checker=None 
    )
    return pipe.to("cpu")

pipe = load_pipeline()

st.title("Text to Image Generator")
st.markdown("Enter a creative prompt and click **Generate** to see the magic!")

user_input = st.text_input("Enter your prompt:")

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter a prompt before generating.")
    else:
        with st.spinner("Generating... please wait..."):
            image = pipe(user_input).images[0]
            st.image(image, caption="Generated Image", use_container_width=True)
