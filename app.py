
import streamlit as st
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io
import tempfile

st.set_page_config(layout="wide")
st.title("Plot Match Verifier (Polygon Matching + OCR)")

reader = easyocr.Reader(['en'])

def extract_images_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        images.append(img)
    return images

def perform_ocr_on_polygon(image):
    bounds_text = reader.readtext(np.array(image), detail=0)
    return " ".join(bounds_text)

uploaded_plot_pdf = st.file_uploader("Upload Multi-Plot PDF", type="pdf")
uploaded_master_pdf = st.file_uploader("Upload Master Plan PDF", type="pdf")

if uploaded_plot_pdf and uploaded_master_pdf:
    plot_images = extract_images_from_pdf(uploaded_plot_pdf)
    master_images = extract_images_from_pdf(uploaded_master_pdf)
    master_image = master_images[0] if master_images else None

    st.subheader("Extracted Plot Images and Matching Status")
    for idx, plot_img in enumerate(plot_images):
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(plot_img, caption=f"Plot {idx+1}", use_column_width=True)
        with col2:
            st.write("🔍 Matching logic can be added later via GPT-4 or OpenCV.")
            ocr_result = perform_ocr_on_polygon(plot_img)
            st.write("📝 OCR Text:")
            st.code(ocr_result)

st.info("Note: This is a local-only MVP. GPT-4 and polygon detection will be added in next version.")
