import streamlit as st
from PIL import Image
import numpy as np
import cv2
import fitz  # PyMuPDF
import easyocr
import tempfile

st.set_page_config(layout="wide")
st.title("🧭 Plot to Master Plan Verifier")

# Initialize OCR reader once
reader = easyocr.Reader(['en'], gpu=False)

def extract_plots_from_pdf(pdf_file):
    extracted_plots = []
    pdf_bytes = pdf_file.read()
    doc = fitz.open("pdf", pdf_bytes)

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 15
        )

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            pad = 10
            x = max(x - pad, 0)
            y = max(y - pad, 0)
            w = min(w + 2*pad, img.shape[1] - x)
            h = min(h + 2*pad, img.shape[0] - y)

            plot_crop = img[y:y+h, x:x+w]
            pil_img = Image.fromarray(cv2.cvtColor(plot_crop, cv2.COLOR_BGR2RGB))
            extracted_plots.append(pil_img)

    return extracted_plots

def perform_ocr_on_image(image):
    image_np = np.array(image)
    result = reader.readtext(image_np, detail=0)
    return result

# Upload section
st.sidebar.header("📤 Upload Files")
multi_plot_pdf = st.sidebar.file_uploader("Upload Multi-Plot PDF", type=["pdf"])
master_plan_pdf = st.sidebar.file_uploader("Upload Master Plan PDF", type=["pdf"])

if multi_plot_pdf and master_plan_pdf:
    with st.spinner("🔍 Extracting plots..."):
        try:
            plots = extract_plots_from_pdf(multi_plot_pdf)
            st.success(f"✅ Extracted {len(plots)} plots.")
        except Exception as e:
            st.error(f"❌ Error extracting plots: {e}")
            plots = []

    if plots:
        st.subheader("📍 Extracted Plot Images")
        for idx, img in enumerate(plots):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(img, caption=f"Plot #{idx+1}", use_column_width=True)
            with col2:
                try:
                    text = perform_ocr_on_image(img)
                    st.markdown("**Detected Text:**")
                    st.write(", ".join(text) if text else "_No text found_")
                except Exception as e:
                    st.warning(f"OCR failed: {e}")

        st.info("🧠 GPT-4 call to compare with master plan is currently **disabled**.\n"
                "This app shows how plots are extracted and OCR-ed for now.")
