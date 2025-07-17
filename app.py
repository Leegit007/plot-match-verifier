import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import cv2
import os
import tempfile
from easyocr import Reader
from PIL import Image
from shapely.geometry import Polygon, Point

st.set_page_config(layout="wide")
st.title("🧠 Plot Image Matcher + Text Extractor")

# Load EasyOCR reader (English, no GPU)
reader = Reader(['en'], gpu=False)

def extract_images_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for i in range(len(doc)):
        pix = doc[i].get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    return images

def detect_polygon_from_dark_lines(image):
    """Detect polygon from contours (assumes dark lines)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) >= 4 and cv2.contourArea(approx) > 10000:
            polygons.append(approx.reshape(-1, 2))
    return polygons

def is_polygon_inside(big_poly, small_poly):
    big = Polygon(big_poly)
    small = Polygon(small_poly)
    return big.contains(small)

def extract_text_from_polygon(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(polygon)], 255)
    masked_img = cv2.bitwise_and(image, image, mask=mask)
    bounds_text = reader.readtext(masked_img, detail=0)
    return " ".join(bounds_text)

# MAIN APP
with st.sidebar:
    plot_pdf = st.file_uploader("📄 Upload Plot Image PDF", type=["pdf"])
    master_pdf = st.file_uploader("🗺️ Upload Master Plan PDF", type=["pdf"])

if plot_pdf and master_pdf:
    with st.spinner("🔍 Processing files..."):
        plot_images = extract_images_from_pdf(plot_pdf.read())
        master_img = extract_images_from_pdf(master_pdf.read())[0]  # Assume one page

        # Detect master polygons once
        master_polygons = detect_polygon_from_dark_lines(master_img)

        results = []

        # Create a temp folder if not exist
        os.makedirs("temp", exist_ok=True)

        for idx, plot_img in enumerate(plot_images):
            temp_path = f"temp/plot_{idx}.png"
            cv2.imwrite(temp_path, cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))

            plot_polygons = detect_polygon_from_dark_lines(plot_img)

            matched = False
            for pp in plot_polygons:
                for mp in master_polygons:
                    if is_polygon_inside(mp, pp):
                        text = extract_text_from_polygon(master_img, mp)
                        results.append((plot_img, text))
                        matched = True
                        break
                if matched:
                    break

            os.remove(temp_path)

    # Show results
    st.subheader("✅ Matched Plots + Extracted Text")
    for img, text in results:
        st.image(img, caption="Matched Plot", use_column_width=True)
        st.info(f"📌 Extracted Text: {text}")
