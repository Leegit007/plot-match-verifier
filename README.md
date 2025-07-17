
# Plot Match Verifier (Streamlit App)

This Streamlit app allows you to:
- Upload a multi-plot PDF and a master plan PDF.
- Extract images of plots from the multi-plot PDF.
- Perform OCR using EasyOCR on each extracted image.
- (Planned) Match each plot image against the master image using GPT-4 vision or OpenCV.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

You can deploy directly on [Streamlit Cloud](https://streamlit.io/cloud) by uploading this repo.
