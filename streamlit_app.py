import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Color Detector", layout="centered")

st.title("🎨 Color Detection App")
st.markdown("Upload an image, click anywhere on it, and get the RGB and closest color name.")

@st.cache_data
def load_colors(csv_path="colors.csv"):
    df = pd.read_csv(csv_path)
    df["R"] = df["R"].astype(int)
    df["G"] = df["G"].astype(int)
    df["B"] = df["B"].astype(int)
    return df

def get_pixel_rgb(img: Image.Image, x: int, y: int):
    """Return RGB tuple at (x, y). PIL uses (x, y) where y is vertical index."""
    img_rgb = img.convert("RGB")
    width, height = img_rgb.size
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return img_rgb.getpixel((x, y))

def find_closest_color_name(rgb, colors_df):
    r, g, b = rgb
    cols = colors_df[["R","G","B"]].values
    dist = np.sqrt(np.sum((cols - np.array([r,g,b]))**2, axis=1))
    idx = np.argmin(dist)
    return colors_df.iloc[idx]

colors_df = load_colors("colors.csv")
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image.  
2. Click on the part of the image whose color you want to detect.  
3. The app will show RGB values, the nearest color name and a color preview.
""")

uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(img, caption="Uploaded image", use_column_width=True)

    st.info("Click anywhere on the image below to detect color.")
    coords = streamlit_image_coordinates(img, key="img_coords")

    if coords is not None:
        x = int(coords['x'])
        y = int(coords['y'])
        st.write(f"Clicked coordinates: (x={x}, y={y})")

        r,g,b = get_pixel_rgb(img, x, y)
        st.markdown("### Detected color")
        row = find_closest_color_name((r,g,b), colors_df)
        color_name = row["color_name"]
        hex_code = row["hex"]

        col1, col2 = st.columns([1,2])
        with col1:
            st.write("**RGB**")
            st.write(f"R: {r}")
            st.write(f"G: {g}")
            st.write(f"B: {b}")
            st.write("**Nearest color**")
            st.write(f"{color_name} ({hex_code})")
        with col2:
            st.write("**Preview**")
            preview = Image.new("RGB", (200,100), (r,g,b))
            st.image(preview, caption="Detected color", use_column_width=False)

        st.markdown("---")
        st.write("**Matched dataset color preview**")
        matched_preview = Image.new("RGB", (200,100), (int(row['R']), int(row['G']), int(row['B'])))
        st.image(matched_preview, caption=f"Dataset color: {color_name} ({row['hex']})", use_column_width=False)

    else:
        st.write("Click on the image to pick a color.")
else:
    st.info("Please upload an image to begin.")

