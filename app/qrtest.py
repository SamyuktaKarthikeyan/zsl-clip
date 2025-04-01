import streamlit as st
import qrcode
import tempfile
import requests
from PIL import Image

UPLOAD_URL = "http://192.168.174.245:5000/"  # Update when deployed



def generate_qr_code(upload_url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(upload_url)
    qr.make(fit=True)

    img = qr.make_image(fill="black", back_color="white")

    # Save QR Code to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name)
    
    return temp_file.name

st.subheader("Upload Image via Mobile")
qr_code_path = generate_qr_code(UPLOAD_URL)
st.image(qr_code_path, caption="Scan to Upload", width=250)

st.write("Scan the QR code to upload an image from your phone.")

# Manually enter uploaded image URL
image_url = st.text_input("Enter Image URL (after upload):")

st.title("Fetch Latest Uploaded Image")

if st.button("Fetch Image"):
    response = requests.get(f"{UPLOAD_URL}/latest-image")
    
    if response.status_code == 200:
        data = response.json()
        image_url = f"{UPLOAD_URL}{data['url']}"
        st.image(image_url, caption="Latest Uploaded Image", use_column_width=True)
    else:
        st.warning("No image uploaded yet!")
