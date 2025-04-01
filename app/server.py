from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import uuid

app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = "C:\\Users\\samyu\\Desktop\\Folder"

# Store the latest filename in a text file
LATEST_FILE = "latest_file.txt"


# Route: Serve the Upload Page (For Testing)
@app.route("/")
def home():
    return render_template("upload.html")


# Route: Handle File Upload (POST)
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Ensure unique filenames to prevent overwriting
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)

    file.save(file_path)

    # Store the latest filename
    with open(LATEST_FILE, "w") as f:
        f.write(unique_filename)

    return jsonify({"message": "File uploaded successfully!", "filename": unique_filename, "url": f"/uploads/{unique_filename}"})


# Route: Fetch the last uploaded image (GET)
@app.route("/latest-image", methods=["GET"])
def latest_image():
    try:
        with open(LATEST_FILE, "r") as f:
            latest_file = f.read().strip()
        
        return jsonify({"filename": latest_file, "url": f"/uploads/{latest_file}"})
    except FileNotFoundError:
        return jsonify({"error": "No file uploaded yet"}), 404


# Route: Serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)  # Allow access from other devices
