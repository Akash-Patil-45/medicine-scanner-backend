import google.generativeai as genai
import PIL.Image
import json
import base64
import io
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Flask App Initialization ---
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) for your app
CORS(app)

# --- Gemini API Configuration ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    print("Gemini model loaded successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None

# --- API Endpoint for Image Processing ---
@app.route('/process_image', methods=['POST'])
def process_image():
    if not model:
        return jsonify({"error": "Gemini model is not configured"}), 500

    # Get the image data from the request
    # The image is sent as a Base64 encoded string
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode the Base64 image
    # The string looks like "data:image/jpeg;base64,..."
    # We need to strip the header part to get the pure Base64 data
    header, encoded = data['image'].split(",", 1)
    image_data = base64.b64decode(encoded)
    
    # Open the image using PIL
    img = PIL.Image.open(io.BytesIO(image_data))

    # The prompt for the Gemini model
    prompt = """
    Analyze the provided image of a medicine strip.
    Identify the medicine name and the expiration date (EXP).
    If any information is not clearly visible, state "Not found".
    Return the result strictly in JSON format with keys "brand_name" and "expiry_date".
    Example: {"brand_name": "Calpol", "expiry_date": "10/2026"}
    """

    try:
        # Send the request to the model
        response = model.generate_content([prompt, img])
        
        # Clean up the response to get the JSON string
        json_str = response.text.strip().replace('```json', '').replace('```', '')
        
        # Parse the JSON string into a Python dictionary
        result = json.loads(json_str)
        return jsonify(result)

    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        return jsonify({'error': 'Failed to process image with Gemini'}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Runs the app on http://127.0.0.1:5000
    app.run(debug=True)