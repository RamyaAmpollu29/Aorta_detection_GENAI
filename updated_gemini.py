import streamlit as st
import json
import base64
import io
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
from PIL import Image
import cv2
from fpdf import FPDF  # <-- PDF generation
import pandas as pd

# ==== Gemini Setup ====
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("API Key not found! Please set GEMINI_API_KEY in your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

#print(f"Using Gemini API Key: {GEMINI_API_KEY}")  # Debugging line to check if the key is loaded
# Create a GenerativeModel instance
model = genai.GenerativeModel('gemini-1.5-flash')
# ==== Utility Functions ====
def convert_image_to_base64(image):
    image_np = np.array(image)
    _, buffer = cv2.imencode('.jpg', image_np)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def convert_json_to_prompt(json_data):
    prompt = (
        "You are an expert AI assistant specializing in pediatric cardiology. Analyze the ultrasound image and generate a detailed, human-readable diagnostic report "
        "focused on detecting aortic malformations in children. These may include aortic coarctation, aortic valve stenosis, aortic root dilation, or abnormal flow patterns. "
        "Do not return JSON or code. Provide your analysis as a clinical paragraph suitable for a doctor's report.\n\n"
        "The image analysis provided the following findings:\n"
    )

    relevant_labels = ["Aorta", "AorticValve", "AorticRoot", "AscendingAorta", "Arch", "DescendingAorta"]

    for label in relevant_labels:
        if label in json_data:
            findings = json_data[label]
            if findings:
                prompt += f"- {label}: {findings}\n"

    prompt += "\nWrite your diagnostic summary in natural language based on these findings."

    return prompt


def convert_text_file_to_prompt(text_file):
    try:
        text = text_file.read().decode("utf-8")
        return f"Additional report information:\n{text}"
    except Exception as e:
        return f"Error reading text file: {e}"

def convert_csv_file_to_prompt(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return f"Additional CSV data:\n{df.to_string()}"
    except Exception as e:
        return f"Error reading CSV file: {e}"

def analyze_with_gemini(image, json_data, user_prompt):
    prompt = convert_json_to_prompt(json_data) + "\n\n" + user_prompt
    image_base64 = convert_image_to_base64(image)
    content = prompt + f"\n\nImage data (base64 encoded): {image_base64}"

    response = model.generate_content(content)

   
    return response.text if response else "No response from Gemini API."

def generate_pdf(text, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Split the text into lines to print in the PDF
    lines = text.split('\n')
    for line in lines:
        pdf.multi_cell(0, 10, line)

    
    # Get PDF content as string, encode to bytes, wrap in BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)

# ==== Streamlit UI ====
st.title("🩺 Aortic Malformations Detection")
st.markdown("Upload an ultrasound image and related files to analyze and detect possible heart abnormalities.")

uploaded_files = st.file_uploader("📤 Upload Files (Image, JSON, Text, CSV, PDF, etc.)", type=["jpg", "jpeg", "png", "json", "txt", "csv", "pdf"], accept_multiple_files=True)

user_prompt = st.text_area("Ask a question or provide a prompt for the analysis (e.g., 'What is the condition of the LA')")

if st.button("Analyze"):
    if not uploaded_files:
        st.error("⚠️ Please upload at least one file.")
    else:
        # Initialize variables
        image_data = None
        json_data = {}
        additional_prompt = ""
        
        # Process uploaded files
        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split('.')[-1].lower()

            if file_extension in ['jpg', 'jpeg', 'png']:
                image_data = uploaded_file.read()
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
                st.success(f"🖼️ Image {uploaded_file.name} loaded successfully.")

            elif file_extension == 'json':
                try:
                    json_data = json.load(uploaded_file)
                    st.success(f"📄 JSON file {uploaded_file.name} loaded successfully.")
                except json.JSONDecodeError:
                    st.error(f"❌ Invalid JSON format in file {uploaded_file.name}.")
                    json_data = {}

            elif file_extension == 'txt':
                additional_prompt += convert_text_file_to_prompt(uploaded_file)

            elif file_extension == 'csv':
                additional_prompt += convert_csv_file_to_prompt(uploaded_file)

            elif file_extension == 'pdf':
                additional_prompt += f"PDF file {uploaded_file.name} uploaded (not processed for content)."

        if image_data:
            # Run analysis with user prompt and additional files' data
            st.info("🔍 Running analysis...")
            analysis_result = analyze_with_gemini(image, json_data, user_prompt + "\n" + additional_prompt)

            # Show in UI
            st.subheader("📝 Gemini Full Analysis")
            st.text_area("Analysis Result", analysis_result, height=300)

            # PDF generation functionality
            pdf_file = generate_pdf(analysis_result, "report.pdf")
            st.download_button(
                label="📥 Download Report as PDF",
                data=pdf_file,
                file_name="analysis_report.pdf",
                mime="application/pdf"
            )
        else:
            st.error("⚠️ No valid image uploaded for analysis.")
