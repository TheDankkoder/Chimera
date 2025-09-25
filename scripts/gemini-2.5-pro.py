from google import genai
from google.genai import types
import json

client = genai.Client()

# Read the JSON file with questions
with open('generated_questions.json', 'r') as f:
    data = json.load(f)

# Open your zebra image
with open('./images/zebra.jpg', 'rb') as f:
    image_bytes = f.read()

# Combine all questions from all images
all_questions = []
for image_key, questions in data["questions"].items():
    all_questions.extend(questions)

# Create the complete expected JSON structure
expected_json = {}
for question in all_questions:
    expected_json[question] = "Yes/No"

# Create a formatted question string with JSON output instruction
question_text = f"""Answer each question with ONLY 'Yes' or 'No' and format your response as a JSON object.

Use this EXACT format with the questions as keys:

{json.dumps(expected_json, indent=4)}

Replace "Yes/No" with your actual answers ("Yes" or "No").
Return ONLY the JSON object, no other text."""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ),
        question_text
    ]
)

# Parse and save the JSON response
try:
    # Clean the response text (remove ```json and ``` if present)
    response_text = response.text.strip()
    if response_text.startswith('```json'):
        response_text = response_text[7:]  # Remove ```json
    if response_text.endswith('```'):
        response_text = response_text[:-3]  # Remove ```
    response_text = response_text.strip()
    
    # Parse the JSON
    response_json = json.loads(response_text)
    
    # Save to file
    with open('grading.json', 'w') as f:
        json.dump(response_json, f, indent=2)
    
    print("Response saved to grading.json")
    
except json.JSONDecodeError as e:
    print(f"Error parsing JSON: {e}")
    print("Raw response:")
    print(response.text)
