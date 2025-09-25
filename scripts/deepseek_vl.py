import torch
from transformers import AutoModelForCausalLM
import json
import os

from deepseek_vl.models import MultiModalityCausalLM, VLChatProcessor
from deepseek_vl.utils.io import load_pil_images

def validate_and_convert_to_json(vlm_answer):
    """
    Parses the VLM's string output, validates it against predefined rules,
    and converts it to a JSON object if valid.
    """
    print("\n--- Starting Validation ---")
    
    required_keys = {"Object", "Part", "Color", "Texture", "Spatial"}
    allowed_parts = {"head", "body", "neck", "legs", "wings", "tail", "nose", "rear"}
    allowed_spatials = {"proportional", "not proportional"}

    parsed_output = {}
    validation_errors = []

    lines = vlm_answer.strip().split('\n')
    for line in lines:
        if ':' in line:
            clean_line = line.replace('**', '')
            key, value = clean_line.split(':', 1)
            key = key.strip()
            value = value.strip()
            parsed_output[key] = value

    found_keys = set(parsed_output.keys())
    if found_keys != required_keys:
        missing = required_keys - found_keys
        extra = found_keys - required_keys
        if missing:
            validation_errors.append(f"Missing required keys: {list(missing)}")
        if extra:
            validation_errors.append(f"Found unexpected keys: {list(extra)}")
    else:
        if ' ' in parsed_output.get('Object', ''): validation_errors.append(f"Object '{parsed_output['Object']}' should not contain adjectives.")
        if ' ' in parsed_output.get('Texture', ''): validation_errors.append(f"Texture '{parsed_output['Texture']}' should be a single word.")
        if parsed_output.get('Part') not in allowed_parts: validation_errors.append(f"Part '{parsed_output.get('Part')}' is not in the allowed list.")
        if parsed_output.get('Spatial') not in allowed_spatials: validation_errors.append(f"Spatial '{parsed_output.get('Spatial')}' must be one of {list(allowed_spatials)}.")

    if not validation_errors:
        print("Validation Successful!")
        return json.dumps(parsed_output, indent=4)
    else:
        print("Validation FAILED:")
        for error in validation_errors:
            print(f"- {error}")
        return None

def process_image(image_path, vl_gpt, vl_chat_processor):
    """
    Runs the full VLM evaluation pipeline for a single image to extract attributes.
    """
    print(f"\n=========================================")
    print(f"Processing Attribute Extraction for: {image_path}")
    print(f"=========================================")

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    conversation = [
        {"role": "User", "content": """Your task is to answer the following five questions about the provided image. Your response MUST be a five-line text block, formatted exactly like the example.

**Questions:**
1.  **Object:** What is the single noun for the main subject? (e.g., 'car', 'dog', no adjectives)
2.  **Part:** From the list ["head", "body", "neck", "legs", "wings", "tail", "nose", "rear"], what is the most prominent part?
3.  **Color:** What is the main color of this part?
4.  **Texture:** What single word best describes the texture? (e.g., 'furry', 'smooth')
5.  **Spatial:** Is the object's form 'proportional' or 'not proportional'?

---
**Example of the required five-line output format:**
Object: car
Part: body
Color: classic red
Texture: metallic
Spatial: proportional
---

**Your Task:**
Now, answer these five questions for the image below, providing the complete five-line response.

<image_placeholder>""", "images": [image_path]},
        {"role": "Assistant", "content": ""},
    ]

    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True).to(vl_gpt.device)
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    outputs = vl_gpt.language_model.generate(inputs_embeds=inputs_embeds, attention_mask=prepare_inputs.attention_mask, pad_token_id=vl_chat_processor.tokenizer.eos_token_id, bos_token_id=vl_chat_processor.tokenizer.bos_token_id, eos_token_id=vl_chat_processor.tokenizer.eos_token_id, max_new_tokens=512, do_sample=False, use_cache=True)
    answer = vl_chat_processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
    print("\n--- Raw Model Output ---")
    print(answer)
    return validate_and_convert_to_json(answer)

def generate_questions_from_attributes(attributes, image_filename):
    """
    Generates specific yes/no questions based on extracted attributes.
    """
    questions = []
    
    obj = attributes.get('Object', '')
    part = attributes.get('Part', '')
    color = attributes.get('Color', '')
    texture = attributes.get('Texture', '')
    spatial = attributes.get('Spatial', '')
    
    # Object + Part question
    questions.append(f"Does the image have a {obj} {part}?")
    
    # Object + Part + Color question
    questions.append(f"Does the {obj} {part} have a {color} color?")
    
    # Object + Part + Texture question
    questions.append(f"Does the {obj} {part} maintain a {texture} texture?")
    
    # Object + Part + Spatial question
    questions.append(f"Is the {obj} {part} {spatial}?")
    
    return questions

def save_questions_to_json(all_questions, output_filename="generated_questions.json"):
    """
    Saves all generated questions to a JSON file.
    """
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, indent=4, ensure_ascii=False)

# --- Main Execution ---

# 1. Load the model and processor once
print("Loading model and processor...")
model_path = "deepseek-ai/deepseek-vl-7b-chat"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(torch.bfloat16).cuda().eval()
print("Model loading complete.")

# 2. Define images for attribute extraction
images = [
    "./images/bear.jpg",
    "./images/elephant.jpg",
]

# 3. Attribute Extraction
attribute_results = {}
print("\n--- Running Attribute Extraction ---")
for img_path in images:
    json_result = process_image(img_path, vl_gpt, vl_chat_processor)
    if json_result:
        filename = os.path.basename(img_path)
        attributes = json.loads(json_result)
        attribute_results[filename] = attributes

print("\n\n=========================================")
print("        Final Attribute Results          ")
print("=========================================")
print(json.dumps(attribute_results, indent=4))

# 4. Generate Questions Based on Attributes
all_questions = {}
for filename, attributes in attribute_results.items():
    questions = generate_questions_from_attributes(attributes, filename)
    all_questions[filename] = questions

# 5. Save Questions to JSON File

output_data = {
    "attributes": attribute_results,
    "questions": all_questions
}

success = save_questions_to_json(output_data, "generated_questions.json")
