from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
import torch

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load fine-tuned Mistral model
MODEL_PATH = "./fine-tuned-mistral-support"  # Ensure this path is correct

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Configure 4-bit quantization to match compute dtype
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16  # Fixes warning for efficient computation
)

# Load model with correct quantization config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # Assigns to GPU automatically
    quantization_config=quantization_config  # Apply correct quantization settings
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define Stopping Criteria (Ensure Responses End Properly)
class StopOnPeriod(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        last_output = input_ids[0][-1].item()
        return tokenizer.decode([last_output]).strip() in [".", "?", "!"]  # Stops if response ends with punctuation

stopping_criteria = StoppingCriteriaList([StopOnPeriod()])

# Chat endpoint
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        # Format input for direct chat (no [INST])
        formatted_input = f"User: {user_message} \nAI: "

        # Tokenize input
        inputs = tokenizer(formatted_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Generate response with stopping criteria
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,  *
                temperature=0.6,
                top_k=30,
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria  
            )

        # Decode and clean response
        bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        return jsonify({'reply': bot_reply})

    except Exception as e:
        return jsonify({'error': f'Failed to generate response: {str(e)}'}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(port=5000, debug=True)
