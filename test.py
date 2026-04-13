from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("./llama2-finetuned")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    exit()

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the base model with quantization
try:
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("Base model loaded successfully.")
except Exception as e:
    print(f"Failed to load base model: {e}")
    exit()

# Load LoRA adapter
try:
    model = PeftModel.from_pretrained(base_model, "./llama2-finetuned")
    print("LoRA adapter loaded successfully.")
except Exception as e:
    print(f"Failed to load LoRA adapter: {e}")
    exit()

# Generate Response
def generate_response(query):
    input_text = f"Query: {query} Response:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    output = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

print(generate_response("Where is the Annual Tech Conference?"))
