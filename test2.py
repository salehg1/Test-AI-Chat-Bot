from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Load the tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load tokenizer: {e}")
    exit()

# Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with quantization
try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        quantization_config=quantization_config,
        device_map="auto"
    )
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=8,  
    lora_alpha=32,  
    target_modules=["q_proj", "v_proj"],  
    lora_dropout=0.1,  
    bias="none",  
    task_type="CAUSAL_LM"  
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# 🔍 Verify LoRA Application
def check_lora_application(model):
    print("\n🔍 Verifying LoRA Application...")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable Parameters: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.2f}%)")

    lora_layers_found = False
    for name, param in model.named_parameters():
        if "lora" in name:
            lora_layers_found = True
            print(f"✅ LoRA Applied: {name} | Trainable: {param.requires_grad}")

    if not lora_layers_found:
        print("❌ No LoRA layers found! Check `target_modules`.")

    non_lora_trainable = [name for name, param in model.named_parameters() if param.requires_grad and "lora" not in name]
    if non_lora_trainable:
        print("\n⚠️ Warning: Some non-LoRA layers are trainable!")
        for layer in non_lora_trainable:
            print(f"🔴 Unexpected trainable layer: {layer}")
    else:
        print("✅ Base model is correctly frozen.")

    print("\n✅ LoRA Debugging Completed.")

# Run LoRA verification
check_lora_application(model)

# Load the dataset
try:
    dataset = load_dataset('json', data_files='university_event_support.json')
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit()

# Split the dataset
try:
    dataset = dataset['train'].train_test_split(test_size=0.1)  
    print("✅ Dataset split into train and test sets.")
except Exception as e:
    print(f"❌ Failed to split dataset: {e}")
    exit()

# Tokenize the dataset
def tokenize_function(examples):
    formatted_texts = [f"Query: {q} Response: {r}" for q, r in zip(examples["query"], examples["response"])]
    tokenized_inputs = tokenizer(formatted_texts, truncation=True, padding="max_length", max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("✅ Dataset tokenized successfully.")
except Exception as e:
    print(f"❌ Failed to tokenize dataset: {e}")
    exit()

# Set training arguments
training_args = TrainingArguments(
    output_dir="./llama2-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=250,
    warmup_steps=50,
    weight_decay=0.0, 
    fp16=True,
)

# Initialize the Trainer
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
    )
    print("✅ Trainer initialized successfully.")
except Exception as e:
    print(f"❌ Failed to initialize Trainer: {e}")
    exit()

# Fine-tune the model
try:
    trainer.train()
    print("✅ Model fine-tuning completed.")
except Exception as e:
    print(f"❌ Failed to fine-tune model: {e}")
    exit()

# Save the model and tokenizer
try:
    model.save_pretrained("./llama2-finetuned")
    tokenizer.save_pretrained("./llama2-finetuned")
    print("✅ Model and tokenizer saved successfully.")
except Exception as e:
    print(f"❌ Failed to save model/tokenizer: {e}")

# 🔥 Inference Function with LoRA Debugging
def generate_response(query):
    print("\n🔍 Checking active LoRA layers before inference...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"✅ LoRA active in: {name}")

    input_text = f"Query: {query} Response:"  # Ensure space after "Query:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    output = model.generate(input_ids, max_length=100, do_sample=True, top_p=0.9, temperature=0.7)
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

# 🔥 Overfitting Test
print("\n🔍 Running Overfitting Test...")
sample_query = tokenized_dataset["train"][0]["query"]
expected_response = tokenized_dataset["train"][0]["response"]

generated_response = generate_response(sample_query)
print(f"📝 Query: {sample_query}")
print(f"✅ Expected Response: {expected_response}")
print(f"🤖 Model Response: {generated_response}")

