import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# ✅ Hugging Face Authentication (Replace with your actual token)
token = "hf_your_actual_token_here"
login(token)

# ✅ Load dataset (Updated with the provided structured JSON)
data = [
    {
      "question": "Where is this event happening?",
      "answer": "This event's location can be found in the event details on the website. You can check there for the exact venue!"
    },
    {
      "question": "How do I register for an event?",
      "answer": "If registration is required, simply log into your account, go to the event page, and click the 'Register' button. If you're not logged in, you'll need to create an account first."
    },
    {
      "question": "Can I attend this event without registering?",
      "answer": "Some events require registration, while others are open to all. Check the event details on the website to see if registration is needed."
    },
    {
      "question": "Is this event open to all students?",
      "answer": "Most events are open to all students, but some are restricted to specific majors. You can check the event details for eligibility."
    },
    {
      "question": "How do I cancel my event registration?",
      "answer": "You can cancel your registration by visiting your registered events section on the website, or I can do it for you! Just let me know which event you'd like to cancel."
    },
    {
      "question": "Will there be any honor guests at this event?",
      "answer": "If there are honor guests attending, their names will be listed in the event details on the website."
    },
    {
      "question": "How many students will attend this event?",
      "answer": "Attendance numbers aren't publicly available, but you can see if the event has limited spots in the event details."
    },
    {
      "question": "Where can I find upcoming events?",
      "answer": "You can check the event calendar on the website for a list of all upcoming events."
    },
    {
      "question": "What happens if an event gets canceled?",
      "answer": "In the rare case that an event is canceled, you’ll receive a notification in the app, and the update will be posted in the announcements section."
    },
    {
      "question": "Can I prove my attendance at an event?",
      "answer": "Yes! If the event requires attendance proof, you can scan the barcode displayed at the event to add it to your attended events."
    },
    {
      "question": "How can I contact support?",
      "answer": "For any issues, you can reach out to support via the contact email provided on the website."
    },
    {
      "question": "Is this event free to attend?",
      "answer": "Most university events are free, but some may require a fee. Check the event details to confirm."
    },
    {
      "question": "Can I bring a guest to the event?",
      "answer": "Some events allow guests, while others are for students only. Please check the event details for guest policies."
    },
    {
      "question": "Is there a dress code for this event?",
      "answer": "If there's a dress code, it will be mentioned in the event details."
    },
    {
      "question": "Can I ask a speaker a question during the event?",
      "answer": "Many events include a Q&A session. Check the event description to see if audience participation is allowed."
    },
    {
      "question": "I can't log into my account. Can you help?",
      "answer": "I can't reset passwords, but you can try the 'Forgot Password' option on the login page. If you're still having trouble, contact support."
    },
    {
      "question": "I registered but didn't get a confirmation. What should I do?",
      "answer": "Your registration should be confirmed via app notification. If you didn't receive it, check your registered events section or contact support."
    },
    {
      "question": "What languages does this assistant support?",
      "answer": "I can assist you in both English and Arabic!"
    },
    {
      "question": "Can you tell me if my friend is attending this event?",
      "answer": "I can’t access student attendance records for privacy reasons. Your friend will have to check their own registered events."
    },
    {
      "question": "Can I see past events I attended?",
      "answer": "Yes! Your attended events history is available in your account under the 'Attended Events' section."
    }
  
]

# Convert to Hugging Face Dataset
dataset = Dataset.from_list(data)

# ✅ Load model and tokenizer (Mistral 7B)
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=token,
    load_in_4bit=True,  # ✅ Enable 4-bit quantization for lower VRAM usage
    device_map="auto"   # ✅ Ensure model is fully on GPU
)

# ✅ Ensure padding token is set
tokenizer.pad_token = tokenizer.eos_token

# ✅ Apply LoRA for Memory-Efficient Fine-Tuning
peft_config = LoraConfig(
    r=8,                
    lora_alpha=16,      
    lora_dropout=0.1,   
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# ✅ Preprocess dataset correctly
def preprocess_function(examples):
    inputs = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# ✅ Reduce dataset size for quicker testing
dataset = dataset.shuffle(seed=42)
dataset = dataset.map(preprocess_function, batched=True, remove_columns=["question", "answer"])

# ✅ Define output folder for saving fine-tuned model
SAVE_PATH = os.path.abspath("./fine-tuned-mistral-support")
os.makedirs(SAVE_PATH, exist_ok=True)

# ✅ Fine-Tuning Parameters
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # ✅ Small batch size for GPU efficiency
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # ✅ Reduces VRAM usage
    num_train_epochs=5,
    save_total_limit=2,
    save_steps=500,
    logging_dir=os.path.join(SAVE_PATH, "logs"),
    logging_steps=10,
    fp16=True,  # ✅ Mixed precision for memory efficiency
    optim="adamw_torch"
)

# ✅ Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # Using same dataset for evaluation (can be split further)
)

try:
    trainer.train()
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print(f"✅ Model successfully saved to {SAVE_PATH}")
except Exception as e:
    print(f"❌ Error during training or saving: {e}")
