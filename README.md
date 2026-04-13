# University Event AI Chat Bot

An AI-powered chatbot designed to answer student questions about university events, registrations, attendance, and campus policies. Built on a fine-tuned large language model served through a Flask REST API.

---

## Features

- Fine-tuned Mistral 7B language model using LoRA (via PEFT)
- 4-bit quantization for efficient GPU inference
- Flask REST API with a `/chat` POST endpoint
- Custom stopping criteria for clean, punctuated responses
- Trained on a structured dataset of 100+ university event Q&A examples
- Bilingual support (English & Arabic)
- CORS-enabled for easy frontend integration

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **ML Framework:** PyTorch, Hugging Face Transformers
- **Fine-tuning:** PEFT (LoRA), Datasets, Accelerate
- **Inference:** 4-bit quantization via BitsAndBytes, CUDA GPU support

---

## Project Structure

```
├── ai_app2.py                  # Flask app — loads model and serves /chat endpoint
├── fine_tune.py                # Fine-tuning script (LoRA training)
├── university_event_support.json  # Training dataset
├── ProjectDataSet.json         # Additional dataset
├── llama2-finetuned/           # Fine-tuned model checkpoints
├── Static/                     # Frontend assets
├── requirements.txt
├── test.py / test2.py          # Basic test scripts
```

> **Note:** The fine-tuned AI model file is too large to be included in this repository. To run the app, you will need to either train the model locally using `fine_tune.py` or download it separately and place it in the `fine-tuned-mistral-support/` directory.

---

## API Usage

**Endpoint:** `POST /chat`

**Request:**
```json
{
  "message": "How do I register for an event?"
}
```

**Response:**
```json
{
  "reply": "You can register for events through the university portal under the Events section."
}
```

---

## How to Run

```bash
pip install -r requirements.txt
python ai_app2.py
```

Server runs on `http://localhost:5000`.

---

## Training

To fine-tune the model on the provided dataset:

```bash
python fine_tune.py
```

Requires a CUDA-compatible GPU and sufficient VRAM for Mistral 7B.
