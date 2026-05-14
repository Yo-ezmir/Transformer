**Author:** Yonatan Azmir | yonatanezmir@gmail.com

# 🧠 Decoder-Only Transformer Web Interface

This project implements a custom **Decoder-Only Transformer** (GPT from scratch) with support for **BOTH character-level and BERT tokenization**.

---

## 🌟 Features

- **Two Model Options:**
  - Character-level (matches PDF, vocab_size=65)
  - BERT subword tokenizer (better text quality, vocab_size=30k)
- **Autoregressive Text Generation:** Generate text from any prompt
- **Model Evaluation:** Calculate perplexity on validation data
- **Web Interface:** Clean UI with model selector dropdown
- **PyTorch + Django Backend:** REST API for model inference



## 📁 Project Structure
```
Transformer/
├── Transformer.ipynb              # Training notebook (run on Colab)
├── requirements.txt               # Python dependencies
├── data/
│   └── tinyshakespeare.txt        # Training data
├── model/
│   ├── decoder_model_bert.pth     # Trained BERT model
│   └── decoder_model_character.pth # Trained character model
├── BE/                            # Django backend
│   ├── manage.py
│   ├── decoder_be/
│   │   ├── settings.py
│   │   └── urls.py
│   └── transform/
│       ├── views.py               # API endpoints
│       ├── utils.py               # BERT tokenizer version
│       └── utils1.py              # Character-level version
└── FE/
    └── index.html                 # Frontend UI
```

## 🚀 Setup

### Step 1 — Train Models on Google Colab

1. Open `Transformer.ipynb` in Google Colab
2. Mount Google Drive and upload `tinyshakespeare.txt` to `MyDrive/data/`
3. Set `MODEL_TYPE = "bert"` → Run all cells
4. Set `MODEL_TYPE = "character"` → Run all cells again
5. Download both `.pth` files from Drive
6. Place them in the `model/` folder

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run Backend

```bash
cd BE
python manage.py runserver
```

Backend runs at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### Step 5 — Open Frontend

Open FE/index.html in your browser

## 🔌 API Endpoints
GET /api/finishtext?prompt=The&model=bert → Generate text (parameters: prompt, model)

GET /api/evaluate?model=bert → Get perplexity (parameter: model)

## ⚠️ Notes
Both models must be trained before running backend

Backend uses the model file matching the selected type

Frontend lets you switch between BERT and Character models

Default hyperparameters: n_embd=128, n_layer=8, block_size=128, dropout=0.1
