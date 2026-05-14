# 🧠 Decoder-Only Transformer Web Interface

This project implements a minimal, custom **Decoder-Only Transformer** (similar to a small GPT) and exposes its functionality via a simple **web interface**.

---

## 🌟 Features

* **Autoregressive Text Generation:**
  Prompts the model with a starting phrase and generates a continuation of the text.

* **Model Evaluation:**
  Provides an endpoint to calculate and display the model's **perplexity** on a validation dataset.

* **Minimalist Interface:**
  A clean, responsive web page allows easy interaction with the model.

* **PyTorch Backend:**
  The model runs securely on a **Python/PyTorch backend (DRF API)**.

---

## 🛠️ Project Structure

```
Transformer/
├── Transformer.ipynb          ← Training notebook (run on Google Colab)
├── requirements.txt           ← Python dependencies
├── data/
│   └── tinyshakespeare.txt    ← Training data
├──     └──train.csv and test.csv
├── model/
│   └── decoder_model3.pth     ← Trained model weights (copied from Colab)
├── BE/                        ← Django backend
│   ├── manage.py
│   ├── decoder_be/
│   │   ├── settings.py
│   │   └── urls.py
│   └── transform/
│       ├── views.py
│       ├── utils.py           ← BERT tokenizer version
│       └── utils1.py          ← Character-level version (active)
└── FE/
    └── index.html             ← Frontend (open directly in browser)
```

---

## 🚀 Setup on Windows

### Step 1 — Train the model on Google Colab

1. Open `Transformer.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Upload `data/tinyshakespeare.txt` to your Google Drive under `MyDrive/data/`
3. Run all cells — training takes a few minutes on Colab's free GPU
4. Download the saved `decoder_model3.pth` file from your Google Drive
5. Place it in the `model/` folder of this project

### Step 2 — Create a virtual environment

Open **PowerShell** or **Command Prompt** in the project root:

```powershell
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Start the backend server

```powershell
cd BE
python manage.py migrate
python manage.py runserver
```

The backend will be running at: `http://127.0.0.1:8000`

### Step 5 — Open the frontend

Open a second terminal (or just use File Explorer) and open:

```
FE/index.html
```

Double-click it to open in your browser, or in PowerShell:

```powershell
start FE\index.html
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/finishtext?prompt=The` | GET | Generate text from a prompt |
| `/api/evaluate` | GET | Evaluate model perplexity on validation data |

---

## ⚠️ Notes

* The `model/` folder must contain `decoder_model3.pth` before starting the backend
* The backend uses `utils1.py` (character-level tokenizer) by default — check `views.py` if you want to switch to the BERT tokenizer version (`utils.py`)
* Both backend and frontend must be running at the same time
















