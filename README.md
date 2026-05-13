# 🧠 Decoder-Only Transformer Web Interface

This project implements a minimal, custom **Decoder-Only Transformer** (similar to a small GPT) and exposes its functionality via a simple **web interface**.

---

## 🌟 Features

* **Autoregressive Text Generation:**  
  Prompts the model with a starting phrase and generates a continuation of the text.

* **Model Evaluation:**  
  Provides an endpoint to calculate and display the model’s **perplexity** on a validation dataset.

* **Minimalist Interface:**  
  A clean, responsive web page allows easy interaction with the model.

* **PyTorch Backend:**  
  The model runs securely on a **Python/PyTorch backend (DRF API)**.

---

## 🛠️ Project Structure

This project is divided into three main components:
### **1. The Transformer Implementation**

The full implementation of the **Transformer Model** is in the notebook called Decoder_only_transformer.ipynb


### **2. Backend (BE) — The Brain**

The backend is a **DRF API** responsible for hosting the trained **PyTorch model** and handling all computation.

* **`transform/views.py`:**  
  The Django application that:  
  - Loads the `decoder_model.pth` file  
  - Defines the model architecture  
  - Exposes two API endpoints:  
    - `/finishtext` – for text generation  
    - `/evaluate` – for model evaluation  

* **`decoder_model.pth`:**  
  This file contains the **saved weights (`state_dict`)** of the trained Transformer model.

### **3.Frontend (FE) — The Interface**
The frontend is a static HTML page that runs in the user's browser, handling the interface and communication with the backend API.

index.html:
A single file containing all HTML, Tailwind CSS, and JavaScript logic for the web application.

---

### 🚀 How to Run the Project
This project requires two separate terminals to run the backend server and frontend interface simultaneously.

#### Start the Backend Server (BE)
Open your first terminal, navigate to the BE directory, and run:
```bash
python manage.py runserver
```
Expected Output:
Running on http://127.0.0.1:5000

#### Access the Frontend (FE)
Open your second terminal or simply open the file in your browser:

```bash
# macOS
open FE/index.html

# Linux
xdg-open FE/index.html

# Windows (PowerShell)
start FE/index.html

```
