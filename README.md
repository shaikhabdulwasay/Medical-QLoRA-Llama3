# ⚕️ Medical QLoRA: Fine-Tuning Llama 3 for Clinical Q&A

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Accelerated%20by-Unsloth-FF4B4B.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-Llama--3--8B--Instruct-green.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
Large Language Models (LLMs) are incredibly powerful, but adapting them to highly specialized domains like medicine traditionally requires massive computational resources. 

**In this project, I engineered an end-to-end pipeline to fine-tune the Llama 3 (8B) model specifically for clinical question-answering.** By leveraging **QLoRA** (Quantized Low-Rank Adaptation) and the **Unsloth** library, I was able to achieve significant domain adaptation while keeping peak VRAM usage under 14GB—meaning this entire pipeline can be trained on a free, consumer-grade GPU like the Google Colab Tesla T4.

### ✨ Key Achievements & What I Built
* **Automated Data Pipeline:** Implemented a robust data ingestion and cleaning pipeline using the `ChatDoctor-HealthCareMagic-100k` dataset. The pipeline strips HTML, filters out incomplete records, enforces minimum length requirements for clinical relevance, and formats the data into the native Llama 3 chat template.
* **Memory-Optimized Training:** Configured a 4-bit NormalFloat (NF4) quantized base model and applied LoRA adapters specifically to the attention and MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). This reduced the trainable parameters to just ~41.9M (0.52% of the model).
* **Accelerated Compute:** Utilized Unsloth to achieve 2x faster training speeds and minimal memory overhead, employing an 8-bit AdamW optimizer and Cosine Learning Rate scheduling.
* **Dynamic Inference:** Built an evaluation script that seamlessly unloads the training graph, clears the GPU cache, and dynamically reloads the base model with the newly trained LoRA adapters for immediate clinical inference.

---

## 🚀 How to Run This Project

This project is optimized to run out-of-the-box on **Google Colab** using their free T4 GPU tier. 

### Prerequisites
* A Google Account (for Google Colab).
* *(Optional but recommended)* A Hugging Face account to download standard datasets without rate limits.

### Step-by-Step Instructions

**1. Set up your environment**
* Go to [Google Colab](https://colab.research.google.com/).
* Create a **New Notebook**.
* Go to `Runtime` > `Change runtime type` in the top menu.
* Select **T4 GPU** under the Hardware Accelerator dropdown and click Save.

**2. Import the Code**
* Copy the entire contents of the `medical_qlora_train.py` file from this repository.
* Paste the code into a single cell (or break it up into logical cells) in your Colab notebook.

**3. Execute the Pipeline**
* Run the cell(s). The script is designed to handle everything automatically:
  1. **Dependencies:** It will install Unsloth, PyTorch, Transformers, TRl, and PEFT.
  2. **Verification:** It will check your GPU constraints to ensure you have at least 8GB of VRAM.
  3. **Data Prep:** It will download, clean, and format the 100k+ medical dataset.
  4. **Training:** It will begin the SFT (Supervised Fine-Tuning) process, outputting loss metrics along the way.
  5. **Evaluation:** Once complete, it will save the adapters, clear the VRAM, reload the model, and run sample clinical questions (e.g., "What are the symptoms of appendicitis?").

### 📁 Outputs
Upon completion, the script generates a `outputs/medical_lora_adapter` directory containing:
* `adapter_config.json`
* `adapter_model.safetensors`

These lightweight weights (~80MB) can be loaded on top of any standard Llama 3 8B model to instantly inject medical domain knowledge.

---

## 📈 Example Output
**Prompt:** *"How is pneumonia diagnosed?"*

**Base Llama 3:** *(Provides a general, standard web-search-style answer).*
**My Fine-Tuned Model:** *(Provides a highly structured, clinically-focused response detailing physical exams, auscultation findings like crackles/rales, and specific imaging modalities like Chest X-rays and CT scans, mirroring the tone of a medical professional).*

DEVELOPED BY SHAIKH ABDUL WASAY
