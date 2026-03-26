# ⚕️ Medical QLoRA: Fine-Tuning Llama 3 for Clinical Q&A

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Unsloth](https://img.shields.io/badge/Accelerated%20by-Unsloth-FF4B4B.svg)](https://github.com/unslothai/unsloth)
[![Model](https://img.shields.io/badge/Model-Llama--3--8B--Instruct-green.svg)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

This project implements a specialized fine-tuning pipeline for the **Llama 3 8B Instruct** model, optimized for the medical domain. By leveraging **QLoRA (Quantized Low-Rank Adaptation)** and the **Unsloth** library, it achieves significant domain adaptation while maintaining high computational efficiency (runnable on a single 16GB GPU like a Tesla T4).

## 📊 Technical Architecture
* **Base Model:** Llama 3 8B Instruct (4-bit NF4 Quantization)
* **Training Method:** QLoRA via Unsloth
* **Dataset:** [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k) (111,726 high-quality clinical Q&A pairs post-cleaning)
* **Optimization:** 8-bit AdamW optimizer with Cosine Learning Rate scheduling

**LoRA Hyperparameters:**
* Rank ($r$): 16
* Alpha ($\alpha$): 16
* Target Modules: All linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
* Trainable Parameters: ~41.9M (0.52% of total)

## 🚀 Quick Start (Google Colab)

This script is optimized for Google Colab. 
1. Open a new Google Colab notebook and select a **T4 GPU** runtime.
2. Upload the `medical_qlora_train.py` script or copy its contents into the notebook.
3. Run the script. It will automatically install Unsloth, mount your drive (optional), prepare the dataset, and begin training.

## 📈 Key Results
* **Memory Efficiency:** Peak VRAM usage was stabilized under 14GB.
* **Qualitative Success:** In clinical inference tests, the model provides accurate, structured responses for medical queries such as appendicitis symptoms and pneumonia diagnosis.

## 📁 Artifacts Produced
The script will output the fine-tuned LoRA adapters to `outputs/medical_lora_adapter` (~80MB), which can be loaded dynamically over the base model for inference.
