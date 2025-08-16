# Veyra – GPT-style Transformer Language Model  
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![GPU Required](https://img.shields.io/badge/GPU-Required-red.svg) 
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) 
![Pull Requests](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)  

Veyra is a **Python GPT-style transformer** you can train to reply however you want. It comes with:  
- 🚀 A launcher script for quick start  
- 📚 ~1,700 Q/A dataset included  
- 🛠️ Tools for training, chatting, and exporting models  
- ⚡ Advanced transformer features (RoPE, GQA, SwiGLU, Flash Attention, RMSNorm)  

> **Requires a GPU to train.**
 
## ✨ Features  
- Configurable Transformer architecture  
- SwiGLU activation, RMSNorm, RoPE, GQA, Flash Attention simulation  
- Training loop with warmup, weight decay, EMA, curriculum learning  
- Text generation with temperature, top-p, top-k, and repetition penalty  
- CLI launcher for chat, train, and export  
- Plug in your own dataset for custom fine-tuning  

## 🛠️ Installation  
git clone https://github.com/Bean758/Veyra-GPT-style-language-model-.git  
cd Veyra-GPT-style-language-model-  
pip install -r requirements.txt  

## 🚀 Usage  
Train: python Veyra.py --train  
Chat: python Veyra.py --chat  
Export: python Veyra.py --export  

## ⚙️ Default Configuration  
Architecture:  
  d_model: 768  
  n_layers: 24  
  n_head: 12  
  n_kv_head: 4  
  max_len: 2048  
Training:  
  epochs: 1000  
  batch_size: 8  
  lr: 0.0006  
  weight_decay: 0.1  
  warmup_steps: 2000  
Features:  
  use_rope: True  
  use_gqa: True  
  use_swiglu: True  
  use_flash_attention: True  
  use_rmsnorm: True  
Generation:  
  gen_len: 256  
  temperature: 0.8  
  top_p: 0.95  
  top_k: 50  
  repetition_penalty: 1.1  

## 📊 Training Sample  
Epoch [1/1000] | Loss: 3.21 | Perplexity: 24.8  
Epoch [2/1000] | Loss: 2.98 | Perplexity: 19.7  
...  

## 📢 Contributing  
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to add.  

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
