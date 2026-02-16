# Lumina4K 🎮✨  
A GPU-accelerated AI texture remastering engine Optimized for games, art, and wallpapers.
Helps to upscale images and game textures while preserving structure and detail. 
Build using PyTorch + Cuda for high performance inference.

---

## 🚀 Features

✔ General Image Upscaling  
✔ Batch Folder Processing  
✔ Game Mode (Preserves Folder Structure for Texture Remastering)  
✔ GPU Acceleration (CUDA)  
✔ Dark Gamer UI  

---

## 🎯 Use Cases

- Remaster old game textures
- Improve low-resolution assets
- Batch upscale image datasets
- Texture enhancement for modding

---

## 🧠 AI Backbone

Lumina4K uses super-resolution deep learning models:

- Real-ESRGAN
- CNN-based Super Resolution
- GAN-based detail enhancement

Inference runs on GPU via PyTorch + CUDA.

---

## 🏗 Architecture

```
app/   → UI + Application Layer  
core/  → AI Inference + Processing  
models/ → Pretrained Weights  
```

---

## 📦 Installation

```bash
git clone https://github.com/Adityaraj614/Lumina4K.git
cd Lumina4K
pip install -r requirements.txt
```

Download model weights and place inside `/models`.

---

## 🖥 Run Application

```bash
python app/main.py
```

---

## 📚 Academic Relevance (Advanced Computer Vision)

This project demonstrates:

- Image Super-Resolution
- Deep Learning Inference Pipelines
- GAN-based Enhancement
- GPU Optimization
- Real-world CV Deployment

---

## 👨‍💻 Author

3rd Year CSE Student  
Focused on AI, Game Tech & Decision Systems
