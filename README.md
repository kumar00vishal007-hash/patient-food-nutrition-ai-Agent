# 🩺 Hospital Patient Diet & Nutrition Analyzer AI

An open-source Vision-Language AI model designed to analyze hospital patients' food items, bowls, and full meal plates. It provides dynamic nutrient breakdown, identifies missing elements, and offers senior nutritionist recommendations without relying on external paid APIs.

---

## 🌟 Key Features
- **Dynamic Food Scanning**: Accurately detects standalone items (e.g., single bowl of dal) as well as full meal thalis.
- **Nutritional & Deficit Analysis**: Identifies Carbohydrates, Proteins, Fiber, Micronutrients, and flags missing essential nutrients.
- **Multilingual Support**: Provides clinical outputs in **Hindi**, **Hinglish**, and **English**.
- **Zero External API Cost**: Powered by the local open-source Vision-Language Model (`vikhyatk/moondream2`).
- **Universal Image Support**: Handles JPG, JPEG, and PNG image formats cleanly via Gradio UI.

---

## 🛠️ Tech Stack
- **Language**: Python 3.10+
- **Vision-Language Model**: `Moondream2` (PyTorch / Hugging Face Transformers)
- **Web UI**: Gradio
- **Image Processing**: Pillow (PIL)

---

## 🚀 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/](https://github.com/)<YOUR-USERNAME>/hospital-diet-nutrition-analyzer.git
   cd hospital-diet-nutrition-analyzer
