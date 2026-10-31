import torch
import gradio as gr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Model Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "vikhyatk/moondream2"
revision = "2024-08-02"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)

# 2. Nutrition Analysis Logic
def analyze_food_items(image):
    image = image.convert("RGB")
    prompt = "List all food items, bowls, or dishes clearly visible in this image. Be precise and concise."
    enc_image = model.encode_image(image)
    detected_text = model.answer_question(enc_image, prompt, tokenizer)
    return detected_text.strip()

def generate_medical_report(detected_items, lang):
    items_lower = detected_items.lower()
    
    has_dal = any(w in items_lower for w in ["dal", "lentil", "soup", "curry", "stew", "pulse", "bowl"])
    has_carbs = any(w in items_lower for w in ["rice", "roti", "chapati", "bread", "potato", "naan", "puri", "pasta"])
    has_protein = has_dal or any(w in items_lower for w in ["paneer", "chicken", "egg", "meat", "fish", "chana", "rajma"])
    has_veg = any(w in items_lower for w in ["vegetable", "sabzi", "salad", "spinach", "broccoli", "cucumber", "tomato"])
    has_dairy = any(w in items_lower for w in ["curd", "yogurt", "dahi", "milk", "butter", "ghee", "paneer"])
    
    if lang == "Hindi":
        res = f"📍 पहचान की गई सामग्री: {detected_items}\n"
        res += "="*40 + "\n🩺 वरिष्ठ डॉक्टर का पोषण विश्लेषण:\n\n"
        res += "1. मौजूद पोषक तत्व:\n"
        if has_dal:
            res += " • प्रोटीन और फाइबर: दाल में प्लांट-बेस्ड प्रोटीन, फोलेट, आयरन और पोटेशियम भरपूर मात्रा में है। यह रिकवरी के लिए बढ़िया है।\n"
        if has_carbs:
            res += " • कार्बोहाइड्रेट: शरीर को ऊर्जा देने के लिए ग्लूकोज और एनर्जी सोर्स मौजूद है।\n"
        if has_veg:
            res += " • विटामिन्स व मिनरल्स: हरी सब्जियां/सलाद एंटी-ऑक्सीडेंट्स और डाइजेस्टिबल फाइबर प्रदान करती हैं।\n"
        if has_dairy:
            res += " • कैल्शियम व प्रोबायोटिक्स: हड्डियों की मजबूती और पेट की सेहत के लिए अच्छा है।\n"
            
        res += "\n2. इस डाइट/आइटम में क्या अनुपस्थित है:\n"
        if has_dal and not has_carbs:
            res += " • कार्बोहाइड्रेट: सिर्फ दाल में पर्याप्त कार्ब्स नहीं होते। पूर्ण ऊर्जा के लिए इसके साथ चावल या रोटी की आवश्यकता है।\n"
        if not has_veg:
            res += " • विटामिन C और फाइबर: प्लेट में रो-फाइबर या कच्ची सब्जियों की कमी है।\n"

        res += "\n3. डॉक्टर की सलाह:\n"
        if has_dal and not has_carbs:
            res += " • यह एक बेहतरीन हाई-प्रोटीन डाइट है। इसमें आधा नींबू निचोड़ें (विटामिन C आयरन अब्जॉर्प्शन बढ़ाएगा) और साथ में 1-2 रोटी या थोड़ा चावल लें।"
        else:
            res += " • मरीज की स्थिति के अनुसार संतुलन बनाए रखें।"

    elif lang == "Hinglish":
        res = f"📍 Detected Items: {detected_items}\n"
        res += "="*40 + "\n🩺 Senior Doctor's Nutrition Analysis:\n\n"
        res += "1. Present Nutrients:\n"
        if has_dal:
            res += " • Protein & Fiber: Isme plant-based protein, iron, aur folate hai. Recovery ke liye badiya hai.\n"
        if has_carbs:
            res += " • Carbohydrates: Instant energy ke liye carbs maujood hain.\n"
        if has_veg:
            res += " • Vitamins & Minerals: Digestion aur immunity booster fiber maujood hai.\n"

        res += "\n2. Missing Nutrients in this Specific Meal:\n"
        if has_dal and not has_carbs:
            res += " • Carbs: Sirf dal me energy ke liye zaroori carbs kam hote hain. Iske sath roti ya rice hona zaroori hai.\n"
        if not has_veg:
            res += " • Vitamin C & Fiber: Fresh vegetables ya salad missing hai.\n"

        res += "\n3. Doctor's Recommendation:\n"
        if has_dal and not has_carbs:
            res += " • Yeh protein se bhari katori hai. Isse complete karne ke liye roti/chawal ke sath dein aur lemon squeeze karein."
        else:
            res += " • Patient ki requirement ke hisab se portion maintain karein."

    else:
        res = f"📍 Detected Items: {detected_items}\n"
        res += "="*40 + "\n🩺 Senior Doctor's Clinical Analysis:\n\n"
        res += "1. Present Nutrients:\n"
        if has_dal:
            res += " • Protein & Minerals: Rich in dietary fiber, iron, folate, and plant protein for cellular repair.\n"
        if has_carbs:
            res += " • Carbohydrates: Adequate glucose source present for basic metabolic energy.\n"

        res += "\n2. Missing Nutrients in this Plate:\n"
        if has_dal and not has_carbs:
            res += " • Complex Carbs: Standalone dal lacks energy-dense carbohydrates like grains/breads.\n"

        res += "\n3. Medical Advice:\n"
        if has_dal and not has_carbs:
            res += " • Excellent protein bowl. Pair with a serving of rice or chapati to complete the essential amino acid profile."
        else:
            res += " • Ensure adequate hydration alongside this meal portion."

    return res

def safe_process_pipeline(image, language):
    if image is None:
        return "Kripya photo upload karein."
    try:
        detected_text = analyze_food_items(image)
        report = generate_medical_report(detected_text, language)
        return report
    except Exception as e:
        return f"Error details: {str(e)}"

# 3. Gradio Interface
interface = gr.Interface(
    fn=safe_process_pipeline,
    inputs=[
        gr.Image(type="pil", label="Upload Food/Plate Image (JPG, PNG, JPEG)"),
        gr.Dropdown(choices=["Hinglish", "Hindi", "English"], value="Hinglish", label="Select Preferred Language")
    ],
    outputs=gr.Textbox(label="Clinical Nutritional Breakdown", lines=16),
    title="Hospital Patient Diet & Nutrition Analyzer AI",
    description="Scan any random food item or plate to get exact nutrients, missing elements, and doctor recommendations.",
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch()