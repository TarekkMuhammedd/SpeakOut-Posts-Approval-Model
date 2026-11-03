import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

@st.cache_resource
def load_model():
    model_id = "tarekkmuhammedd/HyperParameterTuned.NileLLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, model

st.title("🇪🇬 Egyptian Arabic Post Approval Classifier")

tokenizer, model = load_model()

user_text = st.text_area("اكتب البوست المصري هنا:", placeholder="مثلاً: البوست ده جامد جدًا")

if st.button("تحليل البوست"):
    if not user_text.strip():
        st.warning("من فضلك اكتب بوست أولاً.")
    else:
        with st.spinner("جاري تحليل البوست..."):
            prompt = f"""<start_of_turn>user
أنت مساعد ذكي، مهمتك هي مراجعة البوست التالي وتحديد إذا كان "موافق" أم "مرفوض".
البوست: "{user_text}"
القرار:<start_of_turn>model
"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    temperature=0.7,
                    top_p=0.9
                )
            new_tokens = outputs[0][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            if any(w in response.lower() for w in ["موافق", "approve", "accepted", "تمام"]):
                st.success("✅ البوست مقبول")
            else:
                st.error("❌ البوست مرفوض")

            st.caption(f"📜 استجابة الموديل: {response}")
