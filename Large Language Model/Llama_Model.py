import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="💬 Chatbot LLaMA 2", layout="wide")
st.title("🤖 Chatbot LLaMA 2-7B ")

# --- TOKEN Hugging Face ---
HF_TOKEN = "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"  # Thay bằng token thật

@st.cache_resource(show_spinner="🔄 Đang tải mô hình...")
def load_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        token=HF_TOKEN
    )
    
    # Tăng tốc bằng torch.compile nếu có
    try:
        model = torch.compile(model)
    except Exception as e:
        st.warning(f"⚠️ Không thể compile mô hình: {e}")

    return tokenizer, model

tokenizer, model = load_model()
st.success("✅ Mô hình đã tải xong!")

# --- Khởi tạo lịch sử ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Nhập prompt ---
user_input = st.chat_input("💬 Nhập câu hỏi của bạn...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    def format_chat(chat_history, last_n_turns=3):
        prompt = ""
        recent_turns = chat_history[-last_n_turns*2:]  # Mỗi lượt gồm user + bot
        for turn in recent_turns:
            if turn["role"] == "user":
                prompt += f"[INST] {turn['content']} [/INST]\n"
            else:
                prompt += f"{turn['content']}\n"
        return prompt.strip()

    prompt = format_chat(st.session_state.chat_history)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with torch.inference_mode():
            response = model.generate(
                **inputs,
                max_new_tokens=64,       # Giới hạn độ dài để tăng tốc
                do_sample=False,         # Không ngẫu nhiên
                temperature=0.0          # Câu trả lời ổn định
            )

    output_text = tokenizer.decode(response[0], skip_special_tokens=True)
    answer = output_text.replace(prompt, "").strip()

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)