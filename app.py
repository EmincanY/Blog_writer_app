import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import os

# Ortam değişkenlerinden API anahtarını al
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Şirket bilgisi (tek seferlik yükleme için)
company_info = "A construction company specializing in cost estimation and unit pricing."

# Üst bilgi
st.markdown("<h5>Bu websitesi Emincan Yılmaz tarafından Muhammed Baş için hazırlanmıştır.</h5>", unsafe_allow_html=True)

# Modellerin yüklenmesi ve seçilmesi
def load_model(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name.startswith("GPT"):
        return client, None  # OpenAI API client'i ile çağrı yapılacak
    elif model_name == "Llama 3.1 - 70B":
        tokenizer = AutoTokenizer.from_pretrained('nvidia/Llama-3.1-Nemotron-70B-Instruct-HF', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
            device_map="auto",
            load_in_8bit=True,
            max_memory={0: "100GB", "cpu": "30GB"},
            trust_remote_code=True
        )
    elif model_name == "SuperNova - Medius Model":
        tokenizer = AutoTokenizer.from_pretrained('arcee-ai/SuperNova-Medius', trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            'arcee-ai/SuperNova-Medius',
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
    return tokenizer, model

# Prompt oluşturma
def create_prompt(company_info, company_info_detail, keywords, topic, length, language="English"):
    detail_mapping = {
        "Çok": "in-depth",
        "Orta": "moderate detail",
        "Az": "minimal mention",
        "Hiç": "no mention"
    }
    detail_level = detail_mapping[company_info_detail]
    return (f"Create a blog post with a title and content in {language} about '{topic}'. "
            f"Include the following keywords: {keywords}. Make the blog {length} words long. "
            f"Provide {detail_level} about the company details: {company_info}.")

# Basit arayüz tasarımı
st.title("Mami İçin Blog Yazısı Oluşturucu")
length = st.number_input("Kelime Sayısı", min_value=100, max_value=1000, step=50, value=300)
topic = st.text_input("Konu", "Average Cost and Worth")
keywords = st.text_input("Anahtar Kelimeler (virgülle ayırarak)", "Construction Payment, Unit Pricing")
language = st.selectbox("Yazılacak Blog Dili", ["English", "Turkish"])
company_info_detail = st.selectbox("Şirketten Bahsetme Düzeyi", ["Çok", "Orta", "Az", "Hiç"])

# Model seçeneklerini ekleyelim
gpt_models = [
    "gpt-3.5-turbo-0125", "gpt-3.5-turbo", "gpt-3.5-turbo-1106",
    "gpt-4-turbo", "gpt-4-turbo-2024-04-09", "gpt-4-turbo-preview", "gpt-4-0125-preview",
    "gpt-4-1106-preview", "gpt-4", "gpt-4-0613",
    "gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-mini", "gpt-4o-mini-2024-07-18"
]
other_models = ["Llama 3.1 - 70B", "SuperNova - Medius Model"]
all_models = gpt_models + other_models

model_choice = st.selectbox("Model Seçin", all_models)

# Prompt oluşturma
prompt = create_prompt(company_info, company_info_detail, keywords, topic, length, language)

# Model çalıştırma ve sonuç gösterme
if st.button("Blog Yazısını Oluştur"):
    st.write("Blog oluşturuluyor...")

    if model_choice in gpt_models:
        # OpenAI API'yi kullanarak çağrı
        assistant = client.beta.assistants.create(
            name="Content Creator",
            instructions="You are a blog post creator that writes about company topics with specific keywords.",
            tools=[{"type": "code_interpreter"}],
            model=model_choice,  # Model ismi artık doğrudan API endpoint'e uyuyor
        )
        
        # Yeni bir thread (sohbet dizisi) başlatma
        thread = client.beta.threads.create()
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )
        
        # Asistan çalıştırma ve sonucu bekleme
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id,
            instructions="Write in a professional tone for a Turkish-speaking audience."
        )

        if run.status == 'completed': 
            # Mesajları listeleyin ve gösterin
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            for message in messages:
                # Yalnızca asistanın yanıtını alıyoruz
                if hasattr(message, 'role') and message.role == 'assistant':
                    # Blog yazısını sadece metin olarak göster
                    blog_content = message.content[0].text.value  # İlk içeriği al ve text'in değerini göster
                    st.write("### Blog Yazısı")
                    st.write(blog_content)
        
    else:
        # Diğer modeller için
        tokenizer, model = load_model(model_choice)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Modelin ürettiği yanıtı `prompt` içermeyecek şekilde alıyoruz
        outputs = model.generate(**inputs, max_new_tokens=length, temperature=0.7, top_p=0.9, do_sample=True)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # `result` değişkeninden prompt'u çıkarıyoruz (eğer prompt'la başlıyorsa)
        if result.startswith(prompt):
            result = result[len(prompt):].strip()  # Prompt'u çıkar
        
        st.write("### Blog Yazısı")
        st.write(result)

# Alt bilgi
st.markdown("<hr><div style='text-align: center;'>Designed by Emincan Yılmaz</div>", unsafe_allow_html=True)
