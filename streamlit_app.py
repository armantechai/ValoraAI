import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

# ==========================================
# Настройки страницы
# ==========================================
st.set_page_config(
    page_title="ValoraAI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 ValoraAI")
st.subheader("Intelligent Real Estate Valuation Platform powered by ML & AI")

# ==========================================
# OpenAI API Key
# ==========================================
load_dotenv()

if "openai_key" not in st.secrets and not os.getenv("OPENAI_API_KEY"):
    st.error("🔑 Добавьте OpenAI API ключ в Streamlit Secrets (секция `openai_key`)")
    st.stop()

client = OpenAI(api_key=st.secrets.get("openai_key") or os.getenv("OPENAI_API_KEY"))

# ==========================================
# Загрузка ресурсов
# ==========================================
@st.cache_resource
def load_resources():
    # Пути относительно папки приложения
    model_path = "model.pkl"
    csv_path = "krisha_full_with_desc.csv"
    index_path = "faiss_index.bin"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(csv_path)

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index(index_path)

    return model, df, embedding_model, index


model, df, embedding_model, index = load_resources()

# ==========================================
# Вспомогательные функции
# ==========================================
def build_document(data):
    return f"""
Комнаты: {data["rooms"]}
Площадь: {data["area"]} м²
Этаж: {data["floor"]}
Всего этажей: {data["total_floors"]}
Мебель: {data["has_furniture"]}
Евроремонт: {data["has_eurorepair"]}
Новостройка: {data["new_building"]}
"""

def retrieve_similar(query, k=5):
    query_vector = embedding_model.encode(query)
    query_vector = np.array([query_vector]).astype("float32")
    
    distances, indices = index.search(query_vector, k)
    return df.iloc[indices[0]]

def rag_explanation(data):
    query = build_document(data)
    similar = retrieve_similar(query, 5)
    
    context = "\n\n".join(similar.apply(lambda x: build_document(x), axis=1))

    prompt = f"""
Ты эксперт недвижимости Казахстана (Алматы, Астана, другие города).

ЦЕЛЕВАЯ КВАРТИРА:
{query}

ПОХОЖИЕ КВАРТИРЫ:
{context}

Объясни по целевой квартире:
- дорого/дешево/нормально относительно рынка
- почему именно такая оценка
- сравнение с похожими объектами
- 3 ключевые причины твоего вывода

Используй только предоставленные данные. Отвечай на русском.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты профессиональный аналитик недвижимости в Казахстане."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

def summarize_listing(data):
    prompt = f"""
Проанализируй эту квартиру и дай краткую карточку:

{build_document(data)}

Верни в формате:
Тип жилья:
Особенности:
Состояние:
Краткое описание:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

# ==========================================
# Sidebar
# ==========================================
st.sidebar.header("Параметры квартиры")

rooms = st.sidebar.selectbox("Комнаты", [1, 2, 3, 4, 5])
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)
has_furniture = st.sidebar.checkbox("Мебель", value=True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")
luxury = st.sidebar.checkbox("Премиум")

# ==========================================
# Анализ
# ==========================================
if st.button("🚀 Анализировать", type="primary"):
    floor_ratio = floor / total_floors if total_floors > 0 else 0

    features = [[
        area, floor, total_floors, floor_ratio,
        int(has_furniture), int(has_eurorepair),
        int(new_building), int(luxury)
    ]]

    predicted_price = model.predict(features)[0]

    data = {
        "rooms": rooms,
        "area": area,
        "floor": floor,
        "total_floors": total_floors,
        "has_furniture": has_furniture,
        "has_eurorepair": has_eurorepair,
        "new_building": new_building
    }

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💰 Предсказанная цена", f"{predicted_price:,.0f} ₸")

    with col2:
        st.subheader("📊 AI анализ рынка")
        with st.spinner("Анализируем рынок..."):
            st.write(rag_explanation(data))

    st.subheader("🧠 Карточка квартиры")
    with st.spinner("Генерируем описание..."):
        st.write(summarize_listing(data))
