import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rank_bm25 import BM25Okapi
import os
import json

# ====================== КОНФИГУРАЦИЯ ======================
st.set_page_config(page_title="ValoraAI", page_icon="🏠", layout="wide")

st.title("🏠 ValoraAI")
st.subheader("Интеллектуальная оценка недвижимости Казахстана")
st.markdown("**ML + Hybrid RAG + Автоматический парсинг объявлений**")

# ====================== OPENAI ======================
api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API ключ не настроен")
    st.stop()

client = OpenAI(api_key=api_key)

# ====================== ЗАГРУЗКА ======================
@st.cache_resource(show_spinner="Загружаем модели...")
def load_resources():
    model = pickle.load(open("model.pkl", "rb"))
    df = pd.read_csv("krisha_full_with_desc.csv")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    index = faiss.read_index("faiss_index.bin")
    
    tokenized_corpus = [str(doc).split() for doc in df.get('description', '').fillna('')]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return model, df, embedding_model, index, bm25

model, df, embedding_model, index, bm25 = load_resources()

# ====================== ПАРСЕР ======================
def parse_listing_text(text):
    prompt = f"""
Извлеки параметры из объявления о продаже квартиры. Верни только JSON.

Текст объявления:
{text}

Формат ответа (строго JSON):
{{
  "rooms": число,
  "area": число,
  "floor": число,
  "total_floors": число,
  "district": "название района или null",
  "has_furniture": true/false,
  "has_eurorepair": true/false,
  "new_building": true/false,
  "luxury": true/false,
  "price": число или null
}}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=400
        )
        parsed = json.loads(response.choices[0].message.content.strip())
        return parsed
    except:
        return None

# ====================== ФУНКЦИИ ======================
def build_document(data):
    return f"""Квартира:
- Комнат: {data.get("rooms")}
- Площадь: {data.get("area")} м²
- Этаж: {data.get("floor")} из {data.get("total_floors")}
- Район: {data.get("district")}
- Мебель: {"Есть" if data.get("has_furniture") else "Нет"}
- Евроремонт: {"Есть" if data.get("has_eurorepair") else "Нет"}
- Новостройка: {"Да" if data.get("new_building") else "Нет"}"""

def hybrid_retrieve(data, k=6):
    query_text = build_document(data)
    query_vec = np.array([embedding_model.encode(query_text)]).astype("float32")
    _, dense_idx = index.search(query_vec, k*3)
    dense_df = df.iloc[dense_idx[0]]
    
    tokenized_query = query_text.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = np.argsort(bm25_scores)[-k*3:][::-1]
    bm25_df = df.iloc[bm25_idx]
    
    combined = pd.concat([dense_df, bm25_df]).drop_duplicates().head(k)
    return combined

def rag_explanation(data, similar_df):
    query = build_document(data)
    context = "\n\n".join([build_document(row) for _, row in similar_df.iterrows()])

    prompt = f"""
Ты — профессиональный риелтор-аналитик в Казахстане.

**ЦЕЛЕВАЯ КВАРТИРА:**
{query}

**ПОХОЖИЕ ОБЪЕКТЫ:**
{context}

Дай чёткий анализ:
- Дорого / Дешево / По рынку?
- Почему?
- 3 ключевые причины
- Рекомендации покупателю (стоит ли брать, на что обратить внимание, риски)

Отвечай уверенно и по делу на русском.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35
    )
    return response.choices[0].message.content

# ====================== SIDEBAR ======================
st.sidebar.header("📋 Параметры квартиры")

# Инициализация session_state
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None

rooms = st.sidebar.selectbox("Комнаты", [1,2,3,4,5], index=1)
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)
district = st.sidebar.selectbox("Район", list(district_coords.keys()), index=0)

has_furniture = st.sidebar.checkbox("Мебель", True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Парсинг объявления")
raw_text = st.sidebar.text_area("Вставьте текст объявления", height=150)

if st.sidebar.button("🔍 Извлечь параметры"):
    if raw_text.strip():
        with st.spinner("Парсим объявление..."):
            parsed = parse_listing_text(raw_text)
            if parsed:
                st.session_state.parsed_data = parsed
                st.sidebar.success("✅ Параметры извлечены!")
            else:
                st.sidebar.error("Не удалось распарсить")
    else:
        st.sidebar.warning("Введите текст")

# Автозаполнение из session_state
if st.session_state.parsed_data:
    p = st.session_state.parsed_data
    rooms = p.get("rooms", rooms)
    area = p.get("area", area)
    floor = p.get("floor", floor)
    total_floors = p.get("total_floors", total_floors)
    district = p.get("district") if p.get("district") in district_coords else district
    has_furniture = p.get("has_furniture", has_furniture)
    has_eurorepair = p.get("has_eurorepair", has_eurorepair)
    new_building = p.get("new_building", new_building)

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать", type="primary"):
    # ... (твой код расчёта predicted_price и data)

    data = {
        "rooms": rooms, "area": area, "floor": floor, "total_floors": total_floors,
        "district": district, "has_furniture": has_furniture,
        "has_eurorepair": has_eurorepair, "new_building": new_building
    }

    similar_df = hybrid_retrieve(data, k=6)

    # Вывод результатов (col1, col2, таблица сравнения) — как в предыдущих версиях

st.caption("ValoraAI • Автоматический парсинг + Hybrid RAG")
