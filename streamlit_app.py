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
st.markdown("**Автоматический парсинг + Hybrid RAG**")

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

# ====================== СПРАВОЧНИК ======================
district_coords = {
    "Алмалинский": (43.2567, 76.9286), "Бостандыкский": (43.2220, 76.8512),
    "Ауэзовский": (43.2560, 76.8300), "Медеуский": (43.2639, 76.9780),
    "Турксибский": (43.3170, 76.9000), "Жетысуский": (43.3000, 76.9500),
    "Наурызбайский": (43.1800, 76.8000), "Алатауский": (43.2500, 76.7500),
    "Астана": (51.1694, 71.4491), "Другой": (43.25, 76.95)
}

# ====================== ПАРСЕР ======================
def parse_listing_text(text):
    prompt = f"""
Извлеки параметры из объявления. Верни только JSON.

Текст:
{text}

Формат:
{{
  "rooms": число,
  "area": число,
  "floor": число,
  "total_floors": число,
  "district": "название района или null",
  "has_furniture": true/false,
  "has_eurorepair": true/false,
  "new_building": true/false,
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
- Рекомендации покупателю

Отвечай уверенно и по делу.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35
    )
    return response.choices[0].message.content

# ====================== SIDEBAR ======================
st.sidebar.header("📋 Параметры квартиры")

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
raw_text = st.sidebar.text_area("Вставьте текст объявления", height=140)

if st.sidebar.button("🔍 Извлечь параметры из текста"):
    if raw_text.strip():
        with st.spinner("Парсим объявление..."):
            parsed = parse_listing_text(raw_text)
            if parsed:
                st.session_state.parsed_data = parsed
                st.sidebar.success("✅ Параметры извлечены и применены!")
            else:
                st.sidebar.error("Не удалось распарсить")
    else:
        st.sidebar.warning("Введите текст")

# Автозаполнение
if st.session_state.parsed_data:
    p = st.session_state.parsed_data
    if isinstance(p.get("rooms"), (int, float)): rooms = int(p.get("rooms"))
    if isinstance(p.get("area"), (int, float)): area = float(p.get("area"))
    if isinstance(p.get("floor"), (int, float)): floor = int(p.get("floor"))
    if isinstance(p.get("total_floors"), (int, float)): total_floors = int(p.get("total_floors"))
    if p.get("district") in district_coords:
        district = p.get("district")
    if p.get("has_furniture") is not None: has_furniture = bool(p.get("has_furniture"))
    if p.get("has_eurorepair") is not None: has_eurorepair = bool(p.get("has_eurorepair"))
    if p.get("new_building") is not None: new_building = bool(p.get("new_building"))

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать", type="primary"):
    floor_ratio = floor / total_floors if total_floors > 0 else 0
    lat, lon = district_coords.get(district, (43.25, 76.95))

    features = [[total_floors, rooms, 0, lon, lat, floor_ratio, floor, 6.0, area]]
    predicted_price = abs(model.predict(features)[0])

    data = {
        "rooms": rooms, "area": area, "floor": floor, "total_floors": total_floors,
        "district": district, "has_furniture": has_furniture,
        "has_eurorepair": has_eurorepair, "new_building": new_building
    }

    similar_df = hybrid_retrieve(data, k=6)

    col1, col2 = st.columns([1.1, 2])

    with col1:
        st.metric("💰 Предсказанная цена", f"{int(predicted_price):,} ₸")
        st.subheader("📍 Параметры")
        st.write(f"**Комнаты:** {rooms}")
        st.write(f"**Площадь:** {area} м²")
        st.write(f"**Этаж:** {floor}/{total_floors}")
        st.write(f"**Район:** {district}")

    with col2:
        st.subheader("📊 AI Анализ рынка")
        with st.spinner("Генерируем отчёт..."):
            analysis = rag_explanation(data, similar_df)
            st.markdown(analysis)

    st.subheader("🔍 Сравнение с похожими квартирами")
    if 'price' in similar_df.columns:
        comp = similar_df[['rooms', 'area', 'floor', 'total_floors', 'price']].copy()
        comp['₸/м²'] = (comp['price'] / comp['area']).round(0)
        comp['Разница'] = (comp['price'] - predicted_price).round(0)
        st.dataframe(comp.style.format({
            "price": "{:,.0f} ₸",
            "₸/м²": "{:,.0f}",
            "Разница": "{:,.0f} ₸"
        }), use_container_width=True, hide_index=True)

st.caption("ValoraAI • Автоматический парсинг объявлений")
