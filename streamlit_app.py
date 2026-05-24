import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ====================== КОНФИГУРАЦИЯ ======================
st.set_page_config(page_title="ValoraAI", page_icon="🏠", layout="wide")

st.title("DEMO")
st.title("🏠 ValoraAI")
st.subheader("Интеллектуальная оценка недвижимости Казахстана")
st.markdown("**Машинное обучение + RAG + Сравнение с рынком**")

# ====================== OPENAI ======================
api_key = st.secrets.get("openai", {}).get("api_key") or os.getenv("OPENAI_API_KEY")
if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API ключ не настроен в Secrets")
    st.stop()

client = OpenAI(api_key=api_key)

# ====================== ЗАГРУЗКА ======================
@st.cache_resource(show_spinner="Загружаем модели...")
def load_resources():
    model = pickle.load(open("model.pkl", "rb"))
    df = pd.read_csv("krisha_full_with_desc.csv")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.bin")
    return model, df, embedding_model, index

model, df, embedding_model, index = load_resources()

# ====================== СПРАВОЧНИК ======================
district_coords = {
    "Алмалинский": (43.2567, 76.9286), "Бостандыкский": (43.2220, 76.8512),
    "Ауэзовский": (43.2560, 76.8300), "Медеуский": (43.2639, 76.9780),
    "Турксибский": (43.3170, 76.9000), "Жетысуский": (43.3000, 76.9500),
    "Наурызбайский": (43.1800, 76.8000), "Алатауский": (43.2500, 76.7500),
    "Астана": (51.1694, 71.4491), "Другой": (43.25, 76.95)
}

# ====================== ФУНКЦИИ ======================
def build_document(data):
    return f"""Комнаты: {data.get("rooms")}
Площадь: {data.get("area")} м²
Этаж: {data.get("floor")}/{data.get("total_floors")}
Мебель: {"Да" if data.get("has_furniture") else "Нет"}
Евроремонт: {"Да" if data.get("has_eurorepair") else "Нет"}
Новостройка: {"Да" if data.get("new_building") else "Нет"}"""

def retrieve_similar(query, k=6):
    query_vector = np.array([embedding_model.encode(query)]).astype("float32")
    _, indices = index.search(query_vector, k)
    return df.iloc[indices[0]]

def rag_explanation(data, similar_df):
    query = build_document(data)
    context = "\n\n".join([build_document(row) for _, row in similar_df.iterrows()])

    prompt = f"""
Ты эксперт недвижимости Казахстана.

ЦЕЛЕВАЯ КВАРТИРА:
{query}

ПОХОЖИЕ КВАРТИРЫ:
{context}

Сравни целевую квартиру с рынком:
- Дорого / Дешево / По рынку?
- Почему?
- 3 ключевые причины
- Краткие рекомендации

Отвечай на русском, понятно и профессионально.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# ====================== SIDEBAR ======================
st.sidebar.header("Параметры квартиры")

rooms = st.sidebar.selectbox("Комнаты", [1,2,3,4,5])
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)
district = st.sidebar.selectbox("Район", list(district_coords.keys()))

has_furniture = st.sidebar.checkbox("Мебель", True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать и сравнить", type="primary"):
    floor_ratio = floor / total_floors if total_floors > 0 else 0
    lat, lon = district_coords[district]
    distance_to_center = 2.5 if district in ["Алмалинский", "Медеуский"] else 6.0

    features = [[total_floors, rooms, 0, lon, lat, floor_ratio, floor, distance_to_center, area]]
    predicted_price = model.predict(features)[0]

    data = {
        "rooms": rooms, "area": area, "floor": floor, "total_floors": total_floors,
        "has_furniture": has_furniture, "has_eurorepair": has_eurorepair,
        "new_building": new_building
    }

    # Получаем похожие квартиры
    similar_df = retrieve_similar(build_document(data), k=6)

    col1, col2 = st.columns([1.2, 2])

    with col1:
        st.metric("💰 Предсказанная цена", f"{int(predicted_price):,} ₸", delta=None)
        st.subheader("📍 Параметры")
        st.json(data, expanded=False)

    with col2:
        st.subheader("📊 AI Анализ и сравнение с рынком")
        with st.spinner("Генерируем аналитику..."):
            analysis = rag_explanation(data, similar_df)
            st.markdown(analysis)

    # ====================== СРАВНЕНИЕ ЦЕН ======================
    st.subheader("🔍 Сравнение с похожими квартирами")

    comparison = similar_df.copy()
    comparison = comparison[['rooms', 'area', 'floor', 'total_floors', 'price']]  # предполагаем, что в df есть колонка 'price'
    comparison['price_per_m2'] = (comparison['price'] / comparison['area']).round(0)
    comparison['difference'] = (comparison['price'] - predicted_price).round(0)

    st.dataframe(
        comparison.style.format({
            "price": "{:,.0f} ₸",
            "price_per_m2": "{:,.0f} ₸/м²",
            "difference": "{:,.0f} ₸"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Средняя цена похожих
    avg_price = similar_df['price'].mean()
    st.info(f"Средняя цена похожих квартир: **{int(avg_price):,} ₸**")

st.caption("ValoraAI © 2026 • ML + RAG • Казахстан")
