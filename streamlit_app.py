import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rank_bm25 import BM25Okapi
import os

# ====================== КОНФИГУРАЦИЯ ======================
st.set_page_config(page_title="ValoraAI", page_icon="🏠", layout="wide")

st.title("🏠 ValoraAI")
st.subheader("Интеллектуальная оценка недвижимости Казахстана")
st.markdown("**ML + Hybrid RAG (Dense + BM25) + Улучшенный AI**")

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
    
    # BM25 подготовка
    tokenized_corpus = [str(doc).split() for doc in df.get('description', df.get('text', '')).fillna('')]
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
    
    # Dense Search
    query_vec = np.array([embedding_model.encode(query_text)]).astype("float32")
    _, dense_idx = index.search(query_vec, k*3)
    dense_df = df.iloc[dense_idx[0]]
    
    # BM25 Search
    tokenized_query = query_text.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_idx = np.argsort(bm25_scores)[-k*3:][::-1]
    bm25_df = df.iloc[bm25_idx]
    
    # Комбинируем и убираем дубли
    combined = pd.concat([dense_df, bm25_df]).drop_duplicates().head(k)
    return combined

def rag_explanation(data, similar_df):
    query = build_document(data)
    context = "\n\n".join([build_document(row) for _, row in similar_df.iterrows()])

    prompt = f"""
Ты — профессиональный риелтор-аналитик с большим опытом на рынке Казахстана.

**ЦЕЛЕВАЯ КВАРТИРА:**
{query}

**ПОХОЖИЕ ПРЕДЛОЖЕНИЯ:**
{context}

Дай подробный и честный анализ:
- Насколько справедлива цена (дорого / дешево / по рынку)?
- Какие факторы сильнее всего влияют на стоимость?
- 3 главные причины твоего вывода.
- Практические рекомендации покупателю.

Отвечай естественно, уверенно и по делу на русском языке.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.35,
        max_tokens=750
    )
    return response.choices[0].message.content

# ====================== SIDEBAR ======================
st.sidebar.header("📋 Параметры квартиры")

rooms = st.sidebar.selectbox("Комнаты", [1,2,3,4,5])
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)
district = st.sidebar.selectbox("Район", list(district_coords.keys()))

has_furniture = st.sidebar.checkbox("Мебель", True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")

st.sidebar.markdown("---")
st.sidebar.subheader("📝 Данные из объявления")
user_description = st.sidebar.text_area("Описание объявления", height=120)
real_price = st.sidebar.number_input("Реальная цена из объявления (₸)", min_value=0, value=0, step=10000)

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать", type="primary"):
    floor_ratio = floor / total_floors if total_floors > 0 else 0
    lat, lon = district_coords[district]
    distance_to_center = 2.5 if district in ["Алмалинский", "Медеуский"] else 6.0

    features = [[total_floors, rooms, 0, lon, lat, floor_ratio, floor, distance_to_center, area]]
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
        if real_price > 0:
            diff = real_price - predicted_price
            st.metric("📌 Цена в объявлении", f"{int(real_price):,} ₸", 
                     delta=f"{int(diff):,} ₸ {'дороже' if diff > 0 else 'дешевле'}")

        st.subheader("📍 Параметры")
        st.write(f"**Комнаты:** {rooms}")
        st.write(f"**Площадь:** {area} м²")
        st.write(f"**Этаж:** {floor}/{total_floors}")
        st.write(f"**Район:** {district}")

    with col2:
        st.subheader("📊 AI Анализ рынка")
        with st.spinner("Генерируем качественный отчёт..."):
            analysis = rag_explanation(data, similar_df)
            st.markdown(analysis)

    st.subheader("🔍 Сравнение с похожими квартирами")
    if 'price' in similar_df.columns:
        comp = similar_df[['rooms', 'area', 'floor', 'total_floors', 'price']].copy()
        comp['₸/м²'] = (comp['price'] / comp['area']).round(0)
        comp['Разница'] = (comp['price'] - predicted_price).round(0)
        st.dataframe(comp.style.format({"price": "{:,.0f} ₸", "₸/м²": "{:,.0f}", "Разница": "{:,.0f} ₸"}), 
                     use_container_width=True, hide_index=True)

st.caption("ValoraAI • Hybrid RAG + Улучшенный промпт")
