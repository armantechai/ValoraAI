import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# ====================== НАСТРОЙКИ ======================
st.set_page_config(
    page_title="ValoraAI",
    page_icon="🏠",
    layout="wide"
)

st.title("DEMO")
st.title("🏠 ValoraAI")
st.subheader("Intelligent Real Estate Valuation Platform powered by ML & AI")

# ====================== OPENAI ======================
api_key = None
if "openai" in st.secrets:
    api_key = st.secrets["openai"].get("api_key") or st.secrets["openai"].get("API_KEY")
elif "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key or not api_key.startswith("sk-"):
    st.error("🔑 OpenAI API ключ не найден!")
    st.info("Добавьте ключ в Settings → Secrets")
    st.stop()

client = OpenAI(api_key=api_key)
st.success("✅ OpenAI подключён", icon="🔑")
st.info("👇 Настрой параметры слева и нажми «Анализировать»")

# ====================== ЗАГРУЗКА МОДЕЛЕЙ ======================
@st.cache_resource(show_spinner="Загружаем модели...")
def load_resources():
    try:
        with st.spinner("Загружаем модель ценообразования..."):
            model = pickle.load(open("model.pkl", "rb"))
        
        with st.spinner("Загружаем базу квартир..."):
            df = pd.read_csv("krisha_full_with_desc.csv")
        
        with st.spinner("Загружаем эмбеддинги..."):
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        with st.spinner("Загружаем индекс поиска..."):
            index = faiss.read_index("faiss_index.bin")
            
        return model, df, embedding_model, index
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.info("Проверь, что все файлы (model.pkl, faiss_index.bin, krisha_full_with_desc.csv) загружены в репозиторий")
        st.stop()

model, df, embedding_model, index = load_resources()

# ====================== СПРАВОЧНИК КООРДИНАТ ======================
district_coords = {
    "Алмалинский": (43.2567, 76.9286),
    "Бостандыкский": (43.2220, 76.8512),
    "Ауэзовский": (43.2560, 76.8300),
    "Медеуский": (43.2639, 76.9780),
    "Турксибский": (43.3170, 76.9000),
    "Жетысуский": (43.3000, 76.9500),
    "Наурызбайский": (43.1800, 76.8000),
    "Алатауский": (43.2500, 76.7500),
    "Астана": (51.1694, 71.4491),
    "Другой": (43.25, 76.95)
}

# ====================== ФУНКЦИИ ======================
def build_document(data):
    return f"""Комнаты: {data.get("rooms")}
Площадь: {data.get("area")} м²
Этаж: {data.get("floor")}
Всего этажей: {data.get("total_floors")}
Мебель: {data.get("has_furniture")}
Евроремонт: {data.get("has_eurorepair")}
Новостройка: {data.get("new_building")}"""

def retrieve_similar(query, k=5):
    query_vector = embedding_model.encode(query)
    query_vector = np.array([query_vector]).astype("float32")
    _, indices = index.search(query_vector, k)
    return df.iloc[indices[0]]

def rag_explanation(data):
    query = build_document(data)
    similar = retrieve_similar(query, 5)
    context = "\n\n".join([build_document(row) for _, row in similar.iterrows()])

    prompt = f"""
Ты эксперт недвижимости Казахстана.

ЦЕЛЕВАЯ КВАРТИРА:
{query}

ПОХОЖИЕ КВАРТИРЫ:
{context}

Оцени: дорого / дешево / нормально? Почему? Приведи 3 причины. Отвечай на русском.
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
    prompt = f"Составь краткую карточку квартиры:\n{build_document(data)}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# ====================== SIDEBAR ======================
st.sidebar.header("Параметры квартиры")

rooms = st.sidebar.selectbox("Комнаты", [1, 2, 3, 4, 5])
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)

district = st.sidebar.selectbox("Район", list(district_coords.keys()))
has_furniture = st.sidebar.checkbox("Мебель", True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")
luxury = st.sidebar.checkbox("Премиум")

# ====================== АНАЛИЗ ======================
if st.button("🚀 Анализировать", type="primary"):
    floor_ratio = floor / total_floors if total_floors > 0 else 0

    lat, lon = district_coords[district]
    
    # Примерное расстояние до центра (упрощённо)
    distance_to_center = 2.5 if district in ["Алмалинский", "Медеуский"] else \
                        5.0 if district in ["Бостандыкский", "Ауэзовский"] else 8.0

    # price_per_m2 оставляем 0 (модель его может игнорировать при predict)
    price_per_m2 = 0

    features = [[
        total_floors,      # 1
        rooms,             # 2
        price_per_m2,      # 3
        lon,               # 4
        lat,               # 5
        floor_ratio,       # 6
        floor,             # 7
        distance_to_center,# 8
        area               # 9
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

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("💰 Предсказанная цена", f"{int(predicted_price):,} ₸")

    with col2:
        st.subheader("📊 AI анализ рынка")
        with st.spinner("Анализируем рынок..."):
            st.markdown(rag_explanation(data))

    st.subheader("🧠 Карточка квартиры")
    with st.spinner("Генерируем описание..."):
        st.markdown(summarize_listing(data))
