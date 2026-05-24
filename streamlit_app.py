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

st.title("🏠 ValoraAI")
st.subheader("Intelligent Real Estate Valuation Platform powered by ML & AI")

# ====================== OPENAI API KEY ======================
api_key = None

# 1. Streamlit Secrets (самый надёжный способ)
if "openai" in st.secrets:
    api_key = st.secrets["openai"].get("api_key") or st.secrets["openai"].get("API_KEY")
elif "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

# 2. Переменная окружения
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

# 3. Проверка ключа
if not api_key or api_key.strip() == "" or not api_key.startswith("sk-"):
    st.error("🔑 **OpenAI API ключ не найден или некорректный!**")
    st.markdown("""
    **Как исправить:**
    1. Открой **Settings → Secrets**
    2. Вставь точно такой текст:

    ```toml
    [openai]
    api_key = "sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
""")
    st.stop()

# Создаём клиента
try:
    client = OpenAI(api_key=api_key)
    st.success("✅ OpenAI успешно подключён", icon="🔑")
except Exception as e:
    st.error(f"Ошибка подключения OpenAI: {e}")
    st.stop()

# ====================== ЗАГРУЗКА МОДЕЛЕЙ ======================
@st.cache_resource
def load_resources():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        df = pd.read_csv("krisha_full_with_desc.csv")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index("faiss_index.bin")
        return model, df, embedding_model, index
    except FileNotFoundError as e:
        st.error(f"❌ Файл не найден: {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        st.stop()

model, df, embedding_model, index = load_resources()

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
            {"role": "system", "content": "Ты профессиональный аналитик недвижимости."},
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
has_furniture = st.sidebar.checkbox("Мебель", True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")
luxury = st.sidebar.checkbox("Премиум")

# ====================== ЗАПУСК АНАЛИЗА ======================
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
