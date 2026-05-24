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

    # Проверка файлов
    required_files = [
        "model.pkl",
        "krisha_full_with_desc.csv",
        "faiss_index.bin"
    ]

    for file in required_files:
        if not os.path.exists(file):
            st.error(f"❌ Не найден файл: {file}")
            st.stop()

    # Загрузка модели
    model = pickle.load(open("model.pkl", "rb"))

    # Датасет
    df = pd.read_csv("krisha_full_with_desc.csv")

    # Проверка description
    if 'description' not in df.columns:
        df['description'] = ''

    df['description'] = (
        df['description']
        .fillna('')
        .astype(str)
    )

    # Embedding модель
    embedding_model = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu"
    )

    # FAISS
    index = faiss.read_index("faiss_index.bin")

    # BM25
    tokenized_corpus = [
        doc.split()
        for doc in df['description']
    ]

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

# ====================== УЛУЧШЕННЫЙ ПАРСЕР ======================
def parse_listing_text(text):
    prompt = f"""
Ты эксперт по парсингу объявлений недвижимости с Krisha.kz.
Извлеки максимум информации из текста ниже и верни **только** JSON.

Текст объявления:
{text}

Верни JSON в таком формате (все поля обязательны, если не знаешь — null):
{{
  "rooms": число или null,
  "area": число или null,
  "floor": число или null,
  "total_floors": число или null,
  "district": "Наурызбайский" или "Алмалинский" и т.д. или null,
  "has_furniture": true/false или null,
  "has_eurorepair": true/false или null,
  "new_building": true/false или null,
  "price": число или null
}}

Будь максимально точным.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        # На всякий случай очищаем возможный markdown
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].strip()
        
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        st.error(f"Ошибка парсинга: {e}")
        return None

# ====================== ОСТАЛЬНЫЕ ФУНКЦИИ ======================
def build_document(data):
    return f"""Квартира:
- Комнат: {data.get("rooms")}
- Площадь: {data.get("area")} м²
- Этаж: {data.get("floor")} из {data.get("total_floors")}
- Район: {data.get("district")}
- Мебель: {"Есть" if data.get("has_furniture") else "Нет"}
- Евроремонт: {"Есть" if data.get("has_eurorepair") else "Нет"}
- Новостройка: {"Да" if data.get("new_building") else "Нет"}"""

# hybrid_retrieve и rag_explanation оставляем как в предыдущей версии
def hybrid_retrieve(data, k=6):

    query_text = build_document(data)

    # Dense поиск
    query_vec = embedding_model.encode(
        query_text,
        normalize_embeddings=True
    )

    query_vec = np.array(
        [query_vec]
    ).astype("float32")

    _, dense_idx = index.search(
        query_vec,
        k * 3
    )

    dense_df = df.iloc[dense_idx[0]]

    # BM25 поиск
    tokenized_query = query_text.split()

    bm25_scores = bm25.get_scores(
        tokenized_query
    )

    bm25_idx = np.argsort(
        bm25_scores
    )[-k*3:][::-1]

    bm25_df = df.iloc[bm25_idx]

    # объединение
    combined = pd.concat([
        dense_df,
        bm25_df
    ])

    combined = (
        combined
        .drop_duplicates()
        .fillna("Не указано")
        .head(k)
    )

    return combined

def rag_explanation(data, similar_df):

    query = build_document(data)

    similar_df = similar_df.fillna(
        "Не указано"
    )

    context = "\n\n".join([
        build_document(row)
        for _, row in similar_df.iterrows()
    ])

    prompt = f"""
Ты профессиональный аналитик недвижимости Казахстана.

Целевая квартира:

{query}

Похожие объекты:

{context}

Сделай анализ:

1. Дорого/дешево/по рынку
2. Почему
3. 3 причины
4. Рекомендации покупателю

Пиши кратко и по делу.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text
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
raw_text = st.sidebar.text_area("Вставьте текст объявления", height=160)

if st.sidebar.button("🔍 Извлечь параметры из текста"):
    if raw_text.strip():
        with st.spinner("Парсим объявление..."):
            parsed = parse_listing_text(raw_text)
            if parsed:
                st.session_state.parsed_data = parsed
                st.sidebar.success("✅ Параметры успешно извлечены!")
            else:
                st.sidebar.error("Не удалось распарсить объявление")
    else:
        st.sidebar.warning("Введите текст объявления")

# Автозаполнение
if st.session_state.parsed_data:
    p = st.session_state.parsed_data
    if p.get("rooms") is not None: rooms = int(p.get("rooms"))
    if p.get("area") is not None: area = float(p.get("area"))
    if p.get("floor") is not None: floor = int(p.get("floor"))
    if p.get("total_floors") is not None: total_floors = int(p.get("total_floors"))
    if p.get("district") in district_coords:
        district = p.get("district")
    if p.get("has_furniture") is not None: has_furniture = bool(p.get("has_furniture"))
    if p.get("has_eurorepair") is not None: has_eurorepair = bool(p.get("has_eurorepair"))
    if p.get("new_building") is not None: new_building = bool(p.get("new_building"))

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать", type="primary"):
    floor_ratio = (
    floor / total_floors
    if total_floors > 0
    else 0
    )

    luxury = int(
    has_eurorepair
    and new_building
    )

    features = [[
    area,
    floor,
    total_floors,
    floor_ratio,
    int(has_furniture),
    int(has_eurorepair),
    int(new_building),
    luxury
    ]]

    st.write("Количество признаков:", len(features[0]))
st.write("Признаки:", features[0])

try:
    st.write("Ожидаемые признаки:")
    st.write(model.feature_names_in_)
except:
    st.write("feature_names_in_ недоступно")
    
    predicted_price = abs(
    model.predict(features)[0]
    )

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

st.caption("ValoraAI • Улучшенный парсер объявлений")
