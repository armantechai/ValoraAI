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
st.markdown("**ML + Hybrid RAG + Продвинутый AI-анализ**")

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
    
    # Подготовка BM25
    tokenized_corpus = [doc.split() for doc in df['description'].fillna('').astype(str)]
    bm25 = BM25Okapi(tokenized_corpus)
    
    return model, df, embedding_model, index, bm25

model, df, embedding_model, index, bm25 = load_resources()

# ====================== СПРАВОЧНИК ======================
district_coords = { ... }  # оставь свой словарь

# ====================== УЛУЧШЕННЫЕ ФУНКЦИИ ======================
def build_document(data):
    """Улучшенное структурированное описание"""
    return f"""Квартира на продажу:
- Комнат: {data.get("rooms")}
- Общая площадь: {data.get("area")} м²
- Этаж: {data.get("floor")} из {data.get("total_floors")}
- Район: {data.get("district", "Не указан")}
- Мебель: {"Есть" if data.get("has_furniture") else "Нет"}
- Евроремонт: {"Есть" if data.get("has_eurorepair") else "Нет"}
- Новостройка: {"Да" if data.get("new_building") else "Нет"}
- Премиум: {"Да" if data.get("luxury", False) else "Нет"}"""

def hybrid_retrieve(data, k=6):
    """Hybrid Search: Dense + BM25"""
    query_text = build_document(data)
    
    # Dense Retrieval (семантический поиск)
    query_vector = np.array([embedding_model.encode(query_text)]).astype("float32")
    _, dense_indices = index.search(query_vector, k*2)
    dense_results = df.iloc[dense_indices[0]]
    
    # BM25 (ключевой поиск)
    tokenized_query = query_text.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[-k*2:][::-1]
    bm25_results = df.iloc[bm25_top_indices]
    
    # Простое комбинирование (можно улучшить reranking'ом позже)
    combined = pd.concat([dense_results, bm25_results]).drop_duplicates().head(k*2)
    
    # Финальная выборка
    return combined.head(k)

def rag_explanation(data, similar_df):
    """Улучшенный промпт для качественных объяснений"""
    query = build_document(data)
    context = "\n\n".join([build_document(row) for _, row in similar_df.iterrows()])

    prompt = f"""
Ты — опытный риелтор-аналитик с 10-летним опытом на рынке Казахстана (Алматы и Астана).

**ЦЕЛЕВАЯ КВАРТИРА:**
{query}

**ПОХОЖИЕ ОБЪЕКТЫ НА РЫНКЕ:**
{context}

Проведи глубокий анализ и ответь максимально понятно и профессионально:

1. **Общая оценка**: Квартира стоит **дорого**, **дешево** или **по рынку**?
2. **Обоснование**: Почему именно такая цена? Какие факторы влияют сильнее всего?
3. **3 ключевые причины** твоего вывода.
4. **Рекомендации** покупателю (стоит ли рассматривать, на что обратить внимание).

Используй естественный, уверенный и человечный язык. Избегай шаблонных фраз.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ты эксперт по недвижимости высшего уровня."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=700
    )
    return response.choices[0].message.content

# ====================== SIDEBAR (оставляем как было) ======================
# ... (тот же код sidebar'а)

# ====================== АНАЛИЗ ======================
if st.button("🚀 Проанализировать", type="primary"):
    # ... (твой код расчёта predicted_price)

    data = { ... }  # как было

    # Используем Hybrid Search
    similar_df = hybrid_retrieve(data, k=6)

    # Вывод (col1, col2) — оставь как в предыдущей версии

    with col2:
        st.subheader("📊 AI Анализ рынка")
        with st.spinner("Генерируем профессиональный отчёт..."):
            analysis = rag_explanation(data, similar_df)
            st.markdown(analysis)

    # Сравнение — оставь как было

st.caption("ValoraAI • Hybrid RAG + Улучшенный промпт")
