import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

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
# OpenAI API Key — ИСПРАВЛЕНО
# ==========================================
api_key = None

# Вариант 1: Streamlit Secrets (рекомендуется)
if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    api_key = st.secrets["openai"]["api_key"]
elif "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

# Вариант 2: Переменная окружения (для локального запуска)
if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("🔑 **OpenAI API ключ не найден!**")
    st.info("""
    **Как исправить:**
    1. Перейди в настройки приложения → "Secrets"
    2. Добавь следующий код:
    
    ```toml
    [openai]
    api_key = "sk-ваш_ключ_здесь"
""")
st.stop()
client = OpenAI(api_key=api_key)
st.success("✅ OpenAI клиент подключён", icon="🔑")
==========================================
Загрузка ресурсов
==========================================
@st.cache_resource
def load_resources():
try:
model = pickle.load(open("model.pkl", "rb"))
df = pd.read_csv("krisha_full_with_desc.csv")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.bin")
return model, df, embedding_model, index
except FileNotFoundError as e:
st.error(f"Файл не найден: {e}")
st.stop()
except Exception as e:
st.error(f"Ошибка загрузки моделей: {e}")
st.stop()
model, df, embedding_model, index = load_resources()
==========================================
Остальные функции (build_document, retrieve_similar и т.д.)
==========================================
def build_document(data):
return f"""
Комнаты: {data.get("rooms", "—")}
Площадь: {data.get("area", "—")} м²
Этаж: {data.get("floor", "—")}
Всего этажей: {data.get("total_floors", "—")}
Мебель: {data.get("has_furniture", "—")}
Евроремонт: {data.get("has_eurorepair", "—")}
Новостройка: {data.get("new_building", "—")}
"""
def retrieve_similar(query, k=5):
query_vector = embedding_model.encode(query)
query_vector = np.array([query_vector]).astype("float32")
distances, indices = index.search(query_vector, k)
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
Оцени: дорого/дешево/нормально? Почему? Сравни с рынком. Приведи 3 причины.
Отвечай на русском, только по фактам.
"""
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[
{"role": "system", "content": "Ты профессиональный риелтор-аналитик в Казахстане."},
{"role": "user", "content": prompt}
],
temperature=0.3
)
return response.choices[0].message.content
def summarize_listing(data):
prompt = f"Сделай краткую карточку квартиры:\n{build_document(data)}"
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": prompt}]
)
return response.choices[0].message.content
==========================================
Sidebar
==========================================
st.sidebar.header("Параметры квартиры")
rooms = st.sidebar.selectbox("Комнаты", [1, 2, 3, 4, 5])
area = st.sidebar.number_input("Площадь (м²)", 10, 500, 60)
floor = st.sidebar.number_input("Этаж", 1, 30, 3)
total_floors = st.sidebar.number_input("Всего этажей", 1, 30, 9)
has_furniture = st.sidebar.checkbox("Мебель", value=True)
has_eurorepair = st.sidebar.checkbox("Евроремонт")
new_building = st.sidebar.checkbox("Новостройка")
luxury = st.sidebar.checkbox("Премиум")
==========================================
Анализ
==========================================
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
try:
st.markdown(rag_explanation(data))
except Exception as e:
st.error(f"Ошибка AI анализа: {e}")
st.subheader("🧠 Карточка квартиры")
with st.spinner("Генерируем описание..."):
try:
st.markdown(summarize_listing(data))
except Exception as e:
st.error(f"Ошибка генерации описания: {e}")
