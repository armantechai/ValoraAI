
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from getpass import getpass

from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ⚠️ Правильный способ работы с API-ключом — через getpass()
# Функция запрашивает ввод так, что символы не отображаются на экране
# Это защищает ключ от случайного показа во время демонстрации
api_key = getpass("Введите ваш OpenAI API ключ (sk-...): ")

# Создаём клиент OpenAI с нашим ключом
# Все дальнейшие запросы делаем через этот объект client
client = openai.OpenAI(api_key=api_key)

print("✅ Клиент OpenAI создан!")
# ==========================================
# Настройки страницы
# ==========================================

st.set_page_config(
    page_title="ValoraAI",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 ValoraAI")
st.subheader(
    "Intelligent Real Estate Valuation Platform powered by ML & AI"
)

# ==========================================
# Загрузка ресурсов
# ==========================================

@st.cache_resource
def load_resources():

    with open("/content/drive/My Drive/model.pkl","rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("/content/drive/MyDrive/krisha_full_with_desc.csv")

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    index = faiss.read_index(
        "/content/drive/My Drive/faiss_index.bin"
    )

    return model, df, embedding_model, index


model, df, embedding_model, index = load_resources()


# ==========================================
# Создание документа
# ==========================================

def build_document(data):

    return f"""
    Комнаты: {data["rooms"]}

    Площадь:
    {data["area"]} м²

    Этаж:
    {data["floor"]}

    Всего этажей:
    {data["total_floors"]}

    Мебель:
    {data["has_furniture"]}

    Евроремонт:
    {data["has_eurorepair"]}

    Новостройка:
    {data["new_building"]}
    """


# ==========================================
# Поиск похожих квартир
# ==========================================

def retrieve_similar(query,k=5):

    query_vector = embedding_model.encode(
        query
    )

    query_vector=np.array(
        [query_vector]
    ).astype("float32")

    distances,indices=index.search(
        query_vector,
        k
    )

    return df.iloc[
        indices[0]
    ]


# ==========================================
# RAG explanation
# ==========================================

def rag_explanation(data):

    query=build_document(data)

    similar=retrieve_similar(
        query,
        5
    )

    context="

".join(

        similar.apply(
            lambda x:
            build_document(x),
            axis=1
        )

    )

    prompt=f"""
    Ты эксперт недвижимости Казахстана.

    ЦЕЛЕВАЯ КВАРТИРА:

    {query}

    ПОХОЖИЕ КВАРТИРЫ:

    {context}

    Объясни:

    - дорого/дешево/нормально
    - почему
    - сравни с рынком
    - 3 причины

    Не придумывай данные.
    """

    response=client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role":"system",
                "content":"Ты аналитик недвижимости"
            },
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content


# ==========================================
# Summary
# ==========================================

def summarize_listing(data):

    prompt=f"""
    Проанализируй:

    {build_document(data)}

    Верни:

    Тип жилья:
    Особенности:
    Состояние:
    Краткое описание:

    Не придумывай.
    """

    response=client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]
    )

    return response.choices[0].message.content


# ==========================================
# Sidebar
# ==========================================

st.sidebar.header(
    "Параметры квартиры"
)

rooms=st.sidebar.selectbox(
    "Комнаты",
    [1,2,3,4,5]
)

area=st.sidebar.number_input(
    "Площадь",
    10,
    500,
    60
)

floor=st.sidebar.number_input(
    "Этаж",
    1,
    30,
    3
)

total_floors=st.sidebar.number_input(
    "Всего этажей",
    1,
    30,
    9
)

has_furniture=st.sidebar.checkbox(
    "Мебель"
)

has_eurorepair=st.sidebar.checkbox(
    "Евроремонт"
)

new_building=st.sidebar.checkbox(
    "Новостройка"
)

luxury=st.sidebar.checkbox(
    "Премиум"
)


# ==========================================
# Анализ
# ==========================================

if st.button("🚀 Анализировать"):

    floor_ratio=floor/total_floors

    features=[[
        area,
        floor,
        total_floors,
        floor_ratio,
        int(has_furniture),
        int(has_eurorepair),
        int(new_building),
        int(luxury)
    ]]

    predicted_price=model.predict(
        features
    )[0]

    data={

        "rooms":rooms,
        "area":area,
        "floor":floor,
        "total_floors":total_floors,
        "has_furniture":has_furniture,
        "has_eurorepair":has_eurorepair,
        "new_building":new_building
    }

    col1,col2=st.columns(2)

    with col1:

        st.metric(
            "💰 Цена",
            f"{predicted_price:,.0f} ₸"
        )

    with col2:

        st.subheader(
            "📊 AI анализ рынка"
        )

        st.write(
            rag_explanation(data)
        )

    st.subheader(
        "🧠 Карточка квартиры"
    )

    st.write(
        summarize_listing(data)
    )
