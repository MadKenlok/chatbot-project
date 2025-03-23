import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Инициализация модели и токенизатора
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Инициализация истории диалога для модели
if "dialogue_history_ids" not in st.session_state:
    st.session_state.dialogue_history_ids = None

# Функция для модификации ответа в зависимости от стиля
def apply_style_to_response(response, style):
    if style == "friendly":
        return response + " 😊"
    elif style == "sarcastic":
        return response + " 🙄 Oh, really?"
    return response  # Нейтральный стиль без изменений

# Функция для генерации ответа
def generate_response(prompt, style="neutral", max_length=100):
    # Токенизируем входной текст
    new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
    if torch.cuda.is_available():
        new_user_input_ids = new_user_input_ids.cuda()

    # Объединяем с историей диалога, если она есть
    if st.session_state.dialogue_history_ids is not None:
        bot_input_ids = torch.cat([st.session_state.dialogue_history_ids, new_user_input_ids], dim=-1)
    else:
        bot_input_ids = new_user_input_ids

    # Генерируем ответ
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=max_length + bot_input_ids.shape[-1],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=100,  # Увеличиваем top_k для разнообразия
        top_p=0.95,
        temperature=0.9  # Увеличиваем temperature для большей креативности
    )

    # Обновляем историю диалога
    st.session_state.dialogue_history_ids = chat_history_ids

    # Декодируем ответ
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Применяем стиль к ответу
    styled_response = apply_style_to_response(response, style)
    return styled_response.strip()

# Заголовок приложения
st.title("Чат-бот с генерацией текста на основе предобученной модели")

# Описание проекта
st.markdown("""
Этот проект представляет собой чат-бот, который генерирует текстовые ответы с использованием предобученной модели DialoGPT-small.  
Вы можете задать стиль ответа: нейтральный, дружелюбный или саркастичный.
""")

# Инициализация истории чата для отображения
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ввод данных
prompt = st.text_input("Введите ваш запрос:", value="How are you today?")
style = st.selectbox("Выберите стиль ответа:", ["neutral", "friendly", "sarcastic"])
max_length = st.slider("Максимальная длина ответа:", min_value=20, max_value=150, value=100)

# Кнопка для генерации ответа
if st.button("Сгенерировать ответ"):
    with st.spinner("Генерация ответа..."):
        response = generate_response(prompt, style=style, max_length=max_length)
    st.success("Ответ сгенерирован!")
    # Добавляем запрос и ответ в начало истории
    st.session_state.chat_history.insert(0, {"prompt": prompt, "style": style, "response": response})

# Отображаем историю чата (новые сообщения сверху)
if st.session_state.chat_history:
    st.write("### История чата")
    for chat in st.session_state.chat_history:
        st.write(f"**Запрос:** {chat['prompt']}")
        st.write(f"**Стиль:** {chat['style']}")
        st.write(f"**Ответ:** {chat['response']}")
        st.write("---")