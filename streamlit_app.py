import streamlit as st
from otrisovka import *
import numpy as np


def func(picture):
    picture_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
    img1 = cv2.imdecode(picture_bytes, cv2.IMREAD_COLOR)

    if img1 is None:
        st.error("Ошибка: Не удалось загрузить изображение. Убедитесь, что файл является корректным изображением.")
        return

    img2 = Image.fromarray(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    res = get_result(img2)

    if not res:
        st.text("Объекты не найдены.")
        return

    # Создание отдельного изображения для каждого обнаруженного объекта
    for score, box, label in res:
        # Создаем новое изображение для каждого объекта
        final_image = img1.copy()  # Копируем оригинальное изображение
        final_image = draw_object_bounding_box(final_image, box, label)  # Рисуем рамку на копии
        st.image(final_image, caption=f"Найден объект: {label} с уверенностью: {round(float(score), 3)}")

st.title('Нахождение объектов')

picture = st.file_uploader(label='Картинки', type=['png', 'jpg'])

if picture:
    st.image(picture)

if st.button('Найти объекты'):
    func(picture)