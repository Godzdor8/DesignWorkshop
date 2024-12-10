import streamlit as st
from otrisovka import *


def func(picture):
    picture_bytes = np.asarray(bytearray(picture.read()), dtype=np.uint8)
    img1 = cv2.imdecode(picture_bytes, 1)
    res = get_result(img1)
    for img, score, label in otrisovka(res):
        st.image(img)
        st.text(f"Найден объект: {label} с уверенностью: {score}")

st.title('Нахождение объектов')

picture = st.file_uploader(label='Картинки', type=['png', 'jpg'])

if picture:
    st.image(picture)

if st.button('Найти объекты'):
    func(picture)