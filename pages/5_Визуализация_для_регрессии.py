import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data= pd.read_csv("energy_task_filtered.csv")

st.title('Визуализация датасета')

st.header('Датасет для регрессии - "Количество включенных электроприборов"')

st.markdown('---')

st.write("Тепловая карта по матрице корреляции между признаками:")

st.image("heat_map.png")

st.write("Диаграмма рассеяния для электроприборов и освещения")

chart_data = pd.DataFrame(np.random.randn(100, 2), columns=["Appliances", "lights"])
st.scatter_chart(chart_data)

st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()
ax.hist(data['Appliances'], bins=20)

st.pyplot(fig)