import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data= pd.read_csv("neo_task_filtered.csv")

st.title('Визуализация датасета')

st.header('Датасет для классификации - "Опасность космических объектов"')

st.markdown('---')

st.write("Диаграмма для максимального и минимального диаметра")

chart_data = pd.DataFrame(np.random.randn(20, 2), columns=["est_diameter_max", "est_diameter_min"])
st.area_chart(chart_data)

st.write("Диаграмма рассеяния для скорости, расстояния и светимости")

chart_data = pd.DataFrame(np.random.randn(100, 3), columns=["relative_velocity", "miss_distance", "absolute_magnitude"])
st.scatter_chart(chart_data)

st.write("Гистограмма предсказываемого признака")

fig, ax = plt.subplots()
ax.hist(data['hazardous'], bins=20)

st.pyplot(fig)