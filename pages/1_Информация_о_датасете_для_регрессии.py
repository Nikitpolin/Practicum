import streamlit as st
import pandas as pd

data= pd.read_csv("energy_task.csv")
df = pd.DataFrame(data)

st.title('Информация о датасетe')

st.header('Датасет для регрессии - "Количество включенных электроприборов"')
st.markdown('---')
st.dataframe(df)

st.subheader('date')
st.markdown('дата в формате: дд-мм-гггг чч:мм')

st.subheader('Appliances')
st.markdown('потребление энергии в Втч (целевая переменная для прогнозирования)')

st.subheader('lights')
st.markdown('энергопотребление светильников в доме в Втч')

st.subheader('T1')
st.markdown('Температура на кухне, в градусах Цельсия.')

st.subheader('RH_1')
st.markdown('Влажность на кухне, в %')

st.subheader('T2')
st.markdown('Температура в жилом помещении, по Цельсию')

st.subheader('RH_2')
st.markdown('Влажность в жилом помещении, в %')

st.subheader('T3')
st.markdown('Температура в прачечной')

st.subheader('RH_3')
st.markdown('Влажность в помещении прачечной, в %')

st.subheader('T4')
st.markdown('Температура в офисном помещении, по Цельсию')

st.subheader('RH_4')
st.markdown('Влажность в офисном помещении, в %')

st.subheader('T5')
st.markdown('Температура в ванной, по Цельсию')

st.subheader('RH_5')
st.markdown('Влажность в ванной, в %')

st.subheader('T6')
st.markdown('Температура снаружи здания (северная сторона), в градусах Цельсия')

st.subheader('RH_6')
st.markdown('Влажность снаружи здания (северная сторона), в %')

st.subheader('T7')
st.markdown('Температура в гладильной комнате, в градусах Цельсия')

st.subheader('RH_7')
st.markdown('Влажность в гладильной, в %')

st.subheader('T8')
st.markdown('Температура в комнате для подростков 2, по Цельсию')

st.subheader('RH_8')
st.markdown('Влажность в комнате подростка 2, в %')

st.subheader('T9')
st.markdown('Температура в комнате родителей, по Цельсию')

st.subheader('RH_9')
st.markdown('Влажность в комнате родителей, в %')

st.subheader('T_out')
st.markdown('Температура снаружи (по метеостанции Шьевр), в градусах Цельсия')

st.subheader('Press_mm_hg')
st.markdown('Давление (по метеостанции Шьевр), мм рт. ст.')

st.subheader('RH_out')
st.markdown('Влажность на улице (по метеостанции Шьевр), в %')

st.subheader('Windspeed')
st.markdown('Скорость ветра (по данным метеостанции Шьевр), м/с')

st.subheader('Visibility')
st.markdown('Видимость (с метеостанции Шьевр), км')

st.subheader('Tdewpoint')
st.markdown('Точка росы (по метеостанции Шьевр), °C км')