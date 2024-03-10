from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

data2= pd.read_csv("neo_task.csv")
df2 = pd.DataFrame(data2)

st.title('Предобработка данных')

st.header('Предобработка датасета для задачи классификации')
st.markdown('---')
st.dataframe(df2)

st.markdown('Вывод числа пропущенных значений:')
code = '''
data.isna().sum().sort_values(ascending=False)
'''
st.code(code, language='python')

st.code(df2.isna().sum().sort_values(ascending=False))

st.markdown('Вывод головы датасета:')
code = '''
data.head()
'''
st.code(code, language='python')
st.dataframe(df2.head(5))


st.markdown('Удалим столбцы, не несущие какую-либо информацию для обучения модели:')
code = '''
data = data.drop(['id', 'name'], axis = 1)

'''
df2 = df2.drop(['id', 'name'], axis = 1)
st.code(code, language='python')

st.dataframe(df2.head(5))

st.markdown('Вывод статистической сводки по датасету:')
code = '''
df2.describe()

'''
data_described = df2.describe()
st.code(code, language='python')

st.dataframe(data_described)

st.markdown('Заполняем пропущенные значения числами в интервале от минимального до максимального:')

code = '''
data['est_diameter_min'] = data['est_diameter_min'].map(lambda x: np.random.uniform(0, 38) if pd.isna(x) else x)
data['est_diameter_max'] = data['est_diameter_max'].map(lambda x: np.random.uniform(0, 84) if pd.isna(x) else x)
data['relative_velocity'] = data['relative_velocity'].map(lambda x: np.random.uniform(203, 230000) if pd.isna(x) else x)
data['miss_distance'] = data['miss_distance'].map(lambda x: np.random.uniform(6745, 74798) if pd.isna(x) else x)
data['absolute_magnitude'] = data['absolute_magnitude'].map(lambda x: np.random.uniform(9, 33) if pd.isna(x) else x)
data.isna().sum().sort_values(ascending=False)
'''
df2['est_diameter_min'] = df2['est_diameter_min'].map(lambda x: np.random.uniform(0, 38) if pd.isna(x) else x)
df2['est_diameter_max'] = df2['est_diameter_max'].map(lambda x: np.random.uniform(0, 84) if pd.isna(x) else x)
df2['relative_velocity'] = df2['relative_velocity'].map(lambda x: np.random.uniform(203, 230000) if pd.isna(x) else x)
df2['miss_distance'] = df2['miss_distance'].map(lambda x: np.random.uniform(6745, 74798) if pd.isna(x) else x)
df2['absolute_magnitude'] = df2['absolute_magnitude'].map(lambda x: np.random.uniform(9, 33) if pd.isna(x) else x)

st.code(code, language='python') 
st.code(df2.isna().sum().sort_values(ascending=False))


st.markdown('Переведем в столбце целевого признака данные из True-False в 1-0')

code = '''
data.loc[(data['hazardous'] == "True"), 'hazardous'] = 1
data.loc[(data['hazardous'] == "False"), 'hazardous'] = 0
data['hazardous'] = data['hazardous'].astype(int)
'''
df2.loc[(df2['hazardous'] == "True"), 'hazardous'] = 1
df2.loc[(df2['hazardous'] == "False"), 'hazardous'] = 0
df2['hazardous'] = df2['hazardous'].astype(int)

st.code(code, language='python')

st.dataframe(df2.head(5))

st.markdown('Проведем квантильную очистку для всех столбцов, содержащих числовые значения')

code = '''
outlier = data[['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude']]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
data = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]
'''

st.code(code, language='python')

st.write("Гистограмма до/после обработки (признак 'est_diameter_min'):")

st.markdown('ДО:')

fig, ax = plt.subplots()
ax.hist(df2['est_diameter_min'], bins=20)

st.pyplot(fig)

outlier = df2[['est_diameter_min', 'est_diameter_max', 'relative_velocity', 'miss_distance', 'absolute_magnitude']]
Q1 = outlier.quantile(0.25)
Q3 = outlier.quantile(0.75)
IQR = Q3-Q1
df2 = outlier[~((outlier < (Q1 - 1.5 * IQR)) |(outlier > (Q3 + 1.5 * IQR))).any(axis=1)]

st.markdown('ПОСЛЕ:')

fig, ax = plt.subplots()
ax.hist(df2['est_diameter_min'], bins=20)

st.pyplot(fig)


st.markdown('Сохраним предобработанный датасет с новым названием "neo_task_filtered.csv"')

code = '''
data.to_csv('neo_task_filtered.csv', index= False)
'''
st.code(code, language='python')


st.markdown('На этом предобработка завершена. В результате мы получили чистый датасет без выбросов и пустых значений, готовый к использованию для решения задачи классификации')