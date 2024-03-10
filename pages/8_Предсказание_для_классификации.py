import pandas as pd 
import numpy as np 
import math
import pickle
from sklearn.model_selection import train_test_split 
import streamlit as st 


df= pd.read_csv('neo_task_filtered.csv')
df=df = df.drop('Unnamed: 0', axis = 1)

if df is not None:
    st.header("Датасет")
    st.dataframe(df)
    st.write("---")
    st.title("Hazardous Prediction") 

    list=[]

    for i in df.columns[:-1]:
        a = st.slider(i,float(df[i].min()), float(math.ceil(df[i].max())),float(df[i].max()/2))
        list.append(a)

    list = np.array(list).reshape(1,-1)
    list=list.tolist()
    st.title("Тип модели обучения: GNB")
    

    button_clicked = st.button("Предсказать")
    if button_clicked:
        with open('gnb.pkl', 'rb') as file:
            knn_model = pickle.load(file)
        if knn_model.predict(list) == 0:
            st.success("Объект не опасен")
            st.markdown('Площадь под кривой auc-roc:')
            st.code(0.912099364594081, language='python')
        else:
            st.success("Объект опасен")
            st.markdown('Площадь под кривой auc-roc:')
            st.code(0.912099364594081, language='python')