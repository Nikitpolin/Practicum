import pandas as pd 
import numpy as np 
import math
import pickle
from sklearn.model_selection import train_test_split 
import streamlit as st 
from sklearn.metrics import * 

df= pd.read_csv('energy_task_filtered.csv')
if df is not None:
    st.header("Датасет")
    st.dataframe(df)
    st.write("---")
    st.title("Appliances Prediction") 

    st.markdown('Для предсказания необходимо выделить целевой признак, а также разделить датасет на обучающую и тестовую выборку:')
    code = '''
    Y = data_filtered["Appliances"]
    X = data_filtered.drop(["Appliances"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lasso = Lasso(alpha=0.5).fit(X_train, y_train)
    y_pred = lasso.predict(X_test)
    '''

    st.code(code, language='python')
    list=[]
    df = df.drop('Appliances', axis = 1)

    for i in df.columns[:]:
        a = st.slider(i,float(df[i].min()), float(math.ceil(df[i].max())),float(df[i].max()/2))
        list.append(a)

    

    list = np.array(list).reshape(1,-1)
    list=list.tolist()
    st.title("Тип модели обучения: Lasso")
    

    button_clicked = st.button("Предсказать")
    if button_clicked:
        with open('lasso.pkl', 'rb') as file:
            lasso = pickle.load(file)
            y_pred = lasso.predict(list)
            st.success(y_pred)
            st.markdown('mean_absolute_error:')
            st.code(16.2877218519619, language='python')
            st.markdown('mean_square_error:')
            st.code(444.9454876473329, language='python')
            st.markdown('root_mean_square_error:')
            st.code(21.093731003483782, language='python')
            st.markdown(' mean_absolute_percentage_error:')
            st.code(7.8837401568263585, language='python')
            st.markdown('R^2:')
            st.code(0.04174319689907935, language='python')
            