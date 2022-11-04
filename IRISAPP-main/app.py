import streamlit as st 
import pandas as pd 

from sklearn import datasets 
from sklearn.ensemble import RandomForestClassifier

st.write("""
**Flor Iris **
""")

st.sidebar.header('Parametro de entrada')

def user_input_features():
   
    sepal_len = st.sidebar.slider('Sepal length',4.3,7.9,5.4)
    sepal_wid = st.sidebar.slider('Sepal width',2.0,4.4,3.4)
    petal_len = st.sidebar.slider('Petal Length',1.0,6.9,1.3)
    petal_wid = st.sidebar.slider('Ptal wodth',0.1,2.5,0.2)

    data = {'sepal_length':sepal_len,'sepal_width':sepal_wid,'petal_length':petal_len,'petal_width':petal_wid}

    features = pd.DataFrame(data,index=[0])

    return features

df = user_input_features()

st.subheader('User input parameters')
st.write(df)

iris = datasets.load_iris() 

X = iris.data
Y = iris.target

clf = RandomForestClassifier()  
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.write('Rotulo de classes e numero de indice correspondente')
st.write(iris.target_names)

st.subheader('Predição')
st.write(iris.target_names[prediction])

st.subheader('Probabilidade')
st.write(prediction_proba)
