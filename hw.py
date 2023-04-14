import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier as r
from sklearn.svm import SVC as s
import matplotlib.pyplot as plt
import seaborn as cns
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import train_test_split as tts
SVM = 'Support Vector Machine'
LR = "Logistic Regression"
RFC = "Random Forest Classifier"
df = pd.read_csv('penguin.csv')
df = df.dropna()
df['Type'] = df['species'].map({'Adelie': 2, 'Chinstrap': 22, 'Gentoo':222})
df['gender'] = df['sex'].map({'Male':22,'Female':2})
df['island'] = df['island'].map({'Biscoe': 2, 'Dream': 22, 'Torgersen':222})
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'gender']]
y = df['Type']
Xt , Xte , yt , yte = tts(X,y , random_state = 42 , test_size = 0.3333333333333)
rm = r(n_jobs=2)
rm.fit(Xt,yt)
svcm = s(kernel = 'linear')
svcm.fit(Xt,yt)
lrm = lr()
lrm.fit(Xt, yt)
lrms = lrm.score(Xt,yt)
rms = rm.score(Xt,yt)
svcms = svcm.score(Xt,yt)
Ad = 'Adelie'
Ch = 'Chinstrap'
Ge = 'Gentoo'
def predict( md , i , blmm , bdmm , flmm , bmg , g ) :
            array1 = np.array([ i, blmm, bdmm, flmm, bmg ,g])
            array1 = array1.reshape(1, -1)
            if md == SVM:
                    output = svcm.predict(array1)   
            elif md == LR:
                    output = lrm.predict(array1)
            elif md == RFC:
                    output = rm.predict(array1)
            if output == 2 :
                    return Ad 
            elif output == 22:
                    return Ch
            elif output == 222:
                    return Ge
def score2(md) :
            if md == SVM:
                    return svcm.score(Xte,yte)  
            elif md == LR:
                    return lrm.score(Xte,yte)
            elif md == RFC:
                    return rm.score(Xte,yte)
                    
import streamlit as st
Male = "Male"
Female = "Female"
Biscoe = "Biscoe"
Dream = "Dream"
Torgersen = "Torgersen"
st.write("""# Penguin Species predictor web app""")
st.sidebar.write("""## Inputs for prediction""")
blmm = st.sidebar.slider("Bill length in mm",30,60)
bdmm = st.sidebar.slider("Bill depth in mm",10,25)
flmm = st.sidebar.slider("Flipper length in mm",170,240)
bdg = st.sidebar.slider("Body mass in g",2650,6400)
g = st.sidebar.selectbox('Select the Gender of the penguin so cute:)', options=[Male,Female])
i = st.sidebar.selectbox('Select the Island of the penguins so cute:)', options=[Biscoe,Dream,Torgersen])
md = st.selectbox('Select Your Model I recommend Either SVM(Support Vector Machine) or LogisticRegresion(lr) because Random Forest really sucks', options=[SVM,LR,RFC])
if g == Male:
        gv = 22
else:
        gv = 2
if i == Biscoe:
        iv = 2
elif i == Dream:
        iv = 22
else:
        iv = 222
def image(p):
        st.subheader('A cute :) image of the species')
        if p == Ad:
                st.image("a.jpg")
        elif p == Ch:
                st.image("c.jpg")
        else:
                st.image("g.png")
if st.button('Predict the cute penguin :)'):
        predcition = predict( md , iv , blmm , bdmm , flmm , bdg , gv )
        st.write(f'The penguin is {predcition} \n \n The accuracy score of this model is {score2(md)}')
        image(predcition)
st.image(['a.jpg','c.jpg','g.png'], width=200)