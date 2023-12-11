import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
    
#Title
st.title('Weapon classifying model') 
st.image('weapon.jpg', width=700)

tab1, tab2, tab3 = st.tabs(['Describtion','About Author', 'Contacts'])
with tab1:
    tab1.write('I am a webapplication who can identify the type of weapons by an image (**gun**, **bomb**, **tank**, **sword**, **knife**, **bow and arrow**), and I am created by **Davrbek Mirzakosimov**')

with tab2:
    tab2.image('profile.png', width=200, caption='Davrbek Mirzakosimov')
    st.write("Davrbek Mirzakosimov Isokjon Ugli, born on July 9, 2002, in Fergana, Uzbekistan, is a fourth-year economics student with an impressive track record. A recipient of the Posco Scholarship in 2022, the Global Korean Scholarship, and the National Scholarship Program Slovakia in 2023, Davrbek is a dedicated scholar with a global perspective. His achievements underscore a commitment to academic excellence and a promising future in economics.")
with tab3:
    st.link_button('Instagram', "https://www.instagram.com/davrbek_sk/")
    st.link_button("Facebook", 'https://www.facebook.com/davrbek.mirzaqosimov')
    st.link_button("Telegram", "https://t.me/Davrbek_ibn_Isoq")




#Uploading
file = st.file_uploader("**Uploading image**", type=['png','jpeg','gif','svg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('weapon_model.pkl')

    # prediction
    pred, pred_id, probs = model.predict(img)

    if pred == 'Handgun' or pred == 'Shotgun' or pred == 'Rifle' :
        st.success(f"Prediction: Gun")
        st.info(f'Probability: {probs[pred_id]*100:.1f}%')
    elif pred == ('Missile'):
        st.success(f"Prediction: Bomb")
        st.info(f'Probability: {probs[pred_id]*100:.1f}%')
    else:
        st.success(f"Prediction: {pred}")
        st.info(f'Probability: {probs[pred_id]*100:.1f}%')
st.slider('**Rate the model**', 0, 10, 2)    
st.text_area("**Write your feedback about the webapplication :**")