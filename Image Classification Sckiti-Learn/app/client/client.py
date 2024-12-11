import streamlit as st
from tempfile import NamedTemporaryFile
import requests
st.title("Car detection Sklear")
st.info("This small application is designed to use (interact) with a trained SVC model for Car Detection")
st.sidebar.write( "Load an image from test samples")
btn_load = st.sidebar.file_uploader('Load Image')
if btn_load:
    # name of the file 
    file_loaded = btn_load.name
    temp_file_path =""
    with NamedTemporaryFile(dir="../server",suffix='.jpg', prefix='temp', delete=False) as f:
        
        f.write(btn_load.getbuffer())
        temp_file_path = f.name.split()
        temp_file_path_ = temp_file_path[-1].split('\\')[-1]
        print(temp_file_path_)
        print('************************************************************************')
        f.close()
    btn_load.close()
    st.image("../server/"+temp_file_path_,use_column_width=True, caption="Selected image")
    # Make a request
    response  = requests.get('http://127.0.0.1:5000/result',params={'image':  temp_file_path_})
    if (response.status_code == 200):
        print('Succes')
        prediction = response.json()
        print(prediction[0])
        st.text("The result of the prediction is :"+prediction[0])
    

