import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import ZEMO

st.title("iFiMAS (The best Financial Markets Analysis System)")

symbol = st.sidebar.text_input("Enter a crypto symbol:")
uploaded_files = st.file_uploader("Choose an image", accept_multiple_files=True)

for uploaded_file in uploaded_files:
    bytes_data = uploaded_file.read()
    st.write("filename:", uploaded_file.name)

Order = st.sidebar.text_input("Enter an Order number:")

st.image(bytes_data, caption='Original Image')
SZ=np.shape(bytes_data)

for i in range(3):
   ZBFSTR=ZEMO.zernike_bf(SZ[0],Order[i],1)
   Z=ZEMO.zernike_mom(np.double(bytes_data),ZBFSTR)
   I=ZEMO.zernike_rec(Z,SZ[0],ZBFSTR)

st.image(I)
