# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 17:33:42 2023

@author: USER
"""

import streamlit as st

from Tabs import news, predict

Tabs = {
        "Home" : news,
        "Prediction" : predict
        }

#sidebar
st.sidebar.title("News Classification")

#option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

#call function
# if page in ["Prediction", "Visualisation"]:
#     Tabs[page].app(data,x,y)
# else:
#     Tabs[page].app()

Tabs[page].app()