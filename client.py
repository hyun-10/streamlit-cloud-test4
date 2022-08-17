# -*- coding: utf-8 -*-
import streamlit as st
from module_AI import *






item_list =['소개','AI전략매매']

item=st.sidebar.selectbox('선택하세요', item_list)

if item=='소개':
    st.title('TeamProject')

    st.subheader('시장 현황')
    st.subheader('AI전략매매')


if item== 'AI전략매매':
    AI()











