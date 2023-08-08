import streamlit as st
from main import main1 as m1
from main import main2 as m2
from sub import intro

st.title('로스트 아크')


item_list = ['item0','item1', 'item2']

item_labels = {'item0':'홈', 'item1':'리퍼', 'item2':'호크아이 '}

FIL = lambda x : item_labels[x]
item = st.sidebar.selectbox('직업을 골라주세요.',  item_list, format_func = FIL )

if item == 'item1':
	m1.main()
elif item == 'item2':
	m2.main()
elif item == 'item0':
	intro.intro()