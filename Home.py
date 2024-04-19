import streamlit as st

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="🤖"
)

st.title("FullstackGPT Home")

st.markdown(
    """
    # Hello!
    
    Welcome to my FullstackGPT Portfolio!
    
    Here are the apps I made:
    
    - [x] [DocumentGPT](/DocumentGPT)
        - [ ] [challenge](/DocumentGPT)
            1. chain에 memory 추가 하기
    - [x] [PrivateGPT](/PrivateGPT)
        - [ ] [challenge](/PrivateGPT)
            1. select box를 이용해 모델 변경을 허용한다.    
    - [x] [QuizGPT](/QuizGPT)
        - [ ] [challenge](/QuizGPT)
            1. function calling을 이용하는 방법으로 변경        
    - [x] [SiteGPT](/SiteGPT)
        - [ ] [challenge](/SiteGPT)
            1. Chatbot처럼 만들기
            2. question들의 cache
            3. 비슷한 question들의 cache            
    - [x] [MeetingGPT](/MeetingGPT)
        - [ ] [challenge](/MeetingGPT)
            1. Q&A tab            
    - [ ] [InvestorGPT](/InvestorGPT)
    """
)