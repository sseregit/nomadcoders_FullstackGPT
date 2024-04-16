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
    - [ ] [PrivateGPT](/PrivateGPT)
        - [ ] [challenge](/PrivateGPT)
            1. select box를 이용해 모델 변경을 허용한다.    
    - [ ] [QuizGPT](/QuizGPT)
    - [ ] [SiteGPT](/SiteGPT)
    - [ ] [MeetingGPT](/MeetingGPT)
    - [ ] [InvestorGPT](/InvestorGPT)
    """
)