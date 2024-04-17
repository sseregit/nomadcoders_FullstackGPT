import streamlit as st
from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.markdown("""
    # SiteGPT
    
    Ask questions about the content of a website.
    
    Start by writing the URL of the website on the sidebar.
""")

with st.sidebar:
    url = st.text_input("Write down a URL", placeholder="https://example.com")

def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()
    footer = soup.find("footer")
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", " ").replace("\xa0"," ").replace("CloseSearch Submit Blog", "")

@st.cache_data
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        filter_urls=[
            r"^(.*\/blog\/).*",
        ],
        parsing_function=parse_page
    )
    loader.requests_per_second = 5
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL")
    else:
        docs = load_website(url)
        st.write(docs)