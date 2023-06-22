import streamlit as st
# pypdf tiktoken  faiss-cpu
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from PIL import Image

if "messages" not in st.session_state:
    st.session_state["messages"] = ""

image = Image.open('images/calsec-logo.jpg')

col1, col2 = st.columns([1, 2])
col1.image(image)
st.header("전자거래학회 논문 챗봇")
st.subheader("김세형(아주대) / 박재형(아주대) / 강주영(아주대)")
st.text('반드시 api 키를 입력하고 엔터를 먼저 눌러주세요.')



API_KEY = st.sidebar.text_input(":blue[Enter Your OPENAI API-KEY :]", 
                    placeholder="본인의 api 키를 입력해 주세요! (sk-...)",
                    type="password")

if API_KEY != "":

    loader = PyPDFLoader('BERTopic과 소셜 네트워크 분석 기반 고령화 단계별 판례분석을 통한 분쟁 유형 도출에 관한 연구.pdf')
    data = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10, separator = "\n",)
    documents = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings(openai_api_key = API_KEY)
    chat_model = OpenAI(temperature=0, openai_api_key = API_KEY)
    db = FAISS.from_documents(documents, embeddings)



    template = """
You are a chatbot having a conversation with a human.
Given the following extracted parts of a long document and a question, create a final answer, Up to 3 sentences.
{context}

{chat_history}
Human: {human_input}
Chatbot:
"""

    usr_input = st.text_input("전자거래학회-논문-챗봇", placeholder="질문을 입력해주세요.")

    prompt_result = st.empty()
    prompt_result.text_area("전자거래학회-논문-챗봇", height=400)

    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context"], 
        template=template
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
    chain = load_qa_chain(chat_model, chain_type="stuff", memory=memory, prompt=prompt)

# 이부분을 초기화하면 될듯.
if st.button("Send"):
    with st.spinner("Generating response..."):
        
        docs = db.similarity_search(usr_input)
        result = chain({"input_documents": docs, "human_input": usr_input}, return_only_outputs=True)
        st.session_state["messages"] += "전자거래학회: " + result['output_text'] + '\n'
        prompt_result.text_area('전자거래학회-논문-챗봇', value =  st.session_state["messages"])
        



if st.button("초기화"):
    st.session_state["messages"] = ""
    prompt_result.text_area('전자거래학회-논문-챗봇', value = "")