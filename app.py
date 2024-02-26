import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdfRead =PdfReader(pdf)
        for page in pdfRead.pages:
            text += page.extract_text()
    return text
 
def get_pdf_chunks(text):
    textSplitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = textSplitter.split_text(text)
    return chunks


def getVectorStore(textChunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=textChunks, embedding=embeddings)
    return vectorstore
     
def getConvoChain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    memory = ConversationBufferMemory(memory_key = 'chat_history' , return_messages=True)
    convoChain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever= vectorstore.as_retriever(),
        memory = memory
    )
    return convoChain


def handleUserInput(userQuestion):
    response = st.session_state.conversation({'question': userQuestion})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
             st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():

    load_dotenv()

    st.set_page_config(page_title ="DochaT")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None 

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None 

    st.header("CHAT WITH YOUR DOCUMENTS! :books:")
    userQuestion = st.text_input ("Ask questions related to the document(s):")
    if userQuestion:
        handleUserInput(userQuestion)

    with st.sidebar:
        st.subheader("Uploaded Documents")
        docs = st.file_uploader("Upload documents", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("processing"):
                #this will get all pdf texts
                rawText = get_pdf_text(docs)
                #st.write(rawText)

                #this will get chunks of text
                textChunks = get_pdf_chunks(rawText)
                #st.write(textChunks)

                #this is getting the vector store
                vectorstore = getVectorStore(textChunks)

                #creating conversation chain
                st.session_state.conversation = getConvoChain(vectorstore)
            

if __name__ == '__main__':
    main()