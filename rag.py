import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)

# Load PDF and preprocess text
pdf_loader = PyPDFLoader('LNCB.pdf')
pdf_pages = pdf_loader.load_and_split()
text_chunks = pdf_pages[0].page_content

text_splitter = CharacterTextSplitter(chunk_size=300)
chunks = text_splitter.create_documents([text_chunks])

# Initialize Google Generative AI embeddings
api_key = 'AIzaSyDH6DE1tCVpzgVeF2laadHs_1qKgInmF3w'
embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
vectors = embedding_model.embed_documents(text_chunks)

# Store chunks and embeddings in Chroma vector store
db = Chroma.from_documents(chunks, embedding_model)
db.persist()

# Connect to ChromaDB and create retriever
db_connection = Chroma(embedding_function=embedding_model)
retriever = db_connection.as_retriever(search_kwargs={"k": 5})

# Define RAG chain
chat_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="I'm a helpful AI assistant. I'll use the provided document to answer your questions."),
    HumanMessagePromptTemplate.from_template("""Answer the following question based on the provided context:

    Context:
    {context}

    Question:
    {question}

    Answer:""")
])

output_parser = StrOutputParser()

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
     "question": RunnablePassthrough()}
    | chat_template
    | (ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-pro-latest")
       | output_parser)
)

# Streamlit app
st.title("Leave No Context Behind Answering AI")
user_question = st.text_input("Enter your question:", "Enter your query")

if st.button("Get Answer"):
    with st.spinner('Generating...'):
        st.info("Fetching the answer...")
        response = rag_chain.invoke(user_question)
        st.success(response)
