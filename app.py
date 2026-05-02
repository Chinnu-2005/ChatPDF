import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint,ChatHuggingFace
from langchain_community import vectorstores
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_classic.prompts import ChatPromptTemplate,MessagesPlaceholder
import os
from langchain_openai import ChatOpenAI

def text_extract_pdf(pdf_file):
    text=""
    for pdf in pdf_file:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len
        )
    
    chunks=text_splitter.split_text(text)
    return chunks


def create_vector_store(text_chunks):

    embedding_model=HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device":"cpu"},  
        encode_kwargs={"normalize_embeddings":True}
    )
    documents=[Document(page_content=chunk) for chunk in text_chunks]
    vector_db=FAISS.from_documents(documents,embedding_model)
    return vector_db

def create_conversation_chain(vector_store):
    retriever=vector_store.as_retriever(search_kwargs={"k":3}) #search 3 relevant pages

    #llm
    
    llm = ChatOpenAI(
        model="Qwen/Qwen2.5-7B-Instruct",       # free, fast, reliable chat model
        openai_api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        openai_api_base="https://router.huggingface.co/v1",
        max_tokens=512,
        temperature=0,
    )

    contextualised_prompt=ChatPromptTemplate.from_messages(
        [
            ("system","You are a helpful assistant for answering questions based on the provided pdf documents.Given chat history and latest user question,Convert to standalone question donot answer it"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm, retriever, contextualised_prompt)

    question_answer_prompt=ChatPromptTemplate.from_messages(
        [
        ("system",
             "Answer using ONLY the provided context. "
         "If not found, say 'I don't know'.\n\n{context}"),
         MessagesPlaceholder("chat_history"),
         ("human","{input}")
        ]
    )

    question_chain=create_stuff_documents_chain(llm,question_answer_prompt)

    rag_chain=create_retrieval_chain(
        history_aware_retriever,
        question_chain
    )

    return rag_chain





def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":books")

    st.header("Chat with Multiple PDFs :books: ")

    # Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_question=st.text_input("Ask a question on your pdf ")
    if user_question and st.session_state.conversation:
        response = st.session_state.conversation.invoke({
            "input": user_question,
            "chat_history": st.session_state.chat_history
        })

        st.write("### Answer:")
        st.write(response["answer"])

        # Update chat history
        st.session_state.chat_history.append(("human", user_question))
        st.session_state.chat_history.append(("ai", response["answer"]))

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your pdfs and click on Process",accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):

                #get pdf text

                raw_text=text_extract_pdf(pdf_docs)


                #text chunks
                text_chunks=get_text_chunks(raw_text)
                #st.write(text_chunks)
                
                #create vector store
                vector_store=create_vector_store(text_chunks)
                
                #create conversational chain
                st.session_state.conversation = create_conversation_chain(vector_store)

                st.success("Ready! Ask your questions.")





if __name__ == "__main__":
    main()