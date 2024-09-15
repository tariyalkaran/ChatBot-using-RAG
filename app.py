import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch the API key and configure generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("API Key is missing. Please set it in the .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text


# Function to split text into manageable chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {e}")
        return []





# Function to create or update the vector store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        index_path = "faiss_index"
        
        # Check if the index file exists
        index_file = os.path.join(index_path, "index.faiss")
        
        # If the index does not exist, create a new vector store
        if not os.path.exists(index_file):
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local(index_path)
            st.success("New FAISS index created and saved.")
        else:
            # Load existing FAISS index if it exists
            vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            # Add new text chunks
            vector_store.add_texts(text_chunks)
            vector_store.save_local(index_path)
            # st.success("FAISS index updated with new chunks.")
    except ValueError as ve:
        st.error(f"Error loading FAISS vector store: {ve}")
    except Exception as e:
        st.error(f"Error in vector store creation or updating: {e}")



# Function to create the conversational chain
def get_conversational_chain():
    try:
        prompt_template = """
         Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """

        model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error setting up the conversation chain: {e}")
        return None


# Function to process user input and provide a response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        # Search for relevant documents in the vector store
        docs = new_db.similarity_search(user_question)

        # Get the conversational chain and generate a response
        chain = get_conversational_chain()
        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            st.write("Reply: ", response["output_text"])
        else:
            st.error("Unable to create the conversational chain.")
    except Exception as e:
        st.error(f"Error processing user input: {e}")


# Main function to run the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💁")

    # User input for questions
    user_question = st.text_input("Ask a Question from the PDF Files")

    # If the user enters a question, process it
    if user_question:
        user_input(user_question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on Submit & Process", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                try:
                    with st.spinner("Processing..."):
                        raw_text = get_pdf_text(pdf_docs)
                        if raw_text:
                            text_chunks = get_text_chunks(raw_text)
                            if text_chunks:
                                get_vector_store(text_chunks)
                                st.success("Processing Completed!")
                            else:
                                st.error("Text chunks could not be generated.")
                        else:
                            st.error("No text extracted from PDFs.")
                except Exception as e:
                    st.error(f"Error during PDF processing: {e}")
            else:
                st.error("Please upload at least one PDF file.")


if __name__ == "__main__":
    main()

