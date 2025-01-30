import streamlit as st
from chromadb import PersistentClient
from PyPDF2 import PdfReader
from scipy.spatial.distance import cosine
import uuid
import os
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def load_pdf_text(file):
    """Extract text from a PDF file."""
    reader = PdfReader(file)
    full_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text.append(text)
    return ' '.join(full_text)

def generate_embeddings(text):

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    return embeddings.embed_query(text)

def store_pdf_embeddings(pdf_embeddings, db_path, collection_name):
    client = chromadb.PersistentClient(db_path)
    collection = client.create_collection(collection_name)

    
    for pdf_id, embedding in pdf_embeddings:
        # Insert embedding with UUID as the identifier
        collection.upsert({"id": pdf_id, "embedding": embedding})


def search_similar_pdfs(query_embedding, db_path, collection_name):

    client =chromadb.PersistentClient(db_path)
    collection = client.get_collection(collection_name)

    all_embeddings =list( collection.find())

    # Calculate cosine similarity and return most similar entries
    similarities = []
    for entry in all_embeddings:
        similarity = 1 - cosine(query_embedding, entry["embedding"])
        similarities.append((entry["id"], similarity))

    # Sort by highest similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]  # Return top 5 similar documents



def main():

    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Menu:")
        st.title('PDF Embedding Storage')
        uploaded_files = st.file_uploader("Choose PDF files, and Click on the Submit & Process Button", accept_multiple_files=True, type=['pdf'])
        
    
        if st.button('Process and Store Embeddings'):
            with st.spinner("Processing..."):
                if uploaded_files:
                    pdf_embeddings = []
                    db_path = './chroma_data.db'  # Adjust path as necessary
                    collection_name = 'pdf_embeddings'
                    for file in uploaded_files:
                        text = load_pdf_text(file)
                        embedding = generate_embeddings(text)
                    # Generate a unique identifier for each PDF file
                        file_uuid = str(uuid.uuid4())
                        pdf_embeddings.append((file_uuid, embedding))

                        store_pdf_embeddings(pdf_embeddings, db_path, collection_name)
                        st.success("PDF embeddings successfully stored in local Chroma DB.")
                else:
                    st.error("Please upload at least one PDF file.")
   

    if user_question:
        query_embedding=generate_embeddings(user_question)
        docs= search_similar_pdfs(query_embedding, './chroma_data.db', 'pdf_embeddings')

        chain = get_conversational_chain()
    
        response = chain(
             {"input_documents":docs, "question": user_question}
             , return_only_outputs=True)

        print(response)
        st.write("Reply: ", response["output_text"])
    
    

if __name__ == "__main__":
    main()