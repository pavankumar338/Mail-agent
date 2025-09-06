
from langchain_groq import ChatGroq
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Gmail API imports
import os
import pickle
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first time.
        if os.path.exists('token.pickle'):
                with open('token.pickle', 'rb') as token:
                        creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                        creds.refresh(Request())
                else:
                        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                        creds = flow.run_local_server(port=0)
                # Save the credentials for the next run
                with open('token.pickle', 'wb') as token:
                        pickle.dump(creds, token)
        service = build('gmail', 'v1', credentials=creds)
        return service




def fetch_and_store_emails(max_results=10):
        service = get_gmail_service()
        # Use Gmail API to get only emails with the 'IMPORTANT' label
        results = service.users().messages().list(userId='me', maxResults=max_results, labelIds=['IMPORTANT']).execute()
        messages = results.get('messages', [])
        emails = []
        for msg in messages:
                msg_data = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
                snippet = msg_data.get('snippet', '')
                headers = msg_data.get('payload', {}).get('headers', [])
                sender = None
                for header in headers:
                        if header['name'].lower() == 'from':
                                sender = header['value']
                                break
                emails.append({
                        'sender': sender,
                        'content': snippet
                })
        with open('emails.json', 'w', encoding='utf-8') as f:
                json.dump(emails, f, ensure_ascii=False, indent=2)


# Fetch emails and store in emails.json if not already present or if you want to refresh
if not os.path.exists('emails.json') or os.path.getsize('emails.json') == 0:
        fetch_and_store_emails(max_results=20)

with open('emails.json', 'r', encoding='utf-8') as f:
        emails_data = json.load(f)



llm=ChatGroq(temperature=0,
        groq_api_key='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',
        model_name='llama-3.3-70b-versatile')




# Assuming 'docs' variable contains the loaded documents from the PDF
# If not, you'll need to load the PDF first as shown in the previous step.

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)



# Create Document objects using 'content' for page_content and 'sender' as metadata
documents = [Document(page_content=email['content'], metadata={'sender': email['sender']}) for email in emails_data]
document_chunks = text_splitter.split_documents(documents)


# Create Document objects using 'content' for page_content and 'sender' as metadata
documents = [Document(page_content=email['content'], metadata={'sender': email['sender']}) for email in emails_data]
document_chunks = text_splitter.split_documents(documents)

# Initialize Hugging Face embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Create a Chroma vector store from the document chunks and embeddings
db = Chroma.from_documents(document_chunks, embeddings)

import streamlit as st
st.title("Email Chatbot (LangChain + Chroma)")
st.write("Ask questions about your emails below:")



# Create a retriever from the Chroma DB
retriever = db.as_retriever()
# Set up the RAG chain
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

user_input = st.text_input("Your question:")
if user_input:
        response = qa_chain.invoke({"query": user_input})
        st.write("Answer:", response['result'])