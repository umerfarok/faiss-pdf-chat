import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import getpass
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PyPDF2 import PdfReader
import sentence_transformers
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from datetime import datetime
import gdown
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import TokenTextSplitter
import faiss
import pickle
from langchain_community.docstore.in_memory import InMemoryDocstore  # ✅ Correct
from langchain.embeddings import OpenAIEmbeddings
from langchain.callbacks import get_openai_callback
import tiktoken

# Load environment variables
load_dotenv()
###############################################################setting openai ai api##################################################################
file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
output_file = "open_ai_key.txt"

# https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
df=pd.read_csv(url)
# st.write(df)

# def download_db():
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
# k=""
# with open(download_db(),'r') as f:
#     f=f.read()
#     # st.write(f)
#     k=f
os.environ["OPENAI_API_KEY"] = df.keys()[0]
#####################################################################################################################################################
# # Load all PDFs in a directory
# pdf_folder = "database"
# loader = DirectoryLoader(pdf_folder, glob="*.pdf", loader_cls=PyPDFLoader)

# # Load documents
# documents = loader.load()

# st.write(f"Loaded {len(documents)} documents from the directory")

# text_splitter = TokenTextSplitter(encoding_name='o200k_base', chunk_size=100, chunk_overlap=20)
# texts = text_splitter.split_documents(documents)
# st.write(texts)
# Assuming you have OpenAI API key set up in your environment
embeddings = OpenAIEmbeddings()
# vectorstore = FAISS.from_documents(documents=texts, embedding=embeddings)
# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()

########################################################################### Loading the vector db ###########################################################
# Load FAISS index
index = faiss.read_index("faiss_index.bin")

# Load metadata
with open("faiss_metadata.pkl", "rb") as f:
    docstore_data = pickle.load(f)

# ✅ Fix: Wrap the docstore dictionary inside InMemoryDocstore
docstore = InMemoryDocstore(docstore_data)

# Load index-to-docstore mapping
with open("faiss_index_to_docstore.pkl", "rb") as f:
    index_to_docstore_id = pickle.load(f)

# ✅ Fix: Ensure FAISS is initialized with proper embeddings
vector_store = FAISS(
    index=index,
    embedding_function=embeddings,  # ✅ Ensure embeddings are passed correctly
    docstore=docstore,  # ✅ Wrap docstore properly
    index_to_docstore_id=index_to_docstore_id
)
# Set up retriever
retriever = vector_store.as_retriever()

##########################################################################setting groq api ###############################################################

GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm_llama3 = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

##############################################################setting opne ai llm ##################################################################

llm_openai = ChatOpenAI(model="gpt-4o-mini")
###########################################################setting RAG document formatting ##############################################################


prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
##############################################################setting prompts ###########################################################################################

def get_feedback(feedback):
    return f"""
    ### What improvements would you like to see in the chatbot?  
    - {feedback["What improvements would you like to see in the chatbot?"]}

    ### What challenges did you face while using the chatbot for academic support, and how do you think these could be addressed?  
    - {feedback["What challenges did you face while using the chatbot for academic support, and how do you think these could be addressed?"]}

    ### Did you face any issues while using the chatbot? If yes, please explain.  
    - {feedback["Did you face any issues while using the chatbot? If yes, please explain."]}

    ### Summary of Responses:

    - **Improvements:** Improvements Here
    - **Challenges:** Challenges Here
    - **Issues:** Issues Here
    """

# Define a custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant answering user queries based on the provided context.
    
    Context:
    {context}
    
    Question:
    {question}

    You can add additional information from your knowledgebase as well.
    Provide a clear and concise response.
    """
    )
############################################################################################################################################################


st.title("Academic Assistant")
selections=st.sidebar.selectbox("☰ Menu", ["Home","AI Assistant", "Feedback"])


query=""
# tokens={}
if selections=="Home":
    st.markdown("""The School Student Assistant Chatbot is an AI-powered virtual assistant designed to help students with their academic and school-related queries. It provides instant responses to common questions, assists with homework, shares important school updates, and offers guidance on schedules, subjects, and extracurricular activities.  
     Key Features:  
    ✅ Homework Assistance – Provides explanations and study resources.  
    ✅ Timetable & Schedule Support – Helps students check class schedules.  
    ✅ School Announcements & Notices – Delivers updates on events and policies.  
    ✅ Subject Guidance – Answers subject-related queries.  
    ✅ Interactive – Allows students to communicate via text.  """)
    
    
if selections=="AI Assistant":
    query=st.text_input("Write Query Here")
    if st.button("Submit") and query!="":
        rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm_openai
        )
        st.subheader("OpenAI GPT Response")
        res=rag_chain.invoke(query)
        st.write(res.content)
        # st.write(res.response_metadata['token_usage']['total_tokens'])
        # tokens["open_ai"]=res.response_metadata['token_usage']['total_tokens']

    
        # # performing a similarity search to fetch the most relevant context
        st.write("")
        st.write("")
        st.write("")
    
        rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm_llama3
        )
        st.subheader("Meta Llama3 GPT Response")
        res=rag_chain.invoke(query)
        st.write(res.content)
        # tokens["open_ai"]=res.response_metadata['token_usage']['total_tokens']
        # tokens_df=pd.DataFrame(tokens.items())
        # tokens_df.to_csv("token_usage.csv")
        # st.write(tokens_df)




if selections=="Feedback":
    
    st.subheader("Welcome to User Feedback Section")
    st.write("Please Leave Feedback [Here](https://docs.google.com/forms/d/e/1FAIpQLSekxnpLx5glG_bYHy54m0IrbBIZxEM37dihnBNOeRMR0n9KUg/viewform?usp=header)")
    sheet_id = '1k1MYDZ7n9sIjPTfXFMFHMwEJOMkmhcWzikFVoXlH2SQ' # replace with your sheet's ID
    
    url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df=pd.read_csv(url)
    # st.write(df.columns)


    col1,col2=st.columns(2)
    with col1:
        ratings_x=df['How satisfied are you with the chatbot\'s overall performance?'].value_counts().index
        ratings_y=df['How satisfied are you with the chatbot\'s overall performance?'].value_counts().values
        st.write("Application Ratings")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=ratings_x, y=ratings_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Ratings")
        ax.set_ylabel("Values")
        ax.set_title("Application Ratings")
        st.pyplot(fig)

    
    with col2:
        effective_resources_x=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().index
        effective_resources_y=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().values
        st.write("Application Effective")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=effective_resources_x, y=effective_resources_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Did the Chatbot Effectively Assist You")
        st.pyplot(fig)
        
        
        
    col1,col2=st.columns(2)
    with col1:

        st.write("Which GPT is Most helpful?")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.pie(df['Which GPT responses do you find the most helpful?'].value_counts().values, labels=df['Which GPT responses do you find the most helpful?'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # Remove extra padding
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_title("Which GPT is Most helpful?")
        # Adjust layout to reduce whitespace
        st.pyplot(fig)

        
        # response_x=df['Which GPT responses do you find the most helpful?'].value_counts().index
        # response_y=df['Which GPT responses do you find the most helpful?'].value_counts().values
        # st.subheader("Application Ratings")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.bar(x=ratings_x,height=ratings_y)
        # sns.barplot(x=response_x, y=response_y, ax=ax, palette="viridis")
        # # ax.set_xticks([1, 2, 3, 4, 5])
        # ax.set_xlabel("Category")
        # ax.set_ylabel("Values")
        # ax.set_title("GPT Response ")
        # st.pyplot(fig)

    
    with col2:

        st.write("Interaction with the Chatbot")
        fig, ax = plt.subplots(figsize=(4,2))
        ax.pie(df['How easy was it to interact with the chatbot?'].value_counts().values, labels=df['How easy was it to interact with the chatbot?'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # Remove extra padding
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_title("Interaction with the Chatbot")
        # Adjust layout to reduce whitespace
        st.pyplot(fig)





        
        # interaction_x=df['How easy was it to interact with the chatbot?'].value_counts().index
        # interaction_y=df['How easy was it to interact with the chatbot?'].value_counts().values
        # st.subheader("Application Effective")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.bar(x=ratings_x,height=ratings_y)
        # sns.barplot(x=interaction_x, y=interaction_y, ax=ax, palette="deep")
        # # ax.set_xticks([1, 2, 3, 4, 5])
        # ax.set_xlabel("Category")
        # ax.set_ylabel("Values")
        # ax.set_title("Application Effective")
        # st.pyplot(fig)

        
        
        
        
    col1,col2=st.columns(2)
    with col1:
        satisfactory_x=df['Was the chatbot\'s response time satisfactory?'].value_counts().index
        satisfactory_y=df['Was the chatbot\'s response time satisfactory?'].value_counts().values
        st.write("Is AI Response Time satisfactory?")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=satisfactory_x, y=satisfactory_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Is AI Response Time satisfactory?")
        st.pyplot(fig)

    
    with col2:
        understand_x=df['Did the chatbot understand your questions correctly?'].value_counts().index
        understand_y=df['Did the chatbot understand your questions correctly?'].value_counts().values
        st.write("Does Chatbot Understand Questions?")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=understand_x, y=understand_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Does Chatbot Understand Questions?")
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        # st.write(df.columns[7])
        response_x=df[df.columns[7]].value_counts().index
        response_y=df[df.columns[7]].value_counts().values
        st.write(df.columns[7])
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=response_x, y=response_y, ax=ax, palette="viridis")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title(df.columns[7])
        st.pyplot(fig)

    
    with col2:
        experience_x=df['Up to what extent this chatbot contributed to your learning experience or academic efficiency?'].value_counts().index
        experience_y=df['Up to what extent this chatbot contributed to your learning experience or academic efficiency?'].value_counts().values
        st.write("Does Chatbot Contributed to Your Learning Experience?")
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.bar(x=ratings_x,height=ratings_y)
        sns.barplot(x=experience_x, y=experience_y, ax=ax, palette="deep")
        # ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("Category")
        ax.set_ylabel("Values")
        ax.set_title("Does Chatbot Contributed to Your Learning Experience?")
        st.pyplot(fig)

    col1,col2=st.columns(2)
    with col1:
        
        st.write("Does Chatbot give Relevant Responses?")
        fig, ax = plt.subplots(figsize=(4,6))
        ax.pie(df["Did the chatbot provide helpful and relevant responses?"].value_counts().values, labels=df["Did the chatbot provide helpful and relevant responses?"].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # Remove extra padding
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_title("Does Chatbot give Relevant Responses?")
        # Adjust layout to reduce whitespace
        st.pyplot(fig)

        
        # relevant_x=df[df.columns[8]].value_counts().index
        # relevant_y=df[df.columns[8]].value_counts().values
        # st.write("Application Ratings")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.bar(x=ratings_x,height=ratings_y)
        # sns.barplot(x=relevant_x, y=relevant_y, ax=ax, palette="viridis")
        # # ax.set_xticks([1, 2, 3, 4, 5])
        # ax.set_xlabel("Category")
        # ax.set_ylabel("Values")
        # ax.set_title("GPT Response ")
        # st.pyplot(fig)

    
    with col2:
        
        
        st.write("Would You Recommend this Chatbot")
        fig, ax = plt.subplots(figsize=(4,6))
        ax.pie(df["Would you recommend this chatbot to fellow students or faculty members for academic support?"].value_counts().values, labels=df["Would you recommend this chatbot to fellow students or faculty members for academic support?"].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # Remove extra padding
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax.set_title("Would You Recommend this Chatbot")
        # Adjust layout to reduce whitespace
        st.pyplot(fig)
        
        
        # recommend_x=df['Would you recommend this chatbot to fellow students or faculty members for academic support?'].value_counts().index
        # recommend_y=df['Would you recommend this chatbot to fellow students or faculty members for academic support?'].value_counts().values
        # st.write("Application Effective")
        # fig, ax = plt.subplots(figsize=(6, 4))
        # # ax.bar(x=ratings_x,height=ratings_y)
        # sns.barplot(x=recommend_x, y=recommend_y, ax=ax, palette="deep")
        # # ax.set_xticks([1, 2, 3, 4, 5])
        # ax.set_xlabel("Category")
        # ax.set_ylabel("Values")
        # ax.set_title("Application Effective")
        # st.pyplot(fig)
        
        
    prompt=get_feedback(df[["What improvements would you like to see in the chatbot?","What challenges did you face while using the chatbot for academic support, and how do you think these could be addressed?","Did you face any issues while using the chatbot? If yes, please explain."]].to_dict())
    st.write(llm_llama3.invoke(prompt).content)
        
        
        # st.subheader("Pie Chart (Matplotlib)")
        # fig, ax = plt.subplots(figsize=(4,2))
        # ax.pie(df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().values, labels=df['Did the chatbot effectively assist you in finding academic resources or answering your study-related questions?'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette("pastel"),radius=0.6, textprops={'fontsize': 8})
        # # Remove extra padding
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        # ax.set_title("Pie Chart Example")
        # # Adjust layout to reduce whitespace
        # st.pyplot(fig)


    

