# import os
# import streamlit as st
# from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from langchain.text_splitter import CharacterTextSplitter
# from app import vectorstore


# st.title("Document Management")

# # File uploader
# uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx','xlsx','xls'])

# if uploaded_file is not None:
#     # Create a temporary directory to store the uploaded file
#     temp_dir = "temp_uploads"
#     os.makedirs(temp_dir, exist_ok=True)
#     file_path = os.path.join(temp_dir, uploaded_file.name)
    
#     # Save the uploaded file temporarily
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     st.success(f"File {uploaded_file.name} successfully uploaded!")

#     # Process the uploaded file
#     if st.button("Process Document"):
#         with st.spinner("Processing document..."):
#             try:
#                 # Load the document based on file type
#                 if uploaded_file.type == "application/pdf":
#                     loader = PyPDFLoader(file_path)
#                 elif uploaded_file.type == "text/plain":
#                     loader = TextLoader(file_path)
#                 else:
#                     st.error("Unsupported file type.")
#                     st.stop()
                
#                 documents = loader.load()
                
#                 # Split the document into chunks
#                 text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#                 texts = text_splitter.split_documents(documents)
                
#                 # Add the chunks to the vectorstore
#                 vectorstore.add_documents(texts)
                
#                 st.success(f"Document processed and added to the knowledge base!")
#             except Exception as e:
#                 st.error(f"An error occurred: {e}")
        
#         # Clean up: remove the temporary file
#         os.remove(file_path)

# # Display current documents in the knowledge base
# # st.subheader("Current Documents in Knowledge Base")
# # # This is a placeholder. You'll need to implement a method to retrieve and display
# # # the list of documents currently in your Chroma database.
# # st.write("Placeholder for document list")

# # # Option to clear the entire knowledge base
# # if st.button("Clear Knowledge Base"):
# #     if st.sidebar.checkbox("Are you sure you want to clear the entire knowledge base? This action cannot be undone."):
# #         try:
# #             # Clear the Chroma database
# #             vectorstore.delete()
# #             st.success("Knowledge base cleared!")
# #         except Exception as e:
# #             st.error(f"An error occurred: {e}")
import os
import streamlit as st
import pandas as pd
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import app

st.title("Document Management")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx', 'xlsx', 'xls'])

if uploaded_file is not None:
    # Create a temporary directory to store the uploaded file
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Save the uploaded file temporarily
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File {uploaded_file.name} successfully uploaded!")

    # Process the uploaded file
    if st.button("Process Document"):
        with st.spinner("Processing document..."):
            try:
                # Load the document based on file type
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                    documents = loader.load()
                elif uploaded_file.type == "text/plain":
                    loader = TextLoader(file_path)
                    documents = loader.load()
                elif uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                         "application/vnd.ms-excel"]:
                    # Read Excel file
                    df = pd.read_excel(file_path)
                    
                    # Convert DataFrame to string representation
                    text_content = df.to_string()
                    
                    # Create a Document object
                    documents = [Document(
                        page_content=text_content,
                        metadata={"source": uploaded_file.name}
                    )]
                else:
                    st.error("Unsupported file type.")
                    st.stop()
                
                # Split the document into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
                
                # Add the chunks to the vectorstore
                try:
                    app.vectorstore.add_documents(texts)
                except Exception as e:
                    err_msg = str(e).lower()
                    if "compaction" in err_msg or "mismatched types" in err_msg:
                        st.warning("ChromaDB corrupted. Resetting and retrying...")
                        app.reset_chroma_db()
                        app.vectorstore.add_documents(texts)
                    else:
                        raise
                
                st.success(f"Document processed and added to the knowledge base!")
            except Exception as e:
                st.error(f"An error occurred: {e}")
            finally:
                # Clean up: remove the temporary file
                os.remove(file_path)

# Display current documents in the knowledge base
# st.subheader("Current Documents in Knowledge Base")
# # This is a placeholder. You'll need to implement a method to retrieve and display
# # the list of documents currently in your Chroma database.
# st.write("Placeholder for document list")

# # Option to clear the entire knowledge base
# if st.button("Clear Knowledge Base"):
#     if st.sidebar.checkbox("Are you sure you want to clear the entire knowledge base? This action cannot be undone."):
#         try:
#             # Clear the Chroma database
#             vectorstore.delete()
#             st.success("Knowledge base cleared!")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")