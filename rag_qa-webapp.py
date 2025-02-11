import streamlit as st

from llama_index.core import Settings, load_index_from_storage, \
    StorageContext, VectorStoreIndex
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from pathlib import Path
import requests

import logging
logging.basicConfig(level=logging.INFO)

st.title('RAG Q&A')

# Define default settings.
if 'defaults' not in st.session_state:
    st.session_state.defaults = {
        'file_url' : 'https://arxiv.org/pdf/1706.03762.pdf',
        'file_path': './data/transformer.pdf',
        'topic': 'the article: Attention is All You Need',
        'model_name': 'mistral',
        'persist_dir': './storage'
        }

# Initiate messages with the greeting message.
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Change the default settings of llama_index to Ollama model.
Settings.llm = Ollama(model=st.session_state.defaults['model_name'],
                      request_timeout=500.0)
Settings.embed_model = OllamaEmbedding(st.session_state.defaults['model_name'])

# Initiate index and the query engine.
if 'query_engine' not in st.session_state:
    logging.info('Attempting to load the index from default directory')
    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=st.session_state.defaults['persist_dir'])
        index = load_index_from_storage(storage_context)
        logging.info('Index located and loaded.')
    except:
        logging.info('Index cannot be loaded. Checking for the file at ' + \
                     st.session_state.defaults['file_path'])
        
        if not Path(st.session_state.defaults['file_path']).exists():
            logging.info('File not found. Downloading from the source...')

            req = requests.get(st.session_state.defaults['file_url'])
            with open(st.session_state.defaults['file_path'], 'wb') as f:
                f.write(req.content)

        loader = PyMuPDFReader()
        documents = loader.load(st.session_state.defaults['file_path'])
        logging.info('File loaded.')
        
        # Display a message while waiting for index to be created.
        waiting_message = 'Creating index from the file. May take a long time!'
        logging.info(waiting_message)
        with st.chat_message('assistant'):
            with st.spinner(waiting_message):
                index = VectorStoreIndex.from_documents(documents)
                st.write('Index created.')

        logging.info('Index created. Saving it locally for future use.')
        index.storage_context.persist(
            persist_dir=st.session_state.defaults['persist_dir'])
    
    query_engine = index.as_query_engine(similarity_top_k=2)
    vector_retriever = index.as_retriever(similarity_top_k=2)
    st.session_state.query_engine = query_engine

# Start of the chat section
topic = st.session_state.defaults['topic']
greeting_message = f"Hi there! I'm here to answer any questions regarding {topic}."
with st.chat_message('assistant'):
    st.write(greeting_message)

# Ask for user input.
prompt = st.chat_input(f'Ask about {topic}')
if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    logging.info(f'User input: {prompt}')

    # Print all messages so far.
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.write(message['content'])

    with st.chat_message('assistant'):
        logging.info('Generating response')
        
        with st.spinner('Running your prompt...'):
            try:
                # Send the user input to the query engine.
                response = st.session_state.query_engine.query(prompt)
                response_message = response.response
                st.write(response_message)
                
                # Append the response message to the messages.
                st.session_state.messages.append({'role': 'assistant', 'content': response_message})
                logging.info(f'Response: {response_message}')

            except Exception as e:
                st.session_state.messages.append({'role': 'assistant', 'content': str(e)})
                st.error('An error occurred while generating the response.')
                logging.error(f'Error: {str(e)}')