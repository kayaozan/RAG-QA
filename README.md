# Retrieval Augmented Generation (RAG) Question Answering
This project demonstrate how to build a question answering system with Retrieval Augmented Generation capabilities. 

## About RAG
Retrieval-Augmented Generation (RAG) is a technique that provides generative AI models with information retrieval capabilities. LLM models generate responses with reference to given document(s).

Models can be adjusted with this technique on a domain based scale. They can generate more relevant, accurate responses on a topic. Also, while fine-tuning requires the models to retrain, this does not require it.

For instance, companies could parse the documents about their process and such. Models can then be used to search information within those documents.

## Dependencies
The script is written in Python. The required software and libraries are as following:
- [Ollama](https://ollama.com/): Lets the user to run LLMs. It is installed locally and the LLM used in this project is downloaded.
- [mistral](https://ollama.com/library/mistral): A relatively small but capable LLM model.
- [LlamaIndex](https://github.com/run-llama/llama_index): a data framework for LLM applications. A bridge between Python and Ollama(or any LLM like ChatGPT).
- [Streamlit](https://streamlit.io/): The library to build the chatbot app.

## Breakdown
- The paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762) has been selected for this project. It seemed fitting for an LLM app.
- The index of the document, a vector representation, is generated. That index is used as a query engine.
- Index is saved due to the long time it takes to create. In future uses, it is simply read and used.
- The user is asked to provide a question about the topic.
- The query engine will search for similar entries in the document. If the question is relevant, it is excepted to find that information.

## Final Look
Here is how the page looks once the script is run by streamlit in a web browser.

![Screenshot 2025-02-11 182845](https://github.com/user-attachments/assets/904f0202-6e3b-4421-b7aa-af92dc7e005c)
