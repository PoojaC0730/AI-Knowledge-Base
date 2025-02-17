# AI Knowledge Base

## Description

This Streamlit application is a personal AI Knowledge Base that empowers users to store, manage, and interact with their documents and notes.  Leveraging LangChain and Hugging Face models, it creates a conversational AI capable of answering questions based on uploaded documents and created notes. The application features a chat interface, note-taking section, document management, and a knowledge base overview.

## Table of Contents

*   [Features](#features)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Dependencies](#dependencies)
*   [Environment Variables](#environment-variables)
*   [License](#license)

## Features

*   **Chat Interface:** Engage with your knowledge base through a conversational chat interface powered by LangChain.
*   **Notes Management:** Create, store, and manage personal notes within the knowledge base.
*   **Document Upload:** Expand the knowledge base by uploading PDF documents.
*   **Knowledge Base Summary:** Obtain a quick overview of the content stored in your knowledge base.
*   **Persistent Storage:** Utilizes ChromaDB for persistent storage of document embeddings.
*   **Conversation History:** Remembers previous conversations with the chatbot.
*   **Document Processing:** Splits large documents into chunks for optimized retrieval and summarization.

## Installation

1.  **Clone the repository:**

    ```
    git clone [your_repository_url]
    cd [your_project_directory]
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```
    *(Create a `requirements.txt` file with all dependencies. Example content below.)*

## Usage

1.  **Run the Streamlit application:**

    ```
    streamlit run app.py
    ```
    *(Assuming your main file is named `app.py`.  If not, adjust the command accordingly.)*

2.  **Access the application in your browser:**

    Open your browser and go to the address shown in the terminal (usually `http://localhost:8501`).

3.  **Interact with the application:**

    *   Use the "Chat" tab to ask questions.
    *   Use the "Notes" tab to create and manage notes.
    *   Use the "Documents" tab to upload PDF documents.
    *   Use the "Knowledge Base" tab to view statistics and summaries.

## Dependencies

*   streamlit
*   pandas
*   datetime
*   json
*   base64
*   typing
*   tempfile
*   uuid
*   hashlib
*   langchain
*   langchain\_community
*   HuggingFaceHub
*   PyPDFLoader


## Environment Variables

*   `HUGGINGFACE_API_TOKEN`: Your Hugging Face API token. Obtain one from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).  This is *essential* for the LangChain integration. Set this as an environment variable in the system.

## Author
Pooja Chaudhari

## License
MIT License