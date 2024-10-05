# Conflict Resolution Assistant

## Overview

The **Conflict Resolution Assistant** is a Streamlit application designed to provide empathetic support and guidance for individuals navigating disputes or disagreements. Utilizing advanced language processing capabilities, this assistant helps users articulate their concerns and find constructive solutions.

## Features

- **Conversational Interface**: Engages users in a friendly, supportive manner.
- **Empathetic Responses**: Uses a specialized prompt to generate compassionate responses tailored to the user's context and emotions.
- **Retrieval-Augmented Generation**: Combines user queries with contextual information to provide informed answers.
- **Easy-to-Use**: Built with Streamlit for a seamless user experience.

## Technologies Used

- **Streamlit**: For creating the interactive web application.
- **LangChain**: A framework for building applications powered by language models.
- **FAISS**: For efficient similarity search and retrieval of context documents.
- **Hugging Face Transformers**: For embeddings and language model integration.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.7+
- Streamlit
- LangChain community packages
- HuggingFace Transformers

You can install the necessary packages using pip:

```bash
pip install streamlit langchain langchain-community huggingface-hub
```

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone <repository-url>
   cd conflict-resolution-assistant
   ```

2. **Download the Vector Store**: Ensure you have the FAISS vector store located in `vectorstore/db_faiss`.

3. **Run the Application**:
   Start the Streamlit server by running:

   ```bash
   streamlit run main.py
   ```

4. **Access the App**: Open your browser and navigate to `http://localhost:8501`.

## Code Explanation

- **Main File**: The main logic is contained in `main.py`, where the application initializes the QA model, handles user input, and generates responses.
- **Custom Prompt Template**: A specialized prompt designed to elicit empathetic responses from the language model based on user context.
- **Model Loading**: The `load_llm()` function initializes the language model used for generating responses.
- **QA Chain Creation**: The `create_qa_chain()` function sets up the RetrievalQA chain, which combines the language model with the document retrieval system.
- **User Intent Recognition**: The `recognize_intent()` function categorizes user messages to tailor responses accordingly.

## Usage

- Type your message in the input field and click "Send."
- The assistant will respond based on the context of your message, providing empathetic and supportive replies.

## Troubleshooting

- If you encounter issues initializing the QA bot, ensure that the FAISS vector store path is correctly set and that all required packages are installed.
- For any other errors, check the terminal for traceback messages to identify the source of the issue.

## Contributing

Contributions are welcome! If you would like to improve the project, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
