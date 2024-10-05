# Conflict Resolution Assistant

## Overview

The **Conflict Resolution Assistant** is a Streamlit application designed to provide empathetic support and guidance for individuals navigating disputes or disagreements. Utilizing advanced language processing capabilities, this assistant helps users articulate their concerns and find constructive solutions.

## Features

- **Conversational Interface**: Engages users in a friendly, supportive manner.
- **Empathetic Responses**: Uses a specialized prompt to generate compassionate responses tailored to the user's context and emotions.
- **Intent Recognition**: Identifies user intents such as greetings, requests for support, or general queries.
- **Retrieval-Augmented Generation**: Combines user queries with contextual information to provide informed answers.
- **Intel-Optimized Performance**: Leverages the Intel Distribution for Python for enhanced performance on Intel hardware.

## Technologies Used

- **Streamlit**: For creating the interactive web application.
- **LangChain**: A framework for building applications powered by language models.
- **FAISS**: For efficient similarity search and retrieval of context documents.
- **Hugging Face Transformers**: For embeddings and language model integration.
- **Intel Distribution for Python**: For optimized performance on Intel hardware.
- **CTransformers**: For running the Llama-2 language model.



## Prerequisites

- Python 3.7+
- Intel Distribution for Python
- Streamlit
- LangChain and LangChain Community packages
- Hugging Face Transformers
- FAISS
- CTransformers

You can install the necessary packages using pip:

```bash
pip install streamlit langchain langchain-community huggingface-hub
```

### Setup Instructions

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Edward-Eughene-Timothy/CONFLICT-RESOLUTION-ASSISTANT.git
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

- **Main File**: The main logic is contained in `model6.py`, where the application initializes the QA model, handles user input, and generates responses.
- **Custom Prompt Template**: A specialized prompt designed to elicit empathetic responses from the language model based on user context.
- **Model Loading**: The `load_llm()` function initializes the language model used for generating responses.
- **QA Chain Creation**: The `create_qa_chain()` function sets up the RetrievalQA chain, which combines the language model with the document retrieval system.
- **User Intent Recognition**: The `recognize_intent()` function categorizes user messages to tailor responses accordingly.

## Usage

- Type your message in the input field and click "Send."
- The assistant will respond based on the context of your message, providing empathetic and supportive replies.

## Troubleshooting

 If you encounter issues with Intel optimizations, ensure you're using the Intel Distribution for Python and that your environment is set up correctly.
- For FAISS-related errors, check that your vectorstore is properly initialized and located in the correct directory.
- If you experience out-of-memory errors, consider reducing the model size or adjusting the `max_new_tokens` parameter in `load_llm()`.

## Contributing

Contributions are welcome! If you would like to improve the project, please feel free to submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
