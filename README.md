# ChatBot-using-RAG


**Overview**
The goal of this project is to create a user-centric and intelligent system that enhances information retrieval from PDF documents through natural language queries. The project focuses on streamlining the user experience by developing an intuitive interface, allowing users to interact with PDF content using language they are comfortable with. To achieve this, we leverage the Retrieval Augmented Generation (RAG) methodology introduced by Meta AI researchers.


**Retrieval Augmented Generation (RAG)**
Introduction
RAG is a method designed to address knowledge-intensive tasks, particularly in information retrieval. It combines an information retrieval component with a text generator model to achieve adaptive and efficient knowledge processing. Unlike traditional methods that require retraining the entire model for knowledge updates, RAG allows for fine-tuning and modification of internal knowledge without extensive retraining.

**Workflow**
Input: RAG takes multiple pdf as input.
VectoreStore: The pdf's are then converted to vectorstore using FAISS and embedding-001 Embeddings model from Google Generative AI.
Memory: Conversation buffer memory is used to maintain a track of previous conversation which are fed to the llm model along with the user query.
Text Generation with Gemini-pro: The embedded input is fed to the Gemini-pro model from the Google's Generative AI suite, which produces the final output.
User Interface: Streamlit is used to create the interface for the application.

**Benefits**
Adaptability: RAG adapts to situations where facts may evolve over time, making it suitable for dynamic knowledge domains.
Efficiency: By combining retrieval and generation, RAG provides access to the latest information without the need for extensive model retraining.
Reliability: The methodology ensures reliable outputs by leveraging both retrieval-based and generative approaches.
Project Features
User-friendly Interface: An intuitive interface designed to accommodate natural language queries, simplifying the interaction with PDF documents.

Seamless Navigation: The system streamlines information retrieval, reducing complexity and enhancing the overall user experience.

**Getting Started**
To use the PDF Intelligence System:

1. Clone the repository to your local machine.
```bash
    git clone https://github.com/tariyalkaran/ChatBot-using-RAG.git

2. Install dependencies.
    pip install -r requirements.txt

3. Run the application.
    streamlit run app.py

4. Open your browser and navigate to http://localhost:8000 to access the user interface.





