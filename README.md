# AI Image-to-Story Creator

A final project submission for the **CM3020 Artificial Intelligence** module, implementing **Project Idea 1: Orchestrating AI models to achieve a goal**. This is a web application built with Streamlit that transforms user-uploaded images into rich, stylistically controlled narratives, complete with audio narration.

The project moves beyond simple text generation to create a sophisticated system for user-directed creativity. It leverages a pipeline of four distinct AI models and an advanced prompt engineering framework to give users granular control over the creative process, turning the AI into a genuine collaborative partner.

![Application Screenshot](./final_app_screenshot.jpg)

## Key Features

-   **Multi-Modal Pipeline:** Orchestrates four AI models for image captioning (BLIP), emotion analysis (RoBERTa), narrative generation (Google Gemini), and speech synthesis (gTTS).
-   **Advanced Creative Controls:** Allows users to direct the story's tone by selecting a genre, an author to emulate, and an emotional override.
-   **Intelligent Prompt Framework:** Implements a "Master Storyteller" framework that provides the AI with a detailed creative brief to avoid clichés and generate high-quality, structured narratives.
-   **Story Threading:** Supports multi-turn conversations, allowing users to start new, separate stories within a single session.
-   **Story Continuation:** Users can add to their current story thread by uploading new images, and the AI will continue the narrative logically.
-   **On-Demand Audio & Export:** Features a non-blocking, cached system to generate audio narration for individual segments or the entire story. The final text can be easily copied or downloaded.

## Technology Stack

-   **Backend:** Python 3.11
-   **Web Framework:** Streamlit
-   **AI/ML Libraries:** PyTorch, Hugging Face Transformers, Google Generative AI
-   **Core Models:** Salesforce/BLIP, DistilRoBERTa, Gemini 1.5 Flash, gTTS

## Setup and Installation

Follow these steps to run the application locally.

**1. Clone the repository:**
```bash
git clone https://github.com/ahmad-io/CM3020-Final-Project-AI-Image-To-Story-Creator.git
cd CM3020-Final-Project-AI-Image-To-Story-Creator
```

**2. Create and activate a virtual environment:**

-   On Windows:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
-   On macOS/Linux:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up your API Key:**
-   Create a file named `.env` in the root of the project folder.
-   Add your Google AI API key to this file as follows:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```

## Usage

1.  Ensure your virtual environment is activated.
2.  Run the application from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in a new browser tab.
4.  Use the "Creative Controls" in the sidebar to set your desired genre and style.
5.  Upload 1-3 images in the main panel to begin.

## License

This project is licensed under the MIT License.