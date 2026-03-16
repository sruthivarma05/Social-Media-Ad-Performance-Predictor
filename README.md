# Social-Media-Ad-Performance-Predictor
AI-powered system that predicts social media ad engagement using machine learning and explains performance using a RAG pipeline with Llama 3.
Overview

EngageAI is an AI-driven analytics tool designed to predict the engagement rate of social media ads before they are published. The system combines traditional machine learning with large language models to both forecast engagement and explain the reasoning behind the prediction.

The project uses historical social media ad data to train a regression model and a vector database to retrieve similar past ads. A local LLM (Llama 3 via Ollama) analyzes this context to provide human-readable insights and suggestions for improving ad performance.

Key Features

Predict engagement rate before posting

Machine learning model trained on historical ad data

Retrieval-Augmented Generation (RAG) for contextual explanations

Vector database search for similar ads

Local LLM inference using Ollama

Interactive Streamlit dashboard

Supports batch insights and performance explanations

System Architecture

Historical ad data is loaded and processed.

A RandomForest regression model is trained to predict engagement rate.

Ads are embedded using Ollama embeddings.

Embeddings are stored in a Chroma vector database.

User describes a new ad or reel.

The ML model predicts engagement.

The vector database retrieves similar historical ads.

The LLM analyzes the prediction and retrieved ads.

The system returns performance insights and suggestions.

Tech Stack

Machine Learning

Scikit-learn

RandomForest Regressor

Feature Engineering

LLM & AI

LangChain

Ollama

Llama 3

Vector Database

ChromaDB

Ollama Embeddings (mxbai-embed-large)

Application

Streamlit

Python

Pandas

Project Structure
.
├── main.py                 # CLI interaction with the AI system
├── train_model.py          # ML model training pipeline
├── predictor.py            # Engagement prediction function
├── vector.py               # Vector database + embeddings setup
├── streamlit_app.py        # Web dashboard
├── engagement_model.pkl    # Trained ML model
├── chroma_langchain_db     # Vector database
└── dataset
    └── social_media_ad_engagement.csv
Installation

Clone the repository

git clone https://github.com/yourusername/engage-ai-social-media-predictor.git
cd engage-ai-social-media-predictor

Install dependencies

pip install -r requirements.txt

Install and run Ollama

https://ollama.ai

Pull required models

ollama pull llama3
ollama pull mxbai-embed-large
Running the Project

Train the ML model

python train_model.py

Run the CLI version

python main.py

Run the Streamlit dashboard

streamlit run streamlit_app.py
Example Workflow

User describes an Instagram reel including content, audience, and posting time.

The ML model predicts engagement rate.

The vector database retrieves similar historical ads.

The LLM analyzes the retrieved ads and prediction.

The system returns:

predicted engagement rate

explanation of expected performance

actionable improvement suggestions

Example Output

Predicted engagement: 6.8%

Explanation: Similar fitness reels posted during evening hours with short captions and emoji usage achieved above-average engagement. The predicted performance is driven by strong audience alignment and optimal posting time.

Suggestions:

Include a call-to-action in the caption

Test posting slightly earlier around peak engagement hours

Future Improvements

Integrate image and video feature extraction

Add sentiment analysis for captions

Support multi-platform campaign analysis

Deploy as a cloud-based API

Real-time engagement prediction

