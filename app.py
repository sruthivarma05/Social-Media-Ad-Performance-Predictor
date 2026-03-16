import streamlit as st
from predictor import predict_engagement
from vector import retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# LLM
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://127.0.0.1:11434"
)

template = """
You are a social media ads performance analyst.

Predicted engagement rate:
{prediction}%

Similar past ads:
{ads}

User question:
{question}

Explain the expected performance and give 1–2 improvement suggestions.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

st.title("📊 Social Media Reel Engagement Predictor")

question = st.text_input(
    "Describe your reel (content, time, audience, spend):" \
    " Low: ~0.5% , Average: ~3.5–4% , High: ~7–8% ,Hard ceiling: ~9% (rare)"
)

if st.button("Predict Engagement"):
    # For now, fixed features (same as main.py)
    features = {
        "platform": "Instagram",
        "ad_format": "Reel",
        "placement": "Reels",
        "caption_length": 120,
        "emoji_count": 3,
        "sentiment_score": 0.6,
        "hour": 19,
        "is_weekend": 1,
        "age_group": "18-24",
        "interest_category": "Fitness",
        "spend": 1500
    }

    prediction = predict_engagement(features)

    ads = retriever.invoke(question)
    ads_text = "\n\n".join(doc.page_content for doc in ads)

    result = chain.invoke({
        "prediction": prediction,
        "ads": ads_text,
        "question": question
    })

    st.subheader("Prediction")
    st.write(f"**Expected engagement:** {prediction}%")

    st.subheader("Explanation")
    st.write(result)
