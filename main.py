from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from predictor import predict_engagement   

# LLM
model = OllamaLLM(
    model="llama3.2",
    base_url="http://127.0.0.1:11434"
)

# UPDATED PROMPT
template = """
You are a social media ads performance analyst.

Predicted engagement rate for this reel:
{prediction}%

Use ONLY the information from similar past ads below to explain the result.
If the information is insufficient, say:
"I don't have enough historical ad data to answer that."

Similar Ads:
{ads}

Question:
{question}

Explain the expected performance and give 1–2 actionable suggestions.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n--------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n")
    if question == "q":
        break

    # 🔹 Example reel features (known before posting)
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

    # 🔹 ML prediction
    prediction = predict_engagement(features)

    # 🔹 Retrieve similar ads
    ads = retriever.invoke(question)
    ads_text = "\n\n".join(doc.page_content for doc in ads)

    # 🔹 Final AI response
    result = chain.invoke({
        "ads": ads_text,
        "prediction": prediction,
        "question": question
    })

    print(result)
