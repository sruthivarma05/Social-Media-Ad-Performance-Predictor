from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv(
    r"C:\Users\sruthi\Pictures\Downloads\social_media_ad_engagement_20000_rows.csv"
)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434"
)

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        document = Document(
            page_content=(
                f"Platform {row['platform']} "
                f"Ad format {row['ad_format']} "
                f"Placement {row['placement']} "
                f"Caption length {row['caption_length']} "
                f"Emoji count {row['emoji_count']} "
                f"Sentiment {row['sentiment_score']} "
                f"Post hour {row['hour']} "
                f"Age group {row['age_group']} "
                f"Interest {row['interest_category']}"
            ),
            metadata={
                "engagement_rate": row["engagement_rate"],
                "platform": row["platform"]
            },
            id=str(i)
        )
        documents.append(document)
        ids.append(str(i))

vector_store = Chroma(
    collection_name="social_media_ads",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
