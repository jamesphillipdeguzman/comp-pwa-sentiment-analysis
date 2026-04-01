import pandas as pd
from transformers import pipeline
from collections import Counter
from datetime import datetime
import re
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

# Now HF_TOKEN is available as an environment variable
hf_token = os.getenv("HF_TOKEN")

# --------------------------
# LOAD AI MODEL (once)
# --------------------------
print("Loading AI model... (first time may take a while)")
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment",
    device=0,              # GPU if available
    use_auth_token=hf_token
)

LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

NEGATIVE_HINTS = [
    "haven't received", "not received", "did not receive",
    "can't", "cannot", "error", "issue", "problem",
    "failed", "missing", "not working", "unable", "delay"
]

POSITIVE_HINTS = [
    "thank", "thanks", "great", "good", "happy", "resolved",
    "appreciate", "excellent", "working", "success", "completed"
]

TOPIC_KEYWORDS = {
    "Payment Issue": ["pay", "payment", "tuition", "fee", "balance"],
    "Certificate Issue": ["certificate", "cert", "transcript", "diploma"],
    "Enrollment Issue": ["enroll", "register", "registration", "course", "class"],
    "Technical Issue": ["error", "portal", "login", "system", "not working"],
    "Academic Inquiry": ["gpa", "graduate", "credits", "degree", "latin honor"]
}

# --------------------------
# SENTIMENT FUNCTION
# --------------------------
def classify_sentiment_mixed(text, topic_hint=None, threshold=0.3):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"

    text_lower = text.lower()
    if any(k in text_lower for k in NEGATIVE_HINTS):
        return "negative"
    if any(k in text_lower for k in POSITIVE_HINTS):
        return "positive"
    if topic_hint in ["Payment Issue", "Certificate Issue", "Enrollment Issue", "Technical Issue"]:
        return "negative"

    scores = sentiment_model(text[:512], return_all_scores=True)
    if isinstance(scores[0], list):
        scores = scores[0]

    label_probs = {LABEL_MAP[s['label']]: s['score'] for s in scores}
    sorted_probs = sorted(label_probs.items(), key=lambda x: x[1], reverse=True)

    top_label, top_score = sorted_probs[0]
    if len(sorted_probs) > 1:
        second_label, second_score = sorted_probs[1]
        if top_score - second_score < threshold:
            return f"mixed ({top_label}/{second_label})"
    return top_label

# --------------------------
# TOPIC FUNCTION
# --------------------------
def classify_topic(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return topic
    return "General Inquiry"

# --------------------------
# CONSENSUS FUNCTION
# --------------------------
def consensus(sentiments):
    counts = Counter(sentiments)
    most_common = counts.most_common()
    if len(most_common) == 1:
        return most_common[0][0]
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return "mixed (" + "/".join([most_common[0][0], most_common[1][0]]) + ")"
    return most_common[0][0]

# --------------------------
# TEXT NORMALIZATION
# --------------------------
def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)     # normalize spaces
    return text

# --------------------------
# MAIN PIPELINE
# --------------------------
def main(batch_size=5000):
    print("Reading Excel file...")
    df = pd.read_excel("input.xlsx")

    # --------------------------
    # CLEANING
    # --------------------------
    df = df[df["user_prompt"].notna() & (df["user_prompt"].str.strip() != "")]
    df["user_prompt"] = df["user_prompt"].astype(str)
    df["topic_filled"] = df.get("topic", pd.Series(["General Inquiry"]*len(df)))
    df["sentiment_original"] = df.get("sentiment", pd.Series([""]*len(df))).fillna("").astype(str).str.lower()

    if "total_duration_seconds" not in df.columns:
        df["total_duration_seconds"] = 0
    if "times_asked" not in df.columns:
        df["times_asked"] = 1

    # --------------------------
    # SENTIMENT PREDICTION
    # --------------------------
    print("Running sentiment classification in batches...")
    sentiments_pred = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        batch = df.iloc[start:end]
        batch_sentiments = [
            classify_sentiment_mixed(row["user_prompt"], topic_hint=row["topic_filled"])
            for _, row in batch.iterrows()
        ]
        sentiments_pred.extend(batch_sentiments)
        print(f"Processed rows {start} to {min(end, len(df))}")
    df["sentiment_predicted"] = sentiments_pred

    # --------------------------
    # TOPIC CLASSIFICATION
    # --------------------------
    print("Classifying topics...")
    df["predicted_topic"] = df["user_prompt"].apply(classify_topic)

    # --------------------------
    # NORMALIZE PROMPTS FOR AGGREGATION
    # --------------------------
    df["user_prompt_clean"] = df["user_prompt"].apply(normalize_text)

    # --------------------------
    # AGGREGATION
    # --------------------------
    print("Aggregating repeated prompts...")
    agg_df = df.groupby("user_prompt_clean").agg(
        student_count=("user_prompt", "count"),
        total_duration=("total_duration_seconds", "sum"),
        total_times_asked=("times_asked", "sum"),
        sentiment_list=("sentiment_predicted", list),
        predicted_topic=("predicted_topic", lambda x: x.mode()[0] if not x.mode().empty else "General Inquiry"),
        original_prompt=("user_prompt", lambda x: x.iloc[0])  # keep one example
    ).reset_index(drop=True)

    agg_df["avg_duration"] = agg_df["total_duration"] / agg_df["total_times_asked"]
    agg_df["consensus_sentiment"] = agg_df["sentiment_list"].apply(consensus)

    # --------------------------
    # TOP 10 REAL ISSUES
    # --------------------------
    top10_issues = agg_df.sort_values("student_count", ascending=False).head(10)


    # Add export timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg_df["export_timestamp"] = current_time
    top10_issues["export_timestamp"] = current_time

    # --------------------------
    # SAVE OUTPUT
    # --------------------------
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    agg_df["export_timestamp"] = current_time
    top10_issues["export_timestamp"] = current_time

    with pd.ExcelWriter("companion-pwa-sentiment-analysis.xlsx") as writer:
        agg_df.to_excel(writer, sheet_name="All_Issues", index=False)
        top10_issues.to_excel(writer, sheet_name="Top_10_Issues", index=False)
    print("✅ Done! Output saved with 2 sheets: 'All_Issues' and 'Top_10_Issues'.")


# --------------------------
# RUN SCRIPT
# --------------------------
if __name__ == "__main__":
    main(batch_size=5000)