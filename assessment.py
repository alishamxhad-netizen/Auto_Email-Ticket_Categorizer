import re
import sys
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Config

DATA_PATH = r"C:\Academic\BASE\support_tickets_dataset.csv"     # columns: id, ticket_id, subject, body, category
TEXT_COLUMNS = ["subject", "body"]
LABEL_COLUMN = "category"
CONFIDENCE_THRESHOLD = 0.60                 # below this -> route to human review
RANDOM_STATE = 42

URGENT_KEYWORDS = [
    "down", "urgent", "not working", "crash", "crashes", "asap",
    "immediately", "critical", "broken", "can't login", "cannot login",
    "blocked", "outage", "emergency",
]

# Text Preprocessing

def clean_text(text: str) -> str:
    """Lowercase, strip URLs/numbers/punctuation/extra whitespace.

    We deliberately keep this lightweight — TfidfVectorizer's own
    stop_words='english' handles stopword removal, so this function focuses
    on normalizing noise that hurts vectorization (case, punctuation, digits,
    stray whitespace).
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)      # URLs
    text = re.sub(r"#\d+", " ", text)                   # ticket/invoice refs like #12345
    text = re.sub(r"[^a-z\s]", " ", text)                # punctuation & digits
    text = re.sub(r"\s+", " ", text).strip()             # collapse whitespace
    return text


def build_corpus(df: pd.DataFrame) -> pd.Series:
    """Combine subject + body into one text field, then clean it.

    Subject and body are concatenated because the subject line often carries
    a disproportionate amount of the classification signal (e.g. 'Refund
    request'), so keeping it in the text the model sees matters.
    """
    combined = df["subject"].fillna("") + " " + df["body"].fillna("")
    return combined.apply(clean_text)


# Priority Tagging 

def tag_priority(raw_text: str) -> str:
    text = raw_text.lower()
    return "urgent" if any(kw in text for kw in URGENT_KEYWORDS) else "normal"


# TRAIN

def train():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = build_corpus(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df[LABEL_COLUMN],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN]
    )

    # TF-IDF: down-weights common words, up-weights words distinctive to a
    # ticket. unigrams + bigrams so short phrases like "not working" survive.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression over Naive Bayes here because we want calibrated,
    # well-behaved predict_proba() outputs for the confidence score / human
    # review threshold — LogisticRegression's probabilities tend to be more
    # reliable for that than MultinomialNB's, which skews overconfident.
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    return model, vectorizer, X_test_vec, y_test


# Evaluate

def evaluate(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.3f}\n")

    print("Classification report (precision / recall / f1 per class):")
    print(classification_report(y_test, y_pred))

    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Confusion matrix (rows = actual, cols = predicted):")
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
    return acc

# Predict 

def predict_ticket(subject: str, body: str, model, vectorizer):
    raw_text = f"{subject} {body}"
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])

    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = int(np.argmax(probs))

    category = classes[best_idx]
    confidence = float(probs[best_idx])
    needs_human_review = confidence < CONFIDENCE_THRESHOLD
    priority = tag_priority(raw_text)

    return {
        "predicted_category": category,
        "confidence": round(confidence, 3),
        "needs_human_review": needs_human_review,
        "priority": priority,
    }


# More Samples

SAMPLE_TICKETS = [
    ("App keeps crashing on startup",
     "Ever since I updated the app this morning it crashes immediately on launch. This is urgent, I can't get any work done."),
    ("Question about my last paycheck",
     "My paycheck this month seems lower than usual and I'm not sure why. Could someone from payroll take a look?"),
    ("Wrong amount charged to my card",
     "I was charged $89.99 but my plan only costs $49.99 a month. Please correct this and refund the difference."),
    ("Do you offer student discounts?",
     "Hi, I'm a student and was wondering if you offer any discounted pricing plans for students like me."),
    ("Something seems off with my account",
     "Not sure who to send this to, but a few things on my account don't look right lately."),
]


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true",
                         help="Drop into a live CLI demo after training.")
    args = parser.parse_args()

    print("Training classifier...")
    model, vectorizer, X_test_vec, y_test = train()

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    evaluate(model, X_test_vec, y_test)

    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS ON 5 NEW, UNSEEN TICKETS")
    print("=" * 60)
    for subject, body in SAMPLE_TICKETS:
        result = predict_ticket(subject, body, model, vectorizer)
        print(f"\nSubject: {subject}")
        print(f"Body:    {body}")
        print(f"  -> Category: {result['predicted_category']}  "
              f"(confidence: {result['confidence']:.0%})")
        print(f"  -> Needs human review: {result['needs_human_review']}")
        print(f"  -> Priority: {result['priority']}")

    if args.interactive:
        print("\n" + "=" * 60)
        print("LIVE DEMO — type a ticket, or 'quit' to exit")
        print("=" * 60)
        while True:
            subject = input("\nSubject: ").strip()
            if subject.lower() in ("quit", "exit"):
                break
            body = input("Body: ").strip()
            result = predict_ticket(subject, body, model, vectorizer)
            print(f"  -> Category: {result['predicted_category']}  "
                  f"(confidence: {result['confidence']:.0%})")
            print(f"  -> Needs human review: {result['needs_human_review']}")
            print(f"  -> Priority: {result['priority']}")


if __name__ == "__main__":
    main()
