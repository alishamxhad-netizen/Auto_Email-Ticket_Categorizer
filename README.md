# Auto Email / Ticket Categorizer

A lightweight NLP classifier that reads an incoming support ticket (subject + body)
and automatically routes it to the correct department — **Billing**, **Technical**,
**HR**, or **General** — mirroring the triage layer used in real enterprise helpdesk
systems.

## Features

- **Text preprocessing** — lowercasing, URL/punctuation/digit stripping, whitespace cleanup
- **TF-IDF vectorization** (unigrams + bigrams) to turn raw text into model-ready features
- **Logistic Regression classifier**, chosen over Naive Bayes for better-calibrated
  probability outputs
- **Evaluation**: accuracy, per-class precision/recall/F1, confusion matrix
- **Confidence score output** — every prediction returns a probability, not just a label
- **"Needs human review" threshold** — predictions below 60% confidence are routed to
  a manual-review queue instead of being auto-assigned
- **Priority tagging** — simple keyword-based urgent/normal tag (e.g. "down", "urgent",
  "not working") layered on top of the category
- **Live demo mode** — type a new ticket at the command line and get an instant prediction

## Project structure

```
.
├── ticket_classifier.py        # main script: preprocessing, training, evaluation, prediction
├── support_tickets_dataset.csv # labeled training data (subject, body, category)
└── README.md
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

## Usage

Train the model, print evaluation metrics, and run predictions on 5 sample tickets:

```bash
python ticket_classifier.py
```

Do all of the above **and** drop into an interactive CLI where you can type your own
tickets:

```bash
python ticket_classifier.py --interactive
```

Example output:

```
Subject: App keeps crashing on startup
Body:    Ever since I updated the app this morning it crashes immediately on launch.
  -> Category: Technical  (confidence: 53%)
  -> Needs human review: True
  -> Priority: urgent
```

## How it works

1. **Load & clean** — subject and body are concatenated and cleaned (lowercase, strip
   URLs/punctuation/digits).
2. **Vectorize** — `TfidfVectorizer` converts cleaned text into weighted word/bigram
   features, down-weighting common words and up-weighting distinctive ones.
3. **Train** — a `LogisticRegression` classifier is trained on an 80/20 train/test split.
4. **Evaluate** — accuracy, classification report, and confusion matrix are printed.
5. **Predict** — for any new ticket, `predict_ticket()` returns the predicted category,
   a confidence score, whether it should go to human review, and a priority tag.

## Dataset

`support_tickets_dataset.csv` contains 540 labeled synthetic support tickets, evenly
split across the four categories (135 each), with columns:

| Column | Description |
|---|---|
| `id` | Row number |
| `ticket_id` | Synthetic ticket reference (e.g. `TCK-3170`) |
| `subject` | Ticket subject line |
| `body` | Ticket body text |
| `category` | Label: Billing / Technical / HR / General |

## Notes & limitations

- The dataset is template-generated, so it's cleaner and more separable than real-world
  tickets — accuracy on this test set is close to 100%, which shouldn't be read as a
  guarantee of real-world performance.
- With more data or time, the next improvements would be: real (human-labeled) tickets,
  a broader/ noisier vocabulary per category, and a calibration check on the confidence
  scores (e.g. a reliability diagram) rather than assuming `predict_proba()` is well
  calibrated out of the box.

## License

For educational / assessment use.
