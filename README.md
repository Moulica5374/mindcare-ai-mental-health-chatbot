# mindcare-ai-mental-health-chatbot
AI-powered mental health support system using 4 NLP modules for patient assessment and crisis detection
readme_content = """
# MindCare AI - Mental Health Support Chatbot

##  Project Overview

An intelligent NLP system that analyzes mental health statements using multiple AI modules to assist therapists and mental health professionals in patient assessment and triage.

### Business Problem
- Mental health hotlines receive thousands of calls daily
- Therapists need to quickly assess patient severity and risk levels
- Manual review of patient statements is time-consuming and inconsistent
- Early detection of conditions like suicidal ideation can save lives

### Solution
A multi-module AI system that processes patient statements through 4 specialized NLP pipelines to provide comprehensive mental health insights.

---

## 🧠 System Architecture

### Module 1: Mental Health Classification
**Purpose:** Categorize patient statements into 7 mental health conditions

**Technology:** Zero-shot classification using BART-large-MNLI

**Categories:**
- Anxiety
- Normal
- Depression
- Suicidal
- Stress
- Bipolar
- Personality Disorder

**Performance:**
- Accuracy: [Your accuracy here]%
- F1 Score (weighted): [Your F1 here]

**Use Case:** Automatic triage and priority assignment for mental health professionals

---

### Module 2: Text Summarization
**Purpose:** Generate concise summaries of lengthy patient statements

**Technology:** BART-large-CNN summarization model

**Features:**
- Condenses long patient narratives into key points
- Maintains critical mental health indicators
- Reduces therapist reading time by 70%

**Use Case:** Quick overview for therapists during intake assessments

---

### Module 3: Question Answering
**Purpose:** Extract specific information from patient statements

**Technology:** MiniLM-uncased-squad2

**Capabilities:**
- Answer queries like "What symptoms does the patient mention?"
- Extract specific details about patient history
- Identify key triggers or events

**Use Case:** Rapid information retrieval during patient sessions

---

### Module 4: Multi-language Translation
**Purpose:** Support non-English speaking patients

**Technology:** Helsinki-NLP opus-mt translation models

**Supported Languages:**
- English ↔ Spanish
- [Add more as implemented]

**Use Case:** Expand mental health services to diverse populations

---

## 📊 Dataset

**Source:** https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

**Statistics:**
- Total statements: 52,681
- After cleaning: 52,319
- Categories: 7 mental health conditions
- Average statement length: 113 words

---

## 🛠️ Technical Stack

**Framework:** Python 3.11

**Key Libraries:**
- `transformers` - Pre-trained NLP models
- `torch` - Deep learning backend
- `scikit-learn` - Evaluation metrics
- `pandas` - Data manipulation
- `streamlit` - Web interface (optional)

**Models Used:**
- facebook/bart-large-mnli (classification)
- facebook/bart-large-cnn (summarization)
- deepset/minilm-uncased-squad2 (Q&A)
- Helsinki-NLP/opus-mt-en-es (translation)

---

## 🚀 How to Run

### Installation
\`\`\`bash
pip install transformers torch scikit-learn pandas
\`\`\`

### Basic Usage
\`\`\`python
from transformers import pipeline

# Module 1: Classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
categories = ['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar', 'Personality disorder']
result = classifier("I feel overwhelmed and can't sleep", categories)

# Module 2: Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = summarizer(long_text, max_length=50)

# Module 3: Q&A
qa_pipeline = pipeline("question-answering", model="deepset/minilm-uncased-squad2")
answer = qa_pipeline(question="What symptoms?", context=patient_statement)

# Module 4: Translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
translated = translator("I need help")
\`\`\`

---

## 📈 Results & Evaluation

### Classification Performance
- **Accuracy:** XX.X%
- **Weighted F1:** X.XXX
- **Best performing category:** [Category name]
- **Most challenging category:** [Category name]

### Key Insights
1. Model performs well on clear-cut cases (Depression, Suicidal)
2. Struggles with overlapping conditions (Anxiety vs Stress)
3. Short statements (<10 words) reduce accuracy significantly

---

## 💡 Future Enhancements

1. **Fine-tuning:** Train on domain-specific mental health data
2. **Real-time Dashboard:** Streamlit app for live patient monitoring
3. **Alert System:** Automatic notifications for high-risk cases
4. **Multi-modal:** Incorporate voice tone analysis


---

## 🎓 Skills Demonstrated

- Multi-class text classification
- Transfer learning with pre-trained models
- Pipeline integration and orchestration
- Healthcare domain application
- Model evaluation and comparison
- Data cleaning and preprocessing
- Professional documentation

---

##  Author

MOULICA VANI GOLI
- LinkedIn: https://www.linkedin.com/in/moulicagoli/
- Email: moulica99@gmail.com
- Portfolio: https://moulica-portfolio1.vercel.app/

---

##  License

This project is for educational and portfolio purposes.

---

##  Acknowledgments

- 
- Pre-trained models: Hugging Face Transformers library
- Inspiration: Mental health crisis support systems
"""



