# ML-Project
Genz Slang Translator

# Slang Translator - Gen Z ↔ Standard English 

A web-based NLP application that translates Gen Z slang to standard English and vice versa. The model combines a curated slang dictionary with a machine learning pipeline (TF-IDF + Multinomial Naive Bayes) to understand informal, trendy expressions and translate them appropriately.

---

## Features

- 🔤 Translate slang to formal English and vice versa
- 🧠 Hybrid approach: dictionary + machine learning
- 📊 Confidence score with each prediction
- 📈 Evaluates model performance (accuracy, precision, recall, F1)
- 🧪 Ready test dataset included
- 💻 Streamlit-ready (optional UI)

---

## Tech Stack

- Python 3.11+
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Streamlit (optional web app)

---

## Project Structure

<pre> 
  SLANGTRANSLATOR/ 
  │ 
  ├── data/ 
  │ ├── slang_dictionary.json 
  │ ├── test_data.json 
  │ └── training_data.json 
  │ 
  ├── models/ 
  │ └── slang_classifier.pkl 
  │ 
  ├── utils/ 
  │ └── text_processor.py 
  │ 
  ├── evaluate_model.py # Script to compute accuracy, precision, F1 
  ├── inspect_model.py # Inspects pickled model pipeline 
  ├── slang_translator.py # Core translation logic 
  ├── app.py # (Optional) Streamlit interface 
  ├── requirements.txt # Dependencies 
  └── README.md 
   </pre>



---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/slang-translator.git
cd slang-translator

2. In VS Code: Open Virtual Environment(optional):

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

3. Install dependencies:

pip install -r requirements.txt

4.Run Evaluation(checking precision, accuracy):

python evaluate_model.py

5. Run app:

locally: streamlit run app.py

