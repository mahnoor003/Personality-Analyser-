# Personality-Analyser-

🧠 Project Overview
This web-based AI application analyzes personality traits from LinkedIn and GitHub profiles using Natural Language Processing (NLP) and BERT-based models. It predicts the Big Five (OCEAN) traits:

Openness

Conscientiousness

Extraversion

Agreeableness

Neuroticism

The tool supports manual or batch input, visual reports, and PDF export.

🔍 Features
📄 Input via text or CSV upload

🤖 Personality prediction using fine-tuned BERT (Minej/bert-base-personality)

📊 Visualizations: Radar charts & bar graphs (via Plotly)

🧾 PDF report generation (using fpdf)

🌓 Dark/Light mode

📱 Mobile-responsive UI

⚠️ Robust error handling

🛠️ Tools & Tech Stack
Frontend: Streamlit, HTML/CSS

Backend: Python

NLP Models: sentence-transformers, transformers

Visualization: Plotly, Matplotlib

PDF Export: FPDF

ML Models: BERT, RandomForest (optional evaluation)

🔬 Methodology
Data Collection: Manual text or CSV from LinkedIn/GitHub

Preprocessing: Cleaning, stopword removal, tokenization

Embedding: Using all-MiniLM-L6-v2

Prediction: Via bert-base-personality

Visualization: Radar plots & bar charts

Evaluation: MAE using RandomForest (pseudo-labeled data)

📊 Sample Results
GitHub users: Higher Conscientiousness

LinkedIn users: Stronger Agreeableness

Clear visual comparisons between platforms

Exportable reports for offline use
