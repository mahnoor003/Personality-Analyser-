# Personality Analyzer from LinkedIn and GitHub


## 📑 Abstract

This web-based application utilizes Artificial Intelligence and Natural Language Processing (NLP) to analyze personality traits from professional profiles such as LinkedIn and GitHub. The system predicts the Big Five personality traits—**Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism**—using fine-tuned BERT models. Developed in **Streamlit**, it offers interactive visualizations, batch analysis, and PDF report generation—ideal for HR, psychologists, and data enthusiasts.


## 📚 Table of Contents

1. [Introduction](#introduction)  
2. [Problem Statement](#problem-statement)  
3. [Proposed Solution](#proposed-solution)  
   - Features  
   - Methodology  
4. [Tools and Technologies](#tools-and-technologies)  
5. [Project Scope](#project-scope)  
6. [Limitations](#limitations)  
7. [Interfaces](#interfaces)  
8. [Code](#code)  
9. [Conclusion](#conclusion)  
10. [References](#references)  


## 1. Introduction

Professional behavior analysis is valuable for recruitment, self-evaluation, and personality insights. This project leverages AI to extract text from LinkedIn/GitHub and predict personality traits using BERT-based models.


## 2. Problem Statement

Although vast professional data is available online, no easily accessible tool translates it into psychological insights. There's a need to interpret this data into personality assessment.


## 3. Proposed Solution

A web app processes LinkedIn/GitHub content, extracts embeddings, and uses AI models to predict traits, presenting results visually and via downloadable PDFs.

### 3.1 Features

- **Text Input**: Manual or CSV upload  
- **Personality Prediction**: BERT-based models  
- **Visualization**: Radar + Bar plots  
- **Platform-Specific Analysis**  
- **PDF Report Generation**  
- **Dark/Light Mode Support**  
- **Mobile Responsive UI**  
- **Error Handling Built-in**

### 3.2 Methodology

- **Data Collection**: Manual or CSV input  
- **Preprocessing**: Cleaning, lemmatization (NLTK)  
- **Embedding**: all-MiniLM-L6-v2 (768-dim)  
- **Prediction**: bert-base-personality + optional RF model  
- **Visualization**: Plotly + FPDF

### 3.3 Work Flow Diagram

![image](https://github.com/user-attachments/assets/b8aad26b-2c73-4736-b2ff-223622c336b0)

### 3.4 Model Evaluation

- MAE-based comparison  
- Zero-shot performance  
- Validated with sample profiles  

### 3.5 Results & Discussion

- GitHub users → Higher **Conscientiousness**  
- LinkedIn users → Higher **Agreeableness**  
- Clear visual dominance indicators  
- Batch PDFs successfully exported  


## 4. Tools and Technologies

- **Language**: Python  
- **NLP**: NLTK, Transformers, Sentence-Transformers  
- **ML**: Scikit-learn, Torch  
- **Visualization**: Plotly, Matplotlib  
- **PDF**: fpdf  
- **UI**: Streamlit  


## 5. Project Scope

An NLP-based web app predicting personality traits using public LinkedIn/GitHub content. It supports manual and batch uploads, interactive plots, and PDF reports for HR, researchers, and individuals.


## 7. Interfaces 

- Dashboard
  ![image](https://github.com/user-attachments/assets/6f090d49-a827-4375-b1e5-bbeb9297b435)

- LinkedIn Analysis
  ![image](https://github.com/user-attachments/assets/e1587dda-f2a1-4baa-8ddc-a6606d4a9294)

- Trait Visualization
  ![image](https://github.com/user-attachments/assets/e465054e-0bae-4f8c-b292-11518cb9c313)

- Report Generation
![image](https://github.com/user-attachments/assets/456f50e3-99e0-472c-8ffa-7b8dc7d2d5fc)

- GitHub Analysis Manual Input
![image](https://github.com/user-attachments/assets/6c8b538a-3ae0-4e00-ac79-49f8a7132813)

- GitHub Analysis All users input
  ![image](https://github.com/user-attachments/assets/a93a26f7-7fb4-4a14-b797-de9d6de4eee8)

- Platform Comparison
  ![image](https://github.com/user-attachments/assets/af63fc11-3e27-4025-aaf6-17747978d59d)
  ![image](https://github.com/user-attachments/assets/c97376bd-b4b1-4c8f-9b56-dd06293d77d2)

- Key Insights  
  ![image](https://github.com/user-attachments/assets/022b2bfb-7f08-4ae4-b586-557748a014f2)
