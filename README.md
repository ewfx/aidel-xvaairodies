# 🚀 AI-Driven Entity Intelligence Risk Analysis

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How to Run](#how-to-run)
- [Team](#team)

---

## 🎯 Introduction
AI-Driven Entity Intelligence Risk Analysis aims to analyse transactions and calculate the risk involved in a transaction between multiple types of
entities, pre-trained on data of corporates and other entities (like govt orgs, NGOs etc), the system provides a robust mechanism towards identifying
shell companies and bogus transactions. 

## 🎥 Demo
📹 [Video Demo](#) demo added in artifact  

## 💡 Inspiration
Being from the risk division, it was natural to pick a problem that identifies the risk associated with transactions and different entities.

## ⚙️ What It Does
The system loads, enhances, and visualizes the dataset. It uses Named Entity Recognition (NER) to process the information obtained from the dataset
and identify named entities, using a specified pre-trained model and incorporating a straightforward aggregation strategy. The model further uses
Isolation Forest to predict risk from the amount. This is followed by making predictions about the transaction input. 

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/deepahuja17/AI_Roadies.git
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt
   ```
3. Run the project  
   ```sh
   python main.py
   ```

NOTE: For pipeline functionality used in the project, install the following manually:
 ```sh
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install .
```

## 👥 Team
- **Deep Ahuja**
- **Vishnu Awasthi**
- **Diksha Shukla**
- **Aditya Saxena**
