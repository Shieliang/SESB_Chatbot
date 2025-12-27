# ⚡ SESB Intelligent Customer Service Bot (RAG + AWS Bedrock)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AWS](https://img.shields.io/badge/AWS-Bedrock%20%7C%20EC2%20%7C%20S3-orange.svg)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red.svg)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green.svg)

> **Note:** This is a personal learning project exploring Generative AI on AWS. It is **not** an official product of SESB (Sabah Electricity Sdn. Bhd.).

## 📖 Introduction

I built this chatbot to challenge myself as a new **AWS AI Practitioner**.

The goal was to solve a real-world problem: **The lack of multi-lingual support for SESB's power supply application process.** While the official guidelines are in English/Malay, many locals prefer **Chinese**.

This bot uses **RAG (Retrieval-Augmented Generation)** to read official PDF guidelines and answer user queries in Chinese (or English/Malay), bridging the language gap.

![Demo Screenshot](https://github.com/Shieliang/SESB_Chatbot/blob/main/Demo.png?raw=true)

## ✨ Key Features

* **Cross-Lingual RAG:** Retrieves information from **English** PDF documents but answers in **Chinese** (customizable via prompt).
* **Context-Aware:** Remembers chat history for follow-up questions (e.g., "What documents do I need for that?").
* **Smart Reset:** A "Clear Chat" button that performs a "hard reset" on the LangChain memory to prevent hallucinations.
* **Direct Downloads:** Generates valid S3 presigned URLs for users to download relevant application forms.

## 🏗️ Architecture

The application is deployed on an **AWS EC2 (t2.micro)** instance.

![Work Flow Screenshot](https://github.com/Shieliang/SESB_Chatbot/blob/main/SESB_Work_Flow.png?raw=true)

### Tech Stack
* **LLM:** Anthropic Claude 3.5 Sonnet (via AWS Bedrock)
* **Embeddings:** Amazon Titan Embeddings v2
* **Vector DB:** FAISS (Local cache for speed)
* **Orchestration:** LangChain
* **Frontend:** Streamlit

## 🛠️ How to Run Locally

If you want to run this code on your own machine or EC2 instance, follow these steps:

### 1. Prerequisites
* An AWS Account with access to **Bedrock** (Claude 3.5 Sonnet enabled).
* Python 3.9 or higher.
* An S3 Bucket containing your PDF documents.

### 2. Installation
Clone the repository:
```bash
git clone [https://github.com/Shieliang/SESB_Chatbot.git](https://github.com/Shieliang/SESB_Chatbot.git)
cd SESB_Chatbot
```

### 3. Install Dependencies
Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Configuration
Configuration:
Option A (Local Machine): Configure your AWS credentials using CLI:
```bash
aws configure
```

Option B (EC2 - Recommended): Attach an IAM Role to your EC2 instance with the following permissions:
* AmazonBedrockFullAccess
* AmazonS3ReadOnlyAccess

### 5. Edit Code
Edit Code:
Open app.py and update the BUCKET_NAME variable with your own S3 bucket name:
```bash
# app.py
BUCKET_NAME = 'your-own-s3-bucket-name'
```

### 6. Run the App
Run the App:
```bash
streamlit run app.py
```

### 🧩 Challenges & Learnings
Building this project taught me a lot about Cloud Architecture. Here are the main challenges I faced:

Deployment Permissions: My code crashed on EC2 initially. I learned that hardcoding keys is bad practice; using IAM Roles is the secure way to grant EC2 access to S3 and Bedrock.

"Zombie Memory": Streamlit's session state can be tricky. I had to implement a logic to explicitly delete the qa_chain object to ensure the "Clear" button truly wipes the AI's memory.

Language Barrier: To ensure accurate retrieval from English docs for Chinese queries, I optimized the LangChain Condense Prompt to handle cross-lingual context switching.

### 🤝 Contributing
This is a beginner project, so the code might not be perfect! Feedback and Pull Requests are welcome.

### 📄 License
This project is open-source and available under the MIT License.