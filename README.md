# RAG-Powered Talent Search Engine

A Retrieval-Augmented Generation (RAG) prototype for semantic resume search and AI-assisted candidate evaluation. Developed as part of the **Elevvo Pathways Natural Language Processing Internship — Task 8**.

## Project Overview

This project helps recruiters search candidate resumes using natural-language queries. It combines semantic embeddings, vector search, and a large language model (LLM) to retrieve relevant candidate profiles and explain how their documented qualifications relate to a query.

The current implementation uses eight fictional candidate profiles for demonstration and testing.

## Features

- **Semantic Resume Search:** Retrieve relevant resumes using natural-language queries.
- **Vector Database:** Store and search resume embeddings with ChromaDB.
- **Text Embeddings:** Use Sentence Transformers to convert resumes and queries into vector representations.
- **AI Candidate Evaluation:** Use Google Gemini to summarize matching evidence and requirements not demonstrated.
- **Recruiter Q&A:** Check whether selected skills are mentioned in a candidate's resume.
- **Basic Bias Keyword Audit:** Flag selected demographic terms for review.
- **Streamlit Interface:** Search candidates and review evidence in a web application.

## Technology Stack

- Python
- Pandas and NumPy
- Sentence Transformers
- ChromaDB
- Google Gemini API
- Streamlit
- Scikit-learn

## Project Structure

```text
rag-talent-search-engine/
├── app.py
├── candidate_resumes.csv
├── requirements.txt
├── README.md
└── rag_talent_search_engine.ipynb
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/rag-talent-search-engine.git
cd rag-talent-search-engine
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Configure the Gemini API Key

Set the `GEMINI_API_KEY` environment variable before starting the application.

On Linux or macOS:

```bash
export GEMINI_API_KEY="your-api-key"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

Never commit your API key to GitHub. For a deployed Streamlit application, configure the key using the platform's secrets settings.

## Run the Application

Ensure `candidate_resumes.csv` is in the same directory as `app.py`.

Start the app:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal.

## Example Search

**Recruiter query:**

> Find a Junior Data Analyst who knows SQL and Tableau.

The system retrieves semantically similar candidate resumes and displays their skills, experience, education, and resume evidence. Gemini can then generate explanations based on the retrieved information.

## Responsible Use

This application is an educational prototype, not a validated recruitment or hiring system.

- Candidate explanations should be checked against the original resume.
- Semantic similarity is not proof that a candidate meets every job requirement.
- The demographic keyword audit is limited and does not establish fairness or the absence of bias.
- Use job-related criteria and human oversight.
- Do not use the prototype to make automated employment decisions.

## Current Limitations

- The demonstration uses only eight fictional resumes.
- Retrieval quality has not been evaluated on a large, real-world resume dataset.
- LLM explanations may contain errors or omit relevant evidence.
- The basic bias audit is not a comprehensive fairness assessment.
- The in-memory vector database may need to be rebuilt when the application restarts.

## Future Improvements

- Evaluate retrieval quality using a larger, suitable public dataset.
- Add retrieval metrics and systematic LLM evaluation.
- Improve fairness testing and documentation.
- Add persistent vector storage and more comprehensive recruiter interactions.
- Deploy the application using Streamlit Community Cloud.

## Author

Developed by Stewart Daniel as part of the Elevvo Pathways Natural Language Processing internship.

---

**Disclaimer:** This project is intended for learning and demonstration purposes only.
