
import os
import json
import re

import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

st.set_page_config(
    page_title="RAG Talent Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styling and alignment ----------
st.markdown(
    """
    <style>
      .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        margin: 0 auto;
      }
      .hero {
        padding: 1.6rem 1.8rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 18px;
        margin-bottom: 1.4rem;
      }
      .hero h1 { margin-bottom: .35rem; }
      .muted { opacity: .75; }
      div[data-testid="stButton"] button {
        width: 100%;
        min-height: 2.7rem;
        border-radius: 10px;
        font-weight: 600;
      }
      div[data-testid="stTextInput"] input {
        min-height: 2.7rem;
        border-radius: 10px;
      }
      div[data-testid="stExpander"] {
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,.25);
        margin-bottom: .6rem;
      }
      .section-gap { margin-top: 1.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🔎 RAG-Powered Talent Search Engine</h1>
      <p class="muted">
        Search fictional resumes with natural language, review supporting
        evidence, and request an AI-generated explanation.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_search_system():
    data = pd.read_csv("candidate_resumes.csv")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="streamlit_resumes")

    # Avoid stale collection contents if the CSV changes between runs.
    existing_ids = set(collection.get(include=[])["ids"])
    desired_ids = set(data["candidate_id"].astype(str).tolist())
    if existing_ids and existing_ids != desired_ids:
        client.delete_collection("streamlit_resumes")
        collection = client.get_or_create_collection(name="streamlit_resumes")

    if collection.count() == 0:
        embeddings = model.encode(
            data["resume_text"].astype(str).tolist(),
            normalize_embeddings=True,
        )
        collection.add(
            ids=data["candidate_id"].astype(str).tolist(),
            embeddings=embeddings.tolist(),
            documents=data["resume_text"].astype(str).tolist(),
            metadatas=data[
                ["name", "job_title", "skills", "experience", "education"]
            ].fillna("").astype(str).to_dict(orient="records"),
        )
    return data, model, collection

def get_gemini_api_key():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            key = None
    return key

try:
    df, model, collection = load_search_system()
except Exception as exc:
    st.error(f"Could not load the resume search system: {exc}")
    st.info("Confirm candidate_resumes.csv is in the same folder as app.py.")
    st.stop()

# ---------- Search controls ----------
st.subheader("1. Search candidates")
left, right = st.columns([4, 1], vertical_alignment="bottom")
with left:
    query = st.text_input(
        "Describe the candidate you need",
        placeholder="e.g. Junior Data Analyst with SQL and Tableau",
        label_visibility="visible",
    )
with right:
    top_k = st.selectbox("Results", options=[3, 5, 8], index=0)

search_col, hint_col = st.columns([1, 3], vertical_alignment="center")
with search_col:
    search_clicked = st.button("Search Candidates", type="primary")
with hint_col:
    st.caption("Search uses semantic similarity; it does not make hiring decisions.")

if search_clicked:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(int(top_k), collection.count()),
        )
        st.session_state["candidates"] = [
            {
                "candidate_id": results["ids"][0][i],
                "resume": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            for i in range(len(results["ids"][0]))
        ]
        st.session_state["query"] = query

if "candidates" in st.session_state:
    candidates = st.session_state["candidates"]

    st.divider()
    st.subheader("2. Matching candidates")
    st.caption(f'Query: “{st.session_state.get("query", "")}”')

    # Balanced, aligned candidate cards in two columns.
    for start in range(0, len(candidates), 2):
        card_cols = st.columns(2, gap="medium")
        for offset, col in enumerate(card_cols):
            idx = start + offset
            if idx >= len(candidates):
                continue
            candidate = candidates[idx]
            metadata = candidate["metadata"]
            with col:
                with st.container(border=True):
                    st.markdown(f"#### {idx + 1}. {metadata['name']}")
                    st.caption(metadata["job_title"])
                    st.markdown("**Skills**")
                    st.write(metadata.get("skills", "Not provided"))
                    metric_cols = st.columns(2)
                    metric_cols[0].markdown("**Experience**")
                    metric_cols[0].write(metadata.get("experience", "Not provided"))
                    metric_cols[1].markdown("**Education**")
                    metric_cols[1].write(metadata.get("education", "Not provided"))
                    with st.expander("View resume evidence"):
                        st.write(candidate["resume"])
                    st.caption(f"Semantic distance: {candidate['distance']:.4f}")

    st.info(
        "These are retrieval matches, not validated hiring recommendations. "
        "Review job-related evidence and use human oversight."
    )

    st.divider()
    st.subheader("3. Ask about a candidate")
    names = [c["metadata"]["name"] for c in candidates]
    q_col, ask_col = st.columns([2, 3], vertical_alignment="bottom")
    with q_col:
        selected_name = st.selectbox("Candidate", names)
    with ask_col:
        question = st.text_input(
            "Question about the resume",
            placeholder="Does this candidate mention SQL and Tableau?",
        )
    if st.button("Check Resume Evidence"):
        candidate = next(c for c in candidates if c["metadata"]["name"] == selected_name)
        searchable_text = (
            candidate["resume"] + " " + candidate["metadata"].get("skills", "")
        ).lower()
        terms = [
            skill for skill in
            ["sql", "tableau", "python", "excel", "power bi", "javascript",
             "react", "machine learning", "postgresql"]
            if skill in question.lower()
        ]
        if terms:
            result_cols = st.columns(min(len(terms), 3))
            for i, term in enumerate(terms):
                with result_cols[i % len(result_cols)]:
                    if term in searchable_text:
                        st.success(f"{term.title()}: mentioned")
                    else:
                        st.warning(f"{term.title()}: not found")
        else:
            st.write("Resume evidence:", candidate["resume"])
            st.caption("Ask about a specific skill to check for evidence.")

    st.divider()
    st.subheader("4. AI candidate evaluation")
    st.caption("Gemini explains evidence from the top retrieved resumes; verify its output.")

    if st.button("Generate AI Evaluation", type="primary"):
        api_key = get_gemini_api_key()
        if not api_key:
            st.error(
                "Gemini API key not configured. Set GEMINI_API_KEY as an "
                "environment variable or in Streamlit secrets."
            )
        else:
            try:
                gemini_client = genai.Client(api_key=api_key)
                evidence = [
                    {
                        "candidate_id": c["candidate_id"],
                        "name": c["metadata"].get("name", ""),
                        "job_title": c["metadata"].get("job_title", ""),
                        "skills": c["metadata"].get("skills", ""),
                        "experience": c["metadata"].get("experience", ""),
                        "resume": c["resume"],
                    }
                    for c in candidates[:3]
                ]
                prompt = f"""
Evaluate the retrieved candidates against this recruiter query:
{st.session_state.get("query", "")}

Candidate evidence:
{json.dumps(evidence)}

Use only the supplied resume evidence. Do not invent qualifications.
Explain matching evidence and requirements not demonstrated. Do not infer
demographic traits or make a final hiring decision.

Return JSON with an "evaluations" array. Each item must contain:
candidate_id, name, matching_evidence (list), missing_requirements (list),
explanation.
"""
                response = gemini_client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config={
                        "temperature": 0,
                        "max_output_tokens": 1000,
                        "response_mime_type": "application/json",
                    },
                )
                evaluation = json.loads(response.text)
                for item in evaluation.get("evaluations", []):
                    with st.expander(item.get("name", "Candidate"), expanded=True):
                        left, right = st.columns(2)
                        with left:
                            st.markdown("**Matching evidence**")
                            st.write(item.get("matching_evidence", []))
                        with right:
                            st.markdown("**Requirements not demonstrated**")
                            st.write(item.get("missing_requirements", []))
                        st.markdown("**Explanation**")
                        st.write(item.get("explanation", "No explanation returned."))
                st.caption(
                    "AI explanations can be inaccurate. Check each statement against "
                    "the original resume and use human judgment."
                )
            except Exception as exc:
                st.error(f"AI evaluation failed: {type(exc).__name__}: {str(exc)[:300]}")

    st.divider()
    st.subheader("5. Basic demographic keyword audit")
    st.caption(
        "This only detects listed keywords in resume text. It is not a complete "
        "bias or fairness assessment."
    )
    demographic_terms = [
        "gender", "race", "ethnicity", "nationality", "religion",
        "marital status", "disability", "date of birth", "age",
    ]
    audit_cols = st.columns(2)
    for i, candidate in enumerate(candidates):
        text = candidate["resume"].lower()
        flagged = [
            term for term in demographic_terms
            if re.search(r"\b" + re.escape(term) + r"\b", text)
        ]
        with audit_cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{candidate['metadata']['name']}**")
                if flagged:
                    st.warning("Review terms: " + ", ".join(flagged))
                else:
                    st.success("No demographic keywords detected")
else:
    st.markdown(
        """
        <div class="muted">
        <b>How to use:</b> enter a role and required skills, select how many
        results to retrieve, then search. The current demo uses fictional resumes.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "Educational prototype using fictional data. Do not use it to make automated "
    "employment decisions or to replace a fair, human-led recruitment process."
)
