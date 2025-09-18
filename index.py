import os, uuid, json
import streamlit as st
from typing import List
from google.cloud import storage, bigquery
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# CONFIG

PROJECT_ID   = "gen-lang-client-0101408499"
REGION       = "us-west1"
BUCKET       = "my-file-vertex"
DATASET      = "document"
DOCS_TABLE   = "documents"
EMB_TABLE    = "documents_embeddings"

EMBED_MODEL = "gen-lang-client-0101408499.document.embedding_model"
TEXT_MODEL  = "gen-lang-client-0101408499.document.gemini_model"

# Auth
bq  = bigquery.Client(project=PROJECT_ID, location=REGION)
gcs = storage.Client(project=PROJECT_ID)

creds = st.secrets["gcp_service_account"]
client = bigquery.Client.from_service_account_info(dict(creds))

# HELPERS

# Uploding to GCS
def upload_to_gcs(local_path: str) -> str:
    """Upload file to GCS and return gs:// URI"""
    bucket = gcs.bucket(BUCKET)
    blob   = bucket.blob(os.path.basename(local_path))
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET}/{blob.name}"

# Chunk PDF
def chunk_pdf(pdf_path: str, size=1000, overlap=200) -> List[dict]:
    """Extract and chunk text from PDF"""
    docs = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = splitter.split_documents(docs)

    doc_id = str(uuid.uuid4())[:8]
    return [
        {"id": doc_id, "chunk_id": i, "text": c.page_content.lower(), "source": pdf_path}
        for i, c in enumerate(chunks)
    ]

# Insert chunks into BigQuery
def insert_chunks(chunks: List[dict]):
    """Insert chunk rows into BigQuery"""
    bq.insert_rows_json(f"{PROJECT_ID}.{DATASET}.{DOCS_TABLE}", chunks)

# Generate embeddings
def generate_embeddings():
    """Generate embeddings for chunks"""
    query = f"""
    INSERT INTO `gen-lang-client-0101408499.document.documents_embeddings`
  (id, chunk_id, text, embedding)
SELECT
  id,
  chunk_id,
  content AS text,
  ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING (
  MODEL `gen-lang-client-0101408499.document.embedding_model`,
  (
    SELECT id, chunk_id, text AS content
    FROM `gen-lang-client-0101408499.document.documents`
  ),
  STRUCT(TRUE AS flatten_json_output, 1408 AS output_dimensionality)
)
WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0;
    """
    bq.query(query).result()

# Answer query
def answer_query(query: str, debug: bool = False) -> str:
    """Retrieve relevant chunks + generate refined answer"""
    retrieve_sql = f"""
    SELECT base.text
    FROM VECTOR_SEARCH(
      TABLE `gen-lang-client-0101408499.document.documents_embeddings`,
      'embedding',
      (
        SELECT ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING (
          MODEL `gen-lang-client-0101408499.document.embedding_model`,
          (SELECT @query AS content),
          STRUCT(TRUE AS flatten_json_output, 1408 AS output_dimensionality)
        )
      ),
      top_k => 5
    );
    """
    retrieved = bq.query(
        retrieve_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("query", "STRING", query)]
        )
    ).to_dataframe()
    chunks = retrieved["text"].tolist()
    context = "\n".join(chunks)

    # Generate refined text
    generate_sql = f"""
    SELECT ml_generate_text_result AS answer
    FROM ML.GENERATE_TEXT(
      MODEL `gen-lang-client-0101408499.document.gemini_model`,
      (SELECT @prompt AS prompt),
      STRUCT(0.7 AS temperature, 1500 AS max_output_tokens)
    );
    """
    prompt = f"""
You are a knowledgeable assistant. Write a clear, structured, and well-organized answer to the question below. 
Base your answer ONLY on the provided context.

Context:
{context}

Question:
{query}

Instructions:
- Write in paragraphs, not bullet points.
- Summarize the context into a cohesive explanation.
- If the context doesn’t contain the answer, say “I don’t know”.

Answer:
"""
    result = bq.query(
        generate_sql,
        job_config=bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("prompt", "STRING", prompt)]
        )
    ).to_dataframe()

    raw_answer = result["answer"][0]
    try:
        return json.loads(raw_answer)["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return raw_answer



# STREAMLIT APP

st.title("Use File With .pdf Extension")

# Upload PDF
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    st.success("PDF uploaded successfully")

    if st.button("Process PDF"):
        st.write("Chunking and embedding...")
        upload_to_gcs("temp.pdf")
        chunks = chunk_pdf("temp.pdf")
        insert_chunks(chunks)
        generate_embeddings()
        st.success("Document processed and stored!")

# Query
user_query = st.text_input("Ask a question about your PDF")
if user_query:
    st.write("🤖 Generating answer...")
    answer = answer_query(user_query, debug=True)
    st.markdown(answer)
