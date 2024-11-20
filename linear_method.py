from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
import random

# Step 1: Set up the embedding model
print("Setting up the embedding model...")
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")

# Step 2: Load the vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(collection_name="new_clerc_docs", embedding_function=sentence_transformer_ef, persist_directory="new_chroma_db")
print("Chroma vector store setup complete.")

# Get 10 random queries from the query dataset with their corresponding doc ids
query_to_doc = {}
with open("processed_CLERC/queryIdToDocId.txt", "r") as f:
    pairs = [line.strip().split('\t') for line in f.readlines()]
    query_to_doc = {qid: did for qid, did in pairs}

random_pairs = random.sample(list(query_to_doc.items()), 10)
random_query_to_doc = dict(random_pairs)
print("10 random query-doc pairs: ", random_query_to_doc)

# Load query dataset - modify this section to load as a list instead of streaming
queryDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "queries/test.single-removed.indirect.tsv"},
    streaming=False,  # Changed to False
    delimiter="\t"
)["data"]

def getQuery(qId):
    for sample in queryDataset:  # Iterate over the dataset directly
        keys = list(sample.keys())
        queryId = str(sample[keys[0]])
        if queryId == qId:
            header = keys[1]
            return sample[header]
    return None  # Add explicit return None if query not found

# Get the queries from the query dataset
queries = [(qid, getQuery(qid)) for qid in random_query_to_doc.keys()]

# OpenAI client
client = OpenAI()

# Use Chroma similarity search to retrieve similar documents based on a query
for query in queries:
    query_id = query[0]
    query_text = query[1]
    print(f"\nProcessing Query ID: {query_id}")
    print(f"Query: {query_text}")

    # Perform similarity search
    results = vectorstore.similarity_search(query_text, k=5)

    # Create a structured dictionary of retrieved documents
    retrieved_docs = {
        result.metadata["doc_id"]: result.page_content for result in results
    }

    print(f"Retrieved Documents: {list(retrieved_docs.keys())}")
    for doc_id, doc_text in retrieved_docs.items():
        print(f"Document ID: {doc_id}, Text Length: {len(doc_text)}")

    # Generate a refined query using the first prompt
    first_prompt = [
        {"role": "system", "content": "You are an assistant to a lawyer writing legal analyses who needs to find case documents to support their texts."},
        {"role": "user", "content": f"""
        You are given a query that is taken from a legal case document with its citation in the middle removed.
        Your job is to generate additional questions that will help find the relevant document of this query,
        which is defined as the document its central citation cites to.

        Query: {query_text}

        Retrieved Documents:
        {retrieved_docs}

        Generate another question building on top of this query that will help a lawyer identify the most relevant document cited in this query.
        Ideally, a document is cited because of its similarity of facts to the query case, but the exact events that occurred in the relevant document
        will not be the same as the query case. The question should prompt the lawyer to achieve the goal of finding the document that is being cited in the query.
        """},
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini", # 200,000 token limit
        messages=first_prompt
    )
    refined_query = completion.choices[0].message.content
    print(f"Refined Query: {refined_query}")

    # Second LLM prompt to identify the most relevant document
    second_prompt = [
        {"role": "system", "content": "You are an assistant to a lawyer writing legal analyses who needs to find case documents to support their texts."},
        {"role": "user", "content": f"""
        Using the refined query below, determine which document from the following retrieved documents is most relevant to the query (you must select one document):

        Refined Query: {refined_query}

        Retrieved Documents (JSON format):
        {retrieved_docs}

        Output the ID of the most relevant document and explain why it is the most relevant.
        """},
    ]
    second_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=second_prompt
    )
    llm_response = second_completion.choices[0].message.content
    print(f"LLM Response: {llm_response}")

    # Break after first query for testing purposes
    break
