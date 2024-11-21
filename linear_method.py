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

# # Get 10 random queries from the query dataset with their corresponding doc ids
# query_to_doc = {}
# with open("processed_CLERC/queryIdToDocId.txt", "r") as f:
#     pairs = [line.strip().split('\t') for line in f.readlines()]
#     query_to_doc = {qid: did for qid, did in pairs}

# random_pairs = random.sample(list(query_to_doc.items()), 10)
# random_query_to_doc = dict(random_pairs)
random_query_to_doc = {'1732': '640953', '962': '6106988', '96': '6187104', '1328': '5403601', '4455': '719610', '3257': '180535', '3786': '1767540', '3298': '5411644', '3133': '9392759', '4833': '3990383'}
print("10 random query-doc pairs: ", random_query_to_doc)

# Load query dataset
queryDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "queries/test.single-removed.indirect.tsv"},
    streaming=False,
    delimiter="\t"
)["data"]

def getQuery(qId):
    for sample in queryDataset:
        keys = list(sample.keys())
        queryId = str(sample[keys[0]])
        if queryId == qId:
            header = keys[1]
            return sample[header]
    return None

# Get the queries
queries = [(qid, getQuery(qid)) for qid in random_query_to_doc.keys()]

# OpenAI client
client = OpenAI()

# Use Chroma similarity search and incorporate LLM refinement if needed
for query in queries:
    query_id = query[0]
    query_text = query[1]
    print(f"\nProcessing Query ID: {query_id}")
    print(f"Query: {query_text}")

    correct_doc_id = random_query_to_doc[query_id]

    # Perform the initial similarity search
    results = vectorstore.similarity_search(query_text, k=10)
    retrieved_docs = {
        result.metadata["doc_id"]: result.page_content for result in results
    }
    doc_ids = list(retrieved_docs.keys())

    print(f"Correct Document ID: {correct_doc_id}")
    print(f"Retrieved Document IDs: {doc_ids}")

    if correct_doc_id in doc_ids:
        print("Correct document found in the baseline retrieval step.")
        continue  # Skip to the next query if the correct document is found

    print("Correct document not found in the baseline retrieval. Proceeding with LLM refinement.")

    # Generate a refined query using the LLM
    first_prompt = [
        {"role": "system", "content": "You are an assistant to a lawyer writing legal analyses who needs to find case documents to support their texts."},
        {"role": "user", "content": f"""
        You are given a query that is taken from a legal case document with its citation in the middle removed and replaced with "REDACTED".
        In the first attempt, the retrieved documents from an embedding-based semantic similarity search did not match the correct document or provide sufficient relevance to the query.

        Your job is to generate a new question that:
        1. Builds on the original query while retaining key unique entities or critical legal terminology to preserve the specificity required for semantic similarity search.
        2. Explores alternative angles, related principles, or broader contexts to increase the likelihood of retrieving the correct document.
        3. Avoids restating or slightly rephrasing the original query in ways that are likely to retrieve the same irrelevant results.
        4. Prioritizes the creation of a concise, standalone question that is semantically distinct from the original query and can directly improve retrieval results when used in an embedding-based search.

        **Note**: The previously retrieved documents were not accurate, so this new question will be fed into a second round of semantic similarity search to attempt to find the correct document.

        Query: {query_text}

        Previously Retrieved Documents (incorrect results):
        {retrieved_docs}

        Generate a new question that balances retaining the specificity of unique entities while exploring a fresh perspective, ensuring it can improve retrieval results in the semantic similarity search.
        """},
    ]
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=first_prompt
    )
    refined_query = completion.choices[0].message.content.strip()
    print(f"Refined Query: {refined_query}")

    # # Append the refined query to the original query
    # combined_query = f"{refined_query} {query_text}"
    # print(f"Combined Query: {combined_query}")

    # Perform a second similarity search with the combined query
    second_results = vectorstore.similarity_search(refined_query, k=10)
    second_retrieved_docs = {
        result.metadata["doc_id"]: result.page_content for result in second_results
    }
    second_doc_ids = list(second_retrieved_docs.keys())

    print(f"Second Retrieved Document IDs: {second_doc_ids}")
    if correct_doc_id in second_doc_ids:
        print("Correct document found in the second retrieval step with the combined query.")
    else:
        print("Correct document still not found after refinement.")

    # Break after processing one query for testing purposes
    # break
