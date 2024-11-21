from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
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

# Use Chroma similarity search and attempt a second retrieval with left half of query if needed
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

    print("Correct document not found in the baseline retrieval. Proceeding with left-half retrieval.")

    # Extract the right half of the query (everything before "REDACTED")
    if "REDACTED" in query_text:
        right_half_query = query_text.split("REDACTED")[1].strip()
        print(f"Right Half Query: {right_half_query}")
    else:
        right_half_query = query_text  # Fallback to the full query if "REDACTED" is not found

    # Perform a second similarity search using the left half of the query
    second_results = vectorstore.similarity_search(right_half_query, k=10)
    second_retrieved_docs = {
        result.metadata["doc_id"]: result.page_content for result in second_results
    }
    second_doc_ids = list(second_retrieved_docs.keys())

    print(f"Second Retrieved Document IDs: {second_doc_ids}")
    if correct_doc_id in second_doc_ids:
        # for result in second_results:
        #     if result.metadata["doc_id"] == correct_doc_id:
        #         print(result.page_content)
        #         break
        print("Correct document found in the second retrieval step with the left half query.")
    else:
        print("Correct document still not found after refinement.")

    # Break after processing one query for testing purposes
    # break
