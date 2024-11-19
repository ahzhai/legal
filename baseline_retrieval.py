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
vectorstore = Chroma(collection_name="clerc_docs", embedding_function=sentence_transformer_ef, persist_directory="chroma_db")
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

# Use Chroma similarity search to retrieve similar documents based on a query
for query in queries:
    print("Query ID: ", query[0])
    correct_doc_id = random_query_to_doc[query[0]]
    print("Correct document: ", correct_doc_id, "\n")
    print("Query: ", query[1])
    results = vectorstore.similarity_search(
        query[1],
        k=2
    )
    print("Results: ", results)
    doc_ids = [result.metadata["doc_id"] for result in results]
    print("Found documents: ", doc_ids)
    print("document: ", results[0].page_content)

    break
    

