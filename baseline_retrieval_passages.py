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
vectorstore = Chroma(collection_name="clerc_passages", embedding_function=sentence_transformer_ef, persist_directory="chroma_db_passages")
print("Chroma vector store setup complete.")



# Get 30 random queries from the query dataset with their corresponding doc ids
query_to_passage = {}
with open("processed_passages/queryIdToPassageIds.txt", "r") as f:
    pairs = [line.strip().split('\t') for line in f.readlines()]
    query_to_passage = {qid: pid for qid, pid in pairs}

random_pairs = random.sample(list(query_to_passage.items()), 50)
random_query_to_passage = dict(random_pairs)
#print("50 random query-passage pairs: ", random_query_to_passage, "\n")

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
queries = [(qid, getQuery(qid)) for qid in random_query_to_passage.keys()]

average_correct = 0
# Use Chroma similarity search to retrieve similar documents based on a query
for query in queries:
    print("Query ID: ", query[0])
    correct_passage_ids = random_query_to_passage[query[0]]
    print("Correct passages: ", correct_passage_ids)
    # print("Query: ", query[1])
    results = vectorstore.similarity_search(
        query[1],
        k=20 # number of passages to retrieve (20 works better than 10)
    )
    # print("Results: ", results)
    passage_ids = [result.metadata["passage_id"] for result in results]
    print("Found passages: ", passage_ids)

    correctly_retrieved = [passage_id for passage_id in passage_ids if passage_id in correct_passage_ids]
    print("Correctly retrieved: ", correctly_retrieved)
    number_correct = len(correctly_retrieved)
    print("Number correct: ", number_correct, "\n")
    average_correct += number_correct
    # print("passage: ", results[0].page_content)

average_correct /= len(queries)
print("Average number correct: ", average_correct)
