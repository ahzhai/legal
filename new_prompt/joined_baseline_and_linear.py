from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
import random
import csv

number_queries = 20
k_value = 10
temp_value = 0.5

# Step 1: Set up the embedding model
print("Setting up the embedding model...")
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")

# Step 2: Load the vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(collection_name="clerc_passages", embedding_function=sentence_transformer_ef, persist_directory="../chroma_db_passages")
print("Chroma vector store setup complete.")

# Get 30 random queries from the query dataset with their corresponding doc ids
query_to_passage = {}
with open("../processed_passages/queryIdToPassageIds.txt", "r") as f:
    pairs = [line.strip().split('\t') for line in f.readlines()]
    query_to_passage = {qid: pid for qid, pid in pairs}

random_pairs = random.sample(list(query_to_passage.items()), number_queries)
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
    for sample in queryDataset:
        keys = list(sample.keys())
        queryId = str(sample[keys[0]])
        if queryId == qId:
            header = keys[1]
            return sample[header]
    return None

# Get the queries from the query dataset
queries = [(qid, getQuery(qid)) for qid in random_query_to_passage.keys()]

# OpenAI client
client = OpenAI()

number_correct_first = 0
number_correct_second = 0
precisions_first = []
recalls_first = []
precisions_second = []
recalls_second = []
number_improved = 0
number_worsened = 0
number_same = 0
# Use Chroma similarity search to retrieve similar documents based on a query
for query in queries:
    print("Query ID: ", query[0])
    #print("Original Query: ", query[1])
    correct_passage_ids = random_query_to_passage[query[0]]
    #print("Correct passages: ", correct_passage_ids)
    # print("Query: ", query[1])
    results = vectorstore.similarity_search(
        query[1],
        k=2*k_value 
    )
    # print("Results: ", results)
    passage_ids = [result.metadata["passage_id"] for result in results]
    #print("Found passages: ", passage_ids)

    correctly_retrieved = [passage_id for passage_id in passage_ids if passage_id in correct_passage_ids]
    #print("Correctly retrieved: ", correctly_retrieved)
    number_correct = len(correctly_retrieved)
    number_correct_first += number_correct
    print("Number correct: ", number_correct)

    if number_correct > 0:
        precisions_first.append(number_correct / len(passage_ids))
        recalls_first.append(number_correct / len(correct_passage_ids))

    # Generate a refined query using the LLM
    first_prompt = [
        {"role": "system", "content": "You are an expert legal assistant specializing in refining search queries to retrieve precise legal case passages. Your objective is to transform broad or ambiguous queries into focused, legally accurate search prompts optimized for semantic embedding-based retrieval systems."},
        {"role": "user", "content": f"""
        You are given a legal query where citations have been removed and replaced with “REDACTED.” Your task is to refine this query to focus on the most relevant legal principles, statutory references, or case law necessary for accurate retrieval.
        
        Key Refinement Strategies:
        1. Legal Precision: Highlight critical legal doctrines, statutory references, or case-specific terminology. Avoid unnecessary or generic legal jargon.
        2. Contextual Relevance: Focus on the core legal issue or argument presented in the query. Exclude extraneous details, procedural background, or verbose phrasing that detracts from retrieval accuracy.
        3. Semantic Clarity: Maintain specific legal identifiers (e.g., “18 U.S.C.A. § 3617,” “Rule 59 motions,” or “restrictive covenant”) and any implied but critical legal terms to enhance contextual alignment.
        4. Retrieval Optimization: Ensure the refined query bridges any gaps between legal context and the semantic understanding of the embedding model by clearly articulating implied relationships or concepts.
      
        Task Instructions:
         Refine the following legal query for retrieval accuracy:

        Original Query:
        "{query[1]}"

        Deliverable:
        Generate a refined query that retains critical legal details, incorporates specific statutory or case law references where applicable, and ensures semantic alignment to retrieve relevant passages effectively. Only output the refined query text without any additional context or explanations.
        """},
    ]
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=first_prompt,
        temperature=temp_value
    )
    refined_query = completion.choices[0].message.content.strip()
    # print(f"Refined Query: {refined_query}")

    # # Append the refined query to the original query
    # combined_query = f"{refined_query} {query_text}"
    # print(f"Combined Query: {combined_query}")

    # Perform a second similarity search with the combined query
    second_results = vectorstore.similarity_search(refined_query, k=k_value)
    second_retrieved_passages = {
        result.metadata["passage_id"]: result.page_content for result in second_results
    }
    second_passage_ids = [result.metadata["passage_id"] for result in second_results]
    #print("Second found passages: ", second_passage_ids)
    added = 0
    for elem in passage_ids:
        if added < 10:
            second_passage_ids.append(elem)
            added += 1
        else:
            break

    second_correctly_retrieved = [passage_id for passage_id in second_passage_ids if passage_id in correct_passage_ids]
    #print("Second correctly retrieved: ", second_correctly_retrieved)
    second_number_correct = len(second_correctly_retrieved)
    print("Second number correct: ", second_number_correct, "\n")
    number_correct_second += second_number_correct

    if second_number_correct > 0:
        precisions_second.append(second_number_correct / len(second_passage_ids))
        recalls_second.append(second_number_correct / len(correct_passage_ids))

    if number_correct < second_number_correct:
        number_improved += 1
    elif number_correct > second_number_correct:
        number_worsened += 1
    else:
        number_same += 1

    # # Break after processing one query for testing purposes
    # break

precisions_first = sum(precisions_first) / len(precisions_first)
recalls_first = sum(recalls_first) / len(recalls_first)
precisions_second = sum(precisions_second) / len(precisions_second)
recalls_second = sum(recalls_second) / len(recalls_second)


print("Number correct first: ", number_correct_first)
print("Number correct second: ", number_correct_second)
print("Precision first: ", precisions_first)
print("Recall first: ", recalls_first)
print("Precision second: ", precisions_second)
print("Recall second: ", recalls_second)
print("Number improved: ", number_improved)
print("Number worsened: ", number_worsened)
print("Number same: ", number_same) 

# write to csv file
with open("joined_baseline_and_linear_prompt_2_results.csv", "a") as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Only write header if file is empty
        writer.writerow(["Number queries", "k value", "Temperature value", "Number correct first", "Number correct second", "Precision first", "Recall first", "Precision second", "Recall second", "Number improved", "Number worsened", "Number same"])
    writer.writerow([2*number_queries, k_value, temp_value, number_correct_first, number_correct_second, precisions_first, recalls_first, precisions_second, recalls_second, number_improved, number_worsened, number_same])
