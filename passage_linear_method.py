from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
import random
import csv

number_queries = 50
k_value = 20
temp_value = 0.25

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
        k=k_value # number of passages to retrieve (20 works better than 10)
    )
    # print("Results: ", results)
    passage_ids = [result.metadata["passage_id"] for result in results]
    #print("Found passages: ", passage_ids)

    correctly_retrieved = [passage_id for passage_id in passage_ids if passage_id in correct_passage_ids]
    #print("Correctly retrieved: ", correctly_retrieved)
    number_correct = len(correctly_retrieved)
    number_correct_first += number_correct
    print("Number correct: ", number_correct)

    precisions_first.append(number_correct / len(passage_ids))
    recalls_first.append(number_correct / len(correct_passage_ids))

    # Generate a refined query using the LLM
    first_prompt = [
        {"role": "system", "content": "You are an assistant to a lawyer writing legal analyses who needs to find relevant case passages to support their texts."},
        {"role": "user", "content": f"""
        You are given a query taken from a legal case document, with its citation in the middle removed and replaced with "REDACTED."
        Task: Refine the given query to focus only on the most critical and relevant aspects necessary for retrieving the correct passage. Improve retrieval accuracy by refining the query to align semantically with the intended legal context.
        
        Key Refinement Strategies:
        1. Precision in Legal Language: Emphasize critical statutory references, case law citations, and legal principles directly relevant to the query. Avoid generalizations or unrelated legal jargon.
        2. Contextual Focus: Narrow the scope to the core legal argument or issue under discussion. Exclude superfluous details, background context, or lengthy phrasing that might dilute the query's focus.
        3. Semantic Clarity: Structure the query to retain specific legal terms, doctrines, or identifiers that make the passage unique (e.g., "18 U.S.C.A. § 3617," "adequate consideration," or "restrictive covenant").
        4. Pragmatic Relevance: Ensure the refined query bridges the gap between the legal reasoning in the original query and the semantic embedding model's understanding by incorporating key terms or concepts that may have been implied but not explicit.
        
        Example of an Incorrectly Retrieved Passage:
        Query: 'it deserves, and finding no harm after reviewing the record as a whole, we must affirm. D. The Post-Judgment Motions Following a twelve-day trial, and roughly an hour of deliberation, the jury returned a verdict for the defendants. Plaintiffs subsequently moved for relief from the adverse judgment and for a new trial under Fed.R.Civ.P. 59 and 60(b)(3) and (6). The district court denied both motions. The Fernandezes contend that their motions should have been granted on the basis of the alleged errors addressed by us in Parts II A-C of this opinion, and because the jury verdict in favor of the municipal defendant, the Town of Brookline, was against the weight of the evidence. The decision to grant or deny a motion under Rule 59 or 60 is committed to the wide discretion of the district court and must be respected absent abuse.  REDACTED MacQuarrie v. Howard Johnson Co., 877 F.2d 126, 131 (1st Cir.1989). To the extent that plaintiffs’ motions were predicated upon allegations of error in the various evidentiary and misconduct-related rulings at trial, our discussion at Parts II A-C of this opinion compels the conclusion that they were properly denied. We likewise are satisfied that the district court properly rejected plaintiffs’ attack on the verdict for the Town of Brookline. As this court repeatedly has stated, we will reverse a court’s decision not to grant a new trial “ ‘only if the verdict is so seriously mistaken, so clearly against the law or the evidence, as to constitute a miscarriage of justice.’ ” MacQuarrie, 877 F.2d at 131 (quoting Levesque v. Anchor Motor Freight, Inc., 832 F.2d 702, 703 (1st Cir.1987)). Plaintiffs sought to establish § 1983 liability against Brookline on the theory that Mr. Fernandez’s'
        Incorrect Passage: 'TORRUELLA, Circuit Judge. This case arises out of the enforcement of the credit advertising provision of the Truth In Lending Act (“TILA”), 15 U.S.C. § 1601 et seq., and Regulation Z, 12 C.F.R. Part 226. The action was filed by the Federal Trade Commission (“FTC”) against appellants Boch Oldsmobile, Inc., and Boch Toyota, Inc. (collectively referred to as “Boch”) under § 5(m)(l)(B) of the Federal Trade Commission Act (“FTCA”). 15 U.S.C. § 45(m)(l)(B). Thereafter a consent decree was entered into, which appellants then sought to amend. This is an appeal from the denial by the district court of said request. We affirm this decision. FACTS Boch sells and services new and used motor vehicles and advertises accordingly. In 1983, FTC informed Boch of certain published advertisement violations of the TILA and Regulation Z credit advertising provisions. In the notices sent to Boch the FTC pointed to Boch's failure to include certain credit terms in their credit sale advertisements. For example, terms for down payments and monthly payments were published without specifying the finance charge rate and the “annual percentage rate” or “A.P.R.” When similar violations appeared in 1984 the FTC again sent Boch notices advising of these and prior advertisement violations. It also included a “synopsis package” that contained, among other items, four published decisions of the FTC involving proceedings brought under § 5(m)(l)(B) of the Federal Trade Commission Act. 15 U.S.C. § 45(m)(l)(B). When Boch failed to bring its advertising into compliance with the FTC’s contentions, a civil suit was filed seeking both civil penalties and injunctive relief under section 5 of the FTCA. 15 U.S.C. § 45. The FTC based its action on' the theory that receipt of the synopsis package gave Boch actual knowledge that the conduct which the FTC complained of had previously been determined to be unfair or deceptive acts or practices under the FTCA. Following a lengthy period of negotiations, Boch agreed to pay civil penalties in the amount of $35,000, to submit to certain injunctive relief, and to execute consent decrees. Accordingly, and without submitting any issues to trial, the district court entered'

        Your Task:
        Refine the query below to enhance its specificity, focus, and semantic alignment with relevant legal passages. 

        Original Query:
        {query[1]}

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
    second_results = vectorstore.similarity_search(refined_query, k=20)
    second_retrieved_passages = {
        result.metadata["passage_id"]: result.page_content for result in second_results
    }
    second_passage_ids = [result.metadata["passage_id"] for result in second_results]
    #print("Second found passages: ", second_passage_ids)

    second_correctly_retrieved = [passage_id for passage_id in second_passage_ids if passage_id in correct_passage_ids]
    #print("Second correctly retrieved: ", second_correctly_retrieved)
    second_number_correct = len(second_correctly_retrieved)
    print("Second number correct: ", second_number_correct, "\n")
    number_correct_second += second_number_correct

    precisions_second.append(second_number_correct / len(second_passage_ids))
    recalls_second.append(second_number_correct / len(correct_passage_ids))

    if number_correct > second_number_correct:
        number_improved += 1
    elif number_correct < second_number_correct:
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
with open("passage_linear_method_results.csv", "a") as f:
    writer = csv.writer(f)
    if f.tell() == 0:  # Only write header if file is empty
        writer.writerow(["Number queries", "k value", "Temperature value", "Number correct first", "Number correct second", "Precision first", "Recall first", "Precision second", "Recall second", "Number improved", "Number worsened", "Number same"])
    writer.writerow([number_queries, k_value, temp_value, number_correct_first, number_correct_second, precisions_first, recalls_first, precisions_second, recalls_second, number_improved, number_worsened, number_same])