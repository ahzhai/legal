from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from openai import OpenAI
import random
import csv

number_queries = 20
k_value = 5  # Top passages retrieved per query refinement
temp_value = 0.5

# Step 1: Set up the embedding model
print("Setting up the embedding model...")
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")

# Step 2: Load the vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(collection_name="clerc_passages", embedding_function=sentence_transformer_ef, persist_directory="chroma_db_passages")
print("Chroma vector store setup complete.")

# Step 3: Load query dataset and get random queries
query_to_passage = {}
with open("processed_passages/queryIdToPassageIds.txt", "r") as f:
    pairs = [line.strip().split('\t') for line in f.readlines()]
    query_to_passage = {qid: pid for qid, pid in pairs}

random_pairs = random.sample(list(query_to_passage.items()), number_queries)
random_query_to_passage = dict(random_pairs)

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

queries = [(qid, getQuery(qid)) for qid in random_query_to_passage.keys()]

# OpenAI client
client = OpenAI()

# Initialize metrics
number_correct_baseline = 0
precisions_baseline = []
recalls_baseline = []

# Step 4: Define LLM reflection prompt
reflection_prompt_template = """
You are an assistant to a lawyer writing legal analyses. 
Below is a list of passages retrieved for a legal query. 
For each passage, reflect and assign a score from 1 to 10 based on how likely it is the correct cited passage in the query. 
A score of 10 means it is highly likely, and a score of 1 means it is highly unlikely.

Query: {query}

Passages:
{passages}

Output:
For each passage, return its ID and a score.
"""

# Step 5: Retrieval and Evaluation
for query_id, query_text in queries:
    print(f"\nQuery ID: {query_id}")
    print(f"Query Text: {query_text}")

    correct_passage_ids = random_query_to_passage[query_id]

    # Baseline retrieval
    baseline_results = vectorstore.similarity_search(query_text, k=3 * k_value)
    baseline_passages = [{"id": res.metadata["passage_id"], "content": res.page_content} for res in baseline_results]

    # Refined retrieval with 3 LLM calls
    refined_queries = []
    for i in range(3):  # Generate 3 refined queries
        refined_prompt = [
            {"role": "system", "content": "You are an assistant to a lawyer writing legal analyses."},
            {"role": "user", "content": f"""
        Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines. More detailed instructions are as follows:

        You are an AI language model assistant to a lawyer writing legal analyses who needs to find relevant case passages to support their texts.
        
        Task: You are given a query taken from a legal case document, with its citation in the middle removed and replaced with "REDACTED."
        Generate five refined versions of the query to refine the given query to focus only on the most critical and relevant aspects necessary for retrieving the correct passage.
        Improve retrieval accuracy by refining the query to align semantically with the intended legal context.

        1. Precision in Legal Language: Emphasize critical statutory references, case law citations, and legal principles directly relevant to the query. Avoid generalizations or unrelated legal jargon.
        2. Contextual Focus: Narrow the scope to the core legal argument or issue under discussion. Exclude superfluous details, background context, or lengthy phrasing that might dilute the query's focus.
        3. Semantic Clarity: Structure the query to retain specific legal terms, doctrines, or identifiers that make the passage unique (e.g., "18 U.S.C.A. § 3617," "adequate consideration," or "restrictive covenant").
        4. Pragmatic Relevance: Ensure the refined query bridges the gap between the legal reasoning in the original query and the semantic embedding model's understanding by incorporating key terms or concepts that may have been implied but not explicit.
        
        Example of a correctly versus incorrectly retrieved passage:
        Query: ‘appellate courts, however, have suggested that this two-year rule of thumb might, in fact, be a bright-line rule in which at-will employment can only serve as adequate consideration if an employee has worked with the employer for at least two years. See Fifield v. Premier Dealer Servs., Inc., 373 Ill.Dec. 379, 993 N.E.2d 938, 943 (2013); Prairie Rheumatology Assoc., 2014 IL App. (3d) 140338 at *4, 388 Ill.Dec. 150, 24 N.E.3d 58. The Illinois Supreme Court has not addressed this question. In the absence of a ruling from the Illinois Supreme Court, this Court “must make a predictive judgment as to how the supreme court of the state would decide the matter if it were presented presently to that tribunal.” Allstate Ins. Co. v. Menards, Inc., 285 F.3d 630, 635 (7th Cir.2002). Since Fifield, three federal courts decisions have considered this very question. In  REDACTED  Judge Castillo, sitting in the Northern District, rejected the bright-line rule from Fifield, and held that fifteen months of employment constituted adequate consideration for a restrictive covenant. Id. at 715-18. Just last week, in Bankers Life and Casualty Co. v. Miller, 14-cv-3165, 2015 WL 515965 (N.D.Ill. Feb. 6, 2015), Judge Shah, also of the Northern District, predicted that “[t]he Illinois Supreme Court would ... reject a rigid approach to determining whether a restrictive covenant was supported by adequate consideration” and “not adopt a bright-line rule requiring continued employment for at least two years in all cases.” Id. at *4. However, in Instant Technology, LLC v. DeFazio, 12 C 491, 40 F.Supp.3d 989, 2014 WL 1759184 (N.D.Ill. May 2, 2014), Judge Holderman, also of the Northern District, predicted that the Illinois Supreme Court would apply the bright-line rule announced in Fifield. Id. at *14. As discussed below, the'
        Correct Passage: ‘(“Illinois courts have repeatedly held that there must be at least two years or more of continued employment to constitute adequate consideration in support of a restrictive covenant.”); Brown & Brown, Inc. v. Mudron, 379 Ill.App.3d 724, 320 Ill.Dec. 293, 887 N.E.2d 437, 440 (3d Dist.2008) (“Illinois courts have generally held that two years or more of continued employment constitutes adequate consideration”). In other cases, employment for a year was considered a “substantial period” of employment. See Mid-Tom Petroleum, Inc. v. Gowen, 243 Ill.App.3d 63, 183 Ill.Dec. 573, 611 N.E.2d 1221, 1226 (1st Dist.1993) (while noting that the issue of consideration was not directly addressed, citing approvingly to two cases involving a post-covenant term of employment of approximately a year). In addition, courts have suggested that “factors other than the time period of the continued employment, such as whether the employee or the employer terminated employment, may need to be considered to properly review the issue of consideration.” Woodfield Grp., Inc. v. DeLisle, 295 Ill.App.3d 935, 230 Ill.Dec. 335, 693 N.E.2d 464, 469 (1st Dist.1998). For example, in McRand, Inc. v. van Beelen, 138 Ill.App.3d 1045, 93 Ill.Dec. 471, 486 N.E.2d 1306 (1st Dist.1985), the Illinois Appellate Court did not constrain itself by applying a bright-line test in considering whether a defendant was employed for a “substantial period” after signing a restrictive covenant. In its analysis to determine whether sufficient consideration was provided to enforce a restrictive covenant, it considered the raises and bonuses received by the defendants, their voluntary resignation, and the increased responsibilities they received after signing a restrictive covenant. McRand, 93 Ill.Dec. 471, 486 N.E.2d at 1314. Given the contradictory holdings of the lower Illinois courts and the lack of a clear direction from the Illinois Supreme Court, this Court does not find it appropriate to apply a bright line rule. While Defendants suggests that Illinois law establishes that at least two years of employment is required to satisfy the “substantial period” of employment requirement, Illinois courts have unequivocally stated their refusal to “limit[ ] the courts’ review to a numerical formula for determining what constitutes substantial’
        Incorrect Passage: ‘Ill.2d 482, 106 Ill.Dec. 8, 505 N.E.2d 314 (1987), the Illinois Supreme Court held that “an employee handbook or other policy statement creates enforceable contractual rights if the traditional requirements for contract formation are present.” Id. 106 Ill.Dec. at 12, 505 N.E.2d at 318. Specifically, the court stated, the policy statement must fulfill the following requirements: First, the language of the policy statement must contain a promise clear enough that an employee would reasonably believe that an offer has been made. Second, the statement must be disseminated to the employee in such a manner that the employee is aware of its contents and reasonably believes it to be an offer. Third, the employee must accept the offer by commencing or continuing to work after learning of the policy statement. Id. The district court found that even if Policy 4223 met the first of these requirements, it clearly did not satisfy the latter two, because Hohmeier did not receive a copy of the policy until her termination meeting. Thus, she “could not have reasonably believed that the policy manual contained an offer of employment, and she could not have based her employment on the language in” that provision. 748 F.Supp. at 660. We agree. An employee who has just been fired could not reasonably believe that a document given to her at the time of termination in conjunction with a seven-page memorandum detailing the reasons for the termination constituted an offer of employment. Hohmeier argues on appeal that her case fulfills the second requirement of Duldu-lao because she “was eager to continue working after having learned of Rule 4223’s guaranty of continued employment.” Appellant’s Br. at 23. This argument is without merit. Even if Hohmeier honestly believed when she was provided with Policy 4223 at her termination meeting that the policy constituted an offer of employment, that belief would not have been “reasonable,” as required by Duldulao. Hohmeier had no contractual right to continued employment under Duldulao. III. Hohmeier also challenges the district court’s grant of summary judgment in favor of defendants on her substantive due process claim based on her’

        Your Task:
        Refine the query below to enhance its specificity, focus, and semantic alignment with relevant legal passages. 

        Original query: {query_text}
        Deliverable: Provide five alternative refined queries, each on a new line. Only output the refined query text without any additional context or explanations.
        Each refined query should retain critical legal details, incorporate specific statutory or case law references where applicable, and ensure semantic alignment to retrieve relevant passages effectively.
        """}
        ]
        refined_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=refined_prompt,
            temperature=temp_value
        )
        refined_query = refined_completion.choices[0].message.content.strip()
        refined_queries.append(refined_query)

    refined_results = []
    for refined_query in refined_queries:  # Retrieve passages for each refined query
        results = vectorstore.similarity_search(refined_query, k=k_value)
        refined_results.extend(
            [{"id": res.metadata["passage_id"], "content": res.page_content} for res in results]
        )

    # Combine and reflect on passages
    all_passages = baseline_passages + refined_results
    passage_text = "\n\n".join([f"ID: {p['id']}\nContent: {p['content'][:200]}..." for p in all_passages])
    reflection_prompt = reflection_prompt_template.format(query=query_text, passages=passage_text)

    reflection_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": reflection_prompt}],
        temperature=temp_value
    )
    reflection_scores = reflection_response.choices[0].message.content.strip().split("\n")

    # Aggregate top 20 passages by score
    scored_passages = []
    for score_line in reflection_scores:
        try:
            passage_id, score = score_line.split(":")
            scored_passages.append((passage_id.strip(), int(score.strip())))
        except ValueError:
            continue

    top_scored_passages = sorted(scored_passages, key=lambda x: x[1], reverse=True)[:20]

    # Evaluate performance
    top_passage_ids = [str(p[1]) for p in top_scored_passages]
    print("Correct Passage IDs: ", correct_passage_ids)
    print("Top Passage IDs: ", top_passage_ids)
    correctly_retrieved = [pid for pid in top_passage_ids if pid in correct_passage_ids]

    # precision = len(correctly_retrieved) / len(top_passage_ids)
    # recall = len(correctly_retrieved) / len(correct_passage_ids)

    # precisions_baseline.append(precision)
    # recalls_baseline.append(recall)

    # print(f"Correctly Retrieved: {correctly_retrieved}")
    print("Number Correctly Retrieved: ", len(correctly_retrieved))
    # print(f"Precision: {precision}, Recall: {recall}")

# Step 6: Summary
avg_precision = sum(precisions_baseline) / len(precisions_baseline)
avg_recall = sum(recalls_baseline) / len(recalls_baseline)

print(f"\nAverage Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")
