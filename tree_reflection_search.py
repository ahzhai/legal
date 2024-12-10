import random
import csv
import math
import logging
from typing import List, Optional
from collections import deque
from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# Customized Output Parser for MultiQueryRetriever
class LineListOutputParser(BaseOutputParser[List[str]]):
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))

# Customized Prompt for MultiQueryRetriever
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""
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

        Original query: {question}
        Deliverable: Provide five alternative refined queries, each on a new line. Only output the refined query text without any additional context or explanations.
        Each refined query should retain critical legal details, incorporate specific statutory or case law references where applicable, and ensure semantic alignment to retrieve relevant passages effectively.
    """
)

# Set up the embedding model
print("Setting up the embedding model...")
embedding_model = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")

# Set up the Chroma vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(
    collection_name="clerc_passages",
    embedding_function=embedding_model,
    persist_directory="chroma_db_passages"
)
print("Chroma vector store setup complete.")

# Initialize LLM and MultiQueryRetriever
llm = ChatOpenAI(temperature=0.5)
output_parser = LineListOutputParser()
llm_chain = QUERY_PROMPT | llm | output_parser
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(),
    llm_chain=llm_chain,
    parser_key="lines"
)

# Tree Search Classes and Methods
class Reflection(BaseModel):
    reflections: str = Field(description="Critique and score of response.")
    score: int = Field(description="Score from 0-10 for response quality.", gte=0, lte=10)
    found_solution: bool = Field(description="Indicates if a solution is found.")

    def as_message(self):
        return HumanMessage(content=f"Reflection: {self.reflections}\nScore: {self.score}")

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0

class Node:
    def __init__(
        self,
        messages: List[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def upper_confidence_bound(self, exploration_weight=1.0):
        if self.visits == 0:
            return float('inf')
        avg_reward = self.value / self.visits
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return avg_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def _mark_tree_as_solved(self):
        node = self
        while node:
            node._is_solved = True
            node = node.parent

# Generate initial response (Root Node)
def generate_initial_response(query: str, correct_passage_ids: List[str]) -> Node:
    results = vectorstore.similarity_search(query, k=10)
    passages = [AIMessage(content=doc.page_content) for doc in results]

    # Debugging: Print retrieved passage IDs
    retrieved_passage_ids = [doc.metadata.get("passage_id") for doc in results]
    print(f"Initial Retrieved Passage IDs: {retrieved_passage_ids}")

    correct_matches = [pid for pid in retrieved_passage_ids if pid in correct_passage_ids]
    print(f"Number of Correctly Retrieved Passages: {len(correct_matches)}")

    reflection = Reflection(
        reflections="Initial retrieval performed.",
        score=5,
        found_solution=bool(correct_matches)
    )
    return Node(messages=[HumanMessage(content=query)] + passages, reflection=reflection)


# Expand and simulate (Child Nodes) with MultiQueryRetriever Integration
def expand_node_with_multiquery(node: Node, query: str) -> List[Node]:
    refined_queries = retriever.invoke(query)
    child_nodes = []

    # print(f"Refined Queries: {refined_queries}")  # Debugging: Print refined queries

    for refined_query in refined_queries:
        try:
            # Ensure refined_query is a string or extract relevant content from a Document
            if isinstance(refined_query, str):
                refined_query_text = refined_query
            elif hasattr(refined_query, "page_content"):  # Handle Document-like objects
                refined_query_text = refined_query.page_content
            else:
                raise TypeError(f"Unexpected query type: {type(refined_query)}")

            # Perform similarity search with the refined query text
            results = vectorstore.similarity_search(refined_query_text, k=10)
            print(f"Retrieved Passage IDs for Refined Query: {refined_query_text[:30]} - {[doc.metadata.get('passage_id') for doc in results]}")

            for result in results:
                messages = node.messages + [AIMessage(content=result.page_content)]
                reflection = Reflection(
                    reflections="Evaluated relevance for passage.",
                    score=7,
                    found_solution=False
                )
                child_nodes.append(Node(messages=messages, reflection=reflection, parent=node))

        except Exception as e:
            print(f"Error handling refined query: {refined_query}")
            print(f"Exception: {e}")

    return child_nodes


# Monte Carlo Tree Search for Passage Retrieval
def perform_lats_with_multiquery(query: str, correct_passage_ids: List[str], max_depth: int = 5) -> Node:
    root = generate_initial_response(query, correct_passage_ids)  # Pass correct_passage_ids
    current_node = root
    depth = 1

    while depth <= max_depth and not current_node._is_solved:
        if not current_node.children:
            current_node.children = expand_node_with_multiquery(current_node, query)

        if not current_node.children:
            print(f"No children generated at depth {depth} for query: {query}")
            break  # Exit the loop if no valid children are generated

        current_node = max(
            current_node.children,
            key=lambda child: child.upper_confidence_bound()
        )
        depth += 1

    return current_node



# Load dataset for queries
query_to_passage = {}
with open("processed_passages/queryIdToPassageIds.txt", "r") as f:
    pairs = [line.strip().split('\t') for line in f.readlines()]
    query_to_passage = {qid: pid for qid, pid in pairs}

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

queries = [(qid, getQuery(qid)) for qid in query_to_passage.keys()]

# Test Tree Search with MultiQuery Integration
precisions_tree = []
recalls_tree = []
num_samples = 5
sample_queries = random.sample(queries, num_samples)

for query_id, query_text in sample_queries:
    print(f"\nTesting Query ID: {query_id}")
    print(f"Original Query: {query_text}")

    correct_passage_ids = query_to_passage[query_id]
    tree_result = perform_lats_with_multiquery(query_text, correct_passage_ids)
    retrieved_passage_ids = [
        msg.metadata.get("passage_id")
        for msg in tree_result.messages
        if hasattr(msg, "metadata") and "passage_id" in msg.metadata
    ]
    correct_tree = [pid for pid in retrieved_passage_ids if pid in correct_passage_ids]

    print(f"Retrieved Passage IDs: {retrieved_passage_ids}")
    print(f"Correctly Retrieved Passage IDs: {correct_tree}")
    print(f"Number of Retrieved Passages: {len(retrieved_passage_ids)}")
    print(f"Number of Correctly Retrieved Passages: {len(correct_tree)}")

    # Calculate precision and recall
    precision = len(correct_tree) / len(retrieved_passage_ids) if retrieved_passage_ids else 0
    recall = len(correct_tree) / len(correct_passage_ids) if correct_passage_ids else 0

    precisions_tree.append(precision)
    recalls_tree.append(recall)

# Summary of performance
precision_tree_avg = sum(precisions_tree) / len(precisions_tree) if precisions_tree else 0
recall_tree_avg = sum(recalls_tree) / len(recalls_tree) if recalls_tree else 0

print(f"\nAverage Precision: {precision_tree_avg}")
print(f"Average Recall: {recall_tree_avg}")
