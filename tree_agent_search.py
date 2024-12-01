import math
from collections import deque
from typing import Optional, List
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field


# Reflection model
class Reflection(BaseModel):
    reflections: str = Field(description="Critique and score of response.")
    score: int = Field(description="Score from 0-10 for response quality.", gte=0, lte=10)
    found_solution: bool = Field(description="Indicates if a solution is found.")

    def as_message(self):
        return HumanMessage(content=f"Reflection: {self.reflections}\nScore: {self.score}")

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


# Node structure for the search tree
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


# Load vector store
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = Chroma(
    collection_name="clerc_passages",
    embedding_function=sentence_transformer_ef,
    persist_directory="chroma_db_passages"
)


# Generate initial response (Root Node)
def generate_initial_response(query: str) -> Node:
    results = vectorstore.similarity_search(query, k=5)
    passages = [AIMessage(content=doc.page_content) for doc in results]
    reflection = Reflection(
        reflections="Initial retrieval performed.",
        score=5,
        found_solution=False
    )
    return Node(messages=[HumanMessage(content=query)] + passages, reflection=reflection)


# Expand and simulate (Child Nodes)
def expand_node(node: Node, query: str) -> List[Node]:
    results = vectorstore.similarity_search(query, k=5)
    child_nodes = []
    for result in results:
        messages = node.messages + [AIMessage(content=result.page_content)]
        reflection = Reflection(
            reflections=f"Evaluated relevance for passage: {result.page_content[:50]}...",
            score=7,
            found_solution=False
        )
        child_nodes.append(Node(messages=messages, reflection=reflection, parent=node))
    return child_nodes


# Monte Carlo Tree Search for Passage Retrieval
def perform_lats(query: str, max_depth: int = 5):
    root = generate_initial_response(query)
    current_node = root
    depth = 1

    while depth <= max_depth and not current_node.is_solved:
        if not current_node.children:
            current_node.children = expand_node(current_node, query)
        # Select the best child based on Upper Confidence Bound
        current_node = max(
            current_node.children,
            key=lambda child: child.upper_confidence_bound()
        )
        depth += 1

    return current_node


# Example Usage
query = "Explain the legal principle behind 28 U.S.C. § 1292(a)(3)."
final_node = perform_lats(query)

# Display Results
print("Final Retrieved Passages:")
for message in final_node.messages:
    print(message.content)
