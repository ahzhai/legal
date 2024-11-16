from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
https://github.com/ahzhai/legal/tree/main

# Step 1: Set up the embedding model
print("Setting up the embedding model...")
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")


# Step 2: Load the vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(collection_name="clerc_docs", embedding_function=sentence_transformer_ef, persist_directory="chroma_db")
print("Chroma vector store setup complete.")


#TODO: Load in query dataset
# Step 3: Use Chroma similarity search to retrieve similar documents based on a query
results = vectorstore.similarity_search(
    "Since Greene, this court has rejected similar charges of misconduct where the government supplied counterfeit credit cards to detect which merchants would accept them. See Citro, 842 F.2d at 1153. In a case where an FBI agent bribed a state senator, we found no misconduct. See United States v. Carpenter, 961 F.2d 824, 829 (9th Cir.1992). Most recently, we declined to dismiss an indictment where the government established fake bank accounts and wired money to Mexican banks suspected of money laundering. See United States v. Gurolla, 333 F.3d 944, 948-49 (9th Cir.2003).", # Chroma will embed this for you
    k=2 # how many results to return
)
print(results)
