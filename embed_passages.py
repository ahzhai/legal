from datasets import load_dataset
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from tqdm.auto import tqdm


# Step 1: Load document IDs from docIds.txt into a set
passage_id_file = "processed_passages/passageIds.txt"
with open(passage_id_file, 'r') as file:
    valid_passage_ids = set(line.strip() for line in file)


# Step 2: Set up the embedding model
print("Setting up the embedding model...")
sentence_transformer_ef = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en-v1.5")
print("Embedding model setup complete.")


# Step 3: Set up the vector store
print("Setting up the Chroma vector store...")
vectorstore = Chroma(collection_name="clerc_passages", embedding_function=sentence_transformer_ef, persist_directory="chroma_db_passages")
print("Chroma vector store setup complete.")


# Step 4: Load the dataset in streaming mode
print("Loading dataset in streaming mode...")
dataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "collection/collection.passage.tsv.gz"},
    streaming=True,
    delimiter="\t"
)["data"]
print("Dataset loaded successfully.")


# Step 5: Begin processing the dataset
print("Beginning to process dataset in chunks...")
# Set batch parameters
batch_size = 100  # Adjust based on your memory
batch_texts = []
batch_metadatas = []
total_processed = 0  # Counter for progress monitoring

progress_bar = tqdm(desc="Processing Chunks", unit="batch")

for idx, sample in enumerate(dataset):
    # Extract the first key and first value
    first_key = list(sample.keys())[0]  # Gets the first key (e.g., '0')
    first_value = str(sample[first_key])  # Convert to string for comparison

    # Only process the sample if first_value is in valid_doc_ids
    if first_value in valid_passage_ids:
        # Extract the second key and second value
        second_key = list(sample.keys())[1]   # Gets the second key (e.g., 'OPINION...')
        second_value = sample[second_key]     # Gets the text for embedding

        # Print current datapoint being processed
        print(f"Processing datapoint {idx + 1}: passage_id={first_value}")

        # Add to batch
        batch_texts.append(second_value) # Concatenate heading and text to embed
        batch_metadatas.append({
            "passage_id": first_value,
            "passage_text": second_value
        })
        total_processed += 1

        # Step 6: Process the batch once batch_size is reached
        if len(batch_texts) >= batch_size:
            print(f"Generating embeddings for batch {total_processed // batch_size + 1}...")
            embeddings_batch = sentence_transformer_ef.embed_documents(batch_texts)
            print("Embeddings generated. Adding to Chroma vector store...")

            # Add the embeddings to Chroma with metadata
            vectorstore.add_texts(
                texts=batch_texts,
                embeddings=embeddings_batch,
                metadatas=batch_metadatas
            )
            print(f"Batch {total_processed // batch_size + 1} added to Chroma.")

            # Clear the batch
            batch_texts = []
            batch_metadatas = []

            # Update progress bar
            progress_bar.update(1)
            print(f"Progress updated. Total processed so far: {total_processed}")


# Step 7: Process any remaining data in the last batch
if batch_texts:
    print("Processing the final batch...")
    embeddings_batch = sentence_transformer_ef.embed_documents(batch_texts)
    vectorstore.add_texts(
        texts=batch_texts,
        embeddings=embeddings_batch,
        metadatas=batch_metadatas
    )
    total_processed += len(batch_texts)
    print("Final batch added to Chroma.")


# Finalize
progress_bar.update(1)
progress_bar.close()

print(f"Total processed samples: {total_processed}")
print("Processing complete.")
