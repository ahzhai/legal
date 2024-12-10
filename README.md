# legal
For baseline retrieval, run baseline_retrieval_passages.py.
For linear retrieval, run linear_method_and_baseline_joined.py.
For tree-based retrieval, run tree_reflection_search.py.

All processed passage IDs contained in our vector database are contained in processed_passages. queryIdToPassageIds.txt maps each query ID in our test dataset to one or more passages that it cites and should retrieve. Due to the extremely large size of our vector database with 80,000 embedded passages, it is unable to be directly stored in this repository. Instead, run embed_passages.py in order to set up your own local version of this database, or reach out to us to share it.
