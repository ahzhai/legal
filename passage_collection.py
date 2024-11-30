# Convert qrels-passage.text.indirect.tsv into a dictionary that maps query ID → list of passage IDs
# Create a list of passage IDs that don’t map to any queries.
from datasets import load_dataset
from collections import defaultdict

# Step 1: Convert qrels-passage.text.indirect.tsv into a dictionary that maps query ID → list of passage IDs
queryIDtoPassageID = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": "qrels/qrels-passage.test.indirect.tsv"},
        streaming=True,
        delimiter="\t"
    )["data"]
queryIdToPassageId_iter = iter(queryIDtoPassageID)



def getQuery(qId):
    print("looking for query id: ", qId)
    queryDataset = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": "queries/test.single-removed.indirect.tsv"},
        streaming=True,
        delimiter="\t"
    )["data"]
    queryDataset_iter = iter(queryDataset)
    for sample in queryDataset_iter:
        keys = list(sample.keys())
        queryId = sample[keys[0]]
        if queryId == qId:
            header = keys[1]
            return sample[header]

def getPassage(pId):
    print("looking for passage id: ", pId)
    passageDataset  = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": "collection/collection.passage.tsv.gz"},
        streaming=True,
        delimiter="\t"
    )["data"]
    # Create an iterator for the dataset
    passageDatasetIter = iter(passageDataset)
    for passage in passageDatasetIter:
        keys = list(passage.keys())
        passageId = passage[keys[0]]
        if passageId == pId:
            passageText = passage[keys[1]]
            return(passageText)
            break


if __name__ == "__main__":

    queryIds = []
    passageIds = []
    queryIdToPassageIds_dict = defaultdict(list)
    for sample in queryIdToPassageId_iter:
        keys = list(sample.keys())
        queryID = sample[keys[0]]
        passageId = sample[keys[2]]
        if queryID not in queryIds:
            queryIds.append(queryID)
        if passageId not in passageIds:
            passageIds.append(passageId)
        queryIdToPassageIds_dict[queryID].append(passageId)

    passagesNotInQueries = []
    passageDataset = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": "collection/collection.passage.tsv.gz"},
        streaming=True,
        delimiter="\t"
    )["data"]
    passageDatasetIter = iter(passageDataset)
    for passage in passageDatasetIter:
        keys = list(passage.keys())
        passageId = passage[keys[0]]
        if passageId not in queryIdToPassageIds_dict.values():
            passagesNotInQueries.append(passageId)
            if len(passagesNotInQueries) >= 28500:
                break


    # Write out all of this to a file
    with open("processed_passages/passageIds.txt", "w") as f:
        for passage in passageIds:
            f.write(str(passage) + "\n")

    with open("processed_passages/queryIds.txt", "w") as f:
        for query in queryIds:
            f.write(str(query) + "\n")

    with open("processed_passages/queryIdToPassageIds.txt", "w") as f:
        for query, passage in queryIdToPassageIds_dict.items():
            f.write(str(query) + "\t" + str(passage) + "\n")

    with open("processed_passages/passagesNotInQueries.txt", "w") as f:
        for passage in passagesNotInQueries:
            f.write(str(passage) + "\n")