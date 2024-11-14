# Convert qrels-doc.text.indirect.tsv into a dictionary that maps query ID → doc ID (3k)
# also have a separate list of all the passage IDs
# Create a list of doc IDs that don’t map to any passages. (27k additional doc ids)
from datasets import load_dataset



queryIdToDocId = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "qrels/qrels-doc.test.indirect.tsv"},
    streaming=True,
    delimiter="\t"
)["data"]
# Create an iterator for the dataset
queryIdToDocId_iter = iter(queryIdToDocId)

queryIds = []
docIds = []
queryIdToDocId_dict = {}
for sample in queryIdToDocId_iter:
    keys = list(sample.keys())
    queryID = sample[keys[0]]
    docId = sample[keys[2]]
    queryIds.append(queryID)
    docIds.append(docId)
    queryIdToDocId_dict[queryID] = docId

print(len(queryIds))
print(len(queryIdToDocId_dict))
print(len(docIds))



docsNotInPassages = []
docDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "collection/collection.doc.tsv.gz"},
    streaming=True,
    delimiter="\t"
)["data"]
docDatasetIter = iter(docDataset)
for doc in docDatasetIter:
    keys = list(doc.keys())
    docId = doc[keys[0]]
    if docId not in queryIdToDocId_dict.values():
        docsNotInPassages.append(docId)
        if len(docsNotInPassages) >= 28500:
            break

print("first 10 docs not in passages: ", docsNotInPassages[:10])
print(len(docsNotInPassages))

# Write out all of this to a file
with open("docIds.txt", "w") as f:
    for doc in docIds:
        f.write(str(doc) + "\n")

with open("queryIds.txt", "w") as f:
    for query in queryIds:
        f.write(str(query) + "\n")

with open("queryIdToDocId.txt", "w") as f:
    for query, doc in queryIdToDocId_dict.items():
        f.write(str(query) + "\t" + str(doc) + "\n")

with open("docsNotInPassages.txt", "w") as f:
    for doc in docsNotInPassages:
        f.write(str(doc) + "\n")




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


def getDoc(dId):
    print("looking for doc id: ", dId)
    docDataset = load_dataset(
        "jhu-clsp/CLERC",
        data_files={"data": "collection/collection.doc.tsv.gz"},
        streaming=True,
        delimiter="\t"
    )["data"]
    # Create an iterator for the dataset
    docDatasetIter = iter(docDataset)
    for doc in docDatasetIter:
        keys = list(doc.keys())
        docId = doc[keys[0]]
        if docId == docIds[0]:
            docText = doc[keys[1]]
            return(docText)
            break


