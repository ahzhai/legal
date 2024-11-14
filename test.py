""" example = {
    '0': 2, 
    'OPINION REGARDING DEFENDANTS’ MOTION FOR JUDGMENT ON THE PLEADINGS GORDON J. QUIST, District Judge. Plaintiff, Charlie Beamon, proceeding pro se, filed a Complaint against Defendants on April 25, 2012, in the 57th District Court of Alegan County, Michigan. Defendants removed the case to this Court on May 8, 2012, alleging federal question jurisdiction, 28 U.S.C. § 1331, on the basis that Plaintiffs claim is governed by the Employee Retirement Income Security Act of 1974 (“ERISA”), as amended, 29 U.S.C. § 1001 et seq. Plaintiffs claim arises under ERISA because he seeks review of a denial of benefits under a group long-term disability policy. See 29 U.S.C. § 1132(a)(1)(B). Defendants have filed a Motion for Judgment on the Pleadings on the grounds that Plaintiff failed to exhaust his administrative remedies prior to filing this action. Plaintiff has responded by filing a Motion for Dismissal of Defendants’ Motion for Judgment on the Pleadings. For the following reasons, the Court will grant Defendants’ motion, deny Plaintiffs motion, and dismiss Plaintiffs Complaint with prejudice. Background In September of 2000, Plaintiff was placed on medical leave by his employer, Murco Foods Inc., for injuries that he sustained at work. Plaintiff applied for and received long-term disability benefits under a group disability insurance policy (Policy) that Fortis Benefits Insurance Company (Fortis) issued to Murco Food Inc. In August of 2002, Plaintiff obtained a workers’ compensation award. At the time, Fortis had a lien on the workers’ compensation award in the amount of $41,867.00 for an overpayment of benefits under the Policy. Through his counsel, Plaintiff obtained an agreement from Fortis to compromise its lien claim for a payment of $20,993.50. In March 2007, Plaintiff received a retroactive Social Security Disability (SSD) award in the amount of $73,226.63. Fortis claimed that the SSD award created an overpayment that it was entitled to recover under the Policy’s Adjustment of Benefits provision. Fortis also determined that Plaintiff was receiving Social Security dependent benefits, which increased the overpayment amount to $88,438.13. Fortis initially attempted to collect the overpayment from Plaintiff through a collection agency, but when those efforts failed, Fortis': 'exercised its option to recover the overpayment through an adjustment of Plaintiffs monthly benefit until the overpayment was fully reimbursed. Fortis subsequently reduced the overpayment after Plaintiff furnished documents to Fortis showing that Plaintiffs wife, who along with Plaintiffs children was then living apart from Plaintiff, was receiving the dependent benefit on behalf of the children. Plaintiff claims that Fortis’ prior agreement to accept $20,933.50 from the workers’ compensation award in satisfaction of its lien for $41,867 bars Fortis from reducfng Plaintiffs monthly benefit to recover the overpayment resulting from the SSD award. II. Motion Standard Defendants bring their instant motion as a Motion for Judgment on the Pleadings pursuant to Federal Rule of Civil Procedure 12(c). A motion under Rule 12(c) is reviewed under the same standard as a motion to dismiss under Rule 12(b)(6). EEOC v. J.H. Routh Packing Co., 246 F.3d 850, 851 (6th Cir.2001). Defendants assert that a Rule 12(c) motion is an appropriate vehicle for dismissal because an ERISA plaintiff has the burden of pleading exhaustion of administrative remedies. As support for their assertion that the burden of pleading exhaustion in an ERISA case is on the plaintiff, Defendants cite Hagen v. VPA, Inc., 428 F.Supp.2d 708 (W.D.Mich.2006), in which the court observed that dismissal was proper because the plaintiff failed to allege exhaustion in his complaint. See id. at 713. Although the Sixth Circuit has not addressed the issue, a number of courts have held that exhaustion of administrative remedies under ERISA is an affirmative defense. For example, in Wilson v. Kimberly-Clark Corp., 254 Fed.Appx. 280 (5th Cir.2007), the Fifth Circuit concluded that exhaustion is an affirmative defense under ERISA. Id. at 287. For guidance, the court looked to the United States Supreme Court’s decision in Jones v. Bock, 549 U.S. 199, 127 S.Ct. 910, 166 L.Ed.2d 798 (2007), which held that exhaustion under the Prison Litigation Reform Act (PLRA) is an affirmative defense rather than a pleading requirement. The court thus held, “[a]l-though Plaintiffs failed to plead that they exhausted administrative remedies, the^ need not have done so here.” Id. Similarly, the Second'
}

counter = 0
for key, value in example.items():
    if counter == 1:
        print(value)
    counter += 1

 """

from datasets import load_dataset

# Import the passage dataset
passageDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "collection/collection.passage.tsv.gz"},
    streaming=True,
    delimiter="\t"
)["data"]
# Create an iterator for the dataset
passageDataset_iter = iter(passageDataset)

queryDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "queries/test.single-removed.indirect.tsv"},
    streaming=True,
    delimiter="\t"
)["data"]
queryDataset_iter = iter(queryDataset)

queryToPassageDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "qrels/qrels-passage.test.indirect.tsv"},
    streaming=True,
    delimiter="\t"
)["data"]
queryToPassageDataset_iter = iter(queryToPassageDataset)

documentDataset = load_dataset(
    "jhu-clsp/CLERC",
    data_files={"data": "collection/collection.doc.tsv.gz"},
    streaming=True,
    delimiter="\t"
)["data"]
documentDataset_iter = iter(documentDataset)

docs = [next(documentDataset_iter) for _ in range(2)]
document1 = docs[0]
document2 = docs[1]
#print(document1)
print(document2)


def getQuery(queryId):
    for sample in queryDataset_iter:
        first = list(sample.keys())[0]
        if first == queryId:
            second = list(sample.keys())[1]
            return sample[second]

def getPassage(passId):
    # Display the samples
    for sample in passageDataset_iter:
        # sample is a dictionary. get first key and its value. value is the passage id
        first  = list(sample.keys())[0]
        passageId = sample[first]
        if passageId == passId:
            nextKey = list(sample.keys())[1]
            return sample[nextKey]



# examplesPrinted = 0
# # queries = [next(queryToPassageDataset_iter) for _ in range(1)]
# for query in queryToPassageDataset_iter:
#     if examplesPrinted >= 2:
#         break
#     queryID = list(query.keys())[0]
#     passageIdKey = list(query.keys())[2]
#     passageId = query[passageIdKey]

#     if passageId < 100000:
#         examplesPrinted += 1
#         print("QueryID: ", queryID)
#         print("PassageID: ", passageId)
#         print("Query: ", getQuery(queryID))
#         print("Passage: ", getPassage(passageId))


#48606

'''
# mappingPID2DID = load_dataset(
#     "jhu-clsp/CLERC",
#     data_files={"data": "collection/mapping.pid2did.tsv"},
#     streaming=True,
#     delimiter="\t"
# )["data"]

# dataset_iter = iter(mappingPID2DID)

# samples = [next(dataset_iter) for _ in range(10)]

# print(samples)
'''