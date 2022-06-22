# DearDr - Distantly Supervised Wikipedia-grounded Theme Extraction
This repo is based on a recent paper for Wikipedia-grounded entity extraction: DearDr 
(Data-efficient auto-regressive document retrieval). Entity extraction is modelled in a similar manner to
[GENRE](https://github.com/facebookresearch/GENRE), a document retriever and entity gruonder that is
trained to predict the document title given a sentence or query.

The DearDr models are trained to resolve Wikipedia hyperlinks without the need for additional training data.
This repo wraps around the original repo used in the research paper with additional dataset readers for Yahoo and SEC filings.

# deardr
