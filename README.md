# Information Retrieval System on Cranfield Dataset

## Problem Definition
An information retrieval system needs to be built which can retrieve documents from a set of documents given a query for the document. We have a naive vector space model, which uses the classical
approach to implement the IR system. However, it fails to retrieve relevant documents for some queries. We need to find and implement methods which can address these issues
and make the model more efficient and accurate.

## Code Description
This repo contains 6 folders corresponding to the 5 different types of models trained on the cranfield dataset and the 6th folder for hypothesis testing between models. 

### baseline_vsm
This is a naive vector space model with acts as a base model on which we make modifications.

### vsm_with_corrections
This model has better tokenized and spell corrected text.

### lsa
This is the model which implements latent sematic indexing.

### query_expansion
This model uses expanded queries.

### word2vec
This model uses Google's pretrained neural net model.

> **_NOTE:_** Each model can be run independent of each other by running the main.py file of the respective model. 

### hyp_testing_data
This folder contains randomly sampled nDCG scores for all 5 models. The [t_test.py](hyp_testing_data/t_test.py) file performs the paired t test between the models.

## Report
The detailed introduction to the problem, methodology, results and conclusions can be found in the [report](report.pdf)



