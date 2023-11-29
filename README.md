# aistage2023
1. Summary

This repository is containing code for relation extraction task in [aistage](https://stages.ai/host/dashboard/) competition.
Relation extraction is the task of predicting relations for entities(subject and object) in a sentence.
For example, given a sentence “**Barack Obama** was born in **Honolulu**, Hawaii.”, a relation classifier is tasked to predict the relation of “bornInCity”.
Relation Extraction is the key component for building relation knowledge graphs, and it is of crucial significance to natural language processing applications such as structured search, sentiment analysis, question answering, and summarization.

3. Experimental results

I used pretrained model **"klue/roberta-large"** for better performance than "klue/bert-base".
Based on the model "klue/roberta-large", I tried two approaches: 1) fine tune the hyperparameters(batch size, gradient accumulate, ddp, epoch), 2) entity type restriction [paper](https://arxiv.org/pdf/2105.08393.pdf).
However, second approach did not work well, showing poorer performance than first approach. 
Results are reported blow.

1) fine tune the hyperparameters

|submission name|batch size per device|gradient accumulate|ddp|epoch|f1|auprc|
|---|---|---|---|---|---|---|
|3|20|5|2|20|69.9202|69.9143|
|4|16|5|1|20|71.3770|68.6663|
|5|24|10|2|40|69.2951|67.7987|

2) entity type restriction

|submission name|batch size per device|gradient accumulate|ddp|epoch|f1|auprc|
|---|---|---|---|---|---|---|
|14|16|5|2|20|67.2689|55.1896|

5. Instructions

├── code

│   ├── __pycache__

│   ├── best_model

│   ├── logs

│   ├── prediction

│   └── results

    
1) ```python train_org.py``` and ```python inference_org.py``` for first approach.
2) ```python train.py``` and ```python inference.py``` for second approach.

6. Approach

I converted the pretrained model from "klue/bert-base" to "klue/roberta-large" for better performance. 
On top of "klue/roberta-large", I tried two kinds of approaches: 1) fine tune the hyperparameters, 2) entity type restriction.

For the first approach, I fine tuned batch size per device, gradient accumulate, ddp, and epoch. 
Interestingly, the model showed worse classification performance with larger whole batch size(batch size per device * gradient accumulate * ddp(num gpu)).

For the second approach, I used entity type restriction strategy.
I splited the task of classifying 30 labels into 10 subtasks with following types: 'no_relation', 'org_person', 'org_place', 'org_org', 'org_numeric', 'person_job', 'person_org', 'person_person', 'person_place', 'person_date'.
First, classify the type of relation with a main model classifying types of 10 labels. Second, given the type of relation, use type-specific model to classify the relation of 30 labels.
Eventhough this approach involves 11 models to be trained, the performance was worse than first approach.
I tried to develop this approach but there was not enough time.
