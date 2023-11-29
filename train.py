import pickle as pickle
import os
import pandas as pd
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, AutoConfig, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from load_data import *
from functools import partial
import pdb

LABEL_TO_TYPE_ID = {'no_relation':0, 'org:top_members/employees':1, 'org:members':1,
       'org:product':2, 'per:title':5, 'org:alternate_names':3,
       'per:employee_of':6, 'org:place_of_headquarters':2, 'per:product':5,
       'org:number_of_employees/members':4, 'per:children':7,
       'per:place_of_residence':8, 'per:alternate_names':7,
       'per:other_family':7, 'per:colleagues':7, 'per:origin':8, 'per:siblings':7,
       'per:spouse':7, 'org:founded':4, 'org:political/religious_affiliation':3,
       'org:member_of':3, 'per:parents':7, 'org:dissolved':4,
       'per:schools_attended':6, 'per:date_of_death':9, 'per:date_of_birth':9,
       'per:place_of_birth':8, 'per:place_of_death':8, 'org:founded_by':1,
       'per:religion':6}
TYPE_ID_TO_TYPE_PAIR = {0: 'no_relation', 1:'org_person', 2:'org_place', 3:'org_org', 4:'org_numeric',
                  5:'per_job', 6:'per_org', 7:'per_per', 8:'per_place', 9:'per_date'}
TYPE_PAIR_TO_TYPE_ID = {v:k for k, v in TYPE_ID_TO_TYPE_PAIR.items()}

N_TYPES = len(TYPE_ID_TO_TYPE_PAIR)

ID_TO_LABEL = {0: {0:'no_relation'},
               1: # 'org_person'
               {0:'org:top_members/employees', 1:'org:members', 2:'org:founded_by'},
               2: # 'org_place'
               {0:'org:place_of_headquarters', 1: 'org:product'},
               3: #'org_org'
               {0:'org:alternate_names', 1:'org:member_of', 2:'org:political/religious_affiliation'},
               4: #'org_numeric'
               {0:'org:founded', 1:'org:dissolved', 2:'org:number_of_employees/members'},
               
               5: #'per_job'
               {0:'per:title', 1:'per:product'},
               6: #'per_org'
               {0:'per:employee_of', 1:'per:schools_attended', 2:'per:religion'},
               7: #'per_per'
               {0:'per:children', 1:'per:alternate_names', 2:'per:other_family', 3:'per:colleagues', 4:'per:siblings', 5:'per:spouse', 6:'per:parents', },
               8: #per_place
               {0:'per:place_of_residence', 1:'per:origin', 2:'per:place_of_birth', 3:'per:place_of_death', },
               9: #per_date
               {0:'per:date_of_death', 1:'per:date_of_birth', },
              }

def klue_re_micro_f1(preds, labels, label_dict):
    """KLUE-RE micro f1 (except no_relation)"""
    # label_list = ['no_relation', 'org:top_members/employees', 'org:members',
    #    'org:product', 'per:title', 'org:alternate_names',
    #    'per:employee_of', 'org:place_of_headquarters', 'per:product',
    #    'org:number_of_employees/members', 'per:children',
    #    'per:place_of_residence', 'per:alternate_names',
    #    'per:other_family', 'per:colleagues', 'per:origin', 'per:siblings',
    #    'per:spouse', 'org:founded', 'org:political/religious_affiliation',
    #    'org:member_of', 'per:parents', 'org:dissolved',
    #    'per:schools_attended', 'per:date_of_death', 'per:date_of_birth',
    #    'per:place_of_birth', 'per:place_of_death', 'org:founded_by',
    #    'per:religion']
    if 'no_relation' in label_dict:
        label_indices = list(range(1, len(label_dict)))
    else:
        label_indices = list((range(len(label_dict))))
    # no_relation_label_idx = label_list.index("no_relation")
    # label_indices = list(range(len(label_list)))
    # label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels, n_class):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(n_class)[labels]

    score = np.zeros((n_class,))
    for c in range(n_class):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred, type_id=None):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  if type_id==None:
    f1 = klue_re_micro_f1(preds, labels, TYPE_ID_TO_TYPE_PAIR)
    auprc = klue_re_auprc(probs, labels, N_TYPES)
  else:
    f1 = klue_re_micro_f1(preds, labels, ID_TO_LABEL[type_id])
    auprc = klue_re_auprc(probs, labels, len(ID_TO_LABEL[type_id]))
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label

def label_to_num_recent(label, type_id=None):
  num_label = []
  if type_id == None:
      dict_label_to_num = LABEL_TO_TYPE_ID
  else:
      dict_label_to_num = {v:k for k,v in ID_TO_LABEL[type_id].items()}

  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label
  

def train(type_id=None):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    
  # load model and tokenizer
  # MODEL_NAME = "bert-base-uncased"
  # MODEL_NAME = "klue/bert-base"
  MODEL_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  # load dataset
  train_dataset = load_data("../dataset/train/train.csv")
  # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
  if type_id != None:
    train_dataset = train_dataset.loc[train_dataset['label'].isin(ID_TO_LABEL[type_id].values())]

  # random split dataset
  n, _ = train_dataset.shape
  n_train = int(n*0.8)
  df_shuffled = train_dataset.sample(frac=1).reset_index(drop=True)
  train_dataset = df_shuffled.iloc[:n_train]
  dev_dataset = df_shuffled.iloc[n_train:]

  train_label = label_to_num_recent(train_dataset['label'].values, type_id)
  dev_label = label_to_num_recent(dev_dataset['label'].values, type_id)

  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer)
  tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(device)
    
  # setting model hyperparameter
  model_config =  AutoConfig.from_pretrained(MODEL_NAME)
  if type_id == None:
    model_config.num_labels = N_TYPES
  else:
    model_config.num_labels = len(ID_TO_LABEL[type_id])

  model =  AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.to(device)
  # model = DistributedDataParallel(model, device_ids=[device], output_device=device)
  
  print(model.config)
  model.parameters
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir='./results_token/{}'.format(type_id),          # output directory
    save_total_limit=1,              # number of total save model.
    save_steps=200,                 # model saving step.
    num_train_epochs=20,              # total number of training epochs
    learning_rate=2e-4,               # learning_rate
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=200,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    eval_steps = 100,            # evaluation step.
    load_best_model_at_end = True ,
    gradient_accumulation_steps = 5,
  )
  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_train_dataset,             # evaluation dataset
    compute_metrics=partial(compute_metrics, type_id=type_id)         # define metrics function
  )

  # train model
  trainer.train()
  model.save_pretrained('./best_model_token/{}'.format(type_id))
    
def main():
  train()
  for i in range(1, 10):
    train(i)

if __name__ == '__main__':
  main()
