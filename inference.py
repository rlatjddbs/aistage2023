from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
import pdb

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

def inference(model, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)

  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def inference_new(model, model1,model2,model3,model4,model5,model6,model7,model8,model9, tokenized_sent, device):
  """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
  """
  dataloader = DataLoader(tokenized_sent, batch_size=1, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  output_type = []
    
  with open('./dict_label_to_num.pkl', 'rb') as f:
      label_type = pickle.load(f)
        
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      type_output = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
      )
      type_logit = type_output[0]
      type_prob = F.softmax(type_logit, dim=-1)
      type_result = np.argmax(type_logit.detach().cpu().numpy(), axis=-1)
      
      prob = torch.ones((1,30)) * 1e-6
      # for v in ID_TO_LABEL.values():
      #   for c in v.values():
      #       prob[0, label_type[c]] = type_prob[0, LABEL_TO_TYPE_ID[c]] / len(v)
            
      if type_result == 0:
        result = 0
        prob[0,0] = 1.
      else: 
          if type_result == 1:
              class_model = model1
          elif type_result == 2:
              class_model = model2
          elif type_result == 3:
              class_model = model3
          elif type_result == 4:
              class_model = model4
          elif type_result == 5:
              class_model = model5
          elif type_result == 6:
              class_model = model6
          elif type_result == 7:
              class_model = model7
          elif type_result == 8:
              class_model = model8
          elif type_result == 9:
              class_model = model9
            
          outputs = class_model(
              input_ids=data['input_ids'].to(device),
              attention_mask=data['attention_mask'].to(device),
              token_type_ids=data['token_type_ids'].to(device)
              )
          logits = outputs[0]
          prob_mini = F.softmax(logits, dim=-1).detach().cpu().numpy()
          ID_TO_LABEL[type_result][np.argmax(prob_mini, axis=-1)[0]]
          for i, v in enumerate(ID_TO_LABEL[type_result[0]].values()):
            prob[0,label_type[v]] = prob_mini[0, i] * type_prob[0, type_result[0]]

      #logits = logits.detach().cpu().numpy()
      prob = F.softmax(prob, dim=-1).detach().cpu().numpy()
      result = np.argmax(prob, axis=-1)[0]

    #output_type.append(type_result)
    output_pred.append(result)
    output_prob.append(prob)
  
  return output_pred, np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer):
  """
    test dataset을 불러온 후,
    tokenizing 합니다.
  """
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  Tokenizer_NAME = "klue/roberta-large"
  tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

  ## load my model
  MODEL_NAME = args.model_dir # model dir.
  print('Loading model...')
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/None')
  print('Loading model 1...')
  model1 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/1')
  print('Loading model 2...')
  model2 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/2')
  print('Loading model 3...')
  model3 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/3')
  print('Loading model 4...')
  model4 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/4')
  print('Loading model 5...')
  model5 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/5')
  print('Loading model 6...')
  model6 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/6')
  print('Loading model 7...')
  model7 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/7')
  print('Loading model 8...')
  model8 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/8')
  print('Loading model 9...')
  model9 = AutoModelForSequenceClassification.from_pretrained(args.model_dir+'/9')
  # model.parameters
  model.to(device)
  model1.to(device)
  model2.to(device)
  model3.to(device)
  model4.to(device)
  model5.to(device)
  model6.to(device)
  model7.to(device)
  model8.to(device)
  model9.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  Re_test_dataset = RE_Dataset(test_dataset ,test_label)

  ## predict answer
  pred_answer, output_prob = inference_new(model,model1,model2,model3,model4,model5,model6,model7,model8,model9,Re_test_dataset, device) # model에서 class 추론
  # pred_answer, output_prob = inference(model,Re_test_dataset, device) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./best_model")
  args = parser.parse_args()
  print(args)
  main(args)
  
