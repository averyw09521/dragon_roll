import json
with open("squad1-sampled-train.json", "r") as f:
  data_train = json.load(f) 
with open("squad1-sampled-dev.json", "r") as f:
  data_dev = json.load(f) 
with open("squad1-sampled-test.json", "r") as f:
  data_test = json.load(f)


print("Finished loading datasets")


import pandas as pd
documents = pd.read_csv("psgs_w100_subset.tsv", sep='\t', header=0, names=['psg_id', 'text', 'title'], index_col='psg_id')

def create_data_frame(data):
  data_train_rows = []
  for row in data_train:
    dataset, question, answers, positive_ctxs, negative_ctxs, hard_negative_ctxs = row.values()
    positive_indices = [item['psg_id'] for item in row['positive_ctxs']]
    negative_indices = [item['psg_id'] for item in row['negative_ctxs']]
    hard_indices = [item['psg_id'] for item in row['hard_negative_ctxs']]
    data_train_rows.append([dataset, question, answers, positive_indices, negative_indices, hard_indices])
  return pd.DataFrame(data_train_rows, columns=['dataset', 'question', 'answer', 'positive_indices', 'negative_indices', 'hard_indice'])
data_train_df = create_data_frame(data_train)
data_dev_df = create_data_frame(data_dev)
data_test_df = create_data_frame(data_test)

print("Finished creating dataframes")

train_contexts, train_questions, train_answers = [], [], []
dev_contexts, dev_questions, dev_answers = [], [], []
for entry in data_train:
  for positive_passage in entry['positive_ctxs']:
    answer = entry['answers'][0]
    index_find = positive_passage['text'].find(answer)
    if index_find == -1: 
      continue
    train_contexts.append(positive_passage['text'])
    train_questions.append(entry['question'])
    train_answers.append({'text': answer, 'answer_start':index_find})

  #for negative_passage in entry['hard_negative_ctxs'][:4]:
  #  train_contexts.append(negative_passage['text'])
  #  train_questions.append(entry['question'])
  #  answer = entry['answers'][0]
  #  train_answers.append({'text': answer, 'answer_start':-1})

for entry in data_dev:
  for positive_passage in entry['positive_ctxs']:
    answer = entry['answers'][0]
    index_find = positive_passage['text'].find(answer)
    if index_find == -1: 
      continue
    dev_contexts.append(positive_passage['text'])
    dev_questions.append(entry['question'])
    dev_answers.append({'text': answer, 'answer_start':index_find})

print("Finished collecting datasets")
  #for negative_passage in entry['hard_negative_ctxs'][:4]:
  ##  dev_contexts.append(negative_passage['text'])
  #  dev_questions.append(entry['question'])
  #  answer = entry['answers'][0]
  #  dev_answers.append({'text': answer, 'answer_start':-1})

def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if start_idx == -1:
            answer['answer_end'] = -1
        elif context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(dev_answers, dev_contexts)

print("Finished adding end_idx")

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
dev_encodings = tokenizer(dev_contexts, dev_questions, truncation=True, padding=True)

def add_token_positions(encodings, answers):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(dev_encodings, dev_answers)

import torch
print("Finished adding token positions")

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
dev_dataset = SquadDataset(dev_encodings)

from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
dev_encodings = tokenizer(dev_contexts, dev_questions, truncation=True, padding=True)

from transformers import DistilBertForQuestionAnswering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    load_best_model_at_end = True,
    save_strategy="steps",
    logging_strategy="steps",
    logging_first_step=True,
    overwrite_output_dir=True
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=dev_dataset             # evaluation dataset
)

trainer.train()
trainer.evaluate()
