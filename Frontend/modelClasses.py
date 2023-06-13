
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
import pandas as pd
import torch
from transformers import AutoModel

#For Web Application 

class CustomBERTOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class CustomBERTModel(nn.Module):
    def __init__(self, model_name, num_labels, hidden_dim, dropout_rate):
        super(CustomBERTModel, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_extra_features + 1, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask, features, keyword_tokens):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
            
        combined_output = torch.cat((pooled_output, features, keyword_tokens), dim=1)

        logits = self.classifier(combined_output)

        return CustomBERTOutput(logits=logits, hidden_states=outputs.hidden_states)

    
def preprocess(texts, tokenizer, features=None, keywords=None):
  texts = texts.tolist() if isinstance(texts, pd.Series) else texts  # Convert to list if needed

  encoded_inputs = tokenizer(
    texts,
    padding='longest',
    truncation=True,
    max_length=512,
    return_tensors='pt',
    add_special_tokens=True
  )
  input_ids = encoded_inputs['input_ids']
  attention_masks = encoded_inputs['attention_mask']

  if not features is None:

    # Encode the keyword column
    keyword_tokens = []
    for keyword in keywords:
        if pd.isnull(keyword):
            keyword_tokens.append(tokenizer.pad_token_id)
        else:
            keyword_tokens.append(tokenizer.encode(keyword, add_special_tokens=False)[0])
    keyword_tokens = torch.tensor(keyword_tokens).unsqueeze(1)

    features_tensor = torch.tensor(features.tolist())
    return input_ids, attention_masks, features_tensor, keyword_tokens

  return input_ids, attention_masks

class netmodel(nn.Module):
  def __init__(self, input_layer=1, num_hidden=1, node_per_hidden=32, droppout=0., LSTM_layers=0, outputs=2):
    super(netmodel, self).__init__()
    self.input_layer = input_layer
    self.num_hidden = num_hidden 
    self.node_per_hidden = node_per_hidden
    self.droppout = droppout 
    self.SLTM_layers = LSTM_layers 
    self.outputs = outputs 
    self.inputfc = nn.Linear(input_layer, node_per_hidden)
    self.hiddenfc = [] 
    for i in range(num_hidden-1):
      self.hiddenfc.append(nn.Linear(node_per_hidden, node_per_hidden))
    self.lastfc = nn.Linear(node_per_hidden, outputs)

  def forward(self, x, debug=False):
    drop = nn.Dropout(p=self.droppout)
    #x = x.view(1,1)
    x = self.inputfc(x)
    x = F.relu(x)
    x = drop(x)
    for i in range(self.num_hidden-1):
      x = self.hiddenfc[i](x)
      x = F.relu(x)
      x = drop(x)
    
    x = self.lastfc(x)
    x = F.softmax(x, dim=0)
    return x 