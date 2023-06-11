
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

class BERTmodel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', outputs=2):
        super(BERTmodel, self).__init__()
        self.model_name = model_name
        self.outputs = outputs
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=outputs)
    
    def forward(self, x, debug=False):
        x = self.bert(x)
        x = F.softmax(x, dim=0)
        return x

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