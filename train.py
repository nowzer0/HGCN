import numpy as np

import torch
from torch import nn
from torch import cuda
from torch.cuda import default_stream
from torch.nn.functional import dropout
from torch.utils.data import random_split, Dataset, DataLoader
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import InconsistencyNewsDataset 
from model import *
from transformers import RobertaModel, RobertaTokenizer
from transformers import BertTokenizer, BertModel
from src.kobert_tokenizer import KoBertTokenizer

import gc
gc.collect()
torch.cuda.empty_cache()

import argparse
parser = argparse.ArgumentParser(description='Detecting-low_quality_docs-Via-HierGraph')

class ArgsBase():
  @staticmethod
  def add_model_specific_args(parent_parser):
      parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
      parser.add_argument("--train_file", default='./Inconsistency_dataset/train.csv', type=str)
      parser.add_argument("--valid_file", default='./Inconsistency_dataset/valid.csv', type=str)
      parser.add_argument("--test_file", default='./Inconsistency_dataset/test.csv', type=str)
      parser.add_argument("--max_len", default=35, type=int)
      parser.add_argument("--sent_len", default=26, type=int) 
      parser.add_argument("--batch_size", default=112, type=int) # --batch_size / --gpu is allocated to each gpu
      return parser

class InconsistencyHeadlineDataModule(pl.LightningDataModule):
  def __init__(self, args):
    super(InconsistencyHeadlineDataModule,self).__init__()
    self.train_file_path = args.train_file
    self.valid_file_path = args.valid_file
    self.test_file_path = args.test_file
    self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    self.max_len = args.max_len
    self.sent_len = args.sent_len
    self.batch_size = args.batch_size
    self.num_workers=40

  def setup(self, stage=None):
    self.train = InconsistencyNewsDataset(file_name=self.train_file_path, tokenizer=self.tokenizer, max_len=self.max_len, max_sent=self.sent_len)
    self.valid = InconsistencyNewsDataset(file_name=self.valid_file_path, tokenizer=self.tokenizer, max_len=self.max_len, max_sent=self.sent_len)
    self.test = InconsistencyNewsDataset(file_name=self.test_file_path, tokenizer=self.tokenizer, max_len=self.max_len, max_sent=self.sent_len)

  def train_dataloader(self):
    train = DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    return train

  def val_dataloader(self):
    valid = DataLoader(self.valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    return valid

  def test_dataloader(self):
    test = DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    return test

class plModel(pl.LightningModule):

  def __init__(self, args): 
    super(plModel,self).__init__()
    self.dim = args.dim
    self.alpha = args.alpha
    self.dataset_size = args.dataset_size
    self.lr = args.lr
    self.max_epochs = args.max_epochs
    self.warm_up = args.warm_up
    self.bsz = args.batch_size
    self.gpus = args.gpus
    self.bert = BertModel.from_pretrained('monologg/kobert')
    self.title_BertPooler = CLSPooler(self.bert)
    self.subtitle_BertPooler = CLSPooler(self.bert)
    self.body_BertPooler = CLSPooler(self.bert)
    self.BertWrapper = BertWrapper(self.body_BertPooler)
    self.caption_BertPooler = CLSPooler(self.bert)
    self.Headline_extractor = Extractor(int(2*args.dim), args.hidden_dim, args.drop_rate)
    self.Body_extractor = Extractor(int(2*args.dim), args.hidden_dim, args.drop_rate)
    self.News_extractor = Extractor(int(2*args.dim), args.hidden_dim, args.drop_rate)
    # Encode Body /w self att
    self.Self_BodyAttention = Body_SelfAttention(self.dim, 8, args.drop_rate, batch=True)
    # Encode Body Matrix with Headline Body Attention
    self.Headline_Bodytext_Att = Headline_Bodytext_Attention(int(args.batch_size/args.gpus), args.sent_len, self.alpha) 
    self.Graph = HGCN(self.dim, args.gcn_layers, args.drop_rate)
    self.classfier = MLPClassifier(args.dim,args.hidden_dim,args.num_label,args.drop_rate)
    self.drop_layer = nn.Dropout(p=args.drop_rate)
    self.loss_function = nn.BCELoss()
    # optional - save hyper-parameters to self.hparams
    self.lr = args.lr
    self.save_hyperparameters()

  def forward(self,inputs):
  
    # 1. Each entity passes BERT
    title = self.title_BertPooler(inputs['title']) 
    subtitle = self.subtitle_BertPooler(inputs['subtitle'])
    bodytext = self.BertWrapper(inputs['body'])
    caption = self.caption_BertPooler(inputs['caption']) 

    # 2. Title, Subtitle Concat
    title_subtitle_cat = torch.cat((title,subtitle),2)

    # 3. Title, Subtitle > Headline
    headline = self.Headline_extractor(title_subtitle_cat)

    # 4. Headline Bodytext Attention
    # Bodytext Multi-head self attention
    bodytext, _ = self.Self_BodyAttention(bodytext)

    # Headline - Bodytext Attention
    bodytext, att_weight = self.Headline_Bodytext_Att(headline,bodytext)

    # Bodytext Caption Concat 
    bodytext_caption_cat = torch.cat((bodytext,caption),2)

    # 5. Bodytext, Caption > Body
    body = self.Body_extractor(bodytext_caption_cat)

    # Headline, Body Concat
    headline_body_cat = torch.cat((headline,body),2)

    # 6. Headline, Body >  Nes
    news = self.News_extractor(headline_body_cat)
   
    # 7. News, Headline, Body, Title, Subtitle, Bodytext, Caption concat > Node_feature matrix 
    hgraph = torch.cat((news,headline,body,title,subtitle,bodytext,caption),1)

    # 8. Graph Convolution
    hgraph_output = self.Graph(hgraph) 

    # 9. Newsvector
    news_after_hgcn = hgraph_output[:,0].unsqueeze(1) 

    # 10. Logit or prob return through the news vector classifier
    probs = self.classfier(news_after_hgcn)

    return probs.squeeze(1)
  
  @staticmethod
  def add_model_specific_args(parent_parser):
    # add model specific args
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--dim", default=768, type=int) # BERT 768 | RoBERTa 768
    parser.add_argument("--hidden_dim", default=768, type=int) # BERT 768 | RoBERTa 768
    parser.add_argument("--gcn_layers", default=4, type=int)
    parser.add_argument("--alpha", default=0.3, type=float) 
    parser.add_argument("--drop_rate", default=0.1, type=float)
    parser.add_argument("--num_label", default=1, type=int)
    return parser

  # accuracy function
  @staticmethod
  def flat_accuracy(preds,labels):
    pred_flat = np.where(preds<0.5,0,1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat).item(),len(labels_flat)

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.lr, correct_bias=True)
    return optimizer

  
  def training_step(self, batch, batch_idx):
    x,y = batch['input'], batch['label']
    logits = self(x)
    loss = self.loss_function(logits,y)
    # log training loss
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch['input'], batch['label']
    logits = self(x)
    loss = self.loss_function(logits, y)
    #logits and label move to flat_accuracy
    logits = logits.detach().cpu().numpy()
    label_ids = y.to('cpu').numpy()
    correct_size,size = self.flat_accuracy(logits,label_ids)
    return loss,correct_size,size

  def validation_epoch_end(self, val_step_outputs):
    losses = []
    correct_size=[]
    size=[]

    for i in range(len(val_step_outputs)):
      #losses, correct_size, size = val_step_outputs
      
      losses.append(val_step_outputs[i][0])  #loss 
      correct_size.append(val_step_outputs[i][1])  #correct_size
      size.append(val_step_outputs[i][2])  #sizego
      
    total_correct=sum(correct_size)
    total_size=sum(size)
    self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
    self.log('val_acc', torch.div(total_correct,total_size),prog_bar=True)
    
  def test_step(self,batch,batch_idx):
    x, y = batch['input'], batch['label']
    logits = self(x)
    loss = self.loss_function(logits, y)
    #logits and label move to flat_accuracy
    logits = logits.detach().cpu().numpy()
    label_ids = y.to('cpu').numpy()
    correct_size,size = self.flat_accuracy(logits,label_ids)
    return loss,correct_size,size

  def test_epoch_end(self, test_step_outputs):
    losses = []
    correct_size=[]
    size=[]

    for i in range(len(test_step_outputs)):
      losses.append(test_step_outputs[i][0])  #loss 
      correct_size.append(test_step_outputs[i][1])  #correct_size
      size.append(test_step_outputs[i][2])  #size

    total_correct=torch.stack(correct_size).sum()
    total_size=torch.stack(size).sum()
    self.log('test_loss', torch.stack(losses).mean(), prog_bar=True)
    self.log('test_acc', torch.div(total_correct,total_size),prog_bar=True)
 
if __name__ == "__main__":
  parser = ArgsBase.add_model_specific_args(parser)
  parser = plModel.add_model_specific_args(parser)
  parser.add_argument("--warm_up", default=800, type=int)
  parser.add_argument("--patience", default=3, type=int)
  parser.add_argument("--min_delta", default=0.001, type=float) 
  parser.add_argument("--checkpoint_path", default='./checkpoints/', type=str)
  parser = pl.Trainer.add_argparse_args(parser)
  args = parser.parse_args()
  args.dataset_size = 194860  # hardcode train dataset size. Needed to compute number of steps for the lr scheduler
  args.gpus=4
  print(args)

  pl.seed_everything(111)
  early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=args.min_delta, patience=args.patience, verbose=False, mode='min')
  checkpoint_callback = ModelCheckpoint(monitor="val_loss",dirpath=args.checkpoint_path, filename=f'HGCN={args.lr}-BSZ={args.batch_size}-Sent_len={args.sent_len}-'+"{epoch:02d}-{val_loss:.3f}",save_top_k=2,mode="min")

  dm = InconsistencyHeadlineDataModule(args)

  model = plModel(args)
  trainer = pl.Trainer.from_argparse_args(args, accelerator='dp' ,callbacks=[early_stop_callback,checkpoint_callback],gpus=8)

  print("Start Training!!")
  trainer.fit(model, dm)
  
  print("Testing")
  test = InconsistencyNewsDataset(args.test_file, tokenizer=KoBertTokenizer.from_pretrained('monologg/kobert'), max_len=args.max_len, max_sent=args.sent_len)
  test_dataloader = DataLoader(test, batch_size=args.batch_size, num_workers=40, shuffle=False)
  trainer.test(model=model,test_dataloaders=test_dataloader)