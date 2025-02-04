from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from torch.nn.parallel import DistributedDataParallel


class TransformerCustomDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]


class DecoderOnlyCustomDataset(Dataset):
        def __init__(self, targets):
            self.targets = targets

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):
            return self.targets[idx]


class TransformerDataset:
    def __init__(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
    # 영 -> 한 
    def forward(self, eng, kor=None, train=True, device='cuda'):
        src_mask = None

        src = self.tokenizer(eng, padding=True, truncation=True, max_length=256, return_tensors="pt")['input_ids']
        src_mask = (src != self.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        # train
        if train and kor is not None:
            tgt = self.tokenizer(kor, padding=True, truncation=True, max_length=256, return_tensors="pt")['input_ids']
            tgt_mask = ((tgt != self.tokenizer.pad_token_id) & (tgt != self.tokenizer.sep_token_id)).unsqueeze(1).unsqueeze(-1)
            seq_len = tgt.size(1)
            tgt_mask = tgt_mask & torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        
        # inference
        else:
            tgt = torch.tensor([[self.tokenizer.eos_token_id]])
            tgt_mask = None

        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)

        return src, tgt, src_mask, tgt_mask
    
    def __call__(self, eng, kor, train=True):
        return self.forward(eng, kor, train)

    def build_dataset(self, config):
        ds = load_dataset(config['dataset_name'])

        # ds_train = ds["train"][:1100000]
        # ds_val = ds["train"][1100000:]
        
        ds_train = ds["train"][:2000]
        ds_val = ds["train"][200:300]
        
        train_engs = ds_train["Skinner's reward is mostly eye-watering."]
        train_kors = ds_train["스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다."]

        val_engs = ds_train["Skinner's reward is mostly eye-watering."]
        val_kors = ds_train["스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다."]
        
        train_dataset = TransformerCustomDataset(train_engs, train_kors)
        val_dataset = TransformerCustomDataset(val_engs, val_kors)
        
        if config['is_DDP']:
            train_sampler = DistributedSampler(train_dataset, num_replicas=config['world_size'], rank=config['rank'])
            val_sampler = DistributedSampler(val_dataset, num_replicas=config['world_size'], rank=config['rank'])
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, sampler=val_sampler, shuffle=False)
        return train_loader, val_loader


class DecoderOnlyDataset:
    def __init__(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        
    def forward(self, text, train=True, device='cuda'):
        if train:
            tgt = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt")['input_ids']
            tgt_mask = ((tgt != self.tokenizer.pad_token_id) & (tgt != self.tokenizer.sep_token_id)).unsqueeze(1).unsqueeze(-1)
            seq_len = tgt.size(1)
            tgt_mask = tgt_mask & torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        
        # inference
        else:
            tgt = torch.tensor([[self.tokenizer.eos_token_id]])
            tgt_mask = None

        tgt = tgt.to(device)
        tgt_mask = tgt_mask.to(device)

        return tgt, tgt_mask
    
    def __call__(self, text, train=True):
        return self.forward(text, train)

    def build_dataset(self, config):
        ds = load_dataset(config['dataset_name'])

        # ds_train = ds["train"][:1100000]
        # ds_val = ds["train"][1100000:]
        
        ds_train = ds["train"][:2000]
        ds_val = ds["train"][200:300]
        
        train_kors = ds_train["스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다."]
        val_kors = ds_train["스키너가 말한 보상은 대부분 눈으로 볼 수 있는 현물이다."]
        
        train_dataset = DecoderOnlyCustomDataset(train_kors)
        val_dataset = DecoderOnlyCustomDataset(val_kors)
        
        if config['is_DDP']:
            train_sampler = DistributedSampler(train_dataset, num_replicas=config['world_size'], rank=config['rank'])
            val_sampler = DistributedSampler(val_dataset, num_replicas=config['world_size'], rank=config['rank'])
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, sampler=train_sampler, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], pin_memory=True, sampler=val_sampler, shuffle=False)
        return train_loader, val_loader


