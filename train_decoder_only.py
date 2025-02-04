# custom
from core.utils import *
from core.make_dataset import *
from core.evaluate import * 
from models.model.model import *

# module
import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter


def data_distributed_parallel(config):
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.cuda_current_device()
    torch.cuda.set_device(device)
    
    config['rank'] = rank
    config['world_size'] = world_size
    config['device'] = device

    print(f"Use GPU: {device} for training")
    return config


def run(config):
    # DDP train 'on' or 'off'
    if config['is_DDP']:
        config = data_distributed_parallel(config)
        device = config['device']
    else:
        device = 'cuda'
    
    output_path = os.path.join(f"./outputs/{config['output_path']}")
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'info.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)

    # seed setting
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # make dataset
    dataset = DecoderOnlyDataset(pretrained_model=config['tokenizer_model_name'])

    # model build
    model = DecoderOnlyModel( 
                        d_model=config['d_model'],
                        max_len=config['max_len'],
                        temperature=config['temperature'],
                        n_head=config['n_head'],
                        ffn_hidden=config['ffn_hidden'],
                        p_drop=config['p_drop'],
                        n_layers=config['n_layers'],
                        tokenizer=dataset.tokenizer,
                        device=device,
                        )
    
    model.to(device)
    model.apply(initialize_weights)
    
    # DDP 인 경우
    if config['is_DDP']:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    
    # retraining
    if config['retrain']:    
        pt_arr = [f for f in os.listdir(output_path) if f.endswith('.pt')]
        
        # Last .pt file filtering
        sorted_files = sorted(pt_arr, key=lambda f: (extract_number(f) is None, extract_number(f)), reverse=True)        
        model.load_state_dict(torch.load(os.path.join(output_path, sorted_files[0])), strict=False)
        
    # criterion : ingnore_index : [pad] index
    criterion = nn.CrossEntropyLoss(
                                    ignore_index = dataset.tokenizer.pad_token_id,
                                    #label_smoothing = 0.1
                                    )
    
    # parameter check
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('number of params:', n_parameters)

    # optimizer
    optimizer = torch.optim.Adam(
                                 lr=config['learning_rate'],
                                 params=model.parameters(),
                                 betas=[0.9, 0.98],
                                 eps=1e-12
                                 )

    # scheduler_custom
    scheduler = CustomizeLearningScheduler(
                                           optimizer=optimizer,
                                           dim_embed=config['d_model'],
                                           warmup_steps=config['warmup_steps'],
                                           verbose=True
                                           )

    # TensorBoard SummaryWriter 
    writer = SummaryWriter(output_path)

    # early stopping
    patience = config['patience']
    patience_count = 0 

    # learning rate 
    lrs = []

    # build dataset
    train_loader, val_loader = dataset.build_dataset(config)

    step = 0
    best_loss = 100000
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        tgt_len = 0 
        for batch_idx, (targets) in enumerate(tqdm(train_loader)):        
            tgt, tgt_mask = dataset(targets)
            tgt_len = max(tgt_len, tgt.shape[1])

            # initialize optimizer  
            optimizer.zero_grad()
    
            output = model(tgt=tgt[:, :-1], tgt_mask=tgt_mask[:, :, :-1, :-1])
            
            output1 = output.contiguous().view(-1, output.shape[-1])
            tgt1 = tgt[:, 1:].contiguous().view(-1)           
            loss = criterion(output1, tgt1)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip'])

            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            print('Epoch: {}, Step: {}%, Loss: {:.4f}'.format(epoch + 1, round((batch_idx / len(train_loader)) * 100, 2), loss.item()))
            writer.add_scalar('Loss/train', loss.item(), step)
            step += 1

        lrs.append(optimizer.param_groups[0]['lr'])
        train_loss = epoch_loss / len(train_loader)
        val_loss, bleu = decoder_only_evaluate(model, val_loader, dataset, criterion, BLEU_mode=config['BLEU_mode'])
        
        # Tensorboard
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print('val_loss : ', val_loss)
        
        # Save pt file
        if best_loss > val_loss and config.get('rank', 0) == 0:
            best_loss = min(best_loss, val_loss)
            if config['is_DDP']:
                torch.save(model.module.state_dict(), os.path.join(output_path, f'{epoch + 1}_{val_loss}.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(output_path, f'{epoch + 1}_{val_loss}.pt'))
            print('가중치 파일이 저장되었습니다.')
            patience_count = 0
        else:
            patience_count += 1
        
        with open(os.path.join(output_path, 'training_results.txt'), 'a') as f:
            f.write(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, BLEU Score: {bleu:.4f}\n')
        
        if patience_count >= patience:
            break

    writer.close()
    
    # Draw learning_rate graph
    draw_learning_rate(lrs, output_path)
    
    

if __name__ == "__main__":
    config = load_config("./config.yaml")
    
    for key, value in config.items():
        print(f"{key}: {value}")
    
    run(config)