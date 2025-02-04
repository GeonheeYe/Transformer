import torch 

def transformer_evaluate(model, val_loader, dataset, criterion, BLEU_mode=False):
    model.eval()
    epoch_loss = 0 
    batch_bleu = []
    with torch.no_grad(): 
        for batch_idx, (srcs, targets) in enumerate(val_loader):
            src, tgt, src_mask, tgt_mask = dataset(srcs, targets)
            output = model(src=src, tgt=tgt[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask[:, :, :-1, :-1])
        
            output1 = output.contiguous().view(-1, output.shape[-1])
            tgt1 = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output1, tgt1)
            epoch_loss += loss.item()
            
            decoded_output = dataset.tokenizer.batch_decode(output.argmax(dim=-1), skip_special_tokens=True)
            decoded_target = dataset.tokenizer.batch_decode(tgt, skip_special_tokens=True)
        
            total_bleu = []
            if BLEU_mode:
                target = [target.split() for target in decoded_target]
                output = [output.split() for output in decoded_output]
                for i in range(len(target)):
                    bleu = BLEU(output[i], [target[i]], 4)
                    total_bleu.append(bleu)

                total_bleu = sum(total_bleu) / len(target)
                batch_bleu.append(total_bleu)

    if BLEU_mode:
        batch_bleu = sum(batch_bleu) / len(batch_bleu)
    else:
        batch_bleu = 0

    if batch_bleu < 0.01:
        batch_bleu = 0
    return epoch_loss / len(val_loader), batch_bleu

def decoder_only_evaluate(model, val_loader, dataset, criterion, BLEU_mode=False):
    model.eval()
    epoch_loss = 0 
    batch_bleu = []
    with torch.no_grad(): 
        for batch_idx, (targets) in enumerate(val_loader):
            tgt, tgt_mask = dataset(targets)
            output = model(tgt=tgt[:, :-1], tgt_mask=tgt_mask[:, :, :-1, :-1])
        
            output1 = output.contiguous().view(-1, output.shape[-1])
            tgt1 = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output1, tgt1)
            epoch_loss += loss.item()
            
            decoded_output = dataset.tokenizer.batch_decode(output.argmax(dim=-1), skip_special_tokens=True)
            decoded_target = dataset.tokenizer.batch_decode(tgt, skip_special_tokens=True)
        
            total_bleu = []
            if BLEU_mode:
                target = [target.split() for target in decoded_target]
                output = [output.split() for output in decoded_output]
                for i in range(len(target)):
                    bleu = BLEU(output[i], [target[i]], 4)
                    total_bleu.append(bleu)

                total_bleu = sum(total_bleu) / len(target)
                batch_bleu.append(total_bleu)

    if BLEU_mode:
        batch_bleu = sum(batch_bleu) / len(batch_bleu)
    else:
        batch_bleu = 0

    if batch_bleu < 0.01:
        batch_bleu = 0
    return epoch_loss / len(val_loader), batch_bleu
