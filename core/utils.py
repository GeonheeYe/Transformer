import yaml
from torch import nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler
import os 


# yaml file load
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# initiallize weights
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)
    

# number sort and return last number 
def extract_number(filename):
    parts = fileame.split('_')
    if parts[0].isdigit():
        return int(parts[0])
    return None


def draw_learning_rate(lrs, outputs):
    plt.plot(lrs)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(outputs, 'learning_rate.jpg'))


# custormizing learning_scheduler
class CustomizeLearningScheduler(_LRScheduler):
    def __init__(self, optimizer, dim_embed, warmup_steps, verbose=False):       
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)
        super().__init__(optimizer, verbose=verbose)
        
    def get_lr(self):
        lr = self.calc_lr(self.last_epoch + 1, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def calc_lr(self, epoch, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(epoch ** (-0.5), epoch * warmup_steps ** (-1.5))
