import random, torch, os, numpy
import config

def save_checkpoint(filename, model, optimizer):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer = None, lr = None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def init_seeds(seed=0):
    print(f'seed = {seed}')
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
mkdir('outputs3/')