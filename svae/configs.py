import os
import sys
svae_path = 'SVAE_LIV/jax'
sys.path.insert(0,svae_path)
import datetime
from configparser import ConfigParser
from dataset import climate_dataset


def create_output_dir(dataset_name, saved_root):
    filename = "{}_{}".format(
        dataset_name, datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss"))
    save_dir = os.path.join(saved_root, filename)
    if not(os.path.isdir(save_dir)):
        os.makedirs(save_dir)
    print('save_dir:\n', save_dir)
    return save_dir

class myconf(ConfigParser):
    def __init__(self,defaults=None):
        ConfigParser.__init__(self, defaults=None)
    def optionxform(self, optionstr):
        return optionstr

def setup(config_file):
    cfg = myconf()
    config_path = os.path.abspath(config_file)
    print(f"Config file found at: {config_path}")
    cfg.read(config_file)
    save_dir = create_output_dir('sst', './saved_model')
    (train_dataloader, val_dataloader, train_num, val_num
    ) = climate_dataset.build_dataloader(cfg, save_dir)
    return cfg, save_dir, train_dataloader, val_dataloader, train_num, val_num




