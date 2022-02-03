# https://www.kaggle.com/polomarco/brats20-3dunet-3dautoencoder
from tqdm import tqdm
import os
import time
from random import randint

import numpy as np
from scipy import stats
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold

import nibabel as nib
import pydicom as pdm
import nilearn as nl
import nilearn.plotting as nlplt
import h5py

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as anim

import seaborn as sns
import imageio
from skimage.transform import resize
from skimage.util import montage

from IPython.display import Image as show_gif
from IPython.display import clear_output
from IPython.display import YouTubeVideo

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


import albumentations as A
from albumentations import Compose, HorizontalFlip
from albumentations.pytorch import ToTensor#, ToTensorV2 

import warnings
warnings.simplefilter("ignore")

from utils.plots import plot_data_overview
from utils.torch_dataset import BratsDataset, get_dataloader
from utils.metric_and_losses import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes, Meter, BCEDiceLoss 
from utils.Unet import UNet3d 

if __name__ == "__main__":
    path_source = '/content/drive/MyDrive/Datasets/brats20-dataset-training-validation'

    sample_filename = f'{path_source}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii'
    sample_filename_mask = f'{path_source}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii'

    sample_img = nib.load(sample_filename)
    sample_img = np.asanyarray(sample_img.dataobj)
    sample_img = np.rot90(sample_img)
    sample_mask = nib.load(sample_filename_mask)
    sample_mask = np.asanyarray(sample_mask.dataobj)
    sample_mask = np.rot90(sample_mask)
    print("img shape ->", sample_img.shape)
    print("mask shape ->", sample_mask.shape)

    plot_data_overview(path_source, sample_img, sample_mask)

    class GlobalConfig:
        root_dir = path_source
        train_root_dir = f'{path_source}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
        test_root_dir = f'{path_source}/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
        path_to_csv = './train_data.csv'
        pretrained_model_path = 'brats2020logs/unet/last_epoch_model.pth'
        train_logs_path = 'brats2020logs/unet/train_log.csv'
        ae_pretrained_model_path = 'brats2020logs/ae/autoencoder_best_model.pth'
        tab_data = 'brats2020logs/data/df_with_voxel_stats_and_latent_features.csv'
        seed = 0
        
    def seed_everything(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
    config = GlobalConfig()
    seed_everything(config.seed)

    survival_info_df = pd.read_csv(f'{path_source}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
    name_mapping_df = pd.read_csv(f'{path_source}/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv')

    name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 

    df = survival_info_df.merge(name_mapping_df, on="Brats20ID", how="right")

    paths = []
    for _, row  in df.iterrows():
        
        id_ = row['Brats20ID']
        phase = id_.split("_")[-2]
        
        if phase == 'Training':
            path = os.path.join(config.train_root_dir, id_)
        else:
            path = os.path.join(config.test_root_dir, id_)
        paths.append(path)
        
    df['path'] = paths

    #split data on train, test, split
    #train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=69, shuffle=True)
    #train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
    train_data["Age_rank"] =  train_data["Age"] // 10 * 10
    train_data = train_data.loc[train_data['Brats20ID'] != 'BraTS20_Training_355'].reset_index(drop=True, )

    skf = StratifiedKFold(
        n_splits=7, random_state=config.seed, shuffle=True
    )
    for i, (train_index, val_index) in enumerate(
            skf.split(train_data, train_data["Age_rank"])
            ):
            train_data.loc[val_index, "fold"] = i

    train_df = train_data.loc[train_data['fold'] != 0].reset_index(drop=True)
    val_df = train_data.loc[train_data['fold'] == 0].reset_index(drop=True)

    test_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
    print("train_df ->", train_df.shape, "val_df ->", val_df.shape, "test_df ->", test_df.shape)
    train_data.to_csv("train_data.csv", index=False)

    dataloader = get_dataloader(dataset=BratsDataset, path_to_csv='train_data.csv', phase='valid', fold=0)
    len(dataloader)


    data = next(iter(dataloader))
    data['Id'], data['image'].shape, data['mask'].shape

    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()
    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor)) 

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask == False, mask),
            cmap='cool', alpha=0.6)

    class Trainer:
        """
        Factory for training proccess.
        Args:
            display_plot: if True - plot train history after each epoch.
            net: neural network for mask prediction.
            criterion: factory for calculating objective loss.
            optimizer: optimizer for weights updating.
            phases: list with train and validation phases.
            dataloaders: dict with data loaders for train and val phases.
            path_to_csv: path to csv file.
            meter: factory for storing and updating metrics.
            batch_size: data batch size for one step weights updating.
            num_epochs: num weights updation for all data.
            accumulation_steps: the number of steps after which the optimization step can be taken
                        (https://www.kaggle.com/c/understanding_cloud_organization/discussion/105614).
            lr: learning rate for optimizer.
            scheduler: scheduler for control learning rate.
            losses: dict for storing lists with losses for each phase.
            jaccard_scores: dict for storing lists with jaccard scores for each phase.
            dice_scores: dict for storing lists with dice scores for each phase.
        """
        def __init__(self,
                    net: nn.Module,
                    dataset: torch.utils.data.Dataset,
                    criterion: nn.Module,
                    lr: float,
                    accumulation_steps: int,
                    batch_size: int,
                    fold: int,
                    num_epochs: int,
                    path_to_csv: str,
                    display_plot: bool = True,
                    ):

            """Initialization."""
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("device:", self.device)
            self.display_plot = display_plot
            self.net = net
            self.net = self.net.to(self.device)
            self.criterion = criterion
            self.optimizer = Adam(self.net.parameters(), lr=lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min",
                                            patience=2, verbose=True)
            self.accumulation_steps = accumulation_steps // batch_size
            self.phases = ["train", "val"]
            self.num_epochs = num_epochs

            self.dataloaders = {
                phase: get_dataloader(
                    dataset = dataset,
                    path_to_csv = path_to_csv,
                    phase = phase,
                    fold = fold,
                    batch_size = batch_size,
                    num_workers = 4
                )
                for phase in self.phases
            }
            self.best_loss = float("inf")
            self.losses = {phase: [] for phase in self.phases}
            self.dice_scores = {phase: [] for phase in self.phases}
            self.jaccard_scores = {phase: [] for phase in self.phases}
            
        def _compute_loss_and_outputs(self,
                                    images: torch.Tensor,
                                    targets: torch.Tensor):
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.net(images)
            loss = self.criterion(logits, targets)
            return loss, logits
            
        def _do_epoch(self, epoch: int, phase: str):
            print(f"{phase} epoch: {epoch} | time: {time.strftime('%H:%M:%S')}")

            self.net.train() if phase == "train" else self.net.eval()
            meter = Meter()
            dataloader = self.dataloaders[phase]
            total_batches = len(dataloader)
            running_loss = 0.0
            self.optimizer.zero_grad()
            for itr, data_batch in tqdm(enumerate(dataloader), leave=False, total=len(dataloader)):
                images, targets = data_batch['image'], data_batch['mask']
                loss, logits = self._compute_loss_and_outputs(images, targets)
                loss = loss / self.accumulation_steps
                if phase == "train":
                    loss.backward()
                    if (itr + 1) % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                running_loss += loss.item()
                meter.update(logits.detach().cpu(),
                            targets.detach().cpu()
                            )
                
            epoch_loss = (running_loss * self.accumulation_steps) / total_batches
            epoch_dice, epoch_iou = meter.get_metrics()
            
            self.losses[phase].append(epoch_loss)
            self.dice_scores[phase].append(epoch_dice)
            self.jaccard_scores[phase].append(epoch_iou)

            return epoch_loss
            
        def run(self):
            for epoch in range(self.num_epochs):
                self._do_epoch(epoch, "train")
                with torch.no_grad():
                    val_loss = self._do_epoch(epoch, "val")
                    self.scheduler.step(val_loss)
                if self.display_plot:
                    self._plot_train_history()
                    
                if val_loss < self.best_loss:
                    print(f"\n{'#'*20}\nSaved new checkpoint\n{'#'*20}\n")
                    self.best_loss = val_loss
                    torch.save(self.net.state_dict(), "best_model.pth")
                print()
            self._save_train_history()
                
        def _plot_train_history(self):
            data = [self.losses, self.dice_scores, self.jaccard_scores]
            colors = ['deepskyblue', "crimson"]
            labels = [
                f"""
                train loss {self.losses['train'][-1]}
                val loss {self.losses['val'][-1]}
                """,
                
                f"""
                train dice score {self.dice_scores['train'][-1]}
                val dice score {self.dice_scores['val'][-1]} 
                """, 
                    
                f"""
                train jaccard score {self.jaccard_scores['train'][-1]}
                val jaccard score {self.jaccard_scores['val'][-1]}
                """,
            ]
            
            clear_output(True)
            with plt.style.context("seaborn-dark-palette"):
                fig, axes = plt.subplots(3, 1, figsize=(8, 10))
                for i, ax in enumerate(axes):
                    ax.plot(data[i]['val'], c=colors[0], label="val")
                    ax.plot(data[i]['train'], c=colors[-1], label="train")
                    ax.set_title(labels[i])
                    ax.legend(loc="upper right")
                    
                plt.tight_layout()
                plt.show()
                
        def load_predtrain_model(self,
                                state_path: str):
            self.net.load_state_dict(torch.load(state_path))
            print("Predtrain model loaded")
            
        def _save_train_history(self):
            """writing model weights and training logs to files."""
            torch.save(self.net.state_dict(),
                    f"last_epoch_model.pth")

            logs_ = [self.losses, self.dice_scores, self.jaccard_scores]
            log_names_ = ["_loss", "_dice", "_jaccard"]
            logs = [logs_[i][key] for i in list(range(len(logs_)))
                            for key in logs_[i]]
            log_names = [key+log_names_[i] 
                        for i in list(range(len(logs_))) 
                        for key in logs_[i]
                        ]
            pd.DataFrame(
                dict(zip(log_names, logs))
            ).to_csv("train_log.csv", index=False)

    model = UNet3d(in_channels=4, n_classes=3, n_channels=8).to('cuda')

    trainer = Trainer(net=model,
                    dataset=BratsDataset,
                    criterion=BCEDiceLoss(),
                    lr=5e-4,
                    accumulation_steps=4,
                    batch_size=1,
                    fold=0,
                    num_epochs=1,
                    path_to_csv = config.path_to_csv,)

    if config.pretrained_model_path is not None:
        trainer.load_predtrain_model(config.pretrained_model_path)
        
        # if need - load the logs.      
        train_logs = pd.read_csv(config.train_logs_path)
        trainer.losses["train"] =  train_logs.loc[:, "train_loss"].to_list()
        trainer.losses["val"] =  train_logs.loc[:, "val_loss"].to_list()
        trainer.dice_scores["train"] = train_logs.loc[:, "train_dice"].to_list()
        trainer.dice_scores["val"] = train_logs.loc[:, "val_dice"].to_list()
        trainer.jaccard_scores["train"] = train_logs.loc[:, "train_jaccard"].to_list()
        trainer.jaccard_scores["val"] = train_logs.loc[:, "val_jaccard"].to_list()

    print('training')
    trainer.run()

    print('finished')

    torch.cuda.current_device()
    torch.cuda.get_device_name(0)