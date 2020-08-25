import os
import time
import datetime
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from apex import amp
from catalyst.data.sampler import BalanceClassSampler

from dataset import *
from network import *
from utils import *
from configs import *

import warnings
warnings.simplefilter('ignore')
seed_everything(RANDOM_SEED)

def trainer(fold):

    logger.info("#" * 50)
    logger.info(f"############### Training fold {fold}... ###############")

    df = pd.read_csv("../input/train_folds_fe_plus_external.csv")
    df_train = df[df.kfold != fold].reset_index(drop=True)
    if not USE_2018:
        logger.info("############### Not using 2017+2018 external data ...")
        df_train = df_train[df_train.kfold != 2018].reset_index(drop=True)
    if not USE_2019:
        logger.info("############### Not using 2019 external data ...")
        df_train = df_train[df_train.kfold != 2019].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    meta_features = [
        'age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9', 
        'sex_0', 'sex_1', 'sex_2', 'anatom_0', 'anatom_1', 'anatom_2', 'anatom_3', 'anatom_4', 'anatom_5', 'anatom_6'
    ]

    training_data_path = f"../input/jpeg-melanoma-{IMG_SIZE}x{IMG_SIZE}-without-hair/train/"
    train_dataset = MelanomaDataset(df=df_train, imfolder=training_data_path, train=True, transforms=train_transform, meta_features=meta_features)
    sampler = BalanceClassSampler(labels=train_dataset.get_labels(), mode="downsampling")
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(depth=EFFICIENT_TYPE).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.4, min_lr=3e-7)
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss()

    model, optim = amp.initialize(model, optim, opt_level="O1", verbosity=0)

    es_path = f"../models/checkpoint_imgsize_{IMG_SIZE}_eff_b{EFFICIENT_TYPE}_fold_{fold}.pt"
    early_stopping = EarlyStopping(patience=ES_PATIENCE, verbose=True, path=es_path)

    for epoch in range(EPOCHS):
        start_time = time.time()
        correct = 0
        epoch_loss = 0

        model.train()
        for x, y in tqdm(train_loader):

            x[0] = torch.tensor(x[0], device=device, dtype=torch.float32)
            x[1] = torch.tensor(x[1], device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            y_smo = y.float() * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

            z = model(x)
            
            loss = criterion(z, y_smo.unsqueeze(1))

            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()

            optim.step()
            optim.zero_grad()
            pred = torch.round(torch.sigmoid(z))
            correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()
            epoch_loss += loss.item()
        train_acc = correct / len(df_train)

        model.eval()  # switch model to the evaluation mode
        valid_preds = torch.zeros((len(df_valid), 1), dtype=torch.float32, device=device)
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            for tta_idx in range(TTA):
                # Predicting on validation set
                valid_dataset = MelanomaDataset(df=df_valid, imfolder=training_data_path, train=True, transforms=get_tta_transform(tta_idx), meta_features=meta_features)
                valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False, pin_memory=True)

                for j, (x_valid, y_valid) in enumerate(valid_loader):
                    x_valid[0] = torch.tensor(x_valid[0], device=device, dtype=torch.float32)
                    x_valid[1] = torch.tensor(x_valid[1], device=device, dtype=torch.float32)
                    y_valid = torch.tensor(y_valid, device=device, dtype=torch.float32)
                    z_valid = model(x_valid)
                    valid_pred = torch.sigmoid(z_valid)
                    valid_preds[j*train_loader.batch_size:j*train_loader.batch_size + x_valid[0].shape[0]] += valid_pred
                valid_preds /= TTA

            valid_acc = accuracy_score(df_valid['target'].values, torch.round(valid_preds.cpu()))
            valid_roc = roc_auc_score(df_valid['target'].values, valid_preds.cpu())

            logger.info('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
                epoch + 1,
                epoch_loss,
                train_acc,
                valid_acc,
                valid_roc,
                str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

            scheduler.step(valid_roc)
            early_stopping(valid_roc, model, logger)
            if early_stopping.early_stop:
                logger.info("========== Early stopping...==========")
                break
    
    logger.info(f"===> The BEST AUC score in folder {fold} is: {early_stopping.best_score:.3f}")

    return valid_preds.cpu()

if __name__ == "__main__":

    FILE = f"../logs/log_imgsize_{IMG_SIZE}_eff_b{EFFICIENT_TYPE}.log"
    if not os.path.exists(FILE):
        os.mknod(FILE)
    logger = get_logger(FILE)

    df = pd.read_csv("../input/train_folds.csv")
    df["oof"] = 0

    for fold_i in range(5):
        df.loc[df.kfold == fold_i, "oof"] = trainer(fold_i)

    oof_auc = roc_auc_score(df.target, df.oof)
    logger.info(f"############## OOF AUC score is {oof_auc:.3f} ##############")
    OOF_NAME = f"oof_imgsize_{IMG_SIZE}_eff_b{EFFICIENT_TYPE}"
    df.to_csv(f"../ensembles/oofs/{OOF_NAME}.csv", index=None)