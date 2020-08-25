import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import *
from network import *
from utils import *
from configs import *

import warnings
warnings.simplefilter('ignore')
seed_everything(RANDOM_SEED)


def inference(fold):

    print("#" * 50)
    print(f"############### Inferencing fold {fold}... ############")

    df_test = pd.read_csv("../input/test_fe.csv")

    meta_features = [
        'age_0', 'age_1', 'age_2', 'age_3', 'age_4', 'age_5', 'age_6', 'age_7', 'age_8', 'age_9', 
        'sex_0', 'sex_1', 'sex_2', 'anatom_0', 'anatom_1', 'anatom_2', 'anatom_3', 'anatom_4', 'anatom_5', 'anatom_6'
    ]

    testing_data_path = f"../input/jpeg-melanoma-{IMG_SIZE}x{IMG_SIZE}-without-hair/test/"
    test_dataset = MelanomaDataset(df=df_test, imfolder=testing_data_path, train=False, transforms=train_transform, meta_features=meta_features)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(depth=EFFICIENT_TYPE).to(device)
    es_path = f"../models/checkpoint_imgsize_{IMG_SIZE}_eff_b{EFFICIENT_TYPE}_fold_{fold}.pt"
    model.load_state_dict(torch.load(es_path))
    preds = torch.zeros((len(df_test), 1), dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        # Predicting on test set
        for tta_idx in range(TTA):
            test_dataset = MelanomaDataset(df=df_test, imfolder=testing_data_path, train=False, transforms=get_tta_transform(tta_idx), meta_features=meta_features)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
            for i, x_test in enumerate(test_loader):
                x_test[0] = torch.tensor(x_test[0], device=device, dtype=torch.float32)
                x_test[1] = torch.tensor(x_test[1], device=device, dtype=torch.float32)
                z_test = model(x_test)
                z_test = torch.sigmoid(z_test)
                preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
        preds /= TTA

    return preds.cpu().numpy().reshape(-1,)


if __name__ == "__main__":

    sub = pd.read_csv("../input/sample_submission.csv")

    for fold_i in range(5):
        sub['target'] += inference(fold_i) / 5
    PRED_NAME = F"pred_imgsize_{IMG_SIZE}_eff_b{EFFICIENT_TYPE}"
    sub.to_csv(f"../ensembles/preds/{PRED_NAME}.csv", index=False)
