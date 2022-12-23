from dataset import *
from dataloader import *
from model import *
from prep import *

from tqdm import tqdm
import pandas as pd

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for img, tabular in tqdm(iter(test_loader)):
            img, tabular = img.to(device), tabular.to(device)

            model_preds = model(img, tabular)

            top_pred = model_preds.argmax(1, keepdim=True).squeeze(1).tolist()

            preds += top_pred

    return preds

def make_submission(preds, path):
    submit = pd.read_csv('/content/GDSC-1st-AIML-Study/Dacon/sample_submission.csv')
    submit['N_category'] = preds
    submit.to_csv(path, index=False)
