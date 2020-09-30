import argparse
import gc
import random
import os
import subprocess
import re
from typing import AnyStr
from pathlib import Path
from copy import deepcopy
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import glob
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from datareader import ClefClassifierDataset
from datareader import PULearningPHEMEDataset
from datareader import collate_batch_transformer_with_weight
from metrics import ClassificationEvaluator
from metrics import plot_label_distribution
from metrics import acc_f1


def train(
        model: torch.nn.Module,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        validation_evaluator: ClassificationEvaluator,
        n_epochs: int,
        device: AnyStr,
        log_interval: int = 1,
        patience: int = 10,
        neg_class_weight: float = None,
        model_dir: str = "local",
        split: str = ''
) -> torch.nn.Module:
    best_loss = float('inf')
    patience_counter = 0
    best_f1 = 0.0
    weights_found = False
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_class_weight, 1.]).to(device))

    # Main loop
    for ep in range(n_epochs):
        # Training loop
        for i, batch in enumerate(tqdm(train_dl)):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            weights = batch[3]

            (logits,) = model(input_ids, attention_mask=masks)
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
            # loss = (loss * weights).sum()
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()
        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)

        # Saving the best model and early stopping
        if F1 > best_f1:
            weights_found = True
            best_model = model.state_dict()
            # best_loss = val_loss
            best_f1 = F1
            torch.save(model.state_dict(), f'{model_dir}/model_{split}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            # Stop training once we have lost patience
            if patience_counter == patience:
                break

    if weights_found == False:
        print("No good weights found, saving weights from last epoch")
        # Save one just in case
        torch.save(model.state_dict(), f'{model_dir}/model_{split}.pth')

    gc.collect()
    return best_f1


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir", help="Directory with Clef test data", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.9)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--pretrained_model", help="Weights to initialize the model with", type=str, default=None)
    parser.add_argument("--clef_test_script", help="Location of the Clef test executable", type=str, required=True)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--model_dir", help="Where to store the saved model", default="local", type=str)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])
    parser.add_argument("--lr", help="Learning rate", type=float, default=2e-5)
    parser.add_argument("--pretrained_clef_model", help="Weights to use for PU learning", type=str, default=None)
    parser.add_argument("--indices_dir", help="If standard splits are being used", type=str, default=None)

    args = parser.parse_args()
    # Set all the seeds
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # See if CUDA available
    device = torch.device("cpu")
    if args.n_gpu > 0 and torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:0")

    # model configuration
    bert_model = 'bert-base-uncased'
    batch_size = 8
    lr = args.lr
    weight_decay = 0.01
    n_epochs = args.n_epochs
    bert_config = BertConfig.from_pretrained(bert_model, num_labels=2)
    splits_dir = Path(args.split_dir)
    all_dsets = [ClefClassifierDataset(split_file,
                                       BertTokenizer.from_pretrained(bert_model),
                                       split_file.name)
                 for split_file in list(splits_dir.glob(f'*.tsv')) + list(splits_dir.glob(f'*.txt'))]
    accs = []
    Ps = []
    Rs = []
    F1s = []
    APs = []
    RRs = []
    best_f1s = []
    # Store labels and logits for individual splits for micro F1
    labels_all = []
    logits_all = []

    # Create save directory for model
    if not os.path.exists(f"{args.model_dir}"):
        os.makedirs(f"{args.model_dir}")

    for i in range(len(all_dsets)):
        test_dset = all_dsets[i]
        if args.indices_dir is None:
            dataset = pd.concat([deepcopy(ds.dataset) for j, ds in enumerate(all_dsets) if j != i])
            # Just need some pheme dataset
            dset = deepcopy(all_dsets[0])
            dset.dataset = dataset
            dset.name = '_'.join([ds.name for j, ds in enumerate(all_dsets) if j != i])
            train_size = int(len(dset) * args.train_pct)
            val_size = len(dset) - train_size
            subsets = random_split(dset, [train_size, val_size])
            base_train_ds = subsets[0]
            val_ds = subsets[1]
            indices = val_ds.indices
            val_ds = deepcopy(val_ds.dataset)
            val_ds.dataset = val_ds.dataset.iloc[indices]
            val_ds.dataset = val_ds.dataset.reset_index(drop=True)
        else:
            # load the indices
            dset_choices = [all_dsets[j] for j in range(len(all_dsets)) if j != i]
            subset_indices = defaultdict(lambda: [[], []])
            with open(f'{args.indices_dir}/train_idx_{test_dset.name}.txt') as f, \
                    open(f'{args.indices_dir}/val_idx_{test_dset.name}.txt') as g:
                for l in f:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][0].append(int(vals[1]))
                for l in g:
                    vals = l.strip().split(',')
                    subset_indices[int(vals[0])][1].append(int(vals[1]))
            train_dataset = pd.concat([dset_choices[d].dataset.iloc[subset_indices[d][0]] for d in subset_indices])
            val_dataset = pd.concat([dset_choices[d].dataset.iloc[subset_indices[d][1]] for d in subset_indices])
            base_train_ds = deepcopy(all_dsets[0])
            base_train_ds.dataset = train_dataset
            base_train_ds.name = '_'.join([ds.name for j, ds in enumerate(all_dsets) if j != i])
            base_train_ds.dataset = base_train_ds.dataset.reset_index(drop=True)

            val_ds = deepcopy(all_dsets[0])
            val_ds.dataset = val_dataset
            val_ds.name = '_'.join([ds.name for j, ds in enumerate(all_dsets) if j != i])
            val_ds.dataset = val_ds.dataset.reset_index(drop=True)
        validation_evaluator = ClassificationEvaluator(val_ds, device)
        base_network = BertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
        base_network.load_state_dict(torch.load(f"{args.pretrained_clef_model}/model_{test_dset.name}.pth"))
        train_ds = PULearningPHEMEDataset(
            base_train_ds,
            val_ds,
            base_network,
            device
        )

        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch_transformer_with_weight
        )
        validation_evaluator = ClassificationEvaluator(val_ds, device)

        # Create the model
        model = BertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
        if args.pretrained_model is not None:
            weights = {k: v for k, v in torch.load(args.pretrained_model).items() if "classifier" not in k}
            model_dict = model.state_dict()
            model_dict.update(weights)
            model.load_state_dict(model_dict)

        # Create the optimizer
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, n_epochs * len(train_dl))

        # Stupid hack: iterate through the dl to get the number of negative samples
        print("Getting number of negative sampeles for weighting")
        n_negs = 0
        for batch in tqdm(train_dl):
            n_negs += sum(batch[2] == 0)
        n_negs = n_negs.cpu().item()
        neg_class_weight = (len(train_ds) - n_negs) / n_negs
        # Train
        best_f1 = train(
            model,
            train_dl,
            optimizer,
            scheduler,
            validation_evaluator,
            n_epochs,
            device,
            args.log_interval,
            neg_class_weight=neg_class_weight,
            model_dir=args.model_dir,
            split=test_dset.name
        )
        best_f1s.append(best_f1)

        # Load the best weights
        model.load_state_dict(torch.load(f'{args.model_dir}/model_{test_dset.name}.pth'))

        evaluator = ClassificationEvaluator(test_dset, device)
        (loss, acc, P, R, F1), plots, (labels, logits) = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True
        )

        # Get the MAP
        scores = np.asarray(logits)[:, 1]
        test_preds_name = f"{test_dset.name[:-4]}_preds.tsv"
        with open(f"{args.model_dir}/{test_preds_name}", 'w') as f:
            for i,score in enumerate(scores):
                f.write(f"{i+1}\t{score:.20f}\n")
        results = subprocess.check_output( f"python {args.clef_test_script} "
                                           f"--gold_file_path={args.split_dir}/{test_dset.name}"
                                           f" --pred_file_path={args.model_dir}/{test_preds_name}",
                                          stderr=subprocess.STDOUT,
                                          shell=True)
        results = results.decode("utf-8")
        with open(f"{args.model_dir}/{test_dset.name[:-4]}_results.txt", 'w') as f:
            f.write(results)

        lines = results.split('\n')
        AP = [l for l in lines if 'AVERAGE PRECISION:' in l][0]
        RR = [l for l in lines if 'RECIPROCAL RANK:' in l][0]

        accs.append(acc)
        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        APs.append(float(re.findall("\d+\.\d+", AP)[0]))
        RRs.append(float(re.findall("\d+\.\d+", RR)[0]))
        labels_all.extend(labels)
        logits_all.extend(logits)
        with open(f'{args.model_dir}/{test_dset.name[:-4]}_pred_lab.txt', 'a+') as f:
            for p, l in zip(np.argmax(logits, axis=-1), labels):
                f.write(f'{i}\t{p}\t{l}\n')

    print(f"Macro avg accuracy: {sum(accs) / len(accs)}")
    print(f"Macro avg P: {sum(Ps) / len(Ps)}")
    print(f"Macro avg R: {sum(Rs) / len(Rs)}")
    print(f"Macro avg F1: {sum(F1s) / len(F1s)}")

    acc, P, R, F1 = acc_f1([l for run in logits_all for l in run], [l for run in labels_all for l in run])

    print(f"Micro avg accuracy: {acc}")
    print(f"Micro avg P: {P}")
    print(f"Micro avg R: {R}")
    print(f"Micro avg F1: {F1}")

    print(f"MAP: {sum(APs) / len(APs)}")
    print(f"MRR: {sum(RRs) / len(RRs)}")
