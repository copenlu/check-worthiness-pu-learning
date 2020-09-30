import argparse
import gc
import os
import random
import glob
from typing import AnyStr
import bisect

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from datareader import PHEMEClassifierDataset
from datareader import collate_batch_transformer
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
        model_dir: str = "local",
        split: str = ''
):
    #best_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0

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

            loss, logits = model(input_ids, attention_mask=masks, labels=labels)

            loss.backward()
            optimizer.step()
            scheduler.step()

        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)

        # Saving the best model and early stopping
        if F1 > best_f1:
            best_model = model.state_dict()
            #best_loss = val_loss
            best_f1 = F1
            torch.save(model.state_dict(), f'{model_dir}/model_{split}.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            # Stop training once we have lost patience
            if patience_counter == patience:
                break

        gc.collect()


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pheme_dir", help="Directory of the PHEME dataset", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.8)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--pretrained_model", help="Weights to initialize the model with", type=str, default=None)
    parser.add_argument("--exclude_splits", nargs='+', help='A list of splits which should be ignored', default=[])
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--run_name", type=str, help="A name for the run", default="pheme-baseline")
    parser.add_argument("--model_dir", help="Where to store the saved model", default="local", type=str)
    parser.add_argument("--tags", nargs='+', help='A list of tags for this run', default=[])

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
    lr = 3e-5
    weight_decay = 0.01
    n_epochs = args.n_epochs
    bert_config = BertConfig.from_pretrained(bert_model, num_labels=2)

    # Create the datasets
    all_dsets = [PHEMEClassifierDataset(topic_dir, BertTokenizer.from_pretrained(bert_model))
            for topic_dir in glob.glob(f'{args.pheme_dir}/**') if not any([exc in topic_dir for exc in args.exclude_splits])]
    accs = []
    Ps = []
    Rs = []
    F1s = []
    # Store labels and logits for individual splits for micro F1
    labels_all = []
    logits_all = []
    #Create save directory for model
    if not os.path.exists(f"{args.model_dir}"):
        os.makedirs(f"{args.model_dir}")

    for i in range(len(all_dsets)):
        test_dset = all_dsets[i]
        dset = ConcatDataset([all_dsets[j] for j in range(len(all_dsets)) if j != i])

        train_size = int(len(dset) * args.train_pct)
        val_size = len(dset) - train_size
        subsets = random_split(dset, [train_size, val_size])

        # Save the indices
        with open(f'{args.model_dir}/train_idx_{test_dset.name}.txt', 'wt') as f, \
                open(f'{args.model_dir}/val_idx_{test_dset.name}.txt', 'wt') as g:
            for idx in subsets[0].indices:
                dataset_idx = bisect.bisect_right(dset.cumulative_sizes, idx)
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - dset.cumulative_sizes[dataset_idx - 1]
                f.write(f'{dataset_idx},{sample_idx}\n')
            for idx in subsets[1].indices:
                dataset_idx = bisect.bisect_right(dset.cumulative_sizes, idx)
                if dataset_idx == 0:
                    sample_idx = idx
                else:
                    sample_idx = idx - dset.cumulative_sizes[dataset_idx - 1]
                g.write(f'{dataset_idx},{sample_idx}\n')

        train_ds = subsets[0]
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_batch_transformer)

        val_ds = subsets[1]
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
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, n_epochs * len(train_dl))

        # Train
        train(
            model,
            train_dl,
            optimizer,
            scheduler,
            validation_evaluator,
            n_epochs,
            device,
            args.log_interval,
            model_dir=args.model_dir,
            split=test_dset.name
        )

        # Load the best weights
        model.load_state_dict(torch.load(f'{args.model_dir}/model_{test_dset.name}.pth'))

        evaluator = ClassificationEvaluator(test_dset, device)
        (loss, acc, P, R, F1), plots, (labels, logits) = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True
        )
        print(f"{test_dset.name} acc: {acc}")
        print(f"{test_dset.name} P: {P}")
        print(f"{test_dset.name} R: {R}")
        print(f"{test_dset.name} F1: {F1}")
        accs.append(acc)
        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        labels_all.extend(labels)
        logits_all.extend(logits)
        with open(f'{args.model_dir}/pred_lab.txt', 'a+') as f:
            for p,l in zip(np.argmax(logits, axis=-1), labels):
                f.write(f'{i}\t{p}\t{l}\n')

    print(f"Macro avg accuracy: {sum(accs) / len(accs)}")
    print(f"Macro avg P: {sum(Ps) / len(Ps)}")
    print(f"Macro avg R: {sum(Rs) / len(Rs)}")
    print(f"Macro avg F1: {sum(F1s) / len(F1s)}")

    acc, P, R, F1 = acc_f1(logits_all, labels_all)

    print(f"Micro avg accuracy: {acc}")
    print(f"Micro avg P: {P}")
    print(f"Micro avg R: {R}")
    print(f"Micro avg F1: {F1}")
