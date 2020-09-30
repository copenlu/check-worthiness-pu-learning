import argparse
import gc
import os
import random
from typing import AnyStr
from copy import deepcopy

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import AdamW
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

from datareader import WikipediaCitationDataset
from datareader import PULearningWikipediaCitationDataset
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
        model_dir: AnyStr = "local"
) -> torch.nn.Module:
    best_loss = float('inf')
    best_f1 = 0
    patience_counter = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

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
            loss = loss_fn(logits.view(-1,2), labels.view(-1))
            loss = (loss * weights).mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

        gc.collect()

        # Inline evaluation
        (val_loss, acc, P, R, F1), _ = validation_evaluator.evaluate(model)

        # Saving the best model and early stopping
        if F1 > best_f1:
            best_model = model.state_dict()
            best_f1 = F1
            torch.save(model.state_dict(), f'{model_dir}/model.pth')
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
    parser.add_argument("--pos_file", help="Positive samples with citations", required=True, type=str)
    parser.add_argument("--neg_file", help="Unlabelled sentences", required=True, type=str)
    parser.add_argument("--train_pct", help="Percentage of data to use for training", type=float, default=0.9)
    parser.add_argument("--n_gpu", help="The number of GPUs to use", type=int, default=0)
    parser.add_argument("--log_interval", help="Number of steps to take between logging steps", type=int, default=1)
    parser.add_argument("--warmup_steps", help="Number of steps to warm up Adam", type=int, default=200)
    parser.add_argument("--n_epochs", help="Number of epochs", type=int, default=2)
    parser.add_argument("--test_files", nargs=2, type=str, action='append',
                        help="A pair of files (positive examples, negative examples) to test on with the final model")
    parser.add_argument("--pretrained_model", help="Model to use to get unlabelled sample weights", type=str, default=None)
    parser.add_argument("--seed", type=int, help="Random seed", default=1000)
    parser.add_argument("--model_dir", help="Where to store the saved model", default="local", type=str)

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
    lr = 2e-5
    weight_decay = 0.01
    n_epochs = args.n_epochs
    bert_config = BertConfig.from_pretrained(bert_model, num_labels=2)

    # Create the datasets
    base_dset = WikipediaCitationDataset(args.pos_file, args.neg_file, BertTokenizer.from_pretrained(bert_model))
    train_size = int(len(base_dset) * args.train_pct)
    val_size = len(base_dset) - train_size
    subsets = random_split(base_dset, [train_size, val_size])

    base_train_ds = subsets[0]

    val_ds = subsets[1]
    indices = val_ds.indices
    val_ds = deepcopy(val_ds.dataset)
    val_ds.dataset = val_ds.dataset.iloc[indices]
    val_ds.dataset = val_ds.dataset.reset_index(drop=True)
    validation_evaluator = ClassificationEvaluator(val_ds, device)
    # Create the base network from which we get the weights for unlabelled samples
    base_network = BertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)
    base_network.load_state_dict(torch.load(args.pretrained_model))
    # TODO Make sure this loads correctly
    train_ds = PULearningWikipediaCitationDataset(
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

    # Create the model to train
    model = BertForSequenceClassification.from_pretrained(bert_model, config=bert_config).to(device)

    # Create the optimizer
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, n_epochs * len(train_dl))

    # Create save directory for model
    if not os.path.exists(f"{args.model_dir}"):
        os.makedirs(f"{args.model_dir}")

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
        model_dir=args.model_dir
    )

    # Load the best weights
    model.load_state_dict(torch.load(f'{args.model_dir}/model.pth'))

    # Now test
    accs = []
    Ps = []
    Rs = []
    F1s = []
    labels_all = []
    logits_all = []
    for i, test_pair in enumerate(args.test_files):
        dset = WikipediaCitationDataset(test_pair[0], test_pair[1], BertTokenizer.from_pretrained(bert_model))
        evaluator = ClassificationEvaluator(dset, device)
        (loss, acc, P, R, F1), plots, (labels, logits) = evaluator.evaluate(
            model,
            plot_callbacks=[plot_label_distribution],
            return_labels_logits=True
        )
        accs.append(acc)
        Ps.append(P)
        Rs.append(R)
        F1s.append(F1)
        labels_all.append(labels)
        logits_all.append(logits)

        with open(f'{args.model_dir}/pred_lab.txt', 'a+') as f:
            for p,l in zip(np.argmax(logits, axis=-1), labels):
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

