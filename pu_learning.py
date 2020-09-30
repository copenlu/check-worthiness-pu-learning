import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm


eps = 10e-6


def estimate_c(model: torch.nn.Module, dl: DataLoader, device):
    val_probs = []
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dl, desc="Estimating p(s=1|y=1)"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            loss, logits = model(input_ids, attention_mask=masks, labels=labels)
            probs = torch.nn.Softmax(dim=-1)(logits).cpu().numpy()
            pos = labels.cpu().numpy() == 1
            val_probs.extend(list(probs[pos][:, 1]))
    # Estimate of c is the average of p(s=1|x) from the positive examples from validation set
    return sum(val_probs) / len(val_probs)

def estimate_class_prior_probability(model: torch.nn.Module, train_dl: DataLoader, val_dl: DataLoader, device):
    model.eval()
    total = 0.
    n_samples = 0
    c_estimate = estimate_c(model, val_dl, device)
    with torch.no_grad():
        for batch in tqdm(train_dl, desc="Estimating p(y=1)"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            loss, logits = model(input_ids, attention_mask=masks, labels=labels)
            probs = torch.nn.Softmax(dim=-1)(logits).cpu().numpy()[:, 1]
            # Correct for probabilities greater than our estimate of c
            probs[probs > c_estimate] = c_estimate - eps

            pos = sum(labels.cpu().numpy() == 1)
            unlab = (labels == 0).cpu().numpy()
            weights_batch = ((1 - c_estimate) / c_estimate) * (probs[unlab] / (1 - probs[unlab]))
            total += pos + weights_batch.sum()
            n_samples += labels.shape[0]

    return total / n_samples


def get_negative_sample_weights(
        dl: DataLoader,
        val_dl: DataLoader,
        base_network: torch.nn.Module,
        device: torch.device
) -> np.ndarray:
    """Gets the weights for the unlabelled samples in the dataset based on a pretrained network

    :param base_network: The network which will estimate the weights for unlabelled samples
    :param: dl: The base dataset to use for running samples through the model
    :param: val_dl: The validation dataset from which we estimate p(s=1|y=1)
    :param: device: The device to run the model on
    :return: A list of weights for unlabelled samples
    """
    # Create the dataloader
    weights = []
    c_estimate = estimate_c(base_network, val_dl, device)
    base_network.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Getting weights for unlabelled samples"):
            batch = tuple(t.to(device) for t in batch)
            input_ids = batch[0]
            masks = batch[1]
            labels = batch[2]
            loss, logits = base_network(input_ids, attention_mask=masks, labels=labels)
            probs = torch.nn.Softmax(dim=-1)(logits).cpu().numpy()[:, 1]
            # Correct for probabilities greater than our estimate of c
            probs[probs > c_estimate] = c_estimate - eps
            # Calculate the weights
            weights_batch = ((1 - c_estimate) / c_estimate) * (probs / (1 - probs))
            weights_batch = np.concatenate([1 - weights_batch[:, np.newaxis], weights_batch[:, np.newaxis]],
                                           axis=1)
            # Ensure weights valid
            assert (weights_batch <= 1).all() and (weights_batch >= 0).all(), "Invalid weights"
            weights.extend(list(weights_batch))
    weights = np.asarray(weights)
    return weights