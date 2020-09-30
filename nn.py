import torch
import numpy as np
from torch import nn


class NonNegativePULoss(nn.Module):

    def __init__(self, prior: float, beta: float = 0., gamma: float = 1.0):
        """Non-Negative positive unlabelled risk estimator. This code is highly adapted from
        https://github.com/kiryor/nnPUlearning/blob/master/pu_loss.py

        :param prior:
        :param beta:
        """
        super(NonNegativePULoss, self).__init__()

        self.beta = beta
        self.prior = prior
        self.gamma = gamma
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits: torch.FloatTensor, labels: torch.IntTensor):
        positive = (labels == 1.).type(torch.cuda.FloatTensor)
        n_positive = max(1, positive.sum().item())

        unlabelled = (labels == 0.).type(torch.cuda.FloatTensor)
        n_unlabelled = max(1, unlabelled.sum().item())

        base_loss = self.loss_fn(logits, labels)
        reverse_loss = self.loss_fn(logits, 0*labels)

        ### This is the basic unbiased risk estimator from du Plessis et al. in NeurIPS 2014
        # Gets the loss of positive samples weighed by the prior probability and proportion of positives in batch
        positive_loss = torch.sum(self.prior * positive / n_positive * base_loss)
        # 1/n_unlabelled * l(u,0) - pi/n_positive * l(p,0) --
        #   I'm still trying to understand how (and why) this would work???
        negative_loss = torch.sum((unlabelled / n_unlabelled - self.prior * positive / n_positive) * reverse_loss)

        ### Here is the non-negative trick that Kiryo et al. introduced in NeurIPS 2017 for flexible estimators
        ### (i.e. neural nets)
        if negative_loss < -self.beta:
            #loss = -self.beta + positive_loss
            loss = -self.gamma * negative_loss
        else:
            loss = positive_loss + negative_loss

        return loss


class BiLSTMNetwork(nn.Module):
    """
    Basic BiLSTM network
    """
    def __init__(
            self,
            pretrained_embeddings: np.ndarray,
            lstm_dim: int,
            dropout_prob: float = 0.1
    ):
        super(BiLSTMNetwork, self).__init__()
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0),
            'bilstm': nn.LSTM(
                pretrained_embeddings.shape[1],
                lstm_dim,
                2,
                batch_first=True,
                dropout=dropout_prob,
                bidirectional=True),
            'ff': nn.Linear(2*lstm_dim, 2)
        })

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['ff'].named_parameters())
        for n,p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens, labels = None):
        # Get embeddings
        embeds = self.model['embeddings'](inputs)
        # Pack padded
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens,
            batch_first=True,
            enforce_sorted=False
        )
        lstm_out, hidden = self.model['bilstm'](lstm_in)
        lstm_out,_ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Get the last output for classification
        ff_in = lstm_out.gather(1, input_lens.view(-1,1,1).expand(lstm_out.size(0), 1, lstm_out.size(2)) - 1).squeeze()
        #ff_in = ff_in.view(-1, lstm_out.size(2))
        logits = self.model['ff'](ff_in).view(-1, 2)
        outputs = (logits,)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs = (loss,) + outputs
        return outputs
