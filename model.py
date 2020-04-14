import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNText(nn.Module):
    def __init__(self, vocab_size, sentence_len, pretrained_embeddings, output_dim=6, mode="static"):
        super(CNNText, self).__init__()
        kernel_sizes = [3, 4, 5]
        num_filters = 100
        embedding_dim = 300

        self.vars = nn.ParameterList()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = (mode == "nonstatic")

        self.relu = nn.ReLU()

        # block 1
        kernel_size = kernel_sizes[0]
        maxpool_kernel_size = sentence_len - kernel_size + 1
        conv1d_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size,
                             stride=1)
        self.maxpool_1 = nn.MaxPool1d(kernel_size=maxpool_kernel_size)
        self.vars.append(nn.Parameter(conv1d_1.weight))
        self.vars.append(nn.Parameter(conv1d_1.bias))

        # block 2
        kernel_size = kernel_sizes[1]
        maxpool_kernel_size = sentence_len - kernel_size + 1
        conv1d_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size,
                             stride=1)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=maxpool_kernel_size)
        self.vars.append(nn.Parameter(conv1d_2.weight))
        self.vars.append(nn.Parameter(conv1d_2.bias))

        # block 3
        kernel_size = kernel_sizes[1]
        maxpool_kernel_size = sentence_len - kernel_size + 1
        conv1d_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size,
                             stride=1)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=maxpool_kernel_size)
        self.vars.append(nn.Parameter(conv1d_3.weight))
        self.vars.append(nn.Parameter(conv1d_3.bias))

        fc = nn.Linear(num_filters * len(kernel_sizes), output_dim)
        self.vars.append(nn.Parameter(fc.weight))
        self.vars.append(nn.Parameter(fc.bias))

    def forward(self, x, vars=None):  # x: (batch, sentence_len)
        x = self.embedding(x)  # embedded x: (batch, sentence_len, embedding_dim)
        x = x.transpose(1, 2)  # x: (batch, embedding_dim, sentence_len)

        idx = 0
        if vars == None:
            vars = self.vars

        y1 = F.conv1d(x, weight=vars[idx], bias=vars[idx + 1])
        y1 = self.relu(y1)
        y1 = self.maxpool_1(y1)
        idx += 2

        y2 = F.conv1d(x, weight=vars[idx], bias=vars[idx + 1])
        y2 = self.relu(y2)
        y2 = self.maxpool_2(y2)
        idx += 2

        y3 = F.conv1d(x, weight=vars[idx], bias=vars[idx + 1])
        y3 = self.relu(y3)
        y3 = self.maxpool_3(y3)
        idx += 2

        y = torch.cat([y1, y2, y3], 2)
        y = y.view(y.size(0), -1)
        feature_extracted = y
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.linear(y, vars[idx], vars[idx + 1])
        res = F.softmax(y, dim=1)
        # if torch.sum(res[0]) != 1:
        #     raise ValueError
        return res, feature_extracted

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

    def set_parameters(self, param):
        for i, s in enumerate(param):
            self.vars[i] = nn.Parameter(s.clone())


class LinearText(nn.Module):
    def __init__(self, feature_len, output_dim=6):
        super(LinearText, self).__init__()

        self.vars = nn.ParameterList()

        fc = nn.Linear(feature_len, output_dim)
        self.vars.append(nn.Parameter(fc.weight))
        self.vars.append(nn.Parameter(fc.bias))

    def forward(self, x, vars=None):  # x: (batch, sentence_len)
        idx = 0
        if vars == None:
            vars = self.vars

        y = F.linear(x, vars[idx], vars[idx + 1])
        res = F.softmax(y, dim=1)

        return res, y

    def zero_grad(self, vars=None):
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        return self.vars

    def set_parameters(self, param):
        for i, s in enumerate(param):
            self.vars[i] = nn.Parameter(s.clone())
