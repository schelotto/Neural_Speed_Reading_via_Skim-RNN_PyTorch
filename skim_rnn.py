import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(size, index):
    mask = torch.LongTensor(*size).fill_(0)
    if isinstance(index, Variable):
        mask = Variable(mask, volatile=index.volatile)
    ret = mask.scatter_(1, index, 1)
    return ret

class SkimRNN(nn.Module):

    def __init__(self, args):
        super(SkimRNN, self).__init__()
        # Model hyper-parameters
        self.embed_dim = args.embed_dim
        self.vocab_size = args.vocab_size
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.large_cell_size = args.large_cell_size
        self.small_cell_size = args.small_cell_size
        self.tau = args.tau
        self.n_class = args.n_class

        # Model modules
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embedding.weight.requires_grad = False
        self.large_rnn = nn.LSTMCell(input_size=self.embed_dim,
                                       hidden_size=self.large_cell_size,
                                       bias=True)
        self.small_rnn = nn.LSTMCell(input_size=self.embed_dim,
                                       hidden_size=self.small_cell_size,
                                       bias=True)

        self.linear = nn.Linear(self.embed_dim + self.large_cell_size, 2)

        self.selu = nn.SELU()
        self.classifier = nn.Sequential(
            nn.Linear(self.large_cell_size, 512),
            nn.SELU(),
            nn.AlphaDropout(args.dropout),
            nn.Linear(512, self.n_class)
        )

    def _initialize(self, batch_size, cell_size):
        return Variable(torch.FloatTensor(batch_size, cell_size).zero_()).cuda()

    def inference(self, x):
        """
        :param x: [batch, len]
        :return: h_state, Q_list
        """
        embed = self.embedding(x) # [batch, len, embed_dim]
        batch_size = x.size()[0]

        h_state_l = self._initialize(batch_size, self.large_cell_size)
        h_state_s = self._initialize(batch_size, self.small_cell_size)
        c_l = self._initialize(batch_size, self.large_cell_size)
        c_s = self._initialize(batch_size, self.small_cell_size)

        Q_ = [] # [batch, len, 2]
        h_ = [] # [batch, len, large_cell_size]

        for t in range(x.size()[1]):
            embed_ = embed[:, t, :]

            h_state_l_, c_l_ = self.large_rnn(embed_, (h_state_l, c_l))
            h_state_s, c_s = self.small_rnn(embed_, (h_state_s, c_s))

            p_t = self.linear(torch.cat([embed_.contiguous().view(-1, self.embed_dim), h_state_l_, c_l_], 1))
            p_t = F.softmax(p_t, 1)
            Q_t = torch.multinomial(p_t, 1)  # [batch, 1]
            Q_t = one_hot((batch_size, 2), Q_t.view(batch_size, -1)).unsqueeze(1).float() # [batch, 1, 2]

            h_state_tilde = torch.transpose(torch.stack(
                            [h_state_l_,
                             torch.cat([h_state_s[:, :self.small_cell_size],
                                        h_state_l[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            c_tilde = torch.transpose(torch.stack(
                            [c_l_,
                             torch.cat([c_s[:, :self.small_cell_size],
                                        c_l_[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            h_state_l = torch.bmm(Q_t, h_state_tilde).squeeze()
            c_l = torch.bmm(Q_t, c_tilde).squeeze()

            h_.append(h_state_l)
            Q_.append(Q_t)

        logits = F.softmax(self.classifier(self.selu(h_state_l)), dim=1)
        h_stack = torch.stack(h_, dim=1)
        Q_stack = torch.stack(Q_, dim=1)

        return logits, h_stack, Q_stack

    def forward(self, x):
        """
        :param x: [batch, len]
        :return: h_state, p_list
        """
        embed = self.embedding(x) # [batch, len, embed_dim]
        batch_size = x.size()[0]

        h_state_l = self._initialize(batch_size, self.large_cell_size)
        h_state_s = self._initialize(batch_size, self.small_cell_size)
        c_l = self._initialize(batch_size, self.large_cell_size)
        c_s = self._initialize(batch_size, self.small_cell_size)

        p_ = []  # [batch, len, 2]
        h_ = []  # [batch, len, large_cell_size]

        for t in range(x.size()[1]):
            embed_ = embed[:, t, :]

            h_state_l_, c_l_ = self.large_rnn(embed_, (h_state_l, c_l))
            h_state_s, c_s = self.small_rnn(embed_, (h_state_s, c_s))

            p_t = self.linear(torch.cat([embed_.contiguous().view(-1, self.embed_dim), h_state_l_, c_l_], 1))
            p_t = F.softmax(p_t, 1)
            g_t = -torch.log(-torch.log(torch.rand(batch_size, 2)))
            g_t = Variable(g_t)
            r_t = F.softmax((torch.log(p_t) + g_t) / self.tau, dim=1).unsqueeze(1)

            h_state_tilde = torch.transpose(torch.stack(
                            [h_state_l_,
                             torch.cat([h_state_s[:, :self.small_cell_size],
                                        h_state_l[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            c_tilde = torch.transpose(torch.stack(
                            [c_l_,
                             torch.cat([c_s[:, :self.small_cell_size],
                                        c_l_[:, self.small_cell_size:self.large_cell_size]],
                                       dim=1)
                             ], dim=2), 1, 2)

            h_state_l = torch.bmm(p_t, h_state_tilde).squeeze()
            c_l = torch.bmm(p_t, c_tilde).squeeze()

            h_.append(h_state_l)
            p_.append(p_t)

        logits = F.softmax(self.classifier(self.selu(h_state_l)), dim=1)
        h_stack = torch.stack(h_, dim=1)
        p_stack = torch.stack(p_, dim=1)

        return logits, h_stack, p_stack