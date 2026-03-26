import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden1_dim,
        hidden2_dim,
        codebook_channel,
        codebook_dim,
        dropout,
        categorical
    ):
        """
        architecture adapted from https://github.com/sebastianstarke/AI4Animation/blob/master/AI4Animation/SIGGRAPH_2024/PyTorch/Library/Modules.py
        """
        super(LinearEncoder, self).__init__()

        self.input_dim = input_dim
        self.codebook_c = codebook_channel
        self.codebook_d = codebook_dim
        self.output_dim = codebook_channel * codebook_dim

        self.dropout = dropout
        self.categorical = categorical

        self.linear1 = nn.Linear(input_dim, hidden1_dim)
        self.linear2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.linear3 = nn.Linear(hidden2_dim, self.output_dim)

    def sample_gumbel(self, tensor, scale, eps=1e-20):
        scale = scale.view(-1, 1, 1, 1) #This is noise scale between 0 and 1
        noise = torch.rand_like(tensor) - 0.5 #This is random noise between -0.5 and 0.5
        samples = scale * noise + 0.5 #This is noise rescaled between 0 and 1 where 0.5 is default for 0 noise
        
        return -torch.log(-torch.log(samples + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, scale):
        y = logits + self.sample_gumbel(logits, scale)

        return F.softmax(y / temperature, dim=-1)
    
    def gumbel_softmax(self, logits, temperature, scale):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, scale)

        y_soft = y.view(logits.shape)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        y_hard = y_hard.view(logits.shape)

        return y_soft, y_hard

    def sample(self, z, knn):
        z = z.view(-1, self.codebook_c, self.codebook_d)
        z = z.unsqueeze(0).repeat(knn.size(0), 1, 1, 1)
        z_soft, z_hard = self.gumbel_softmax(z, 1.0, knn)
        z_soft = z_soft.view(-1, self.output_dim)
        z_hard = z_hard.view(-1, self.output_dim)
        
        return z_soft, z_hard
        
    def forward(self, z, knn):
        z = F.dropout(z, self.dropout, training=self.training)
        z = self.linear1(z)
        z = F.elu(z)

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.linear2(z)
        z = F.elu(z)

        z = F.dropout(z, self.dropout, training=self.training)
        z = self.linear3(z)

        z_soft, z_hard = None, None
        if self.categorical:
            z_soft, z_hard = self.sample(z, knn)

        return z, z_soft, z_hard
    
class LinearDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        input_dim: int = -1,
        hidden_dim: list = [1024, 1024],
        output_dim: list = 3,
        skips=[5],
    ):
        """
        architecture adapted from https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
        """
        super(LinearDecoder, self).__init__()

        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.skips = skips

        input_layer = [nn.Linear(input_dim, hidden_dim[0])]
        hidden_layers = [nn.Linear(hidden_dim[0], hidden_dim[-1])]
        for i in range(2, n_layers):
            if i not in self.skips:
                hidden_layers.append(nn.Linear(hidden_dim[-1], hidden_dim[-1]))
            else:
                hidden_layers.append(nn.Linear(hidden_dim[-1] + input_dim, hidden_dim[-1]))
        self.mlp = nn.ModuleList(input_layer + hidden_layers)
        self.output_pos_layer = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        h = x.clone()
        for i, _ in enumerate(self.mlp):
            h = self.mlp[i](h)
            h = F.relu(h)
            if i+1 in self.skips:
                h = torch.cat([x, h], -1)
        rst_pos = self.output_pos_layer(h)

        return {"deform_p": rst_pos}