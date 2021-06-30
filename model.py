import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, steps=2):
        super(GraphLayer, self).__init__()

        self.steps = steps

        self.encode = nn.Linear(input_dim, output_dim, bias=False)

        self.z0 = nn.Linear(output_dim, output_dim, bias=False)
        self.z1 = nn.Linear(output_dim, output_dim, bias=False)

        self.r0 = nn.Linear(output_dim, output_dim, bias=False)
        self.r1 = nn.Linear(output_dim, output_dim, bias=False)

        self.h0 = nn.Linear(output_dim, output_dim, bias=False)
        self.h1 = nn.Linear(output_dim, output_dim, bias=False)

        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.xavier_uniform_(self.encode.weight)
        torch.nn.init.xavier_uniform_(self.z0.weight)
        torch.nn.init.xavier_uniform_(self.z1.weight)
        torch.nn.init.xavier_uniform_(self.r0.weight)
        torch.nn.init.xavier_uniform_(self.r1.weight)
        torch.nn.init.xavier_uniform_(self.h0.weight)
        torch.nn.init.xavier_uniform_(self.h1.weight)

    def forward(self, inputs, adj_matrix, mask):
        # TODO: Add Dropout from line 219 of layers (Original Code)

        # print("inputs.dtype:", inputs.dtype)
        # print("adj_matrix.dtype:", adj_matrix.dtype)
        # print("mask.dtype:", mask.dtype)


        x = self.encode(inputs)
        x = mask.float() * self.relu(x)

        for _ in range(self.steps):
            # TODO Dropout : L56 layers.py
            a = torch.matmul(adj_matrix.float(), x.float())
            # update gate
            z0 = self.z0(a)
            z1 = self.z1(x.float())
            z = torch.sigmoid(z0 + z1)
            # reset gate
            r0 = self.r0(a)
            r1 = self.r1(x)
            r = torch.sigmoid(r0 + r1)
            # update embeddings
            h0 = self.h0(a)
            h1 = self.h1(x * r)
            h = F.relu(mask * (h0 + h1))
            # Update x for next iteration
            x = h * z + x * (1 - z)

        return x


class ReadoutLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReadoutLayer, self).__init__()

        self.att = nn.Linear(input_dim, 1, bias=False)

        self.emb = nn.Linear(input_dim, input_dim, bias=False)
        # self.mlp = nn.Linear(input_dim, output_dim, bias=False)

        torch.nn.init.xavier_uniform_(self.att.weight)
        torch.nn.init.xavier_uniform_(self.emb.weight)
        # torch.nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self, inputs, mask):
        x = inputs
        att = torch.sigmoid(self.att(x))
        emb = torch.relu(self.emb(x))
        n = torch.sum(mask, dim=1)
        m = (mask - 1) * 1e9

        # Graph Summation
        g = mask * att * emb
        g = (torch.sum(g, dim=1) / n) + torch.max(g + m, dim=1).values

        return g


class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.graph = GraphLayer(input_dim=input_dim, output_dim=hidden_dim)
        self.readout = ReadoutLayer(input_dim=hidden_dim, output_dim=output_dim)

        self.mlp = nn.Linear(hidden_dim, output_dim, bias=False)
        torch.nn.init.xavier_uniform_(self.mlp.weight)

    def forward(self, inputs, adj_matrix, mask):
        graph = self.graph(inputs, adj_matrix, mask)
        g_doc = self.readout(graph, mask)

        output = self.mlp(g_doc)

        return output




# class GNNmodel(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(GNNmodel, self).__init__()
#
#         self.gnn = GNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).\
#             load_state_dict(torch.load('model_save/mr.ckpt'))
#
#     def forward(self, inputs, adj_matrix, mask):
#         out = self.gnn(inputs, adj_matrix, mask)
#         return out



