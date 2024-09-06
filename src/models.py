import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATEncoder(nn.Module):
    ''' The GAT encoder module. It takes in a graph batch and returns the mu and logvar vectors for each frame. '''

    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        #self.gatenc3 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=True)
        #self.gatenc4 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=True)

        #self.res_conn = nn.ModuleList()  # residual connections
        #for _ in range(1):
        #    self.res_conn.append(nn.Linear(self.n_hidden * attention_hidden, self.n_hidden * attention_hidden))
        #    self.res_conn.append(nn.ReLU())



        #self.out = nn.Linear(self.n_hidden * attention_hidden, self.n_out)

        


    def forward(self, x, edge_index, frame_mask):

        # data type of the input
        x = self.gatenc1(x, edge_index)
        x1 = self.relu(x)
        x = self.gatenc2(x1, edge_index)
        x = self.relu(x)
        #x = self.res_conn[0](x) + x1
        #x = self.res_conn[1](x)
        #x = self.gatenc3(x, edge_index)
        #x = self.relu(x)
        #x = self.res_conn[4](x) + x
        #x = self.res_conn[5](x)
        #x = self.gatenc4(x, edge_index)
        #x = self.relu(x)
        #x = self.res_conn[6](x) + x
        #x = self.res_conn[7](x)
        

        # Aggrgate the node features for each frame, Only interested in the ENC-DEC model
        #x = global_mean_pool(x, frame_mask) 
        # Keep only where the frame mask is 1
        #x = x[frame_mask] 

        #x = self.out(x)
        #x = self.relu(x)

        return x


import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

class GATDecoder(nn.Module):
    ''' The GAT decoder module. It takes in latent vectors and reconstructs the graph for each frame. '''

    def __init__(self, n_latent, n_hidden, n_out):
        super(GATDecoder, self).__init__()

        self.n_out = n_out
        self.num_nodes = 28
        self.hidden1 = nn.Linear(n_latent, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()

        self.out = nn.Linear(n_hidden, n_out * self.num_nodes)

    def forward(self, z):
        # Expand latent vectors to match the graph structure
        x = self.hidden1(z)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        
        x = self.out(x)
        x = x.view(-1, self.num_nodes, self.n_out)
        return x


class GraphAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, edge_index, frame_mask):
        embbed = self.encoder(x, edge_index, frame_mask)
        return self.decoder(embbed)

    def loss(self, x, recon_x):
        # Reconstruction loss
        # convert to 2D, concatenating the first two dimensions
        recon_x = recon_x.view(-1, recon_x.size(-1))
        recon_loss = nn.MSELoss()(recon_x, x)
        return recon_loss


        
############# VARIAITONAL AUTOENCODER ####################

class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(GraphVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, frame_mask):
        mu, logvar = self.encoder(x, edge_index, frame_mask)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, edge_index, frame_mask), mu, logvar

    def loss(self, x, recon_x, mu, logvar):
        # Reconstruction loss
        recon_loss = nn.MSELoss()(recon_x, x)
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss 


######### SIMPLE LINEAR CLASSIFIER ON THE LATENT SPACE ##########

class ClassificationHead(nn.Module):
    def __init__(self, n_latent, nhid, nout):
        super(ClassificationHead, self).__init__()
        self.hidden1 = nn.Linear(n_latent, nhid)
        self.hidden2 = nn.Linear(nhid, nout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
     

    def forward(self, z):
        x = self.hidden1(z)
        x = self.relu(x)
        x = self.hidden2(x)
        return self.softmax(x)
    
class GraphClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super(GraphClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, batch):
        x, edge_index, frame_mask, graph_batch = batch.x, batch.edge_index, batch.frame_mask, batch.batch
        embbed = self.encoder(x, edge_index, frame_mask)
        # readout the embeddings for each graph
        embbed = global_mean_pool(embbed, graph_batch, size = graph_batch.max().item()+1)
        # concatenate the embeddings for each frame
        
        return self.classifier(embbed)

    def loss(self, y, y_pred):
        return nn.CrossEntropyLoss()(y_pred, y) # Overlapping multi-class classification
    
    def accuracy(self, y, y_pred, threshold=0.5):
        y_pred = (y_pred > threshold).float()
        return torch.sum(y == y_pred).item() / len(y)
        
    
class SimpleMLPforGraph(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(SimpleMLPforGraph, self).__init__()
        self.hidden1 = nn.Linear(n_in, n_hid)
        self.hidden2 = nn.Linear(n_hid, n_hid)
        self.out = nn.Linear(n_hid, n_out)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, batch):
        x, edge_index, frame_mask, graph_batch = batch.x, batch.edge_index, batch.frame_mask, batch.batch
        x = x[frame_mask == 2]
        x = x.flatten()
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.out(x)
        return self.softmax(x)
    


    
