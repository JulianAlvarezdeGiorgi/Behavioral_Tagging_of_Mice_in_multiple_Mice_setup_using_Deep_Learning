import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATEncoder(nn.Module):
    ''' The GAT encoder module. It takes in a graph batch and returns the mu and logvar vectors for each frame.
    Parameters:
        - nout: int, the dimension of the latent space
        - nhid: int, the number of hidden units in the GAT layers
        - attention_hidden: int, the number of attention heads in the GAT layers
        - n_in: int, the number of input features
        - n_layers: int, the number of GAT layers with residual connections
        - dropout: float, the dropout rate
    '''
    def __init__(self, nout, nhid, attention_heads, n_in, n_layers, dropout):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_heads = attention_heads
        self.n_hidden = nhid
        self.n_out = nout
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        
        self.GAT_layers = nn.ModuleList()
        #self.res_conn = nn.ModuleList()  # residual connections
        self.GAT_layers.append(GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_heads, dropout=self.dropout, concat=True)) 
        for _ in range(self.n_layers-2):
            self.GAT_layers.append(GATv2Conv(in_channels=self.n_hidden * self.attention_heads, out_channels=self.n_hidden, heads=self.attention_heads, dropout=self.dropout, concat=True))
            #self.res_conn.append(nn.Linear(self.n_hidden * self.attention_heads, self.n_hidden * self.attention_heads))

        self.GAT_layers.append(GATv2Conv(in_channels=self.n_hidden * self.attention_heads, out_channels=self.n_out, heads=self.attention_heads, dropout=self.dropout, concat=False))

        

    def forward(self, x, edge_index):

        x = self.GAT_layers[0](x, edge_index)
        x = self.relu(x)
        for i in range(1, self.n_layers-2):
            x1 = self.GAT_layers[i](x, edge_index)
            x1 = self.relu(x1)
            x = x + x1
        x = self.GAT_layers[-1](x, edge_index)
        x = self.relu(x)
        return x


class GATEncoder_old(nn.Module):
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
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_out, heads=self.attention_hidden, dropout=self.dropout, concat=False)
        #self.gatenc3 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=False)
        #self.gatenc4 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=True)

        self.res_conn = nn.ModuleList()  # residual connections
        for _ in range(1):
            self.res_conn.append(nn.Linear(self.n_hidden * attention_hidden, self.n_hidden * attention_hidden))
            self.res_conn.append(nn.ReLU())



        #self.out = nn.Linear(self.n_hidden * attention_hidden, self.n_out)



    def forward(self, x, edge_index, frame_mask):

        # data type of the input
        x = self.gatenc1(x, edge_index)
        x1 = self.relu(x)
        x = self.res_conn[0](x) + x1
        x2 = self.res_conn[1](x)
        x = self.gatenc2(x2, edge_index)
        x = self.relu(x)

        return x

class GATEncoder_v2(nn.Module):
    ''' The GAT encoder module. It takes in a graph batch and returns the mu and logvar vectors for each frame. '''

    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoder_v2, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc3 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc4 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_out, heads=attention_hidden, dropout=self.dropout, concat=False)

        self.res_conn = nn.ModuleList()  # residual connections
        for _ in range(2):
            self.res_conn.append(nn.Linear(self.n_hidden * attention_hidden, self.n_hidden * attention_hidden))
            self.res_conn.append(nn.ReLU())



        #self.out = nn.Linear(self.n_hidden * attention_hidden, self.n_out)

        


    def forward(self, x, edge_index, frame_mask):

        # data type of the input
        x = self.gatenc1(x, edge_index)
        x1 = self.relu(x)
        x = self.gatenc2(x1, edge_index)
        x = self.relu(x)
        x = self.res_conn[0](x) + x1
        x2 = self.res_conn[1](x)
        x = self.gatenc3(x2, edge_index)
        x = self.relu(x)
        x = self.res_conn[2](x) + x2
        x3 = self.res_conn[3](x)
        x = self.gatenc4(x3, edge_index)
        x = self.relu(x)
        #x = self.res_conn[4](x) + x3
        #x = self.res_conn[5](x)
        

        # Aggrgate the node features for each frame, Only interested in the ENC-DEC model
        #x = global_mean_pool(x, frame_mask) 
        # Keep only where the frame mask is 1
        #x = x[frame_mask] 

        #x = self.out(x)
        #x = self.relu(x)

        return x
    
class GATEncoder_v3(nn.Module):
    ''' Without residual connections '''
    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoder_v3, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_out, heads=self.attention_hidden, dropout=self.dropout, concat=False)




    def forward(self, x, edge_index, frame_mask):
            
        # data type of the input
        x = self.gatenc1(x, edge_index)
        x = self.relu(x)
        x = self.gatenc2(x, edge_index)
        x = self.relu(x)
        

        return x

class GATEncoder_vfollowing(nn.Module):
    ''' The GAT encoder module. It takes in a graph batch and returns the mu and logvar vectors for each frame. '''

    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoder_vfollowing, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc3 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_hidden, heads=attention_hidden, dropout=self.dropout, concat=False)
        #self.gatenc4 = GATv2Conv(in_channels=self.n_hidden * attention_hidden, out_channels=self.n_out, heads=attention_hidden, dropout=self.dropout, concat=False)

        self.res_conn = nn.ModuleList()  # residual connections
        for _ in range(1):
            self.res_conn.append(nn.Linear(self.n_hidden * attention_hidden, self.n_hidden * attention_hidden))
            self.res_conn.append(nn.ReLU())



        #self.out = nn.Linear(self.n_hidden * attention_hidden, self.n_out)

        


    def forward(self, x, edge_index, frame_mask):

        # data type of the input
        x = self.gatenc1(x, edge_index)
        x1 = self.relu(x)
        x = self.gatenc2(x1, edge_index)
        x = self.relu(x)
        x = self.res_conn[0](x) + x1
        x2 = self.res_conn[1](x)
        x = self.gatenc3(x2, edge_index)
        x = self.relu(x)
        # x = self.res_conn[2](x) + x2
        # x3 = self.res_conn[3](x)
        # x = self.gatenc4(x3, edge_index)
        # x = self.relu(x)
        #x = self.res_conn[4](x) + x3
        #x = self.res_conn[5](x)
        

        # Aggrgate the node features for each frame, Only interested in the ENC-DEC model
        #x = global_mean_pool(x, frame_mask) 
        # Keep only where the frame mask is 1
        #x = x[frame_mask] 

        #x = self.out(x)
        #x = self.relu(x)

        return x
    
class GATEncoder_v3(nn.Module):
    ''' Without residual connections '''
    def __init__(self, nout, nhid, attention_hidden, n_in, dropout):
        super(GATEncoder_v3, self).__init__()
        self.dropout = dropout
        self.n_in = n_in
        self.attention_hidden = attention_hidden
        self.n_hidden = nhid
        self.n_out = nout
        self.relu = nn.ReLU()
        
        self.gatenc1 = GATv2Conv(in_channels=self.n_in, out_channels=self.n_hidden, heads=self.attention_hidden, dropout=self.dropout, concat=True)
        self.gatenc2 = GATv2Conv(in_channels=self.n_hidden * self.attention_hidden, out_channels=self.n_out, heads=self.attention_hidden, dropout=self.dropout, concat=False)




    def forward(self, x, edge_index, frame_mask):
            
        # data type of the input
        x = self.gatenc1(x, edge_index)
        x = self.relu(x)
        x = self.gatenc2(x, edge_index)
        x = self.relu(x)
        

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
        self.hidden2 = nn.Linear(nhid, nhid)
        self.hidden3 = nn.Linear(nhid, nout)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
     

    def forward(self, z):
        x = self.hidden1(z)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        return x
    
class GraphClassifier(nn.Module):
    def __init__(self, encoder, classifier, readout = 'mean'):
        super(GraphClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.readout = readout

    def forward(self, batch):
        x, edge_index, frame_mask, graph_batch = batch.x, batch.edge_index, batch.frame_mask, batch.batch
        embbed = self.encoder(x, edge_index)
        if self.readout == 'mean':
            embbed = self.mean_pooling_per_graph(embbed, graph_batch, frame_mask)
        elif self.readout == 'max':
            embbed = self.max_pooling_per_graph(embbed, graph_batch, frame_mask)
        elif self.readout == 'concatenate':
            embbed = self.concatenate_per_graph(embbed, graph_batch, frame_mask)
        # concatenate the embeddings for each frame
        return self.classifier(embbed)
    
    @staticmethod
    def concatenate_per_graph(embbed, batch, frame_mask):
        ''' Concatenate the embeddings per graph, only the central frame '''
        out = []
        for i in range(batch.max()+1):
            out.append(embbed[batch==i][frame_mask[batch==i] == frame_mask[batch==i].median()].flatten())
        return torch.stack(out)
    
    @staticmethod
    def mean_pooling_per_graph(embbed, batch, frame_mask):
        ''' Mean pooling of the embeddings per graph, only the central frame '''
        out = []
        for i in range(batch.max()+1):
            out.append(embbed[batch==i][frame_mask[batch==i] == frame_mask[batch==i].median()].mean(dim=0))
        return torch.stack(out)
    
    @staticmethod
    def max_pooling_per_graph(embbed, batch, frame_mask):
        ''' Max pooling of the embeddings per graph, only the central frame '''
        out = []
        for i in range(batch.max()+1):
            out.append(embbed[batch==i][frame_mask[batch==i] == frame_mask[batch==i].median()].max(dim=0))
        return torch.stack(out)
    
    @staticmethod
    def attention_readout(embbed, batch, frame_mask, readout = 'mean'):
        ''' Attention readout of the embeddings per graph. Normal readouts will be applied per frame, then the attention will be applied to the frames representation to build the graph representation '''
        out = []
        for i in range(batch.max()+1):
            # For each frame, apply the readout
            frame_embbed = []
            for j in frame_mask[batch==i].unique():
                if readout == 'mean':
                    frame_embbed.append(embbed[batch==i][frame_mask[batch==i] == j].mean(dim=0))
                elif readout == 'max':
                    frame_embbed.append(embbed[batch==i][frame_mask[batch==i] == j].max(dim=0))
                elif readout == 'sum':
                    frame_embbed.append(embbed[batch==i][frame_mask[batch==i] == j].sum(dim=0))
            
            # Apply the attention mechanism
            frame_embbed = torch.stack(frame_embbed)
        return
    
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
        x_per_graph = self.concatenate_per_graph(x, graph_batch)
        x = self.hidden1(x_per_graph)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.out(x)
        return x
    
    @staticmethod
    def concatenate_per_graph(embbed, batch):
        ''' Concatenate the embeddings per graph '''
        out = []
        for i in range(batch.max()+1):
            out.append(embbed[batch==i].flatten())
        return torch.stack(out)
    



###### NEW MODEL #########

class GATLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, dropout):
        super(GATLayer, self).__init__()
        self.gat1 = GATv2Conv(input_dim, hidden_dim, heads = heads, dropout= dropout)
        self.gat2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads = heads, concat=False, dropout= dropout)

        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.gat1(x, edge_index))
        x = torch.relu(self.gat2(x, edge_index))
        return x



# Now modify the forward method to handle batches of sequences
class GAT_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_hidden_dim, num_classes, num_nodes, heads, dropout=0.5):
        ''' A GAT-LSTM model for sequence classification. 
        Parameters:
            - input_dim: int, the number of input features
            - hidden_dim: int, the number of hidden units in the GAT layers
            - lstm_hidden_dim: int, the number of hidden units in the LSTM layer
            - num_classes: int, the number of classes
            - num_nodes: int, the number of nodes in the graph
            - heads: int, the number of attention heads in the GAT layers
            - dropout: float, the dropout rate 
        '''
        super(GAT_LSTM, self).__init__()
        self.num_nodes = num_nodes
        self.gcn = GATLayer(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.lstm = nn.LSTM(hidden_dim * num_nodes, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)

    def forward(self, batch):
        batch_size = len(batch)
        sequence_len = len(batch[0])  # assuming all sequences are the same length

        gcn_out = []
        for seq in batch:
            # Each element in batch is a sequence of graphs (Data objects)
            seq_out = []
            for graph in seq:
                # Process each graph frame in the sequence with GCN
                x = self.gcn(graph)
                # Flatten the output to pass into LSTM
                seq_out.append(x.view(-1))  # Flatten node features for LSTM
            gcn_out.append(torch.stack(seq_out))  # Stack the sequence

        gcn_out = torch.stack(gcn_out)  # Batch all sequences Shape 
        lstm_out, (h_n, c_n) = self.lstm(gcn_out)  # Pass through LSTM

        # Use the final hidden state of the LSTM to classify
        out = self.fc(lstm_out[:, -1, :])  # Use the last LSTM output for classification
        return out
