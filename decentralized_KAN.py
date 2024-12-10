import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Decentralized_layer(nn.Module):
    def __init__(self, in_dim, h, connections):
        super(Decentralized_layer, self).__init__()
        h = [1] + h
        self.univariate_nn = nn.Sequential()
        layers = []
        self.masks = []
        for layer in range(1,len(h)):
            layers.append(nn.Linear(h[layer -1] * in_dim, h[layer] * in_dim))
            nn.init.xavier_uniform_(layers[-1].weight)
            layers.append(nn.BatchNorm1d(h[layer] * in_dim))
            layers.append(nn.LeakyReLU())
            self.masks.append(self.hidden_sparistiy_masks(h[layer] * in_dim, h[layer -1] * in_dim, h[layer - 1],h[layer]))
        self.univariate_nn = nn.Sequential(*layers)
        print(self.univariate_nn)
        self.multiply_weight_masks()
        self.fc2 = nn.Linear(h[-1] * in_dim, in_dim)
        print(self.fc2)
        if connections is not None:
            self.connection_mask = self.get_connection_mask(h[-1], in_dim, connections)
            self.multiply_connection_weight_masks()

    def multiply_connection_weight_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),2):
                self.fc2.weight.mul_(self.connection_mask)

    def multiply_grad_connection_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),2):
                self.fc2.weight.grad.mul_(self.connection_mask)

    def get_connection_mask(self, h, in_dim, connections):
        mask = torch.zeros(in_dim, h * in_dim)
        for i in range(in_dim):
            for j in range(in_dim):
                if j in connections[i]:
                    #print(mask[j,i * h:(i+1) * h])
                    mask[j,i * h:(i+1)* h] = 1
                #else:
                    #print(j, connections[i])
        print(mask)
        return mask

    def multiply_weight_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),3):
                self.univariate_nn[i].weight.mul_(self.masks[i // 3])

    def multiply_grad_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),3):
                self.univariate_nn[i].weight.grad.mul_(self.masks[i // 3])

    def hidden_sparistiy_masks(self, out_dim, in_dim, input_neurons, output_neurons):
        mask = torch.zeros(out_dim, in_dim)
        for i in range(0,in_dim):
            mask[i*output_neurons:output_neurons*(i + 1) , i*input_neurons:(i + 1)*input_neurons] = 1
        return mask

    def forward(self, x):
        hidden = self.univariate_nn(x)
        output = self.fc2(hidden)
        return output

class Neural_Kan(nn.Module):
    def __init__(self, in_dim, iters, h, connections):
        super(Neural_Kan, self).__init__()
        self.layers = nn.Sequential()
        for i in range(iters):
            self.layers.append(Decentralized_layer(in_dim = in_dim, h = h, connections=connections))
    def forward(self,x):
        return self.layers(x)
    
    def fit(self,dataloader, epochs=10, lr=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        self.train()
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                #print(inputs, targets)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                for models in self.layers:
                #    for layer in models.univariate_nn:
                #        try:
                #            print(torch.max(layer.weight.grad), torch.min(layer.weight.grad))
                #        except:
                #            continue
                    models.multiply_grad_masks()
                    models.multiply_grad_connection_masks()
                optimizer.step()
                scheduler.step()
            loss_list.append(total_loss)
            avg_loss = total_loss / len(dataloader)
            #avg_loss /= 32 
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, learning_rate {scheduler.get_last_lr()}")
            plt.plot(loss_list)
def f(x):
    # Assume x is a 1D tensor of size (n,)
    return torch.stack([
        torch.norm(x, p = 3),
        torch.norm(x, p = 2),
        torch.norm(x, p = 1),
        torch.min(x),
        torch.max(x)
    ])

def generate_data(num_samples, input_dim):
    data = torch.randn(num_samples, input_dim)
    target = torch.stack([f(x) for x in data])  # Apply the function to each sample
    return data, target
#def generate_data(num_samples, input_dim):
#    return torch.randn(num_samples, input_dim), torch.stack([f(data[i]) for i in range(num_samples)])
#    data = torch.randn(num_samples, input_dim)
#    target = torch.mean(data, axis = 1)# * torch.exp(torch.sum(torch.abs(data), axis = 1))  # Apply the function to each sample
    #target.reshape(num_samples,1).repeat(1, input_dim)
    
    return data, 
input_dim = 5
model = Neural_Kan(in_dim = input_dim, iters = 4, h = [16,32,64,32,16], connections = [[0,1,2,3],[0,1,2,4],[0,1,2,3,4],[0,2,3,4], [1,2,3,4]])      
num_samples = 1000 
data, target = generate_data(num_samples, input_dim)
print(data.shape, target.shape)
train_dataset = torch.utils.data.TensorDataset(data, target)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False)
model.fit(train_dataloader, epochs=100)
#for models in model.layers:
#    for lin in models.univariate_nn:
#        try:
#            print(lin.weight)
#        except:
#            continue