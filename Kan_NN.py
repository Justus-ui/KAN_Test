import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn as nn
import matplotlib.pyplot as plt

class SparseNeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, h = [8,4]):
        super(SparseNeuralNetwork, self).__init__()
        h = [1] + h
        self.univariate_nn = nn.Sequential()
        layers = []
        self.masks = []
        for layer in range(1,len(h)):
            layers.append(nn.Linear(h[layer -1] * in_dim, h[layer] * in_dim))
            layers.append(nn.ReLU())
            self.masks.append(self.hidden_sparistiy_masks(h[layer] * in_dim, h[layer -1] * in_dim, h[layer - 1],h[layer]))
        self.univariate_nn = nn.Sequential(*layers)
        print(self.univariate_nn)
        self.multiply_weight_masks()
        self.fc2 = nn.Linear(h[-1] * in_dim, out_dim)

    def multiply_weight_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),2):
                self.univariate_nn[i].weight.mul_(self.masks[i // 2])

    def multiply_grad_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),2):
                self.univariate_nn[i].weight.grad.mul_(self.masks[i // 2])

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
    def __init__(self, shape, h):
        super(Neural_Kan, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(shape) - 1):
            print(shape[i], shape[i + 1])
            self.layers.append(SparseNeuralNetwork(in_dim = shape[i], out_dim = shape[i + 1], h = h))
    def forward(self,x):
        return self.layers(x)
    
    def fit(self,dataloader, epochs=100, lr=0.001):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10*len(dataloader))
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.MSELoss()
        self.train()
        loss_list = []
        for epoch in range(epochs):
            total_loss = 0.0
            for inputs, targets in dataloader:
                scheduler.step()
                #print(scheduler.get_lr())
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                for models in self.layers:
                    models.multiply_grad_masks()
                optimizer.step()
            loss_list.append(total_loss)
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")
        plt.plot(loss_list)
        plt.show()

    def get_dataloader(self,f,num_samples=1000, in_dim=2, out_dim=1, batch_size = 32):
        X = torch.randn(num_samples, in_dim) 
        train_dataset = torch.utils.data.TensorDataset(X, f(X))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader

if __name__ == "__main__":
    model = Neural_Kan(shape = [2,1,1], h = [8,32,8])
    def f(X):
        return torch.sin(torch.sum(X, dim=1, keepdim=True))
    dataloader = model.get_dataloader(f)
    print(dataloader)
    model.fit(dataloader = dataloader, epochs=1000, lr=0.001)
        

            
