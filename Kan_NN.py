import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

class SparseNeuralNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, h = [8,4]):
        super(SparseNeuralNetwork, self).__init__()
        h = [1] + h
        self.univariate_nn = nn.Sequential()
        layers = []
        self.masks = []
        self.num_funcs = 2
        for layer in range(1,len(h)):
            layers.append(nn.Linear(h[layer -1] * in_dim, h[layer] * in_dim))
            #layers.append(nn.BatchNorm1d(h[layer] * in_dim, affine=True))
            layers.append(nn.ReLU())
            self.masks.append(self.hidden_sparistiy_masks(h[layer] * in_dim, h[layer -1] * in_dim, h[layer - 1],h[layer]))
        self.univariate_nn = nn.Sequential(*layers)
        print(self.univariate_nn)
        self.multiply_weight_masks()
        self.fc2 = nn.Linear(h[-1] * in_dim, out_dim)

    def multiply_weight_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),self.num_funcs):
                self.univariate_nn[i].weight.mul_(self.masks[i // self.num_funcs])

    def multiply_grad_masks(self):
        with torch.no_grad():
            for i in range(0,len(self.univariate_nn),self.num_funcs):
                self.univariate_nn[i].weight.grad.mul_(self.masks[i // self.num_funcs])

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
    """
    Class:
    shape: list, describing tuple (n_1,...,n_N)
    h: shape of univariate Neural Networks. 
    """
    def __init__(self, shape, h):
        super(Neural_Kan, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(shape) - 1):
            print(shape[i], shape[i + 1])
            self.layers.append(SparseNeuralNetwork(in_dim = shape[i], out_dim = shape[i + 1], h = h))

    def forward(self,x):
        return self.layers(x)
    
    def test_loss(self,dataloader):
        self.eval()
        criterion = nn.MSELoss(reduction = 'mean')
        total_loss = 0.
        with torch.no_grad():
            for inputs, targets in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        return total_loss

    def fit(self,dataloader, dataloader_test, epochs=100, lr=0.001, weight_decay = 1e-3):
        print("new version")
        optimizer = optim.RAdam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,min_lr  = 1e-5)
        criterion = nn.MSELoss(reduction = 'mean')
        self.loss_list = []
        avg_test_loss_list = []
        for epoch in range(epochs):
            self.train()
            total_loss = 0.0
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                loss.backward()
                for models in self.layers:
                    models.multiply_grad_masks()
                optimizer.step() 
            test_loss = self.test_loss(dataloader_test)
            #scheduler.step()
            #scheduler.step(test_loss)
            #print()
            avg_testloss = test_loss / len(dataloader_test)
            avg_test_loss_list.append(avg_testloss)
            self.loss_list.append(total_loss)
            avg_loss = total_loss / len(dataloader)
            #Lr:{scheduler.get_last_lr()}
            print(f" Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.7f},Test_Loss: {avg_testloss:.7f}")
        #plt.plot(self.loss_list)
        #plt.show()
        return avg_test_loss_list

    def get_dataloader(self,f,num_samples=1000, in_dim=2, out_dim=1, batch_size = 32):
        X = torch.rand(num_samples, in_dim) 
        train_dataset = torch.utils.data.TensorDataset(X, f(X))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader

if __name__ == "__main__":
    def f(x):
    # Fixed exponents between 0 and 1
        alpha = 0.5
        beta = 0.7
        gamma = 0.3
        delta = 0.8
        epsilon = 0.6
        omega = 0.4

        # First term: (x^alpha + (1 - x)^beta)^gamma
        term1 = 10*torch.sum(x ** alpha, dim = 1)
        term2 = 2*torch.sum((1.0001 - x) ** beta, dim = 1)
        # Second term: (sin(2πx)^delta)^epsilon
        term3 = torch.abs(torch.sin(2 * torch.pi * torch.sum(x**delta, dim = 1)))
        term4= torch.sum(x**0.3, dim = 1)
        result = (torch.abs((torch.sin(2 * torch.pi * (term1 ** .5)) + torch.cos(20 * torch.pi * (term2 ** .4))))**omega + torch.abs((term3**.67 + term4**0.1)))
        return torch.reshape(result, [result.shape[0], 1])
    #def f(X):
    #    return torch.sum(X, dim=1, keepdim=True)
    in_dim = 10
    model = Neural_Kan(shape = [in_dim,4,2,1], h = [64])
    dataloader = model.get_dataloader(f, in_dim=in_dim, num_samples=1000, batch_size=32)
    dataloader_test = model.get_dataloader(f, in_dim=in_dim, num_samples=200, batch_size=20)
    print("dataloader",len(dataloader_test),len(dataloader))
    h = [32,64,32]
    widths = [[64],[128],[1024], [2048], [4096]]
    #widths = [[4,8],[2,4]]
    decays = [0.1, 1e-2, 1e-5,0]
    epics = 200
    for h in widths:#, [64], [128], [1024]]:#, [8,16,8], [16,32,16], [32,64,32], [64,128,64]]:
        plt.figure()
        Losses = np.zeros((len(decays), epics))
        for i,decay in enumerate(decays):
            model = Neural_Kan(shape = [in_dim,4,2,1], h = h)
            loss = model.fit(dataloader = dataloader,dataloader_test = dataloader_test, epochs=epics, lr=1e-3, weight_decay= decay)
            Losses[i,:] = loss
            plt.plot(loss, label = f"{decay}")
        np.save(f'no_batch_norm/{decays}_{h}', Losses)
        plt.title(f"Effect of L2 Reg. on testloss (uni. NNs {h}, 10 Inputs)")
        plt.yscale('log')
        plt.ylabel("Test Loss: Root Mean Squared Error (RMSE)")
        plt.xlabel("epoch")
        plt.legend()
        plt.show(block=False)
    plt.show()
    #valid = torch.rand(10, in_dim)
    #test = model(valid)
    #print(test.shape, "test")
    #print(torch.sum(valid, dim = 1).shape)
    #print(torch.sum((test.flatten() - torch.sum(valid, dim = 1))**2) / 10)
    #print(test, valid)
    #print((test - torch.sum(valid, dim = 0)).shape)
    #model = Neural_Kan(shape = [2,1,1], h = [8,32,8])
    #def f(X):
    #    return torch.sin(torch.sum(X, dim=1, keepdim=True))
    #dataloader = model.get_dataloader(f)
    #dataloader_test = model.get_dataloader(f, batch_size =1)
    #print(dataloader)
    #model.fit(dataloader = dataloader,dataloader_test = dataloader_test, epochs=200, lr=1e-3)
        

            
