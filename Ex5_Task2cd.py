import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28, 64)  # the image size is 28 by 28
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return F.log_softmax(x, dim=1)

    
#-------------------------------------------------------------------------------------------

def Hv_computer(w, loss, v):
    
    with torch.enable_grad():
        fristGrads = torch.autograd.grad(loss, w, create_graph=True)
        fGv = [torch.dot(fG.flatten(), v.flatten()) for fG,v in zip(fristGrads, v)]
        sums = torch.Tensor([0.])
        for i in fGv:
            sums = sums + i
        return torch.autograd.grad([sums], w)
    
#-------------------------------------------------------------------------------------------

def main():
    trainset = datasets.MNIST('', train=True, download=True, 
                           transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))
    testset = datasets.MNIST('', train=False, download=True, 
                           transform=transforms.Compose([
                                    transforms.ToTensor()
                                ]))


    trainloader  = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, pin_memory=True)


    net = Net() # inital network
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # create a Adam optimizer

    net.train() # set netowrk to traning mode
    epochs = 2
    for epoch in range(epochs):
        for data in trainloader:
            X, y = data
            # training process
            optimizer.zero_grad()    # clear the gradient calculated previously
            predicted = net(X.view(-1, 28 * 28))  # put the mini-batch training data to Nerual Network, and get the predicted labels
            loss = F.nll_loss(predicted, y)  # compare the predicted labels with ground-truth labels
            loss.backward()      # compute the gradient
            optimizer.step()     # optimize the network
        print(f'epoch:{epoch}, loss:{loss}')

#-------------------------------------------------------------------------------------------
    """
    model.train()" and "model.eval()" activates and deactivates Dropout and BatchNorm, so it is quite important. 
    "with torch.no_grad()" only deactivates gradient calculations, but doesn't turn off Dropout and BatchNorm.
    Your model accuracy will therefore be lower if you don't use model.eval() when evaluating the model.
    """
    net.eval() # evaluation mode

    # Evaluation the trainig data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            X, y = data
            output = net(X.view(-1, 28 * 28))
            correct += (torch.argmax(output, dim=1) == y).sum().item() # 計算此次batch有多少預測正確；item()是將Tensor資料型態轉成 Python資料型態，否則Tensor型態無法與Python互相運算
            total += y.size(0) # total加上每次batch數量

    print(f'Training data Accuracy: {correct}/{total} = {round(correct/total, 3)}')

    # Evaluation the testing data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            X, y = data
            output = net(X.view(-1, 28 * 28))
            correct += (torch.argmax(output, dim=1) == y).sum().item()
            total += y.size(0)

    print(f'testing data Accuracy: {correct}/{total} = {round(correct/total, 3)}')

# -------------------------------------------------------------------------------------
    X = testset[0][0]
    y = torch.tensor([testset[0][1]])

    output = net(X.view(-1, 28 * 28))
    Error = F.nll_loss(output, y)
    v = [torch.rand_like(p) for p in net.parameters()]
    Hv = Hv_computer(list(net.parameters()), [Error], v)

    print('shape of Hv: ',len(Hv))
    for i in range(8):
        print('layer ',i,' : ',Hv[i].size())
    print('Hv:', Hv)



if __name__ == '__main__':
    main()
