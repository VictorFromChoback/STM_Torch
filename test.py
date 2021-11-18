from src.algo_stm import STM_Method
import torch
from src.rosenbrock import optimize_rosenbrock
import numpy as np

class Net(torch.nn.Module):
    
    def __init__(self):
        
        super(Net, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=0)
    
        self.relu = torch.nn.ReLU()
        
        self.dropout = torch.nn.Dropout(0.5)
        
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=0)
        
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
    
        self.fc = torch.nn.Linear(32 * 5 * 5, 10)
        
        
    def forward(self, x):
        
        # conv 1
        out = self.conv1(x)
        out = self.relu(out)
    
        # maxpool
        out = self.maxpool(out)
        out = self.dropout(out)
        
        # conv 2
        out = self.conv2(out)
        out = self.relu(out)
        
        # maxpool
        out = self.maxpool(out)
        
        # to linear
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


def sumq(xy):
    return xy[0] ** 2 + xy[1] ** 2


def func(start_point, optim, iterations: int, **optimizer_kwargs):

    xy = torch.tensor(start_point, requires_grad=True)
    optimizer = optim([xy], **optimizer_kwargs)

    path = np.empty((iterations + 1, 2))
    path[0] = start_point

    for iteration in range(1, iterations):
        optimizer.zero_grad()
        loss = sumq(xy)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(xy, 1)
        optimizer.step()

        path[iteration] = xy.data
    
    return path


def main():

    # obj = Net()
    
    # start = [0.2, 0.3]
    # params = [torch.tensor(start, requires_grad=True)]
    # opt = STM_Method(obj.parameters())
    # opt.step()

    # _ = optimize_rosenbrock(np.array([3, 3], dtype=float), torch.optim.Adam, 10, lr=1e-8)
    # _ = optimize_rosenbrock(np.array([3, 3], dtype=float), STM_Method, 30, lr=1e-3)
    _ = func(np.array([3, 3], dtype=float), STM_Method, 30, lr=1e-2)


    print('Try jupyter')

if __name__ == '__main__':
    main()