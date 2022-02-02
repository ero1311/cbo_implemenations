from turtle import forward
import torch.nn as nn
import torch
import timeit

class FashionCNN(nn.Module):

    def __init__(self):
        super(FashionCNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, _, _, _ = x.shape
        conv = self.relu(self.conv(x))
        pool = self.max_pool(conv)
        #drop = self.drop(pool)
        fc1 = self.relu(self.fc1(pool.view(B, -1)))
        out = self.fc2(fc1)

        return out

class FashionFC(nn.Module):

    def __init__(self):
        super(FashionFC, self).__init__()
        self.fc = nn.Linear(784, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        B, _, _, _ = x.shape
        return self.relu(self.fc(x.view(B, -1)))

class FashionAE(nn.Module):

    def __init__(self):
        super(FashionAE, self).__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(32, 1, 2, stride=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        enc1 = self.relu(self.enc1(x))
        enc2 = self.relu(self.enc2(enc1))
        dec1 = self.relu(self.dec1(enc2))
        dec2 = self.relu(self.dec2(dec1))

        return dec2

if __name__ == '__main__':
    x = torch.randn(64, 1, 28, 28)
    model = FashionAE()
    start = timeit.default_timer()
    result = model(x)
    print("Inference on CPU: {} seconds".format(timeit.default_timer() - start))
    print(result.shape)
    state_dict = model.state_dict()
    for w_name, w_vals in state_dict.items():
        print(w_name, w_vals.shape)