#%%
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from ray import tune
from ray.tune.schedulers import ASHAScheduler

#%%

# the standardized random number
myrand = 4
np.random.seed(myrand)
torch.manual_seed(myrand)

def xorshift128():
    x = 123456789
    y = 362436069
    z = 521288629
    w = 88675123

    def _random():
        nonlocal x, y, z, w
        t = x ^ ((x << 11) & 0xFFFFFFFF)  # 32bit
        x, y, z = y, z, w
        w = (w ^ (w >> 19)) ^ (t ^ (t >> 8))
        return w

    return _random


# convert an AxBxN np array to AxBxLx(N-L)
def strided(a, L):
    s = a.strides
    nd0 = a.shape[0]-L+1
    shape_out = (nd0, L)+a.shape[1:]

    strides_out = (s[0],) + s
    return np.lib.stride_tricks.as_strided(a, shape=shape_out, strides=strides_out)

#take in a 1xN array, return bwxN where bw is the bit width
#LSB first binary
def bit_array(arr_in):
    BIT_WIDTH = np.ceil(np.log2(np.amax(arr_in))).astype(int)
    bits_out = inputs[:,None] & (1 << np.arange(BIT_WIDTH))
    bits_out[bits_out > 0]=1
    return bits_out


#%%
rng = xorshift128()
#generate 2 000 000 random numbers
inputs = np.array([rng() for x in range(2_000_000)])
rand_bits = bit_array(inputs)

windowed_bits = strided(rand_bits,5)

y = windowed_bits[:,-1,:] #last one in every window
x = windowed_bits[:,:-1,:] #first four in every window
X = x.reshape([x.shape[0], x.shape[1]*x.shape[2]])
test_count = 100_000
X_train = torch.Tensor(X[test_count:]).to('cuda')
X_test = torch.Tensor(X[:test_count]).to('cuda')
y_train = torch.Tensor(y[test_count:]).to('cuda')
y_test = torch.Tensor(y[:test_count]).to('cuda')
#%%
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=10000)
test_loader = DataLoader(test_dataset, batch_size=test_count)


model = torch.nn.Sequential(
    torch.nn.Linear(128, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024,32),
    torch.nn.Sigmoid()
).to('cuda')
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3, betas=(0.9,0.98), eps=1.5e-07)
criterion = torch.nn.BCELoss()


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 100 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

# %%

# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
print(running_loss)
# %%
