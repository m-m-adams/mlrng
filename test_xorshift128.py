# %%
# from functools import partial
from cgi import test
import sched
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.bayesopt import BayesOptSearch


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

# take in a 1xN array, return bwxN where bw is the bit width
# LSB first binary


def bit_array(arr_in):
    BIT_WIDTH = np.ceil(np.log2(np.amax(arr_in))).astype(int)
    bits_out = arr_in[:, None] & (1 << np.arange(BIT_WIDTH))
    bits_out[bits_out > 0] = 1
    return bits_out


def gen_data(n=2_000_000):
    rng = xorshift128()
    # generate 2 000 000 random numbers
    inputs = np.array([rng() for _ in range(n)])
    rand_bits = bit_array(inputs)
    return strided(rand_bits, 5)


def preprocess(raw_rng):
    y = raw_rng[:, -1, :]  # last one in every window
    x = raw_rng[:, :-1, :]  # first four in every window
    X = x.reshape([x.shape[0], x.shape[1]*x.shape[2]])

    test_count = 100_000
    X_train = torch.Tensor(X[test_count:])
    X_test = torch.Tensor(X[:test_count])
    y_train = torch.Tensor(y[test_count:])
    y_test = torch.Tensor(y[:test_count])
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    return train_dataset, test_dataset


class Net(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.input = torch.nn.Linear(128, hidden)

        self.output = torch.nn.Linear(hidden, 32)

    def forward(self, x):
        x = F.relu(self.input(x))
        return torch.sigmoid(self.output(x))


def train(model, optimizer, loader, device, criterion=torch.nn.BCELoss(), progress=False):
    # train a single epoch
    running_loss = 0
    for i, data in enumerate(loader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if (i+1) % 50 == 0:    # print every 50 mini-batches
            if progress:
                print(
                    f'[{i + 1:5d}] loss: {running_loss / 50:.3f} processed: {i*len(data[0])}')
            running_loss = 0.0


def validate(model, loader, device, criterion=nn.BCELoss()):
    val_loss = 0.0
    val_steps = 0
    total = 0
    correct = 0
    #  no gradients needed
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.round(model(inputs))
            loss = criterion(outputs, labels)
            val_loss += loss.cpu().numpy()
            correct += (outputs == labels).sum().item()
            val_steps += 1
            total += labels.size(0)*labels.size(1)
    return ((val_loss / val_steps), correct / total)


def train_tune(config, train_dataset=None, checkpoint_dir=None):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    test_abs = int(len(train_dataset) * 0.8)
    train_subset, val_subset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs])
    train_loader = DataLoader(train_subset, batch_size=10000)
    val_loader = DataLoader(val_subset, batch_size=10000)

    model = Net(1024).to(device)

    optimizer = torch.optim.NAdam(
        model.parameters(), lr=config["lr"], eps=config["eps"],
        betas=(config["beta1"], config["beta2"]))
    criterion = torch.nn.BCELoss()

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    for epoch in range(5):
        train(model, optimizer, train_loader, device,
              criterion=criterion)

        loss, accuracy = validate(
            model, val_loader, device, criterion=criterion)

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=loss, accuracy=accuracy)


def tune(train_dataset):
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "eps": tune.loguniform(1e-7, 1e-5),
        "beta1": tune.uniform(0.8, 0.9),
        "beta2": tune.uniform(0.85, 0.99)
    }
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=10,
        grace_period=1,
        reduction_factor=2)

    search = BayesOptSearch(metric="accuracy", mode="max")

    print("starting ray tune")
    analysis = tune.run(tune.with_parameters(train_tune, train_dataset=train_dataset),
                        config=search_space,
                        num_samples=50,
                        resources_per_trial={"gpu": 0.5},
                        progress_reporter=reporter,
                        scheduler=scheduler,
                        search_alg=search
                        )

    best_trial = analysis.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    return best_trial.config


def train_complete(config, train_dataset, epochs):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    test_abs = int(len(train_dataset) * 0.8)
    train_subset, val_subset = random_split(
        train_dataset, [test_abs, len(train_dataset) - test_abs])
    train_loader = DataLoader(train_subset, batch_size=10000)
    val_loader = DataLoader(val_subset, batch_size=10000)

    model = Net(1024).to(device)

    optimizer = torch.optim.NAdam(
        model.parameters(), lr=config["lr"], eps=config["eps"],
        betas=(config["beta1"], config["beta2"]))
    criterion = torch.nn.BCELoss()
    last_loss = 1
    for epoch in range(epochs):
        train(model, optimizer, train_loader, device,
              criterion=criterion, progress=True)

        loss, accuracy = validate(
            model, val_loader, device, criterion=criterion)

        if loss > last_loss:
            return model
        last_loss = loss

        path = os.path.join("checkpoints", 'epoch'+str(epoch))
        print(f"finished epoch {epoch} with accuracy {accuracy}")
        torch.save((model.state_dict(), optimizer.state_dict()), path)
    return model


def main():
    # per RFC 1149.5
    myrand = 4
    np.random.seed(myrand)
    torch.manual_seed(myrand)
    raw_rng = gen_data()

    train_dataset, test_dataset = preprocess(raw_rng)
    test_loader = DataLoader(test_dataset, batch_size=10000)
    best_conf = {'lr': 0.0009968361945817176, 'eps': 4.80867121379225e-07,
                 'beta1': 0.8627089454715666, 'beta2': 0.947364904619059}
    if sys.argv[-1] == "tune":
        print("tuning")
        best_conf = tune(train_dataset)

    model = train_complete(best_conf, train_dataset, 50)
    torch.save(model, "trained_model.pt")

    results = validate(model, test_loader, "cuda")
    print(results)


if __name__ == "__main__":
    main()
