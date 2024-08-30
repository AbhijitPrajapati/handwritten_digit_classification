import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
from architecture import DigitClassifier
from torch.nn.functional import one_hot
from data import traindl, testdl
from config import *
from util import plot_metrics
from early_stopping import EarlyStopping


# run every epoch to train
# uses training data
def train_epoch(model, loss_func, optimizer, scheduler):
    model.train()
    epoch_training_losses = []
    for x, y in traindl:
        xb, yb, = x.cuda(), one_hot(y, num_classes=10).float().cuda()
        pred = model(xb)
        loss = loss_func(pred, yb)
        epoch_training_losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
    scheduler.step()
    return sum(epoch_training_losses) / len(epoch_training_losses)

    
# run after train_epoch each epoch in order to validate model on unseen data
# uses testing data
def validate_epoch(model, loss_func):
    model.eval()
    epoch_validation_losses = []
    # validation with no grad to reduce runtime
    with torch.no_grad():
        for x, y in testdl:
            xb, yb, = x.cuda(), one_hot(y, num_classes=10).float().cuda()
            pred = model(xb)

            loss = loss_func(pred, yb)
            epoch_validation_losses.append(loss.item())

    avg_validation_loss = sum(epoch_validation_losses) / len(epoch_validation_losses)
    return avg_validation_loss

def main():
    model = DigitClassifier()
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate, (beta1, beta2), eps=epsilon)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    early_stopping = EarlyStopping(patience, min_delta)

    losses = []
    # training loop
    for epoch in range(1, num_epochs + 1):
        train_epoch(model, loss_function, optimizer, scheduler)
        avg_loss = validate_epoch(model, loss_function)
        early_stopping(avg_loss)
        if early_stopping.early_stop(): break
        print(f'Epoch: {epoch}/{num_epochs}\nLoss: {avg_loss}')
        losses.append(avg_loss)

    plot_metrics(losses)

    save_model = int(input('Confirm save? [1/0] '))
    if save_model:
        torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    main()
