from torch.cuda import is_available

batch_size = 64
learning_rate = 0.0001
beta1, beta2 = 0.9, 0.999
epsilon = 1e-8
num_epochs = 30
gamma = 1
patience = 5
min_delta = 0.01
device = 'cuda' if is_available() else 'cpu'
