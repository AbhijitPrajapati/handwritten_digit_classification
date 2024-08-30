from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from config import batch_size

trainds = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor()            
)
testds = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

traindl = DataLoader(trainds, batch_size=batch_size, shuffle=True)
testdl = DataLoader(testds, batch_size=batch_size, shuffle=False)