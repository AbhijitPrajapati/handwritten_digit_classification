import matplotlib.pyplot as plt
from torch import argmax

# pytorch tensor in shape 1x28x28
def display_image(img):
    plt.imshow(img[0], cmap='gray')
    plt.axis('off')
    plt.show()

def plot_metrics(*args):     
    x = range(len(args[0]))
    for metric in args:
        plt.plot(x, metric)
    plt.title('Stock Price Training Graph')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.show()

def predict(model, x):
    out = model(x)
    digit = argmax(out)
    return digit.item()