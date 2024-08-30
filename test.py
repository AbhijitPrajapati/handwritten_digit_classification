from torch import load
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import ImageDraw, Image, ImageOps, ImageFilter
import tkinter as tk
from train import DigitClassifier
from util import predict, display_image

class DrawingApp:
    def __init__(self, root):
        self.model = DigitClassifier()
        self.model.load_state_dict(load('model.pt', weights_only=True))

        self.root = root
        
        self.canvas = tk.Canvas(self.root, bg='white', width=800, height=800)
        self.canvas.pack()

        self.label = tk.Label(root, text='', font=('Arial', 24))
        self.label.pack(pady=20)

        self.create_new_image()

        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)
        self.root.bind('<BackSpace>', self.clear)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-30, y-30, x+30, y+30, fill='black', outline='black')
        self.draw.ellipse([x-30, y-30, x+30, y+30], fill='black')

    def on_button_release(self, event):
        # model input shape: (1, 1, 28, 28)
        input_tensor = self.get_image().unsqueeze(0)
        pred_digit = predict(self.model, input_tensor)
        self.label.configure(text=f'Prediction: {pred_digit}')
        
    
    def create_new_image(self):
        self.image = Image.new('L', (800, 800), 'white')
        self.draw = ImageDraw.Draw(self.image)

    def clear(self, event):
        self.canvas.delete('all')
        self.create_new_image()
    
    def get_image(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        img = img.crop(bbox)
        out = Image.new('L', (28, 28), color='black')
        paste_pos = ((28 - img.width) // 2, (28 - img.height) // 2)
        out.paste(img, paste_pos)
        out = out.filter(ImageFilter.GaussianBlur(1))
        out = ToTensor()(out)
        return out



def main():
    
    
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()



if __name__ == '__main__':
    main()
