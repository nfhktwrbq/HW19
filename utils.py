from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt

RESCALE_SIZE = 128

class PicDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files, transformList):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)   
        self.transformList = transformList
        self.len_ = len(self.files)    
                      
    def __len__(self):
        return self.len_
      
    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image
  
    def __getitem__(self, index):
        # для преобразования изображений в тензоры PyTorch и нормализации входа
        transform = transforms.Compose([  
            transforms.Resize((RESCALE_SIZE , RESCALE_SIZE )),          
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        xPath, yPath = self.files[index]
        x = self.load_sample(xPath)
        x = self._prepare_sample(x)

        y = self.load_sample(yPath)
        y = self._prepare_sample(y)

        if 'RANDOM_HOR_FLIP' in self.transformList:
            hflip = np.random.sample() < 0.5
            if hflip:
                x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                y.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        
        if 'RANDOM_CROP' in self.transformList:
            x0 = np.random.sample() *50
            x1 = np.random.sample() *50
            y0 = np.random.sample() *50
            y1 = np.random.sample() *50
            width, height = x.size
            x = x.crop((x0, y0, width-x1, height-y1))
            y = y.crop((x0, y0, width-x1, height-y1))

        if 'RANDOM_ROTATION' in self.transformList:
            angle = np.random.sample() * 360
            x = x.rotate(angle)
            y = y.rotate(angle)

        if 'RANDOM_VERT_CROP' in self.transformList:
            y0 = np.random.sample() *50
            y1 = np.random.sample() *50
            width, height = x.size
            x = x.crop((0, y0, width, height-y1))
            y = y.crop((0, y0, width, height-y1))


        x = transform(x)
        #x = np.array(x / 255, dtype='float32')
        #x = torch.tensor(x)
        #x = x/255
        
        
        y = transform(y) 
        #y = y/255
        #y = np.array(y / 255, dtype='float32')
        #y = torch.tensor(y)

        return x, y
        
    def _prepare_sample(self, image):
        #image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return image
"""
def printPics(inp, outp):
    plt.figure(figsize=(18, 6))
    for i in range(min([6, len(inp)])):        
        plt.subplot(2, 6, i+1)
        plt.axis("off") 
        X = inp[i]
        Y = outp[i]
        X = X.numpy().transpose((1, 2, 0))
        X = (X-X.min())/(X.max()-X.min()) 
        plt.imshow(X)
        plt.subplot(2, 6, i+7)
        plt.axis("off")
        Y = Y.numpy().transpose((1, 2, 0))
        Y = (Y-Y.min())/(Y.max()-Y.min()) 
        plt.imshow(Y/(Y.max() - Y.min()) - Y.min())
    plt.show()"""


def printPics(inp, outp, real):
    real = (real-real.min())/(real.max()-real.min()) 
    inp = (inp-inp.min())/(inp.max()-inp.min()) 
    outp = (outp-outp.min())/(outp.max()-outp.min())         
    plt.figure(figsize=(18, 9))
    for k in range(min([6, len(outp)])):
        plt.subplot(3, 6, k+1)
        plt.imshow(np.rollaxis(inp[k].numpy(), 0, 3), cmap='gray')
        plt.title('Input')
        plt.axis('off')            
        plt.subplot(3, 6, k+7)            
        plt.imshow(np.rollaxis(outp[k].numpy(), 0, 3), cmap='gray')
        plt.title('Output')
        plt.axis('off')   
        plt.subplot(3, 6, k+13)            
        plt.imshow(np.rollaxis(real[k].numpy(), 0, 3), cmap='gray')
        plt.title('Real')
        plt.axis('off')  
    plt.show()    

def test(model, test_loader, device):
    with torch.no_grad():       
        outputs = []
        for inputs, _  in test_loader:
            inputs = inputs.to(device)
            model.eval()
            output = model(inputs).cpu()
            outputs.append(output)
            printPics(inputs.cpu(), output)
    return outputs