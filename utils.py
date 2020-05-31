from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import PIL

RESCALE_SIZE = 128

class PicDataset(Dataset):
    """
    Датасет с картинками, который паралельно подгружает их из папок
    производит скалирование и превращение в торчевые тензоры
    """
    def __init__(self, files):
        super().__init__()
        # список файлов для загрузки
        self.files = sorted(files)   

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

        hflip = np.random.sample() < 0.5
        if hflip:
            x.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            y.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        
        crop = np.random.sample() < 0.5
        if crop:
            x0 = np.random.sample() *50
            x1 = np.random.sample() *50
            y0 = np.random.sample() *50
            y1 = np.random.sample() *50
            width, height = x.size
            x = x.crop((x0, y0, width-x1, height-y1))
            y = y.crop((x0, y0, width-x1, height-y1))

        x = transform(x)
        x = np.array(x / 255, dtype='float32')
        x = torch.tensor(x)
        
        
        
        y = transform(y) 
        y = np.array(y / 255, dtype='float32')
        y = torch.tensor(y)

        return x, y
        
    def _prepare_sample(self, image):
        #image = image.resize((RESCALE_SIZE, RESCALE_SIZE))
        return image