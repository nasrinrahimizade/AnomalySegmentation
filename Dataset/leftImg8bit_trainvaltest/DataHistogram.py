import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import Counter
from torchvision import transforms


EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def is_label(filename):
    return filename.endswith("_labelTrainIds.png")

def image_path_city(root, name):
    return os.path.join(root, f'{name}')

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')
        
        self.images_root += subset
        self.labels_root += subset

        print (self.images_root)
        #self.filenames = [image_basename(f) for f in os.listdir(self.images_root) if is_image(f)]
        self.filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.images_root)) for f in fn if is_image(f)]
        self.filenames.sort()

        #[os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(".")) for f in fn]
        #self.filenamesGt = [image_basename(f) for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenamesGt = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root)) for f in fn if is_label(f)]
        self.filenamesGt.sort()

        self.co_transform = co_transform # ADDED THIS


    def __getitem__(self, index):
        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        if self.co_transform is not None:
            image = self.co_transform(image)
            label = self.co_transform(label)
            
            # image, label = self.co_transform(image, label)

        return image, label

    def __len__(self):
        return len(self.filenames)

def main():
    # Example data and labels
    dataset_path = 'D:\COMPUTER_DEPARTMENT\\3RD_YEAR\AML\AML_Project_v2.0\Dataset\leftImg8bit_trainvaltest'
    transform = transforms.ToTensor()

    dataset = cityscapes(dataset_path,co_transform=transform,subset='val')
    # print(dataset.__len__())
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    label_counts = Counter()
    print('Done with 0 %',end='\r')
    i =0
    for _, labels in loader:
        print(f'Done with {len(labels)*i*100/dataset.__len__()} %',end='\r')
        i+=1
        label_counts.update(labels.view(-1).tolist() )
    print('\n')
    print(label_counts)
    
    # Compute class weights
    total_count = sum(label_counts.values())
    num_classes = len(label_counts)
    class_weights = {label: total_count / count for label, count in label_counts.items()}
    weights = [class_weights[i] for i in range(num_classes)]
    weights_tensor = torch.tensor(weights, dtype=torch.float)






if __name__ == '__main__':
    main()
    