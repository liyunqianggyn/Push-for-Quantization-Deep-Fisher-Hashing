import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from PIL import Image
import os
import os.path
from models.DFH_loss import DFHLoss_margin
import matplotlib.image as mpimg
from cal_map import calculate_top_map, calculate_map, compress
from center import Relaxcenter, Discretecenter

# Hyper Parameters
num_epochs = 150
batch_size = 128
epoch_lr_decrease = 50
learning_rate = 0.01
num_classes = 80

mu =1
vul = 1
nta = 1
eta = 0.5
margin = 0

encode_length_all = [16, 32,48,64]
len_encodeL_all = len(encode_length_all)


mAP_all = np.zeros([len_encodeL_all, 1])
mAP_all_top = np.zeros([len_encodeL_all, 1])


def pil_loader(path):
   
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MSCOCO(torch.utils.data.Dataset):

    def __init__(self, root,
                 transform=None, target_transform=None, train=True, database_bool=False):
        self.loader = default_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train_coco.txt' 
        elif database_bool:
            self.base_folder = 'database_coco.txt'
        else:
            self.base_folder = 'test_coco.txt'

        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)
        # fo = open(file, 'rb')

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                # print lines.split()
                if not lines:
                    break
                pos_tmp = lines.split()[0]
                # print pos_tmp
#                pos_tmp = os.path.join(self.root, pos_tmp)
                pos_tmp = os.path.join(self.root, pos_tmp[39:])
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        # self.train_labels.reshape()
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        self.train_labels.reshape((-1, num_classes))

        self.onehot_targets = torch.from_numpy(self.train_labels).float()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]
        #target = int(np.where(target == 1)[0])

        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.train_data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return self.onehot_targets

class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.alex = torchvision.models.alexnet(pretrained=True)
        self.alex.classifier = nn.Sequential(*list(self.alex.classifier.children())[:6])
        self.fc_plus = nn.Linear(4096, encode_length)

    def forward(self, x):
        x = self.alex.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alex.classifier(x)
        x = self.fc_plus(x)

        return x

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MSCOCO(root='./data/coco/',
                            train=True,
                            transform=train_transform)

    test_dataset = MSCOCO(root='./data/coco/',
                            train=False,
                            transform=test_transform)

    database_dataset = MSCOCO(root='./data/coco/',
                            train=False,
                            transform=test_transform,
                            database_bool=True)


    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)


    cnn = CNN(encode_length=encode_length)
    # Loss and Optimizer
    criterion = DFHLoss_margin(eta, margin)
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
            
    best_top = 0.0
    best = 0.0    


    # Initialize for centers
    N = len(train_loader.dataset)
    U = torch.zeros(encode_length, N)
    train_targets = train_loader.dataset.get_onehot_targets()
    S = (train_targets @ train_targets.t() > 0).float()
    Y = train_targets.t()
    
    # multi-label process
    Multi_Y = Y.sum(0).expand(Y.size())
    Multi_Y = 1./Multi_Y
    Y = Multi_Y*Y
    
    Relax_center = torch.zeros(encode_length, num_classes) 
    CenTer = Relax_center
    

    # Train the Model
    for epoch in range(num_epochs):
        cnn.cuda().train()
        adjust_learning_rate(optimizer, epoch)
        for i, (images, labels, index) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().long())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            U_batch = cnn(images).t()


            # Prepare
            U[:, index] = U_batch.cpu().data
            batchY = Y[:, index]
            batchS = S[:, index]


            # B-step            
            batchB = (mu * CenTer@batchY + U_batch.cpu()).sign()

            # C-step: two methods - relax and discrete
            """
            First: relax method
            """
            """
            CenTer, Relax_center = Relaxcenter(Variable(batchY.cuda(), requires_grad=False), \
                                                   Variable(batchB.cuda(), requires_grad=False), \
                                                   Variable(Relax_center.cuda(), requires_grad=True), mu, vul, nta);
            """
            """
            Second: discrete method
            """                      
            CenTer = Discretecenter(Variable(batchY.cuda(), requires_grad=False), \
                                                   Variable(batchB.cuda(), requires_grad=False), \
                                                   Variable(CenTer.t().cuda(), requires_grad=True), mu, vul);

            # U-step+ Backward + Optimize                     
            loss = criterion(U_batch, Variable(U.cuda()), Variable(batchS.cuda()), Variable(batchB.cuda()))
        
            loss.backward()
            optimizer.step()


 
        # Test the Model
        if (epoch + 1) % 10 == 0:
            cnn.eval()
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn, classes=num_classes)
            
            print(np.shape(retrievalB))
            print(np.shape(retrievalL))
            print(np.shape(queryB))
            print(np.shape(queryL))


            print('-----calculate top 5000 map-------')
            result = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=5000)
            print(result)

            if result > best_top:
                best_top = result
                mAP_all_top[code_index, 0] = result
                print('-------------Best mAP for all bits-------------')
                print(mAP_all_top) 



if __name__ == '__main__':
    for code_index in range(len_encodeL_all):
        encode_length = encode_length_all[code_index] 
        print('-------------encode_length----------------')
        print(encode_length_all)
        main() 
