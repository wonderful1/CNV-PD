import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os 
import sys
import numpy as np

#########################################
#ref: https://www.cnblogs.com/denny402/p/7520063.html
# python pytorch_CNN.class.py  <testtxt> <title> <class_num> <model> <outdir>
testtxt=sys.argv[1] # ab path/txt of test_data
title=sys.argv[2]
class_num=int(sys.argv[3]) # out put class 2/4
CNN=sys.argv[4]
outdir=sys.argv[5] #outdir


# -----------------ready the dataset--------------------------
def default_loader(path):
    return Image.open(path).convert('RGB')
class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

test_data=MyDataset(txt=testtxt, transform=transforms.ToTensor())
test_loader = DataLoader(dataset=test_data, batch_size=64)

#########################
#-----------------create the Net------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, class_num)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        res = conv4_out.view(conv4_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net()
#print(model)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.to(device)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()



#####################
#-----------------predition and evaluation------------------------
#load mode
model.load_state_dict(torch.load(CNN))
model.eval()

# evaluation--------------------------------
eval_loss = 0.
eval_acc = 0.
pre_list=[]
for batch_x, batch_y in test_loader:
    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    out = model(batch_x)
    loss = loss_func(out, batch_y)
    eval_loss += loss.item()
    pred = torch.max(out, 1)[1]
    num_correct = (pred == batch_y).sum()
    pre_np=pred.numpy()
    pre_li=pre_np.tolist()
    pre_list.append(pre_li)
    eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(test_data)), eval_acc / (len(test_data))))


f = open(outdir+"/"+title+"."+str(class_num)+".pre.txt",'a')
for i in pre_list:
    f.write(str(i))
print("save file successfully！")

