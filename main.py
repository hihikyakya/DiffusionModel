import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms.functional import normalize
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import math

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RestoreNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.main1_1=self.convblock(256,128,3,1,1)
    self.main1_2=self.convblock(128,64,3,1,1)
    self.main1_3=self.convblock(64,32,3,1,1)
    self.main1_4=self.convblock(32,3,3,1,1)
    self.sig=nn.Sigmoid()
    self.main2_1=self.convblock(3,32,3,1,1)
    self.main2_2=self.convblock(32,64,3,1,1)
    self.main2_3=self.convblock(64,128,3,1,1)
    self.main2_4=self.convblock(128,256,3,1,1)
    self.tanh=nn.Tanh()
  def convblock(self,in_channel,out_channel,kernel_size,stride,padding,groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding,bias=False),#8->16
        nn.LeakyReLU(0.2),
        nn.InstanceNorm2d(out_channel,affine=True),
    )
  def forward(self,input,noize,amount):
    x=self.main2_1(input+amount*noize)
    x=self.main2_2(x)
    x=self.main2_3(x)
    x=self.main2_4(x)
    x=self.sig(x)
    x=self.main1_1(x)
    x=self.main1_2(x)
    x=self.main1_3(x)
    x=self.main1_4(x)
    output=self.tanh(x)
    return output
def test(model,content,noize,batch_size,amount=1):
  with torch.no_grad():
    topil=transforms.ToPILImage()
    feature=model(content,noize,amount)
    img_grid_content = torchvision.utils.make_grid(
          feature[:batch_size], normalize=True
      )
    plt.imshow(topil(img_grid_content))
    plt.show()
def train(model,args):
  resize=args.resize
  batch_size=args.batch_size
  lr=args.lr
  epochs=args.epochs
  transform=transforms.Compose([transforms.ToTensor(),transforms.Resize([resize,resize]),
                                      transforms.Normalize([0.5 for i in range(3)],[0.5 for i in range(3)])
                                      ])
  dataset=datasets.ImageFolder(root='/content/drive/MyDrive/Contents',transform=transform)
  dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
  criterion=nn.MSELoss()
  m=model().to(device)
  m.train()
  optim_m=optim.Adam(m.parameters(),betas=(0.5,0.9),lr=lr)
  amount=1/epochs

  for epoch in range(epochs):
    for batch,(content,_) in enumerate(dataloader):
      if(content.shape[0]==batch_size):
        content=content.to(device)
        NOIZE=torch.rand(batch_size,3,resize,resize).to(device)
        feature=m(content,NOIZE,amount)
        loss=criterion(feature,content)
        loss.backward()
        optim_m.step()
        optim_m.zero_grad()
        test(m,content,NOIZE,batch_size)
    print(f"loss: {loss.item()}")
    amount=math.sqrt(epoch/epochs)

import argparse

if(__name__=="__main__"):
  parser = argparse.ArgumentParser(description='for train the model')
  parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        metavar="E",
        help="input epochs for training (default: 20)",
    )
  parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
  parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="input learning rate for training (default: 0.01)",
    )
  parser.add_argument(
        "--resize",
        type=int,
        default=128,
        metavar="RS",
        help="input resize for training (default: 128)",
    )
  train(RestoreNet,parser.parse_args(args=[]))