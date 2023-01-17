from __future__ import print_function
import argparse
import os
from math import log10
from grad import GradLoss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from guided_filter_pytorch.guided_filter import GuidedFilter
#from PIL import Image
#import torch
#import torch.utils.data as data
#import torchvision.transforms as transforms
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision import transforms as trans
from utils import is_image_file, load_img



from networks import define_G, define_D, GANLoss, get_scheduler, update_learning_rate
from data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--nepochs', type=int, default=50, help='saved model of which epochs')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = "dataset/"
train_set = get_training_set(root_path + opt.dataset, opt.direction)
test_set = get_test_set(root_path + opt.dataset, opt.direction)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)

device = torch.device("cuda:1" if opt.cuda else "cpu")
#device = torch.device("cuda:1")
#model_path = "checkpoint/{}/netG1_model_epoch_50.pth".format(opt.dataset, opt.nepochs)

#net_g1 = torch.load(model_path).to(device)
print('===> Building models')
#net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
net_g1 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
#net_g2 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
#net_g3 = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)

net_d = define_D(opt.input_nc+ opt.output_nc , opt.ndf, 'basic', gpu_id=device)
#net_d1 = define_D(opt.input_nc+ opt.output_nc , opt.ndf, 'basic', gpu_id=device)
#net_d2 = define_D(opt.input_nc+ opt.output_nc , opt.ndf, 'basic', gpu_id=device)
#net_d3 = define_D(opt.input_nc+ opt.output_nc, opt.ndf, 'basic', gpu_id=device)
#net_d4 = define_D(opt.input_nc+ opt.output_nc, opt.ndf, 'basic', gpu_id=device)

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
#optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_g1 = optim.Adam(net_g1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_g2 = optim.Adam(net_g2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_g3 = optim.Adam(net_g3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

#optimizer_xfm = optim.Adam(xfm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_ifm = optim.Adam(ifm.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_d1 = optim.Adam(net_d1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_d2 = optim.Adam(net_d2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_d3 = optim.Adam(net_d3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizer_d4 = optim.Adam(net_d4.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#net_g_scheduler = get_scheduler(optimizer_g, opt)
net_g1_scheduler = get_scheduler(optimizer_g1, opt)
#net_g2_scheduler = get_scheduler(optimizer_g2, opt)
#net_g3_scheduler = get_scheduler(optimizer_g3, opt)
#xfm_scheduler = get_scheduler(optimizer_xfm, opt)
#ifm_scheduler = get_scheduler(optimizer_ifm, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)
#net_d1_scheduler = get_scheduler(optimizer_d1, opt)
#net_d2_scheduler = get_scheduler(optimizer_d2, opt)
#net_d3_scheduler = get_scheduler(optimizer_d3, opt)
#net_d4_scheduler = get_scheduler(optimizer_d4, opt)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    # train
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        #real_a,real_b= batch[0].to(device), batch[1].to(device)
        real_a,real_b, B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121= batch[0].to(device), batch[1].to(device),batch[2].to(device), batch[3].to(device),batch[4].to(device), batch[5].to(device), batch[6].to(device),batch[7].to(device),batch[8].to(device), batch[9].to(device),batch[10].to(device), batch[11].to(device),batch[12].to(device), batch[13].to(device),batch[14].to(device), batch[15].to(device),batch[16].to(device), batch[17].to(device),batch[18].to(device), batch[19].to(device),batch[20].to(device), batch[21].to(device),batch[22].to(device), batch[23].to(device),batch[24].to(device), batch[25].to(device),batch[26].to(device), batch[27].to(device),batch[28].to(device), batch[29].to(device),batch[30].to(device), batch[31].to(device),batch[32].to(device), batch[33].to(device),batch[34].to(device), batch[35].to(device),batch[36].to(device), batch[37].to(device),batch[38].to(device), batch[39].to(device),batch[40].to(device), batch[41].to(device),batch[42].to(device), batch[43].to(device),batch[44].to(device), batch[45].to(device),batch[46].to(device), batch[47].to(device),batch[48].to(device), batch[49].to(device),batch[50].to(device), batch[51].to(device),batch[52].to(device), batch[53].to(device),batch[54].to(device), batch[55].to(device),batch[56].to(device), batch[57].to(device),batch[58].to(device), batch[59].to(device),batch[60].to(device), batch[61].to(device)
        xfm = DWTForward(J=1, mode='zero', wave='haar').cuda(1)  # Accepts all wave types available to PyWavelets
        ifm = DWTInverse(mode='zero', wave='haar').cuda(1)
        YY, Yh = xfm(real_b)
#        print(num0)
#        print(num1)
#        print(num2)
     #   YL, YLL = xfm(real_a)
#        qwe=GradLoss().cuda(1)
        
        
#        qwe=GradLoss().cuda(1)
#        pq_a=qwe(real_a)
#        pq_b=qwe(real_b)
#        fake_N=pq_a-pq_b
        #print(real_a.size())
        #real_a,real_b= batch[0].to(device), batch[1].to(device),
#        real_a = transforms.ToTensor()(real_a)
#        real_b = transforms.ToTensor()(real_b)
#      #  w_offset = random.randint(0, max(0, 512 - 256 - 1))
#      #  h_offset = random.randint(0, max(0, 512 - 256 - 1))
#    
##        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
##        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
#          for i in range(0,448):       
#            for j in range(0,448):
#               (x1,y1,x2,y2)=(i,j,i+64,j+64)
#               box=(x1,y1,x2,y2)
#               ImCrop = input.crop(box)
#               (x,y) = ImCrop.size #read image size
#                 = ImCrop.resize((x,y),Image.ANTIALIAS)
#        real_a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_a)
#        real_b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(real_b)
#        if random.random() < 0.5:
#            idx = [i for i in range(real_a.size(2) - 1, -1, -1)]
#            idx = torch.LongTensor(idx)
#            real_a = real_a.index_select(2, idx)
#            real_b = real_b.index_select(2, idx)
        fake_bb = net_g1(real_a,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
        XX = real_a-fake_bb
        YlL, YhL = xfm(XX)
        fake_b1=YhL[0][:,:,0,:,:]
        fake_b2=YhL[0][:,:,1,:,:]
        fake_b3=YhL[0][:,:,2,:,:]
        fake_b0=YlL
#        YYB, YhB = xfm(fake_abB)
#        fake_b0= net_g(fake_abB,YYB,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)        
      #  fake_b1,fake_b2,fake_b3=net_g1(real_a,YhL[0][:,:,0,:,:], YhL[0][:,:,1,:,:], YhL[0][:,:,2,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
#        Y0L= torch.stack((fake_b1,fake_b2,fake_b3), axis=2)         
#        Y0h=list()
#        Y0h.append(Y0L)   
       # fake_YH =ifm((YlL, Y0h))

        

#        print(fake_b1.size())
#        print(fake_b2.size())
#        print(fake_b3.size())
#        print(fake_b0)
#        print(YlL)
#        print(type(YlL))
       
#        fake_b2= net_g2(YhL[0][:,:,1,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
#        fake_b3= net_g3(YhL[0][:,:,2,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)

       # print(fake_b1.size())
#
     
       # XX = ifm((fake_b0, Y0h)) 

       # print(fake_b0.size())
     #   XX= net_g1(fake_abB)
        #print(YY.size())
      #  fake_b3,fake_b0,fake_b1,fake_b2 =batch[0].to(device),batch[1].to(device),batch[2].to(device),batch[3].to(device)
        #print(real_a.size())
        #nd_ct=real_a-fake_b1
        #grad=real_a
        #nd_ct=grad-fake_N
       # fake_b11=fake_b1-fake_b1
        #fake_b11 = net_g1(nd_ct,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
        #fake_b11 = net_g1(real_a)
        #print(pr.size())
        
        #fake_b1 = net_g1(real_a,fake_b)
       # fake_ab0 = torch.cat((real_a, fake_b), 1)
       # fake_b1 = net_g1(fake_ab0)
        ######################
        # (1) Update D network
        ######################
        #if epoch<=25:
        optimizer_d.zero_grad()
#        optimizer_d1.zero_grad()
#        optimizer_d2.zero_grad()
#        optimizer_d3.zero_grad()
#        optimizer_d4.zero_grad()
        # train with fake
#        fake_ab = torch.cat((real_a, fake_b), 1)
#        pred_fake = net_d.forward(fake_ab.detach())
#        loss_d_fake = criterionGAN(pred_fake, False)
#        if epoch<=5:
#            fake_ab1 = torch.cat((real_a, ndct), 1)
#            pred_fake1 = net_d1.forward(fake_ab1.detach())
#            loss_d1_fake = 0.5*criterionGAN(pred_fake1, False)
#           
#            fake_ab2 = torch.cat((real_a, fake_b1), 1)
#            pred_fake2 = net_d2.forward(fake_ab2.detach())
#            loss_d2_fake = 0.5*criterionGAN(pred_fake2, False)

        # train with real
#        real_ab = torch.cat((real_a, real_b), 1)
#        pred_real = net_d.forward(real_ab)
#        loss_d_real = criterionGAN(pred_real, True)
#            n_e=pr-hr_y
#            real_ab1 = torch.cat((real_a, nd_ct), 1)
#            pred_real1 = net_d1.forward(real_ab1)
#            loss_d1_real = 0.5*criterionGAN(pred_real1, True)
#            
#            real_ab2 = torch.cat((real_a, pr), 1)
#            pred_real2 = net_d2.forward(real_ab2)
#            loss_d2_real = 0.5*criterionGAN(pred_real2, True)

        fake_ab = torch.cat((real_a, XX), 1)
#        fake_ab = XX
        pred_fake = net_d.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)

#        fake_ab1 = torch.cat((YY, fake_b0), 1)
#        #fake_ab1 = fake_b0
#        pred_fake1 = net_d1.forward(fake_ab1.detach())
#        loss_d1_fake = criterionGAN(pred_fake1, False)
#       
#        fake_ab2 = torch.cat((Yh[0][:,:,0,:,:], fake_b1), 1)
#        #fake_ab2 = fake_b1
#        pred_fake2 = net_d2.forward(fake_ab2.detach())
#        loss_d2_fake = criterionGAN(pred_fake2, False)
#
#        fake_ab3 = torch.cat((Yh[0][:,:,1,:,:], fake_b2), 1)
#        #fake_ab3 = fake_b2
#        pred_fake3 = net_d3.forward(fake_ab3.detach())
#        loss_d3_fake = criterionGAN(pred_fake3, False)
#       
#        fake_ab4 = torch.cat((Yh[0][:,:,2,:,:], fake_b3), 1)
#       # fake_ab4 = fake_b3
#        pred_fake4 = net_d4.forward(fake_ab4.detach())
#        loss_d4_fake = criterionGAN(pred_fake4, False)

    # train with real
        real_ab = torch.cat((real_a, real_b), 1)
#        real_ab = real_b
        pred_real = net_d.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
#        real_ab1 = torch.cat((YY, YY), 1)
#        #real_ab1 = YY
#        pred_real1 = net_d1.forward(real_ab1)
#        loss_d1_real = criterionGAN(pred_real1, True)
#        
#        real_ab2 = torch.cat((Yh[0][:,:,0,:,:], Yh[0][:,:,0,:,:]), 1)
#        #real_ab2 =  Yh[0][:,:,0,:,:]
#        pred_real2 = net_d2.forward(real_ab2)
#        loss_d2_real = criterionGAN(pred_real2, True)
#        
#        real_ab3 = torch.cat((Yh[0][:,:,1,:,:], Yh[0][:,:,1,:,:]), 1)
#        #real_ab3 =  Yh[0][:,:,1,:,:]
#        pred_real3 = net_d3.forward(real_ab3)
#        loss_d3_real = criterionGAN(pred_real3, True)
#        
#        real_ab4 = torch.cat((Yh[0][:,:,2,:,:], Yh[0][:,:,2,:,:]), 1)
#        #real_ab4 =  Yh[0][:,:,2,:,:]
#        pred_real4 = net_d4.forward(real_ab4)
#        loss_d4_real = criterionGAN(pred_real4, True)
    
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
#        loss_d1 = (loss_d1_fake + loss_d1_real)* 0.5#xiechengd2-real
#        loss_d2 = (loss_d2_fake + loss_d2_real)* 0.5 
#        loss_d3 = (loss_d3_fake + loss_d3_real)* 0.5#xiechengd2-real
#        loss_d4 = (loss_d4_fake + loss_d4_real)* 0.5 
        loss_d.backward()
#        loss_d1.backward(retain_graph=True)
#        loss_d2.backward(retain_graph=True)
#        loss_d3.backward(retain_graph=True)
#        loss_d4.backward(retain_graph=True)       
        optimizer_d.step()
#        optimizer_d1.step()
#        optimizer_d2.step()
#        optimizer_d3.step()
#        optimizer_d4.step()
        ######################
        # (2) Update G network
        ######################

        #optimizer_g.zero_grad()
        optimizer_g1.zero_grad()
#        optimizer_g2.zero_grad()
#        optimizer_g3.zero_grad()
       # optimizer_xfm.zero_grad()
       # optimizer_ifm.zero_grad()


        # First, G(A) should fake the discriminator
#        fake_ab = torch.cat((real_a, fake_b), 1)
#        pred_fake = net_d.forward(fake_ab)
#        loss_g_gan = criterionGAN(pred_fake, True)
        
#        fake_ab1 = torch.cat((real_a, fake_b1), 1)
#        pred_fake1 = net_d1.forward(fake_ab1.detach())
#        0.5*criterionGAN(pred_fake1, False)+0.5*
#        if epoch<=5:
#            fake_ab1 = torch.cat((real_a, ndct), 1)
#            pred_fake1 = net_d1.forward(fake_ab1.detach())
#            loss_g1_gan = 0.5*criterionGAN(pred_fake1, True)
#            
#    
#            

        fake_ab = torch.cat((real_a, XX), 1)
#        fake_ab = XX
        pred_fake = net_d.forward(fake_ab.detach())
        loss_g_gan = criterionGAN(pred_fake, True)            

#        fake_ab1 = torch.cat((YY, fake_b0), 1)
#        #fake_ab1 = fake_b0
#        pred_fake1 = net_d1.forward(fake_ab1.detach())
#        loss_g1_gan =criterionGAN(pred_fake1, True)
#        
#
#        
#        fake_ab2 = torch.cat((Yh[0][:,:,0,:,:], fake_b1), 1)
#        #fake_ab2 = fake_b1
#        pred_fake2 = net_d2.forward(fake_ab2.detach())
#        loss_g2_gan = criterionGAN(pred_fake2, True)
##
#        fake_ab3 = torch.cat((Yh[0][:,:,1,:,:], fake_b2), 1)
#        #fake_ab3 = fake_b2
#        pred_fake3 = net_d3.forward(fake_ab3.detach())
#        loss_g3_gan = criterionGAN(pred_fake3, True)
#        
#
#        
#        fake_ab4 = torch.cat((Yh[0][:,:,2,:,:], fake_b3), 1)
#        #fake_ab4 = fake_b3
#        pred_fake4 = net_d4.forward(fake_ab4.detach())
#        loss_g4_gan =criterionGAN(pred_fake4, True)
# 
#        h = torch.zeros([4,3,Yh[0].size(3),Yh[0].size(3)]).float().cuda(1)
#
#        h[0,:,:,:] = Yl.to(device)
#        h[1,:,:,:] = Yh[0][:,:,0,:,:].to(device)
#        h[2,:,:,:] = Yh[0][:,:,1,:,:]
#        h[3,:,:,:] = Yh[0][:,:,2,:,:]
##
##        hL = torch.zeros([4,3,YhL[0].size(3),YhL[0].size(3)]).float().cuda(1)
##        hL[0,:,:,:] = YlL.to(device)
##        hL[1,:,:,:] = YhL[0][:,:,0,:,:].to(device)
##        hL[2,:,:,:] = YhL[0][:,:,1,:,:]
##        hL[3,:,:,:] = YhL[0][:,:,2,:,:]
#        #print(Yh.size())
      #  Yhh = torch.zeros([1,3,3,256,256]).float().cuda(1)

        #YYY= torch.cat((Yh[0][:,:,0,:,:],Yh[0][:,:,1,:,:],Yh[0][:,:,2,:,:]), 1)   
        #YY= torch.cat((fake_b1,fake_b2,fake_b3), 2)  

      #  pq=qwe(real_b)
       # pqX=qwe(XX)
#        # Second, G(A) = B
       # loss_g_l2 = criterionL1(pq, pqX) 
       # loss_g_l21 = criterionL1(pq_x, pq_xX) 
       # loss_g_l22 = criterionL1(pq_y, pq_yX) 
        #loss_g_l23 = criterionL1(hr_y, pro) * opt.lamb
        loss_g_0 = criterionL1(YY, fake_b0) 
        loss_g_l = criterionL1(Yh[0][:,:,0,:,:], fake_b1) 
        loss_g_lM = criterionMSE(YY, fake_b0) 
        loss_g_2 = criterionL1(Yh[0][:,:,1,:,:], fake_b2)
        loss_g_3 = criterionL1(Yh[0][:,:,2,:,:], fake_b3) 
       # loss_g_l21 = criterionL1(fake_YH, real_b) 
       # loss_g00 = criterionMSE(fake_abB, real_b)
        loss_g000 = criterionL1(fake_bb, real_a-real_b)
        loss_g0 = criterionL1(real_b, XX)
        loss_g_00 = criterionMSE(real_b, XX)
        #loss_g_l3 = criterionL1(nd_ct, fake_b11) * opt.lamb
        #loss_g_l211 = criterionL1(pr1, pro1) 
        #loss_g_l222 = criterionL1(pro1, hr_y1) * opt.lamb
        #loss_g_l233 = criterionL1(hr_y1, hr_yo1) * opt.lamb
#        +
 
     #   loss_g1 = loss_g_3+loss_g_2+loss_g_l+0.4*loss_g2_gan+0.4*loss_g3_gan+0.2*loss_g4_gan+loss_g_l2+loss_g_l21+loss_g00
       # loss_g =2*loss_g_0+2*loss_g0+loss_g_3+loss_g_2+loss_g_l+loss_g_gan+loss_g00+loss_g_00#+loss_g3_gan+loss_g4_gan+loss_g2_gan+loss_g1_gan
       # loss_g =  loss_g_0+0.5*loss_g_gan+loss_g0+0.5*loss_g1_gan+loss_g_l21+loss_g_lM
       
        loss_g1 =  loss_g_3+loss_g_2+loss_g_l+2*loss_g0+loss_g_gan+2*loss_g_lM+loss_g000+loss_g_0+loss_g_00
       # loss_g1 =  loss_g_gan+loss_g_0+loss_g000
#        loss_g3 = loss_g4_gan+loss_g_3+loss_g00
        #loss_xfm =  loss_g1_gan+loss_g2_gan+loss_g3_gan+loss_g3_gan+loss_g_0+loss_g_l+loss_g_2+loss_g_3+loss_g00
       # loss_ifm =  loss_g1_gan+loss_g2_gan+loss_g3_gan+loss_g3_gan+loss_g_0+loss_g_l+loss_g_2+loss_g_3+loss_g00
        
       # loss_xfm.backward(retain_graph=True)
       # loss_ifm.backward(retain_graph=True)
    #    loss_g.backward(retain_graph=True)
        loss_g1.backward()
#        loss_g2.backward(retain_graph=True)
#        loss_g3.backward(retain_graph=True)
       # optimizer_g.step()
        optimizer_g1.step()
#        optimizer_g2.step()
#        optimizer_g3.step()
        
      #  optimizer_xfm.step()
      #  optimizer_ifm.step()

        print("===> Epoch[{}]({}/{}):Loss_G1: {:.4f} Loss_D: {:.4f}".format(
              epoch, iteration, len(training_data_loader),loss_g1.item(),loss_d.item()))
#        print("===> Epoch[{}]({}/{}):Loss_G: {:.4f} ".format(
#              epoch, iteration, len(training_data_loader), loss_g.item()))

    update_learning_rate(net_d_scheduler, optimizer_d)
    update_learning_rate(net_g1_scheduler, optimizer_g1)
#    update_learning_rate(net_d1_scheduler, optimizer_d1)
#    update_learning_rate(net_d2_scheduler, optimizer_d2)
#    update_learning_rate(net_d3_scheduler, optimizer_d3)  
#    update_learning_rate(net_d4_scheduler, optimizer_d4)  
#    # test
    avg_psnr = 0
   # avg_psnr1 = 0
    for batch in testing_data_loader:
        input, target,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121= batch[0].to(device), batch[1].to(device),batch[2].to(device), batch[3].to(device),batch[4].to(device), batch[5].to(device), batch[6].to(device),batch[7].to(device),batch[8].to(device), batch[9].to(device),batch[10].to(device), batch[11].to(device),batch[12].to(device), batch[13].to(device),batch[14].to(device), batch[15].to(device),batch[16].to(device), batch[17].to(device),batch[18].to(device), batch[19].to(device),batch[20].to(device), batch[21].to(device),batch[22].to(device), batch[23].to(device),batch[24].to(device), batch[25].to(device),batch[26].to(device), batch[27].to(device),batch[28].to(device), batch[29].to(device),batch[30].to(device), batch[31].to(device),batch[32].to(device), batch[33].to(device),batch[34].to(device), batch[35].to(device),batch[36].to(device), batch[37].to(device),batch[38].to(device), batch[39].to(device),batch[40].to(device), batch[41].to(device),batch[42].to(device), batch[43].to(device),batch[44].to(device), batch[45].to(device),batch[46].to(device), batch[47].to(device),batch[48].to(device), batch[49].to(device),batch[50].to(device), batch[51].to(device),batch[52].to(device), batch[53].to(device),batch[54].to(device), batch[55].to(device),batch[56].to(device), batch[57].to(device),batch[58].to(device), batch[59].to(device),batch[60].to(device), batch[61].to(device)
        #input, target= batch[0].to(device), batch[1].to(device)
       # YYYL, YhYL = xfm(input)
       # YYY, YhY = xfm(target)
      #  YlLY, YhLY = xfm(input)     
      #  XXX = xfm(prediction, YhhH))
        prediction = net_g1(input,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
        XXX0 = input-prediction
#        prediction0 = net_g(predictionB,YYYL,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
##        prediction1,prediction2,prediction3 = net_g1(input,YhYL[0][:,:,0,:,:], YhYL[0][:,:,1,:,:], YhYL[0][:,:,2,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
#        YLLL= torch.stack((prediction1,prediction2,prediction3), axis=2)
#        YhhH=list()
#        YhhH.append(YLLL)           
#       # fake_Y= ifm((YYYL, YhhH))      
#
#
##        prediction1 = net_g1(YhLY[0][:,:,0,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
##        prediction2 = net_g2(YhLY[0][:,:,1,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
##        prediction3 = net_g3(YhLY[0][:,:,2,:,:],B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)        
# 
##
#     
#        XXX0 = ifm((prediction0, YhhH))
      #  XXX = net_g1(XXX0)
        #prediction1 = net_g1(input,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
        mse = criterionMSE(XXX0, target)
        #mse1 = criterionMSE(prediction2, target)
       # mse1 = criterionMSE(prediction2,target)
        psnr = 10 * log10(1 / mse.item())
       # psnr1 = 10 * log10(1 / mse1.item())
        avg_psnr += psnr
       # avg_psnr1 += psnr1
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
#    
    #checkpoint
    if epoch % 10 == 0:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
            os.mkdir(os.path.join("checkpoint", opt.dataset))
       # net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_g1_model_out_path = "checkpoint/{}/netG1_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_g2_model_out_path = "checkpoint/{}/netG2_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_g3_model_out_path = "checkpoint/{}/netG3_model_epoch_{}.pth".format(opt.dataset, epoch)
      #  xfm_model_out_path = "checkpoint/{}/xfm_model_epoch_{}.pth".format(opt.dataset, epoch)
      #  ifm_model_out_path = "checkpoint/{}/ifm_model_epoch_{}.pth".format(opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_d1_model_out_path = "checkpoint/{}/netD1_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_d2_model_out_path = "checkpoint/{}/netD2_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_d3_model_out_path = "checkpoint/{}/netD3_model_epoch_{}.pth".format(opt.dataset, epoch)
#        net_d4_model_out_path = "checkpoint/{}/netD4_model_epoch_{}.pth".format(opt.dataset, epoch)
       # torch.save(net_g, net_g_model_out_path)
        torch.save(net_g1, net_g1_model_out_path)
#        torch.save(net_g2, net_g2_model_out_path)
#        torch.save(net_g3, net_g3_model_out_path)
     #   torch.save(net_xfm, net_xfm_model_out_path)
    #    torch.save(net_ifm, net_ifm_model_out_path)
        torch.save(net_d, net_d_model_out_path)    
#        torch.save(net_d1, net_d1_model_out_path)
#        torch.save(net_d2, net_d2_model_out_path)
#        torch.save(net_d3, net_d3_model_out_path)
#        torch.save(net_d4, net_d4_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))
