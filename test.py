from __future__ import print_function
import argparse
import os

from torchvision.utils import make_grid

import torch
import torchvision.transforms as transforms
from guided_filter_pytorch.guided_filter import GuidedFilter
from utils import is_image_file, load_img, save_img
from PIL import Image
import glob
from grad import GradLoss
import numpy as np
# Testing settings
#coding=gbk
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=50, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:1" if opt.cuda else "cpu")

#model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)
model1_path = "checkpoint/{}/netG1_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

#net_g = torch.load(model_path).to(device)
net_g1 = torch.load(model1_path).to(device)

if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)
    image_dir1 = "dataset/{}/test/a/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img1 = load_img(image_dir1 + image_name)
    i=0
    j=0
    num0=0
    num1=0
    num2=0
   

    for i in range(0,8):       
        for j in range(0,8):
            (x1,y1,x2,y2)=(i*64,j*64,i*64+64,j*64+64)
            box=(x1,y1,x2,y2)
            ImCrop = img.crop(box)
            (x,y) = ImCrop.size #read image size
            B0= ImCrop.resize((x,y),Image.ANTIALIAS)
            #print(img.size())
            row=0
            col=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix=0
            for row in range(64):  
                for col in range(64): 
                    r, g, ba = B0.getpixel((row, col))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value = r*30+g*59+ba*11
                    if(value!=0):
                        pix=pix+1
            if(pix/(64*64)>0.9):
                num0=num0+1
                B00 = transforms.ToTensor()(B0)
                if(num0==1):
                    B1=B00
                elif(num0>1):
                    B1 = torch.cat((B1,B00), 0)  
            elif(0.015<pix/(64*64)<=0.9):  
                for ii in range(0,2):       
                    for jj in range(0,2):
                        (x11,y11,x22,y22)=(ii*32+i*64,jj*32+j*64,ii*32+i*64+32,jj*32+j*64+32)
                        box1=(x11,y11,x22,y22)
                        ImCrop1 = img.crop(box1)
                        (xx,yy) = ImCrop1.size #read image size
                        C0= ImCrop1.resize((xx,yy),Image.ANTIALIAS)
                        row1=0
                        col1=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix1=0
                        for row1 in range(32):  
                            for col1 in range(32):  
                                rr, gg, baa = C0.getpixel((row1, col1))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value1 = rr*30+gg*59+baa*11
                                if(value1!=0):
                                    pix1=pix1+1
                        if(pix1/(32*32)>0.8):
                            num1=num1+1
                            C00 = transforms.ToTensor()(C0)
                            if(num1==1):
                                C1=C00
                            elif(num1>1):
                                C1 = torch.cat((C1,C00), 0)
                        elif(0.01<pix1/(32*32)<=0.8):
                            for iii in range(0,2):       
                                for jjj in range(0,2):
                                    (x111,y111,x222,y222)=(iii*16+ii*32+i*64,jjj*16+jj*32+j*64,iii*16+ii*32+i*64+16,jjj*16+jj*32+j*64+16)
                                    box11=(x111,y111,x222,y222)
                                    ImCrop11 = img.crop(box11)
                                    (xxx,yyy) = ImCrop11.size #read image size
                                    D0= ImCrop11.resize((xxx,yyy),Image.ANTIALIAS)
                                    row11=0
                                    col11=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix11=0
                                    for row11 in range(16):  
                                        for col11 in range(16):  
                                            rrr, ggg, baa1 = D0.getpixel((row11, col11))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value11 = rrr*30+ggg*59+baa1*11
                                            if(value11!=0):
                                                pix11=pix11+1
                                    if(pix11/(16*16)>0.5):
                                        num2=num2+1
                                        D00= transforms.ToTensor()(D0)
                                        if(num2==1):
                                            D1=D00
                                        elif(num2>1):
                                            D1 = torch.cat((D1,D00), 0) 
    i4=0
    j4=0
    num04=0
    num14=0
    num24=0
    
    for i4 in range(0,7):       
        for j4 in range(0,7):
            (x14,y14,x24,y24)=(i4*64+4,j4*64+4,i4*64+64+4,j4*64+64+4)
            box4=(x14,y14,x24,y24)
            ImCrop4 = img.crop(box4)
            (x4,y4) = ImCrop4.size #read image size
            B04= ImCrop4.resize((x4,y4),Image.ANTIALIAS)
            #print(img.size())
            row4=0
            col4=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix4=0
            for row4 in range(64):  
                for col4 in range(64):  
                    r4, g4, ba4 = B04.getpixel((row4, col4))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value4 = r4*30+g4*59+ba4*11
                    if(value4!=0):
                        pix4=pix4+1
            if(pix4/(64*64)>0.9):
                num04=num04+1
                B004 = transforms.ToTensor()(B04)
                if(num04==1):
                    B14=B004
                elif(num04>1):
                    B14 = torch.cat((B14,B004), 0)  
            elif(0.015<pix4/(64*64)<=0.9):  
                for ii4 in range(0,2):       
                    for jj4 in range(0,2):
                        (x114,y114,x224,y224)=(ii4*32+i4*64+4,jj4*32+j4*64+4,ii4*32+i4*64+32+4,jj4*32+j4*64+32+4)
                        box14=(x114,y114,x224,y224)
                        ImCrop14 = img.crop(box14)
                        (xx4,yy4) = ImCrop14.size #read image size
                        C04= ImCrop14.resize((xx4,yy4),Image.ANTIALIAS)
                        row14=0
                        col14=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix14=0
                        for row14 in range(32):  
                            for col14 in range(32):  
                                rr4, gg4, baa4 = C04.getpixel((row14, col14))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value14 = rr4*30+gg4*59+baa4*11
                                if(value14!=0):
                                    pix14=pix14+1
                        if(pix14/(32*32)>0.8):
                            num14=num14+1
                            C004 = transforms.ToTensor()(C04)
                            if(num14==1):
                                C14=C004
                            elif(num14>1):
                                C14 = torch.cat((C14,C004), 0)
                        elif(0.01<pix14/(32*32)<=0.8):
                            for iii4 in range(0,2):       
                                for jjj4 in range(0,2):
                                    (x1114,y1114,x2224,y2224)=(iii4*16+ii4*32+i4*64+4,jjj4*16+jj4*32+j4*64+4,iii4*16+ii4*32+i4*64+16+4,jjj4*16+jj4*32+j4*64+16+4)
                                    box114=(x1114,y1114,x2224,y2224)
                                    ImCrop114 = img.crop(box114)
                                    (xxx4,yyy4) = ImCrop114.size #read image size
                                    D04= ImCrop114.resize((xxx4,yyy4),Image.ANTIALIAS)
                                    row114=0
                                    col114=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix114=0
                                    for row114 in range(16):  
                                        for col114 in range(16):  
                                            rrr4, ggg4, baa14 = D04.getpixel((row114, col114))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value114 = rrr4*30+ggg4*59+baa14*11
                                            if(value114!=0):
                                                pix114=pix114+1
                                    if(pix114/(16*16)>0.5):
                                        num24=num24+1
                                        D004= transforms.ToTensor()(D04)
                                        if(num24==1):
                                            D14=D004
                                        elif(num24>1):
                                            D14 = torch.cat((D14,D004), 0)
#        i40=0
#        j40=0
#        num040=0
#        num140=0
#        num240=0
#        
#        for i40 in range(0,7):       
#            for j40 in range(0,7):
#                (x140,y140,x240,y240)=(i40*64+4,j40*64+4,i40*64+64+4,j40*64+64+4)
#                box40=(x140,y140,x240,y240)
#                ImCrop40 = img.crop(box40)
#                (x40,y40) = ImCrop40.size #read image size
#                B040= ImCrop40.resize((x40,y40),Image.ANTIALIAS)
#                #print(img.size())
#                row40=0
#                col40=0
##                height = B0.size[0]
##                weight = B0.size[1]
#                pix40=0
#                for row40 in range(64):  
#                    for col40 in range(64):  
#                        r40, g40, ba40 = B040.getpixel((row40, col40))
##                        g = B0[row, col, 1]
##                        ba = B0[row, col, 2]
#                        value40 = r40*30+g40*59+ba40*11
#                        if(value40!=0):
#                            pix40=pix40+1
#                if(pix40/(64*64)>0.8):
#                    num040=num040+1
#                    B0040 = transforms.ToTensor()(B040)
#                    if(num040==1):
#                        B140=B0040
#                    elif(num04>1):
#                        B140 = torch.cat((B140,B0040), 0)  
#                elif(0.01<pix40/(64*64)<=0.8):  
#                    for ii40 in range(0,2):       
#                        for jj40 in range(0,2):
#                            (x1140,y1140,x2240,y2240)=(ii40*32+i40*64+4,jj40*32+j40*64+4,ii40*32+i40*64+32+4,jj40*32+j40*64+32+4)
#                            box140=(x1140,y1140,x2240,y2240)
#                            ImCrop140 = img.crop(box140)
#                            (xx40,yy40) = ImCrop140.size #read image size
#                            C040= ImCrop140.resize((xx40,yy40),Image.ANTIALIAS)
#                            row140=0
#                            col140=0
##                            height1 = C0.size[0]
##                            weight1 = C0.size[1]
#                            pix140=0
#                            for row140 in range(32):  
#                                for col140 in range(32):  
#                                    rr40, gg40, baa40 = C040.getpixel((row140, col140))
##                                    gg = C0[row1, col1, 1]
##                                    baa = C0[row1, col1, 2]
#                                    value140 = rr40*30+gg40*59+baa40*11
#                                    if(value140!=0):
#                                        pix140=pix140+1
#                            if(pix140/(32*32)>0.8):
#                                num140=num140+1
#                                C0040 = transforms.ToTensor()(C040)
#                                if(num140==1):
#                                    C140=C0040
#                                elif(num140>1):
#                                    C140 = torch.cat((C140,C0040), 0)
#                            elif(0.1<pix140/(32*32)<=0.8):
#                                for iii40 in range(0,2):       
#                                    for jjj40 in range(0,2):
#                                        (x11140,y11140,x22240,y22240)=(iii40*16+ii40*32+i40*64+4,jjj40*16+jj40*32+j40*64+4,iii40*16+ii40*32+i40*64+16+4,jjj40*16+jj40*32+j40*64+16+4)
#                                        box1140=(x11140,y11140,x22240,y22240)
#                                        ImCrop1140 = img.crop(box1140)
#                                        (xxx40,yyy40) = ImCrop1140.size #read image size
#                                        D040= ImCrop1140.resize((xxx40,yyy40),Image.ANTIALIAS)
#                                        row1140=0
#                                        col1140=0
##                                        height11 = D0.size[0]
##                                        weight11 = D0.size[1]
#                                        pix1140=0
#                                        for row1140 in range(16):  
#                                            for col1140 in range(16):  
#                                                rrr40, ggg40, baa140 = D040.getpixel((row1140, col1140))
##                                                ggg = D0[row11, col11, 1]
##                                                baa1 = D0[row11, col11, 2]
#                                                value1140 = rrr40*30+ggg40*59+baa140*11
#                                                if(value1140!=0):
#                                                    pix1140=pix1140+1
#                                        if(pix1140/(16*16)>0.5):
#                                            num240=num240+1
#                                            D0040= transforms.ToTensor()(D040)
#                                            if(num240==1):
#                                                D140=D0040
#                                            elif(num240>1):
#                                                D140 = torch.cat((D140,D0040), 0)
    i40=0
    j40=0
    num040=0
    num140=0
    num240=0
    
    for i40 in range(0,7):       
        for j40 in range(0,8):
            (x140,y140,x240,y240)=(i40*64+4,j40*64,i40*64+64+4,j40*64+64)
            box40=(x140,y140,x240,y240)
            ImCrop40 = img.crop(box40)
            (x40,y40) = ImCrop40.size #read image size
            B040= ImCrop40.resize((x40,y40),Image.ANTIALIAS)
            #print(img.size())
            row40=0
            col40=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix40=0
            for row40 in range(64):  
                for col40 in range(64):  
                    r40, g40, ba40 = B040.getpixel((row40, col40))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value40 = r40*30+g40*59+ba40*11
                    if(value40!=0):
                        pix40=pix40+1
            if(pix40/(64*64)>0.9):
                num040=num040+1
                B0040 = transforms.ToTensor()(B040)
                if(num040==1):
                    B140=B0040
                elif(num040>1):
                    B140 = torch.cat((B140,B0040), 0)  
            elif(0.015<pix40/(64*64)<=0.9):  
                for ii40 in range(0,2):       
                    for jj40 in range(0,2):
                        (x1140,y1140,x2240,y2240)=(ii40*32+i40*64+4,jj40*32+j40*64,ii40*32+i40*64+32+4,jj40*32+j40*64+32)
                        box140=(x1140,y1140,x2240,y2240)
                        ImCrop140 = img.crop(box140)
                        (xx40,yy40) = ImCrop140.size #read image size
                        C040= ImCrop140.resize((xx40,yy40),Image.ANTIALIAS)
                        row140=0
                        col140=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix140=0
                        for row140 in range(32):  
                            for col140 in range(32):  
                                rr40, gg40, baa40 = C040.getpixel((row140, col140))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value140 = rr40*30+gg40*59+baa40*11
                                if(value140!=0):
                                    pix140=pix140+1
                        if(pix140/(32*32)>0.8):
                            num140=num140+1
                            C0040 = transforms.ToTensor()(C040)
                            if(num140==1):
                                C140=C0040
                            elif(num140>1):
                                C140 = torch.cat((C140,C0040), 0)
                        elif(0.01<pix140/(32*32)<=0.8):
                            for iii40 in range(0,2):       
                                for jjj40 in range(0,2):
                                    (x11140,y11140,x22240,y22240)=(iii40*16+ii40*32+i40*64+4,jjj40*16+jj40*32+j40*64,iii40*16+ii40*32+i40*64+16+4,jjj40*16+jj40*32+j40*64+16)
                                    box1140=(x11140,y11140,x22240,y22240)
                                    ImCrop1140 = img.crop(box1140)
                                    (xxx40,yyy40) = ImCrop1140.size #read image size
                                    D040= ImCrop1140.resize((xxx40,yyy40),Image.ANTIALIAS)
                                    row1140=0
                                    col1140=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix1140=0
                                    for row1140 in range(16):  
                                        for col1140 in range(16):  
                                            rrr40, ggg40, baa140 = D040.getpixel((row1140, col1140))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value1140 = rrr40*30+ggg40*59+baa140*11
                                            if(value1140!=0):
                                                pix1140=pix1140+1
                                    if(pix1140/(16*16)>0.5):
                                        num240=num240+1
                                        D0040= transforms.ToTensor()(D040)
                                        if(num240==1):
                                            D140=D0040
                                        elif(num240>1):
                                            D140 = torch.cat((D140,D0040), 0)        
    i41=0
    j41=0
    num041=0
    num141=0
    num241=0
    
    for i41 in range(0,8):       
        for j41 in range(0,7):
            (x141,y141,x241,y241)=(i41*64,j41*64+4,i41*64+64,j41*64+64+4)
            box41=(x141,y141,x241,y241)
            ImCrop41 = img.crop(box41)
            (x41,y41) = ImCrop41.size #read image size
            B041= ImCrop41.resize((x41,y41),Image.ANTIALIAS)
            #print(img.size())
            row41=0
            col41=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix41=0
            for row41 in range(64):  
                for col41 in range(64):  
                    r41, g41, ba41 = B041.getpixel((row41, col41))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value41 = r41*30+g41*59+ba41*11
                    if(value41!=0):
                        pix41=pix41+1
            if(pix41/(64*64)>0.9):
                num041=num041+1
                B0041 = transforms.ToTensor()(B041)
                if(num041==1):
                    B141=B0041
                elif(num041>1):
                    B141 = torch.cat((B141,B0041), 0)  
            elif(0.015<pix41/(64*64)<=0.9):  
                for ii41 in range(0,2):       
                    for jj41 in range(0,2):
                        (x1141,y1141,x2241,y2241)=(ii41*32+i41*64,jj41*32+j41*64+4,ii41*32+i41*64+32,jj41*32+j41*64+32+4)
                        box141=(x1141,y1141,x2241,y2241)
                        ImCrop141 = img.crop(box141)
                        (xx41,yy41) = ImCrop141.size #read image size
                        C041= ImCrop141.resize((xx41,yy41),Image.ANTIALIAS)
                        row141=0
                        col141=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix141=0
                        for row141 in range(32):  
                            for col141 in range(32):  
                                rr41, gg41, baa41 = C041.getpixel((row141, col141))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value141 = rr41*30+gg41*59+baa41*11
                                if(value141!=0):
                                    pix141=pix141+1
                        if(pix141/(32*32)>0.8):
                            num141=num141+1
                            C0041 = transforms.ToTensor()(C041)
                            if(num141==1):
                                C141=C0041
                            elif(num141>1):
                                C141 = torch.cat((C141,C0041), 0)
                        elif(0.01<pix141/(32*32)<=0.8):
                            for iii41 in range(0,2):       
                                for jjj41 in range(0,2):
                                    (x11141,y11141,x22241,y22241)=(iii41*16+ii41*32+i41*64,jjj41*16+jj41*32+j41*64+4,iii41*16+ii41*32+i41*64+16,jjj41*16+jj41*32+j41*64+16+4)
                                    box1141=(x11141,y11141,x22241,y22241)
                                    ImCrop1141 = img.crop(box1141)
                                    (xxx41,yyy41) = ImCrop1141.size #read image size
                                    D041= ImCrop1141.resize((xxx41,yyy41),Image.ANTIALIAS)
                                    row1141=0
                                    col1141=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix1141=0
                                    for row1141 in range(16):  
                                        for col1141 in range(16): 
                                            rrr41, ggg41, baa141 = D041.getpixel((row1141, col1141))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value1141 = rrr41*30+ggg41*59+baa141*11
                                            if(value1141!=0):
                                                pix1141=pix1141+1
                                    if(pix1141/(16*16)>0.5):
                                        num241=num241+1
                                        D0041= transforms.ToTensor()(D041)
                                        if(num241==1):
                                            D141=D0041
                                        elif(num241>1):
                                            D141 = torch.cat((D141,D0041), 0)     
                                           
    i80=0
    j80=0
    num080=0
    num180=0
    num280=0
    
    for i80 in range(0,7):       
        for j80 in range(0,8):
            (x180,y180,x280,y280)=(i80*64+8,j80*64,i80*64+64+8,j80*64+64)
            box80=(x180,y180,x280,y280)
            ImCrop80 = img.crop(box80)
            (x80,y80) = ImCrop80.size #read image size
            B080= ImCrop80.resize((x80,y80),Image.ANTIALIAS)
            #print(img.size())
            row80=0
            col80=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix80=0
            for row80 in range(64):  
                for col80 in range(64):  
                    r80, g80, ba80 = B080.getpixel((row80, col80))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value80 = r80*30+g80*59+ba80*11
                    if(value80!=0):
                        pix80=pix80+1
            if(pix80/(64*64)>0.9):
                num080=num080+1
                B0080 = transforms.ToTensor()(B080)
                if(num080==1):
                    B180=B0080
                elif(num080>1):
                    B180 = torch.cat((B180,B0080), 0)  
            elif(0.015<pix80/(64*64)<=0.9):  
                for ii80 in range(0,2):       
                    for jj80 in range(0,2):
                        (x1180,y1180,x2280,y2280)=(ii80*32+i80*64+8,jj80*32+j80*64,ii80*32+i80*64+32+8,jj80*32+j80*64+32)
                        box180=(x1180,y1180,x2280,y2280)
                        ImCrop180 = img.crop(box180)
                        (xx80,yy80) = ImCrop180.size #read image size
                        C080= ImCrop180.resize((xx80,yy80),Image.ANTIALIAS)
                        row180=0
                        col180=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix180=0
                        for row180 in range(32):  
                            for col180 in range(32):  
                                rr80, gg80, baa80 = C080.getpixel((row180, col180))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value180 = rr80*30+gg80*59+baa80*11
                                if(value180!=0):
                                    pix180=pix180+1
                        if(pix180/(32*32)>0.8):
                            num180=num180+1
                            C0080 = transforms.ToTensor()(C080)
                            if(num180==1):
                                C180=C0080
                            elif(num180>1):
                                C180 = torch.cat((C180,C0080), 0)
                        elif(0.01<pix180/(32*32)<=0.8):
                            for iii80 in range(0,2):       
                                for jjj80 in range(0,2):
                                    (x11180,y11180,x22280,y22280)=(iii80*16+ii80*32+i80*64+8,jjj80*16+jj80*32+j80*64,iii80*16+ii80*32+i80*64+16+8,jjj80*16+jj80*32+j80*64+16)
                                    box1180=(x11180,y11180,x22280,y22280)
                                    ImCrop1180 = img.crop(box1180)
                                    (xxx80,yyy80) = ImCrop1180.size #read image size
                                    D080= ImCrop1180.resize((xxx80,yyy80),Image.ANTIALIAS)
                                    row1180=0
                                    col1180=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix1180=0
                                    for row1180 in range(16):  
                                        for col1180 in range(16):  
                                            rrr80, ggg80, baa180 = D080.getpixel((row1180, col1180))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value1180 = rrr80*30+ggg80*59+baa180*11
                                            if(value1180!=0):
                                                pix1180=pix1180+1
                                    if(pix1180/(16*16)>0.5):
                                        num280=num280+1
                                        D0080= transforms.ToTensor()(D080)
                                        if(num280==1):
                                            D180=D0080
                                        elif(num280>1):
                                            D180 = torch.cat((D180,D0080), 0)                                                                              
                                           
    i8=0
    j8=0
    num08=0
    num18=0
    num28=0
    
    for i8 in range(0,7):       
        for j8 in range(0,7):
            (x18,y18,x28,y28)=(i8*64+8,j8*64+8,i8*64+64+8,j8*64+64+8)
            box8=(x18,y18,x28,y28)
            ImCrop8 = img.crop(box8)
            (x8,y8) = ImCrop8.size #read image size
            B08= ImCrop8.resize((x8,y8),Image.ANTIALIAS)
            #print(img.size())
            row8=0
            col8=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix8=0
            for row8 in range(64):  
                for col8 in range(64):  
                    r8, g8, ba8 = B08.getpixel((row8, col8))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value8 = r8*30+g8*59+ba8*11
                    if(value8!=0):
                        pix8=pix8+1
            if(pix8/(64*64)>0.9):
                num08=num08+1
                B008 = transforms.ToTensor()(B08)
                if(num08==1):
                    B18=B008
                elif(num08>1):
                    B18 = torch.cat((B18,B008), 0)  
            elif(0.015<pix8/(64*64)<=0.9):  
                for ii8 in range(0,2):       
                    for jj8 in range(0,2):
                        (x118,y118,x228,y228)=(ii8*32+i8*64+8,jj8*32+j8*64+8,ii8*32+i8*64+32+8,jj8*32+j8*64+32+8)
                        box18=(x118,y118,x228,y228)
                        ImCrop18 = img.crop(box18)
                        (xx8,yy8) = ImCrop18.size #read image size
                        C08= ImCrop18.resize((xx8,yy8),Image.ANTIALIAS)
                        row18=0
                        col18=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix18=0
                        for row18 in range(32):  
                            for col18 in range(32):  
                                rr8, gg8, baa8 = C08.getpixel((row18, col18))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value18 = rr8*30+gg8*59+baa8*11
                                if(value18!=0):
                                    pix18=pix18+1
                        if(pix18/(32*32)>0.8):
                            num18=num18+1
                            C008 = transforms.ToTensor()(C08)
                            if(num18==1):
                                C18=C008
                            elif(num18>1):
                                C18 = torch.cat((C18,C008), 0)
                        elif(0.01<pix18/(32*32)<=0.8):
                            for iii8 in range(0,2):       
                                for jjj8 in range(0,2):
                                    (x1118,y1118,x2228,y2228)=(iii8*16+ii8*32+i8*64+8,jjj8*16+jj8*32+j8*64+8,iii8*16+ii8*32+i8*64+16+8,jjj8*16+jj8*32+j8*64+16+8)
                                    box118=(x1118,y1118,x2228,y2228)
                                    ImCrop118 = img.crop(box118)
                                    (xxx8,yyy8) = ImCrop118.size #read image size
                                    D08= ImCrop118.resize((xxx8,yyy8),Image.ANTIALIAS)
                                    row118=0
                                    col118=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix118=0
                                    for row118 in range(16):  
                                        for col118 in range(16):  
                                            rrr8, ggg8, baa18 = D08.getpixel((row118, col118))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value118 = rrr8*30+ggg8*59+baa18*11
                                            if(value118!=0):
                                                pix118=pix118+1
                                    if(pix118/(16*16)>0.5):
                                        num28=num28+1
                                        D008= transforms.ToTensor()(D08)
                                        if(num28==1):
                                            D18=D008
                                        elif(num28>1):
                                            D18 = torch.cat((D18,D008), 0)  
                                              
    i81=0
    j81=0
    num081=0
    num181=0
    num281=0
    
    for i81 in range(0,8):       
        for j81 in range(0,7):
            (x181,y181,x281,y281)=(i81*64,j81*64+8,i81*64+64,j81*64+64+8)
            box81=(x181,y181,x281,y281)
            ImCrop81 = img.crop(box81)
            (x81,y81) = ImCrop81.size #read image size
            B081= ImCrop81.resize((x81,y81),Image.ANTIALIAS)
            #print(img.size())
            row81=0
            col81=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix81=0
            for row81 in range(64):  
                for col81 in range(64):  
                    r81, g81, ba81 = B081.getpixel((row81, col81))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value81 = r81*30+g81*59+ba81*11
                    if(value81!=0):
                        pix81=pix81+1
            if(pix81/(64*64)>0.9):
                num081=num081+1
                B0081 = transforms.ToTensor()(B081)
                if(num081==1):
                    B181=B0081
                elif(num081>1):
                    B181 = torch.cat((B181,B0081), 0)  
            elif(0.015<pix81/(64*64)<=0.9):  
                for ii81 in range(0,2):       
                    for jj81 in range(0,2):
                        (x1181,y1181,x2281,y2281)=(ii81*32+i81*64,jj81*32+j81*64+8,ii81*32+i81*64+32,jj81*32+j81*64+32+8)
                        box181=(x1181,y1181,x2281,y2281)
                        ImCrop181 = img.crop(box181)
                        (xx81,yy81) = ImCrop181.size #read image size
                        C081= ImCrop181.resize((xx81,yy81),Image.ANTIALIAS)
                        row181=0
                        col181=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix181=0
                        for row181 in range(32):  
                            for col181 in range(32):  
                                rr81, gg81, baa81 = C081.getpixel((row181, col181))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value181 = rr81*30+gg81*59+baa81*11
                                if(value181!=0):
                                    pix181=pix181+1
                        if(pix181/(32*32)>0.8):
                            num181=num181+1
                            C0081 = transforms.ToTensor()(C081)
                            if(num181==1):
                                C181=C0081
                            elif(num181>1):
                                C181 = torch.cat((C181,C0081), 0)
                        elif(0.01<pix181/(32*32)<=0.8):
                            for iii81 in range(0,2):       
                                for jjj81 in range(0,2):
                                    (x11181,y11181,x22281,y22281)=(iii81*16+ii81*32+i81*64,jjj81*16+jj81*32+j81*64+8,iii81*16+ii81*32+i81*64+16,jjj81*16+jj81*32+j81*64+16+8)
                                    box1181=(x11181,y11181,x22281,y22281)
                                    ImCrop1181 = img.crop(box1181)
                                    (xxx81,yyy81) = ImCrop1181.size #read image size
                                    D081= ImCrop1181.resize((xxx81,yyy81),Image.ANTIALIAS)
                                    row1181=0
                                    col1181=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix1181=0
                                    for row1181 in range(16):  
                                        for col1181 in range(16):  
                                            rrr81, ggg81, baa181 = D081.getpixel((row1181, col1181))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value1181 = rrr81*30+ggg81*59+baa181*11
                                            if(value1181!=0):
                                                pix1181=pix1181+1
                                    if(pix1181/(16*16)>0.5):
                                        num281=num281+1
                                        D0081= transforms.ToTensor()(D081)
                                        if(num281==1):
                                            D181=D0081
                                        elif(num281>1):
                                            D181 = torch.cat((D181,D0081), 0)    
    
    i12=0
    j12=0
    num012=0
    num112=0
    num212=0
    
    for i12 in range(0,7):       
        for j12 in range(0,7):
            (x112,y112,x212,y212)=(i12*64+12,j12*64+12,i12*64+64+12,j12*64+64+12)
            box12=(x112,y112,x212,y212)
            ImCrop12 = img.crop(box12)
            (x12,y12) = ImCrop12.size #read image size
            B012= ImCrop12.resize((x12,y12),Image.ANTIALIAS)
            #print(img.size())
            row12=0
            col12=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix12=0
            for row12 in range(64):  
                for col12 in range(64):  
                    r12, g12, ba12 = B012.getpixel((row12, col12))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value12 = r12*30+g12*59+ba12*11
                    if(value12!=0):
                        pix12=pix12+1
            if(pix12/(64*64)>0.9):
                num012=num012+1
                B0012 = transforms.ToTensor()(B012)
                if(num012==1):
                    B112=B0012
                elif(num012>1):
                    B112 = torch.cat((B112,B0012), 0)  
            elif(0.015<pix12/(64*64)<=0.9):  
                for ii12 in range(0,2):       
                    for jj12 in range(0,2):
                        (x1112,y1112,x2212,y2212)=(ii12*32+i12*64+12,jj12*32+j12*64+12,ii12*32+i12*64+32+12,jj12*32+j12*64+32+12)
                        box112=(x1112,y1112,x2212,y2212)
                        ImCrop112 = img.crop(box112)
                        (xx12,yy12) = ImCrop112.size #read image size
                        C012= ImCrop112.resize((xx12,yy12),Image.ANTIALIAS)
                        row112=0
                        col112=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix112=0
                        for row112 in range(32):  
                            for col112 in range(32):  
                                rr12, gg12, baa12 = C012.getpixel((row112, col112))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value112 = rr12*30+gg12*59+baa12*11
                                if(value112!=0):
                                    pix112=pix112+1
                        if(pix112/(32*32)>0.8):
                            num112=num112+1
                            C0012 = transforms.ToTensor()(C012)
                            if(num112==1):
                                C112=C0012
                            elif(num112>1):
                                C112 = torch.cat((C112,C0012), 0)
                        elif(0.01<pix112/(32*32)<=0.8):
                            for iii12 in range(0,2):       
                                for jjj12 in range(0,2):
                                    (x11112,y11112,x22212,y22212)=(iii12*16+ii12*32+i12*64+12,jjj12*16+jj12*32+j12*64+12,iii12*16+ii12*32+i12*64+16+12,jjj12*16+jj12*32+j12*64+16+12)
                                    box1112=(x11112,y11112,x22212,y22212)
                                    ImCrop1112 = img.crop(box1112)
                                    (xxx12,yyy12) = ImCrop1112.size #read image size
                                    D012= ImCrop1112.resize((xxx12,yyy12),Image.ANTIALIAS)
                                    row1112=0
                                    col1112=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix1112=0
                                    for row1112 in range(16):  
                                        for col1112 in range(16):
                                            rrr12, ggg12, baa112 = D012.getpixel((row1112, col1112))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value1112 = rrr12*30+ggg12*59+baa112*11
                                            if(value1112!=0):
                                                pix1112=pix1112+1
                                    if(pix1112/(16*16)>0.5):
                                        num212=num212+1
                                        D0012= transforms.ToTensor()(D012)
                                        if(num212==1):
                                            D112=D0012
                                        elif(num212>1):
                                            D112 = torch.cat((D112,D0012), 0)  
                                          
    i120=0
    j120=0
    num0120=0
    num1120=0
    num2120=0
    
    for i120 in range(0,7):       
        for j120 in range(0,8):
            (x1120,y1120,x2120,y2120)=(i120*64+12,j120*64,i120*64+64+12,j120*64+64)
            box120=(x1120,y1120,x2120,y2120)
            ImCrop120 = img.crop(box120)
            (x120,y120) = ImCrop120.size #read image size
            B0120= ImCrop120.resize((x120,y120),Image.ANTIALIAS)
            #print(img.size())
            row120=0
            col120=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix120=0
            for row120 in range(64):  
                for col120 in range(64):  
                    r120, g120, ba120 = B0120.getpixel((row120, col120))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value120 = r120*30+g120*59+ba120*11
                    if(value120!=0):
                        pix120=pix120+1
            if(pix120/(64*64)>0.9):
                num0120=num0120+1
                B00120 = transforms.ToTensor()(B0120)
                if(num0120==1):
                    B1120=B00120
                elif(num0120>1):
                    B1120 = torch.cat((B1120,B00120), 0)  
            elif(0.015<pix120/(64*64)<=0.9):  
                for ii120 in range(0,2):       
                    for jj120 in range(0,2):
                        (x11120,y11120,x22120,y22120)=(ii120*32+i120*64+12,jj120*32+j120*64,ii120*32+i120*64+32+12,jj120*32+j120*64+32)
                        box1120=(x11120,y11120,x22120,y22120)
                        ImCrop1120 = img.crop(box1120)
                        (xx120,yy120) = ImCrop1120.size #read image size
                        C0120= ImCrop1120.resize((xx120,yy120),Image.ANTIALIAS)
                        row1120=0
                        col1120=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix1120=0
                        for row1120 in range(32):  
                            for col1120 in range(32):  
                                rr120, gg120, baa120 = C0120.getpixel((row1120, col1120))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value1120 = rr120*30+gg120*59+baa120*11
                                if(value1120!=0):
                                    pix1120=pix1120+1
                        if(pix1120/(32*32)>0.8):
                            num1120=num1120+1
                            C00120 = transforms.ToTensor()(C0120)
                            if(num1120==1):
                                C1120=C00120
                            elif(num1120>1):
                                C1120 = torch.cat((C1120,C00120), 0)
                        elif(0.01<pix1120/(32*32)<=0.8):
                            for iii120 in range(0,2):       
                                for jjj120 in range(0,2):
                                    (x111120,y111120,x222120,y222120)=(iii120*16+ii120*32+i120*64+12,jjj120*16+jj120*32+j120*64,iii120*16+ii120*32+i120*64+16+12,jjj120*16+jj120*32+j120*64+16)
                                    box11120=(x111120,y111120,x222120,y222120)
                                    ImCrop11120 = img.crop(box11120)
                                    (xxx120,yyy120) = ImCrop11120.size #read image size
                                    D0120= ImCrop11120.resize((xxx120,yyy120),Image.ANTIALIAS)
                                    row11120=0
                                    col11120=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix11120=0
                                    for row11120 in range(16):  
                                        for col11120 in range(16):  
                                            rrr120, ggg120, baa1120 = D0120.getpixel((row11120, col11120))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value11120 = rrr120*30+ggg120*59+baa1120*11
                                            if(value11120!=0):
                                                pix11120=pix11120+1
                                    if(pix11120/(16*16)>0.5):
                                        num2120=num2120+1
                                        D00120= transforms.ToTensor()(D0120)
                                        if(num2120==1):
                                            D1120=D00120
                                        elif(num2120>1):
                                            D1120 = torch.cat((D1120,D00120), 0)                                                                                                   
    i121=0
    j121=0
    num0121=0
    num1121=0
    num2121=0
    
    for i121 in range(0,8):       
        for j121 in range(0,7):
            (x1121,y1121,x2121,y2121)=(i121*64,j121*64+12,i121*64+64,j121*64+64+12)
            box121=(x1121,y1121,x2121,y2121)
            ImCrop121 = img.crop(box121)
            (x121,y121) = ImCrop121.size #read image size
            B0121= ImCrop121.resize((x121,y121),Image.ANTIALIAS)
            #print(img.size())
            row121=0
            col121=0
#                height = B0.size[0]
#                weight = B0.size[1]
            pix121=0
            for row121 in range(64):  
                for col121 in range(64):  
                    r121, g121, ba121 = B0121.getpixel((row121, col121))
#                        g = B0[row, col, 1]
#                        ba = B0[row, col, 2]
                    value121 = r121*30+g121*59+ba121*11
                    if(value121!=0):
                        pix121=pix121+1
            if(pix121/(64*64)>0.9):
                num0121=num0121+1
                B00121 = transforms.ToTensor()(B0121)
                if(num0121==1):
                    B1121=B00121
                elif(num0121>1):
                    B1121 = torch.cat((B1121,B00121), 0)  
            elif(0.015<pix121/(64*64)<=0.9):  
                for ii121 in range(0,2):       
                    for jj121 in range(0,2):
                        (x11121,y11121,x22121,y22121)=(ii121*32+i121*64,jj121*32+j121*64+12,ii121*32+i121*64+32,jj121*32+j121*64+32+12)
                        box1121=(x11121,y11121,x22121,y22121)
                        ImCrop1121 = img.crop(box1121)
                        (xx121,yy121) = ImCrop1121.size #read image size
                        C0121= ImCrop1121.resize((xx121,yy121),Image.ANTIALIAS)
                        row1121=0
                        col1121=0
#                            height1 = C0.size[0]
#                            weight1 = C0.size[1]
                        pix1121=0
                        for row1121 in range(32):  
                            for col1121 in range(32):  
                                rr121, gg121, baa121 = C0121.getpixel((row1121, col1121))
#                                    gg = C0[row1, col1, 1]
#                                    baa = C0[row1, col1, 2]
                                value1121 = rr121*30+gg121*59+baa121*11
                                if(value1121!=0):
                                    pix1121=pix1121+1
                        if(pix1121/(32*32)>0.8):
                            num1121=num1121+1
                            C00121 = transforms.ToTensor()(C0121)
                            if(num1121==1):
                                C1121=C00121
                            elif(num1121>1):
                                C1121 = torch.cat((C1121,C00121), 0)
                        elif(0.01<pix1121/(32*32)<=0.8):
                            for iii121 in range(0,2):       
                                for jjj121 in range(0,2):
                                    (x111121,y111121,x222121,y222121)=(iii121*16+ii121*32+i121*64,jjj121*16+jj121*32+j121*64+12,iii121*16+ii121*32+i121*64+16,jjj121*16+jj121*32+j121*64+16+12)
                                    box11121=(x111121,y111121,x222121,y222121)
                                    ImCrop11121 = img.crop(box11121)
                                    (xxx121,yyy121) = ImCrop11121.size #read image size
                                    D0121= ImCrop11121.resize((xxx121,yyy121),Image.ANTIALIAS)
                                    row11121=0
                                    col11121=0
#                                        height11 = D0.size[0]
#                                        weight11 = D0.size[1]
                                    pix11121=0
                                    for row11121 in range(16):  
                                        for col11121 in range(16):  
                                            rrr121, ggg121, baa1121 = D0121.getpixel((row11121, col11121))
#                                                ggg = D0[row11, col11, 1]
#                                                baa1 = D0[row11, col11, 2]
                                            value11121 = rrr121*30+ggg121*59+baa1121*11
                                            if(value11121!=0):
                                                pix11121=pix11121+1
                                    if(pix11121/(16*16)>0.5):
                                        num2121=num2121+1
                                        D00121= transforms.ToTensor()(D0121)
                                        if(num2121==1):
                                            D1121=D00121
                                        elif(num2121>1):
                                            D1121 = torch.cat((D1121,D00121), 0)  
    img = transform(img)
    img1 = transform(img1)
    
    input = img.unsqueeze(0).to(device)
    input1 = img1.unsqueeze(0).to(device)
    B1= B1.unsqueeze(0).to(device)
    C1= C1.unsqueeze(0).to(device)
    D1= D1.unsqueeze(0).to(device)
    #num0= num0.unsqueeze(0).to(device)
   # num1= num1.unsqueeze(0).to(device)
   # num2= num2.unsqueeze(0).to(device)
    B14= B14.unsqueeze(0).to(device)
    C14= C14.unsqueeze(0).to(device)
    D14= D14.unsqueeze(0).to(device)
#    num04= num04.unsqueeze(0).to(device)
#    num14= num14.unsqueeze(0).to(device)
#    num24=num24.unsqueeze(0).to(device)
    B140= B140.unsqueeze(0).to(device)
    C140= C140.unsqueeze(0).to(device)
    D140= D140.unsqueeze(0).to(device)
#    num040= num040.unsqueeze(0).to(device)
#    num140= num140.unsqueeze(0).to(device)
#    num240= num240.unsqueeze(0).to(device)
    B141= B141.unsqueeze(0).to(device)
    C141= C141.unsqueeze(0).to(device)
    D141= D141.unsqueeze(0).to(device)
#    num041= num041.unsqueeze(0).to(device)
#    num141= num141.unsqueeze(0).to(device)
#    num241= num241.unsqueeze(0).to(device)
    B180= B180.unsqueeze(0).to(device)
    C180= C180.unsqueeze(0).to(device)
    D180= D180.unsqueeze(0).to(device)
#    num080= num080.unsqueeze(0).to(device)
#    num180= num180.unsqueeze(0).to(device)
#    num280= num280.unsqueeze(0).to(device)
    B18=B18.unsqueeze(0).to(device)
    C18= C18.unsqueeze(0).to(device)
    D18= D18.unsqueeze(0).to(device)
#    num08=  num08.unsqueeze(0).to(device)
#    num18= num18.unsqueeze(0).to(device)
#    num28=num28.unsqueeze(0).to(device)
    B181= B181.unsqueeze(0).to(device)
    C181= C181.unsqueeze(0).to(device)
    D181= D181.unsqueeze(0).to(device)
#    num081= num081.unsqueeze(0).to(device)
#    num181= num181.unsqueeze(0).to(device)
#    num281= num281.unsqueeze(0).to(device)
    B112= B112.unsqueeze(0).to(device)
    C112= C112.unsqueeze(0).to(device)
    D112= D112.unsqueeze(0).to(device)
#    num012= num012.unsqueeze(0).to(device)
#    num112= num112.unsqueeze(0).to(device)
#    num212= num212.unsqueeze(0).to(device)
    B1120= B1120.unsqueeze(0).to(device)
    C1120= C1120.unsqueeze(0).to(device)
    D1120= D1120.unsqueeze(0).to(device)
#    num0120= num0120.unsqueeze(0).to(device)
#    num1120= num1120.unsqueeze(0).to(device)
#    num2120= num2120.unsqueeze(0).to(device)
    B1121= B1121.unsqueeze(0).to(device)
    C1121= C1121.unsqueeze(0).to(device)
    D1121= D1121.unsqueeze(0).to(device)
#    num0121= num0121.unsqueeze(0).to(device)
#    num1121= num1121.unsqueeze(0).to(device)
#    num2121= num2121.unsqueeze(0).to(device)

    out  = net_g1(input,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
    #
    #pq_b=qwe(real_b)
   # fake_N=pq_a-pq_b
   # nd_ct=input-input1
    #nd_ct=input-out
  #  out1  = net_g1(input,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121)
    #pr=pr.transpose(1,0).reshape((-1,1,512,512))
  #  print(input.size())
    out1=input-out
    out_img =out1.detach().squeeze(0).cpu()
    #out_img = out_img.detach().squeeze(0).cpu()
    #out_img=out_img.transpose(1,0).reshape((-1,3,512,512))
    
    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
