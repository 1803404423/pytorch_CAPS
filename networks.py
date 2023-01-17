import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
#from torchsummary import summary
#from collections import OrderedDict
from guided_filter_pytorch.guided_filter import GuidedFilter

class SobelOperator(nn.Module):
    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

#        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, :, :, :] / 4)
#        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, :, :, :] / 4)

        self.register_buffer('conv_x', torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])[None, None, :, :] / 3)
        self.register_buffer('conv_y', torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])[None, None, :, :] / 3)
        
    def forward(self, x):
        b, c, h, w = x.shape
        if c > 1:
            x = x.view(b*c , 1, h, w)

        grad_x = F.conv2d(x, self.conv_x, bias=None, stride=1, padding=1)
        grad_y = F.conv2d(x, self.conv_y, bias=None, stride=1, padding=1)

        x = torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.epsilon)

        x = x.view(b, c, h, w)

        return x


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.sobel = SobelOperator(1e-4)

    def forward(self, pr, gt):
        gt_sobel = self.sobel(gt)
        pr_sobel = self.sobel(pr)
        grad_loss = F.l1_loss(gt_sobel, pr_sobel)
        return grad_loss


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class NLBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, mode='embedded', 
                 dimension=3, bn_layer=True):
        """Implementation of Non-Local Block with 4 different pairwise functions but doesn't include subsampling trick
        args:
            in_channels: original channel size (1024 in the paper)
            inter_channels: channel size inside the block if not specifed reduced to half (512 in the paper)
            mode: supports Gaussian, Embedded Gaussian, Dot Product, and Concatenation
            dimension: can be 1 (temporal), 2 (spatial), 3 (spatiotemporal)
            bn_layer: whether to add batch norm
        """
        super(NLBlockND, self).__init__()

        assert dimension in [1, 2, 3]
        
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        
        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                    conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                    bn(self.in_channels)
                )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                    nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
            
    def forward(self, x):
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """

        #batch_size = x.size(0)
        batch_size = 1
        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x.view(batch_size, self.in_channels, -1)
            phi_x = x.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "concatenate":
            theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
            
            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)
            
            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))
        
        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1) # number of position in x
            f_div_C = f / N
        
        y = torch.matmul(f_div_C, g_x)
        
        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(1, self.inter_channels, *x.size()[2:])
        
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x

        return z

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


# update learning rate (called once every epoch)
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_id='cuda:1'):
    net.to(gpu_id)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:1'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    #net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    net = Unet2(input_nc, output_nc)
    
    return init_net(net, init_type, init_gain, gpu_id)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class Unet2(nn.Module):
        
    def __init__(self, in_ch, out_ch):
        super(Unet2, self).__init__()

        self.conv1 = DoubleConv(in_ch, 16)
       
        self.conv11 = DoubleConv(3, 16)
        self.conv12 = DoubleConv(2*3, 16)
        self.conv13 = DoubleConv(3*3, 16)
        self.conv14 = DoubleConv(4*3, 16)
        self.conv15 = DoubleConv(5*3, 16)
        self.conv16 = DoubleConv(6*3, 16)
        self.conv17 = DoubleConv(7*3, 16)
        self.conv18 = DoubleConv(8*3, 16)
        self.conv19 = DoubleConv(9*3, 16)
        self.conv110 = DoubleConv(10*3, 16)
        self.conv111 = DoubleConv(11*3, 16)
        self.conv112 = DoubleConv(12*3, 16)
        self.conv113 = DoubleConv(13*3, 16)
        self.conv114 = DoubleConv(14*3, 16)
        self.conv115 = DoubleConv(15*3, 16)
        self.conv116 = DoubleConv(16*3, 16)
        self.conv117 = DoubleConv(17*3, 16)
        self.conv118 = DoubleConv(18*3, 16)
        self.conv119 = DoubleConv(19*3, 16)
        self.conv120 = DoubleConv(20*3, 16)
        self.conv121 = DoubleConv(21*3, 16)
        self.conv122 = DoubleConv(22*3, 16)
        self.conv123 = DoubleConv(23*3, 16)
        self.conv124 = DoubleConv(24*3, 16)
        self.conv125 = DoubleConv(25*3, 16)
        self.conv126 = DoubleConv(26*3, 16)
        self.conv127 = DoubleConv(27*3, 16)
        self.conv128 = DoubleConv(28*3, 16)
        self.conv129 = DoubleConv(29*3, 16)
        self.conv130 = DoubleConv(30*3, 16)
        self.conv131 = DoubleConv(31*3, 16)
        self.conv132 = DoubleConv(32*3, 16)
        self.conv133 = DoubleConv(33*3, 16)
        self.conv134 = DoubleConv(34*3, 16)
        self.conv135 = DoubleConv(35*3, 16)
        self.conv136 = DoubleConv(36*3, 16)
        self.conv137 = DoubleConv(37*3, 16)
        self.conv138 = DoubleConv(38*3, 16)
        self.conv139 = DoubleConv(39*3, 16)
        self.conv140 = DoubleConv(40*3, 16)
        self.conv141 = DoubleConv(41*3, 16)
        self.conv142 = DoubleConv(42*3, 16)
        self.conv143 = DoubleConv(43*3, 16)
        self.conv144 = DoubleConv(44*3, 16)
        self.conv145 = DoubleConv(45*3, 16)
        self.conv146 = DoubleConv(46*3, 16)
        self.conv147 = DoubleConv(47*3, 16)
        self.conv148 = DoubleConv(48*3, 16)
        self.conv149 = DoubleConv(49*3, 16)
        self.conv150 = DoubleConv(50*3, 16)
        self.conv151 = DoubleConv(51*3, 16)
        self.conv152 = DoubleConv(52*3, 16)
        self.conv153 = DoubleConv(53*3, 16)
        self.conv154 = DoubleConv(54*3, 16)
        self.conv155 = DoubleConv(55*3, 16)
        self.conv156 = DoubleConv(56*3, 16)
        self.conv157 = DoubleConv(57*3, 16)
        self.conv158 = DoubleConv(58*3, 16)
        self.conv159 = DoubleConv(59*3, 16)
        self.conv160 = DoubleConv(60*3, 16)
        self.conv161 = DoubleConv(61*3, 16)
        self.conv162 = DoubleConv(62*3, 16)
        self.conv163 = DoubleConv(63*3, 16)
        self.conv164 = DoubleConv(64*3, 16)
        self.conv165 = DoubleConv(65*3, 16)
        self.conv166 = DoubleConv(66*3, 16)
        self.conv167 = DoubleConv(67*3, 16)
        self.conv168 = DoubleConv(68*3, 16)
        self.conv169 = DoubleConv(69*3, 16)
        self.conv170 = DoubleConv(70*3, 16)
        self.conv171 = DoubleConv(71*3, 16)
        self.conv172 = DoubleConv(72*3, 16)
        self.conv173 = DoubleConv(73*3, 16)
        self.conv174 = DoubleConv(74*3, 16)
        self.conv175 = DoubleConv(75*3, 16)
        self.conv176 = DoubleConv(76*3, 16)
        self.conv177 = DoubleConv(77*3, 16)
        self.conv178 = DoubleConv(78*3, 16)
        self.conv179 = DoubleConv(79*3, 16)
        self.conv180 = DoubleConv(80*3, 16)
        self.conv181 = DoubleConv(81*3, 16)
        self.conv182 = DoubleConv(82*3, 16)
        self.conv183 = DoubleConv(83*3, 16)
        self.conv184 = DoubleConv(84*3, 16)
        self.conv185 = DoubleConv(85*3, 16)
        self.conv186 = DoubleConv(86*3, 16)
        self.conv187 = DoubleConv(87*3, 16)
        self.conv188 = DoubleConv(88*3, 16)
        self.conv189 = DoubleConv(89*3, 16)
        self.conv190 = DoubleConv(90*3, 16)
        self.conv191 = DoubleConv(91*3, 16)
        self.conv192 = DoubleConv(92*3, 16)
        self.conv193 = DoubleConv(93*3, 16)
        self.conv194 = DoubleConv(94*3, 16)
        self.conv195 = DoubleConv(95*3, 16)
        self.conv196 = DoubleConv(96*3, 16)
        self.conv197 = DoubleConv(97*3, 16)
        self.conv198 = DoubleConv(98*3, 16)
        self.conv199 = DoubleConv(99*3, 16)
        self.conv1100 = DoubleConv(100*3, 16)
        self.conv1101 = DoubleConv(101*3, 16)
        self.conv1102 = DoubleConv(102*3, 16)
        self.conv1103 = DoubleConv(103*3, 16)
        self.conv1104 = DoubleConv(104*3, 16)
        self.conv1105 = DoubleConv(105*3, 16)
        self.conv1106 = DoubleConv(106*3, 16)
        self.conv1107 = DoubleConv(107*3, 16)
        self.conv1108 = DoubleConv(108*3, 16)
        self.conv1109 = DoubleConv(109*3, 16)
        self.conv1110 = DoubleConv(110*3, 16)
        self.conv1111 = DoubleConv(111*3, 16)
        self.conv1112 = DoubleConv(112*3, 16)
        self.conv1113 = DoubleConv(113*3, 16)
        self.conv1114 = DoubleConv(114*3, 16)
        self.conv1115 = DoubleConv(115*3, 16)
        self.conv1116 = DoubleConv(116*3, 16)
        self.conv1117 = DoubleConv(117*3, 16)
        self.conv1118 = DoubleConv(118*3, 16)
        self.conv1119 = DoubleConv(119*3, 16)
        self.conv1120 = DoubleConv(120*3, 16)
        self.conv1121 = DoubleConv(121*3, 16)
        self.conv1122 = DoubleConv(122*3, 16)
        self.conv1123 = DoubleConv(123*3, 16)
        self.conv1124 = DoubleConv(124*3, 16)
        self.conv1125 = DoubleConv(125*3, 16)
        self.conv1126 = DoubleConv(126*3, 16)
        self.conv1127 = DoubleConv(127*3, 16)
        self.conv1128 = DoubleConv(128*3, 16)
        self.conv1129 = DoubleConv(129*3, 16)
        self.conv1130 = DoubleConv(130*3, 16)
        self.conv1131 = DoubleConv(131*3, 16)
        self.conv1132 = DoubleConv(132*3, 16)
        self.conv1133 = DoubleConv(133*3, 16)
        self.conv1134 = DoubleConv(134*3, 16)
        self.conv1135 = DoubleConv(135*3, 16)
        self.conv1136 = DoubleConv(136*3, 16)
        self.conv1137 = DoubleConv(137*3, 16)
        self.conv1138 = DoubleConv(138*3, 16)
        self.conv1139 = DoubleConv(139*3, 16)
        self.conv1140 = DoubleConv(140*3, 16)
        self.conv1141 = DoubleConv(141*3, 16)
        self.conv1142 = DoubleConv(142*3, 16)
        self.conv1143 = DoubleConv(143*3, 16)
        self.conv1144 = DoubleConv(144*3, 16)
        self.conv1145 = DoubleConv(145*3, 16)
        self.conv1146 = DoubleConv(146*3, 16)
        self.conv1147 = DoubleConv(147*3, 16)
        self.conv1148 = DoubleConv(148*3, 16)
        self.conv1149 = DoubleConv(149*3, 16)
        self.conv1150 = DoubleConv(150*3, 16)
        self.conv1151 = DoubleConv(151*3, 16)
        self.conv1152 = DoubleConv(152*3, 16)
        self.conv1153 = DoubleConv(153*3, 16)
        self.conv1154 = DoubleConv(154*3, 16)
        self.conv1155 = DoubleConv(155*3, 16)
        self.conv1156 = DoubleConv(156*3, 16)
        self.conv1157 = DoubleConv(157*3, 16)
        self.conv1158 = DoubleConv(158*3, 16)
        self.conv1159 = DoubleConv(159*3, 16)
        self.conv1160 = DoubleConv(160*3, 16)
        self.conv1161 = DoubleConv(161*3, 16)
        self.conv1162 = DoubleConv(162*3, 16)
        self.conv1163 = DoubleConv(163*3, 16)
        self.conv1164 = DoubleConv(164*3, 16)
        self.conv1165 = DoubleConv(165*3, 16)
        self.conv1166 = DoubleConv(166*3, 16)
        self.conv1167 = DoubleConv(167*3, 16)
        self.conv1168 = DoubleConv(168*3, 16)
        self.conv1169 = DoubleConv(169*3, 16)
        self.conv1170 = DoubleConv(170*3, 16)
        self.conv1171 = DoubleConv(171*3, 16)
        self.conv1172 = DoubleConv(172*3, 16)
        self.conv1173 = DoubleConv(173*3, 16)
        self.conv1174 = DoubleConv(174*3, 16)
        self.conv1175 = DoubleConv(175*3, 16)
        self.conv1176 = DoubleConv(176*3, 16)
        self.conv1177 = DoubleConv(177*3, 16)
        self.conv1178 = DoubleConv(178*3, 16)
        self.conv1179 = DoubleConv(179*3, 16)
        self.conv1180 = DoubleConv(180*3, 16)
        self.conv1181 = DoubleConv(181*3, 16)
        self.conv1182 = DoubleConv(182*3, 16)
        self.conv1183 = DoubleConv(183*3, 16)
        self.conv1184 = DoubleConv(184*3, 16)
        self.conv1185 = DoubleConv(185*3, 16)
        self.conv1186 = DoubleConv(186*3, 16)
        self.conv1187 = DoubleConv(187*3, 16)
        self.conv1188 = DoubleConv(188*3, 16)
        self.conv1189 = DoubleConv(189*3, 16)
        self.conv1190 = DoubleConv(190*3, 16)
        self.conv1191 = DoubleConv(191*3, 16)
        self.conv1192 = DoubleConv(192*3, 16)
        self.conv1193 = DoubleConv(193*3, 16)
        self.conv1194 = DoubleConv(194*3, 16)
        self.conv1195 = DoubleConv(195*3, 16)
        self.conv1196 = DoubleConv(196*3, 16)
        self.conv1197 = DoubleConv(197*3, 16)
        self.conv1198 = DoubleConv(198*3, 16)
        self.conv1199 = DoubleConv(199*3, 16)
        self.conv1200 = DoubleConv(200*3, 16)
        self.conv1201 = DoubleConv(201*3, 16)
        self.conv1202 = DoubleConv(202*3, 16)
        self.conv1203 = DoubleConv(203*3, 16)
        self.conv1204 = DoubleConv(204*3, 16)
        self.conv1205 = DoubleConv(205*3, 16)
        self.conv1206 = DoubleConv(206*3, 16)
        self.conv1207 = DoubleConv(207*3, 16)
        self.conv1208 = DoubleConv(208*3, 16)
        self.conv1209 = DoubleConv(209*3, 16)
        self.conv1210 = DoubleConv(210*3, 16)
        self.conv1211 = DoubleConv(211*3, 16)
        self.conv1212 = DoubleConv(212*3, 16)
        self.conv1213 = DoubleConv(213*3, 16)
        self.conv1214 = DoubleConv(214*3, 16)
        self.conv1215 = DoubleConv(215*3, 16)
        self.conv1216 = DoubleConv(216*3, 16)
        self.conv1217 = DoubleConv(217*3, 16)
        self.conv1218 = DoubleConv(218*3, 16)
        self.conv1219 = DoubleConv(219*3, 16)
        self.conv1220 = DoubleConv(220*3, 16)
        self.conv1221 = DoubleConv(221*3, 16)
        self.conv1222 = DoubleConv(222*3, 16)
        self.conv1223 = DoubleConv(223*3, 16)
        self.conv1224 = DoubleConv(224*3, 16)
        self.conv1225 = DoubleConv(225*3, 16)
        self.conv1226 = DoubleConv(226*3, 16)
        self.conv1227 = DoubleConv(227*3, 16)
        self.conv1228 = DoubleConv(228*3, 16)
        self.conv1229 = DoubleConv(229*3, 16)
        self.conv1230 = DoubleConv(230*3, 16)
        self.conv1231 = DoubleConv(231*3, 16)
        self.conv1232 = DoubleConv(232*3, 16)
        self.conv1233 = DoubleConv(233*3, 16)
        self.conv1234 = DoubleConv(234*3, 16)
        self.conv1235 = DoubleConv(235*3, 16)
        self.conv1236 = DoubleConv(236*3, 16)
        self.conv1237 = DoubleConv(237*3, 16)
        self.conv1238 = DoubleConv(238*3, 16)
        self.conv1239 = DoubleConv(239*3, 16)
        self.conv1240 = DoubleConv(240*3, 16)
        self.conv1241 = DoubleConv(241*3, 16)
        self.conv1242 = DoubleConv(242*3, 16)
        self.conv1243 = DoubleConv(243*3, 16)
        self.conv1244 = DoubleConv(244*3, 16)
        self.conv1245 = DoubleConv(245*3, 16)
        self.conv1246 = DoubleConv(246*3, 16)
        self.conv1247 = DoubleConv(247*3, 16)
        self.conv1248 = DoubleConv(248*3, 16)
        self.conv1249 = DoubleConv(249*3, 16)
        self.conv1250 = DoubleConv(250*3, 16)
        self.conv1251 = DoubleConv(251*3, 16)
        self.conv1252 = DoubleConv(252*3, 16)
        self.conv1253 = DoubleConv(253*3, 16)
        self.conv1254 = DoubleConv(254*3, 16)
        self.conv1255 = DoubleConv(255*3, 16)
        self.conv1256 = DoubleConv(256*3, 16)
        self.conv1257 = DoubleConv(257*3, 16)
        self.conv1258 = DoubleConv(258*3, 16)
        self.conv1259 = DoubleConv(259*3, 16)
        self.conv1260 = DoubleConv(260*3, 16)
        self.conv1261 = DoubleConv(261*3, 16)
        self.conv1262 = DoubleConv(262*3, 16)
        self.conv1263 = DoubleConv(263*3, 16)
        self.conv1264 = DoubleConv(264*3, 16)
        self.conv1265 = DoubleConv(265*3, 16)
        self.conv1266 = DoubleConv(266*3, 16)
        self.conv1267 = DoubleConv(267*3, 16)
        self.conv1268 = DoubleConv(268*3, 16)
        self.conv1269 = DoubleConv(269*3, 16)
        self.conv1270 = DoubleConv(270*3, 16)
        self.conv1271 = DoubleConv(271*3, 16)
        self.conv1272 = DoubleConv(272*3, 16)
        self.conv1273 = DoubleConv(273*3, 16)
        self.conv1274 = DoubleConv(274*3, 16)
        self.conv1275 = DoubleConv(275*3, 16)
        self.conv1276 = DoubleConv(276*3, 16)
        self.conv1277 = DoubleConv(277*3, 16)
        self.conv1278 = DoubleConv(278*3, 16)
        self.conv1279 = DoubleConv(279*3, 16)
        self.conv1280 = DoubleConv(280*3, 16)
        self.conv1281 = DoubleConv(281*3, 16)
        self.conv1282 = DoubleConv(282*3, 16)
        self.conv1283 = DoubleConv(283*3, 16)
        self.conv1284 = DoubleConv(284*3, 16)
        self.conv1285 = DoubleConv(285*3, 16)
        self.conv1286 = DoubleConv(286*3, 16)
        self.conv1287 = DoubleConv(287*3, 16)
        self.conv1288 = DoubleConv(288*3, 16)
        self.conv1289 = DoubleConv(289*3, 16)
        self.conv1290 = DoubleConv(290*3, 16)
        self.conv1291 = DoubleConv(291*3, 16)
        self.conv1292 = DoubleConv(292*3, 16)
        self.conv1293 = DoubleConv(293*3, 16)
        self.conv1294 = DoubleConv(294*3, 16)
        self.conv1295 = DoubleConv(295*3, 16)
        self.conv1296 = DoubleConv(296*3, 16)
        self.conv1297 = DoubleConv(297*3, 16)
        self.conv1298 = DoubleConv(298*3, 16)
        self.conv1299 = DoubleConv(299*3, 16)
        self.conv1300 = DoubleConv(300*3, 16)
        self.conv1301 = DoubleConv(301*3, 16)
        self.conv1302 = DoubleConv(302*3, 16)
        self.conv1303 = DoubleConv(303*3, 16)
        self.conv1304 = DoubleConv(304*3, 16)
        self.conv1305 = DoubleConv(305*3, 16)
        self.conv1306 = DoubleConv(306*3, 16)
        self.conv1307 = DoubleConv(307*3, 16)
        self.conv1308 = DoubleConv(308*3, 16)
        self.conv1309 = DoubleConv(309*3, 16)
        self.conv1310 = DoubleConv(310*3, 16)
        self.conv1311 = DoubleConv(311*3, 16)
        self.conv1312 = DoubleConv(312*3, 16)
        self.conv1313 = DoubleConv(313*3, 16)
        self.conv1314 = DoubleConv(314*3, 16)
        self.conv1315 = DoubleConv(315*3, 16)
        self.conv1316 = DoubleConv(316*3, 16)
        self.conv1317 = DoubleConv(317*3, 16)
        self.conv1318 = DoubleConv(318*3, 16)
        self.conv1319 = DoubleConv(319*3, 16)
        self.conv1320 = DoubleConv(320*3, 16)
        self.conv1321 = DoubleConv(321*3, 16)
        self.conv1322 = DoubleConv(322*3, 16)
        self.conv1323 = DoubleConv(323*3, 16)
        self.conv1324 = DoubleConv(324*3, 16)
        self.conv1325 = DoubleConv(325*3, 16)
        self.conv1326 = DoubleConv(326*3, 16)
        self.conv1327 = DoubleConv(327*3, 16)
        self.conv1328 = DoubleConv(328*3, 16)
        self.conv1329 = DoubleConv(329*3, 16)
        self.conv1330 = DoubleConv(330*3, 16)
        self.conv1331 = DoubleConv(331*3, 16)
        self.conv1332 = DoubleConv(332*3, 16)
        self.conv1333 = DoubleConv(333*3, 16)
        self.conv1334 = DoubleConv(334*3, 16)
        self.conv1335 = DoubleConv(335*3, 16)
        self.conv1336 = DoubleConv(336*3, 16)
        self.conv1337 = DoubleConv(337*3, 16)
        self.conv1338 = DoubleConv(338*3, 16)
        self.conv1339 = DoubleConv(339*3, 16)
        self.conv1340 = DoubleConv(340*3, 16)
        self.conv1341 = DoubleConv(341*3, 16)
        self.conv1342 = DoubleConv(342*3, 16)
        self.conv1343 = DoubleConv(343*3, 16)
        self.conv1344 = DoubleConv(344*3, 16)
        self.conv1345 = DoubleConv(345*3, 16)
        self.conv1346 = DoubleConv(346*3, 16)
        self.conv1347 = DoubleConv(347*3, 16)
        self.conv1348 = DoubleConv(348*3, 16)
        self.conv1349 = DoubleConv(349*3, 16)
        self.conv1350 = DoubleConv(350*3, 16)
        self.conv1351 = DoubleConv(351*3, 16)
        self.conv1352 = DoubleConv(352*3, 16)
        self.conv1353 = DoubleConv(353*3, 16)
        self.conv1354 = DoubleConv(354*3, 16)
        self.conv1355 = DoubleConv(355*3, 16)
        self.conv1356 = DoubleConv(356*3, 16)
        self.conv1357 = DoubleConv(357*3, 16)
        self.conv1358 = DoubleConv(358*3, 16)
        self.conv1359 = DoubleConv(359*3, 16)
        self.conv1360 = DoubleConv(360*3, 16)
        self.conv1361 = DoubleConv(361*3, 16)
        self.conv1362 = DoubleConv(362*3, 16)
        self.conv1363 = DoubleConv(363*3, 16)
        self.conv1364 = DoubleConv(364*3, 16)
        self.conv1365 = DoubleConv(365*3, 16)
        self.conv1366 = DoubleConv(366*3, 16)
        self.conv1367 = DoubleConv(367*3, 16)
        self.conv1368 = DoubleConv(368*3, 16)
        self.conv1369 = DoubleConv(369*3, 16)
        self.conv1370 = DoubleConv(370*3, 16)
        self.conv1371 = DoubleConv(371*3, 16)
        self.conv1372 = DoubleConv(372*3, 16)
        self.conv1373 = DoubleConv(373*3, 16)
        self.conv1374 = DoubleConv(374*3, 16)
        self.conv1375 = DoubleConv(375*3, 16)
        self.conv1376 = DoubleConv(376*3, 16)
        self.conv1377 = DoubleConv(377*3, 16)
        self.conv1378 = DoubleConv(378*3, 16)
        self.conv1379 = DoubleConv(379*3, 16)
        self.conv1380 = DoubleConv(380*3, 16)
        self.conv1381 = DoubleConv(381*3, 16)
        self.conv1382 = DoubleConv(382*3, 16)
        self.conv1383 = DoubleConv(383*3, 16)
        self.conv1384 = DoubleConv(384*3, 16)
        self.conv1385 = DoubleConv(385*3, 16)
        self.conv1386 = DoubleConv(386*3, 16)
        self.conv1387 = DoubleConv(387*3, 16)
        self.conv1388 = DoubleConv(388*3, 16)
        self.conv1389 = DoubleConv(389*3, 16)
        self.conv1390 = DoubleConv(390*3, 16)
        self.conv1391 = DoubleConv(391*3, 16)
        self.conv1392 = DoubleConv(392*3, 16)
        self.conv1393 = DoubleConv(393*3, 16)
        self.conv1394 = DoubleConv(394*3, 16)
        self.conv1395 = DoubleConv(395*3, 16)
        self.conv1396 = DoubleConv(396*3, 16)
        self.conv1397 = DoubleConv(397*3, 16)
        self.conv1398 = DoubleConv(398*3, 16)
        self.conv1399 = DoubleConv(399*3, 16)
        self.conv1400 = DoubleConv(400*3, 16)
        self.conv1401 = DoubleConv(401*3, 16)
        self.conv1402 = DoubleConv(402*3, 16)
        self.conv1403 = DoubleConv(403*3, 16)
        self.conv1404 = DoubleConv(404*3, 16)
        self.conv1405 = DoubleConv(405*3, 16)
        self.conv1406 = DoubleConv(406*3, 16)
        self.conv1407 = DoubleConv(407*3, 16)
        self.conv1408 = DoubleConv(408*3, 16)
        self.conv1409 = DoubleConv(409*3, 16)
        self.conv1410 = DoubleConv(410*3, 16)
        self.conv1411 = DoubleConv(411*3, 16)
        self.conv1412 = DoubleConv(412*3, 16)
        self.conv1413 = DoubleConv(413*3, 16)
        self.conv1414 = DoubleConv(414*3, 16)
        self.conv1415 = DoubleConv(415*3, 16)
        self.conv1416 = DoubleConv(416*3, 16)
        self.conv1417 = DoubleConv(417*3, 16)
        self.conv1418 = DoubleConv(418*3, 16)
        self.conv1419 = DoubleConv(419*3, 16)
        self.conv1420 = DoubleConv(420*3, 16)
        self.conv1421 = DoubleConv(421*3, 16)
        self.conv1422 = DoubleConv(422*3, 16)
        self.conv1423 = DoubleConv(423*3, 16)
        self.conv1424 = DoubleConv(424*3, 16)
        self.conv1425 = DoubleConv(425*3, 16)
        self.conv1426 = DoubleConv(426*3, 16)
        self.conv1427 = DoubleConv(427*3, 16)
        self.conv1428 = DoubleConv(428*3, 16)
        self.conv1429 = DoubleConv(429*3, 16)
        self.conv1430 = DoubleConv(430*3, 16)
        self.conv1431 = DoubleConv(431*3, 16)
        self.conv1432 = DoubleConv(432*3, 16)
        self.conv1433 = DoubleConv(433*3, 16)
        self.conv1434 = DoubleConv(434*3, 16)
        self.conv1435 = DoubleConv(435*3, 16)
        self.conv1436 = DoubleConv(436*3, 16)
        self.conv1437 = DoubleConv(437*3, 16)
        self.conv1438 = DoubleConv(438*3, 16)
        self.conv1439 = DoubleConv(439*3, 16)
        self.conv1440 = DoubleConv(440*3, 16)
        self.conv1441 = DoubleConv(441*3, 16)
        self.conv1442 = DoubleConv(442*3, 16)
        self.conv1443 = DoubleConv(443*3, 16)
        self.conv1444 = DoubleConv(444*3, 16)
        self.conv1445 = DoubleConv(445*3, 16)
        self.conv1446 = DoubleConv(446*3, 16)
        self.conv1447 = DoubleConv(447*3, 16)
        self.conv1448 = DoubleConv(448*3, 16)
        self.conv1449 = DoubleConv(449*3, 16)
        self.conv1450 = DoubleConv(450*3, 16)
        self.conv1451 = DoubleConv(451*3, 16)
        self.conv1452 = DoubleConv(452*3, 16)
        self.conv1453 = DoubleConv(453*3, 16)
        self.conv1454 = DoubleConv(454*3, 16)
        self.conv1455 = DoubleConv(455*3, 16)
        self.conv1456 = DoubleConv(456*3, 16)
        self.conv1457 = DoubleConv(457*3, 16)
        self.conv1458 = DoubleConv(458*3, 16)
        self.conv1459 = DoubleConv(459*3, 16)
        self.conv1460 = DoubleConv(460*3, 16)
        self.conv1461 = DoubleConv(461*3, 16)
        self.conv1462 = DoubleConv(462*3, 16)
        self.conv1463 = DoubleConv(463*3, 16)
        self.conv1464 = DoubleConv(464*3, 16)
        self.conv1465 = DoubleConv(465*3, 16)
        self.conv1466 = DoubleConv(466*3, 16)
        self.conv1467 = DoubleConv(467*3, 16)
        self.conv1468 = DoubleConv(468*3, 16)
        self.conv1469 = DoubleConv(469*3, 16)
        self.conv1470 = DoubleConv(470*3, 16)
        self.conv1471 = DoubleConv(471*3, 16)
        self.conv1472 = DoubleConv(472*3, 16)
        self.conv1473 = DoubleConv(473*3, 16)
        self.conv1474 = DoubleConv(474*3, 16)
        self.conv1475 = DoubleConv(475*3, 16)
        self.conv1476 = DoubleConv(476*3, 16)
        self.conv1477 = DoubleConv(477*3, 16)
        self.conv1478 = DoubleConv(478*3, 16)
        self.conv1479 = DoubleConv(479*3, 16)
        self.conv1480 = DoubleConv(480*3, 16)
        self.conv1481 = DoubleConv(481*3, 16)
        self.conv1482 = DoubleConv(482*3, 16)
        self.conv1483 = DoubleConv(483*3, 16)
        self.conv1484 = DoubleConv(484*3, 16)
        self.conv1485 = DoubleConv(485*3, 16)
        self.conv1486 = DoubleConv(486*3, 16)
        self.conv4187 = DoubleConv(487*3, 16)
        self.conv1488 = DoubleConv(488*3, 16)
        self.conv1489 = DoubleConv(489*3, 16)
        self.conv1490 = DoubleConv(490*3, 16)
        self.conv1491 = DoubleConv(491*3, 16)
        self.conv1492 = DoubleConv(492*3, 16)
        self.conv1493 = DoubleConv(493*3, 16)
        self.conv1494 = DoubleConv(494*3, 16)
        self.conv1495 = DoubleConv(495*3, 16)
        self.conv1496 = DoubleConv(496*3, 16)
        self.conv1497 = DoubleConv(497*3, 16)
        self.conv1498 = DoubleConv(498*3, 16)
        self.conv1499 = DoubleConv(499*3, 16)
        self.conv1500 = DoubleConv(500*3, 16)
        self.conv1501 = DoubleConv(501*3, 16)
        self.conv1502 = DoubleConv(502*3, 16)
        self.conv1503 = DoubleConv(503*3, 16)
        self.conv1504 = DoubleConv(504*3, 16)
        self.conv1505 = DoubleConv(505*3, 16)
        self.conv1506 = DoubleConv(506*3, 16)
        self.conv1507 = DoubleConv(507*3, 16)
        self.conv1508 = DoubleConv(508*3, 16)
        self.conv1509 = DoubleConv(509*3, 16)
        self.conv1510 = DoubleConv(510*3, 16)
        self.conv1511 = DoubleConv(511*3, 16)
        self.conv1512 = DoubleConv(512*3, 16)
        self.conv1513 = DoubleConv(513*3, 16)
        self.conv1514 = DoubleConv(514*3, 16)
        self.conv1515 = DoubleConv(515*3, 16)
        self.conv1516 = DoubleConv(516*3, 16)
        self.conv1517 = DoubleConv(517*3, 16)
        self.conv1518 = DoubleConv(518*3, 16)
        self.conv1519 = DoubleConv(519*3, 16)
        self.conv1520 = DoubleConv(520*3, 16)
        self.conv1521 = DoubleConv(521*3, 16)
        self.conv1522 = DoubleConv(522*3, 16)
        self.conv1523 = DoubleConv(523*3, 16)
        self.conv1524 = DoubleConv(524*3, 16)
        self.conv1525 = DoubleConv(525*3, 16)
        self.conv1526 = DoubleConv(526*3, 16)
        self.conv1527 = DoubleConv(527*3, 16)
        self.conv1528 = DoubleConv(528*3, 16)
        self.conv1529 = DoubleConv(529*3, 16)
        self.conv1530 = DoubleConv(530*3, 16)
        self.conv1531 = DoubleConv(531*3, 16)
        self.conv1532 = DoubleConv(532*3, 16)
        self.conv1533 = DoubleConv(533*3, 16)
        self.conv1534 = DoubleConv(534*3, 16)
        self.conv1535 = DoubleConv(535*3, 16)
        self.conv1536 = DoubleConv(536*3, 16)
        self.conv1537 = DoubleConv(537*3, 16)
        self.conv1538 = DoubleConv(538*3, 16)
        self.conv1539 = DoubleConv(539*3, 16)
        self.conv1540 = DoubleConv(540*3, 16)
        self.conv1541 = DoubleConv(541*3, 16)
        self.conv1542 = DoubleConv(542*3, 16)
        self.conv1543 = DoubleConv(543*3, 16)
        self.conv1544 = DoubleConv(544*3, 16)
        self.conv1545 = DoubleConv(545*3, 16)
        self.conv1546 = DoubleConv(546*3, 16)
        self.conv1547 = DoubleConv(547*3, 16)
        self.conv1548 = DoubleConv(548*3, 16)
        self.conv1549 = DoubleConv(549*3, 16)
        self.conv1550 = DoubleConv(550*3, 16)
        self.conv1551 = DoubleConv(551*3, 16)
        self.conv1552 = DoubleConv(552*3, 16)
        self.conv1553 = DoubleConv(553*3, 16)
        self.conv1554 = DoubleConv(554*3, 16)
        self.conv1555 = DoubleConv(555*3, 16)
        self.conv1556 = DoubleConv(556*3, 16)
        self.conv1557 = DoubleConv(557*3, 16)
        self.conv1558 = DoubleConv(558*3, 16)
        self.conv1559 = DoubleConv(559*3, 16)
        self.conv1560 = DoubleConv(560*3, 16)
        self.conv1561 = DoubleConv(561*3, 16)
        self.conv1562 = DoubleConv(562*3, 16)
        self.conv1563 = DoubleConv(563*3, 16)
        self.conv1564 = DoubleConv(564*3, 16)
        self.conv1565 = DoubleConv(565*3, 16)
        self.conv1566 = DoubleConv(566*3, 16)
        self.conv1567 = DoubleConv(567*3, 16)
        self.conv1568 = DoubleConv(568*3, 16)
        self.conv1569 = DoubleConv(569*3, 16)
        self.conv1570 = DoubleConv(570*3, 16)
        self.conv1571 = DoubleConv(571*3, 16)
        self.conv1572 = DoubleConv(572*3, 16)
        self.conv1573 = DoubleConv(573*3, 16)
        self.conv1574 = DoubleConv(574*3, 16)
        self.conv1575 = DoubleConv(575*3, 16)
        self.conv1576 = DoubleConv(576*3, 16)
        self.conv1577 = DoubleConv(577*3, 16)
        self.conv1578 = DoubleConv(578*3, 16)
        self.conv1579 = DoubleConv(579*3, 16)
        self.conv1580 = DoubleConv(580*3, 16)
        self.conv1581 = DoubleConv(581*3, 16)
        self.conv1582 = DoubleConv(582*3, 16)
        self.conv1583 = DoubleConv(583*3, 16)
        self.conv1584 = DoubleConv(584*3, 16)
        self.conv1585 = DoubleConv(585*3, 16)
        self.conv1586 = DoubleConv(586*3, 16)
        self.conv1587 = DoubleConv(587*3, 16)
        self.conv1588 = DoubleConv(588*3, 16)
        self.conv1589 = DoubleConv(589*3, 16)
        self.conv1590 = DoubleConv(590*3, 16)
        self.conv1591 = DoubleConv(591*3, 16)
        self.conv1592 = DoubleConv(592*3, 16)
        self.conv1593 = DoubleConv(593*3, 16)
        self.conv1594 = DoubleConv(594*3, 16)
        self.conv1595 = DoubleConv(595*3, 16)
        self.conv1596 = DoubleConv(596*3, 16)
        self.conv1597 = DoubleConv(597*3, 16)
        self.conv1598 = DoubleConv(598*3, 16)
        self.conv1599 = DoubleConv(599*3, 16)
        self.conv1600 = DoubleConv(600*3, 16)
        self.conv1601 = DoubleConv(601*3, 16)
        self.conv1602 = DoubleConv(602*3, 16)
        self.conv1603 = DoubleConv(603*3, 16)
        self.conv1604 = DoubleConv(604*3, 16)
        self.conv1605 = DoubleConv(605*3, 16)
        self.conv1606 = DoubleConv(606*3, 16)
        self.conv1607 = DoubleConv(607*3, 16)
        self.conv1608 = DoubleConv(608*3, 16)
        self.conv1609 = DoubleConv(609*3, 16)
        self.conv1610 = DoubleConv(610*3, 16)
        self.conv1611 = DoubleConv(611*3, 16)
        self.conv1612 = DoubleConv(612*3, 16)
        self.conv1613 = DoubleConv(613*3, 16)
        self.conv1614 = DoubleConv(614*3, 16)
        self.conv1615 = DoubleConv(615*3, 16)
        self.conv1616 = DoubleConv(616*3, 16)
        self.conv1617 = DoubleConv(617*3, 16)
        self.conv1618 = DoubleConv(618*3, 16)
        self.conv1619 = DoubleConv(619*3, 16)
        self.conv1620 = DoubleConv(620*3, 16)
        self.conv1621 = DoubleConv(621*3, 16)
        self.conv1622 = DoubleConv(622*3, 16)
        self.conv1623 = DoubleConv(623*3, 16)
        self.conv1624 = DoubleConv(624*3, 16)
        self.conv1625 = DoubleConv(625*3, 16)
        self.conv1626 = DoubleConv(626*3, 16)
        self.conv1627 = DoubleConv(627*3, 16)
        self.conv1628 = DoubleConv(628*3, 16)
        self.conv1629 = DoubleConv(629*3, 16)
        self.conv1630 = DoubleConv(630*3, 16)
        self.conv1631 = DoubleConv(631*3, 16)
        self.conv1632 = DoubleConv(632*3, 16)
        self.conv1633 = DoubleConv(633*3, 16)
        self.conv1634 = DoubleConv(634*3, 16)
        self.conv1635 = DoubleConv(635*3, 16)
        self.conv1636 = DoubleConv(636*3, 16)
        self.conv1637 = DoubleConv(637*3, 16)
        self.conv1638 = DoubleConv(638*3, 16)
        self.conv1639 = DoubleConv(639*3, 16)
        self.conv1640 = DoubleConv(640*3, 16)
        self.conv1641 = DoubleConv(641*3, 16)
        self.conv1642 = DoubleConv(642*3, 16)
        self.conv1643 = DoubleConv(643*3, 16)
        self.conv1644 = DoubleConv(644*3, 16)
        self.conv1645 = DoubleConv(645*3, 16)
        self.conv1646 = DoubleConv(646*3, 16)
        self.conv1647 = DoubleConv(647*3, 16)
        self.conv1648 = DoubleConv(648*3, 16)
        self.conv1649 = DoubleConv(649*3, 16)
        self.conv1650 = DoubleConv(650*3, 16)
        self.conv1651 = DoubleConv(651*3, 16)
        self.conv1652 = DoubleConv(652*3, 16)
        self.conv1653 = DoubleConv(653*3, 16)
        self.conv1654 = DoubleConv(654*3, 16)
        self.conv1655 = DoubleConv(655*3, 16)
        self.conv1656 = DoubleConv(656*3, 16)
        self.conv1657 = DoubleConv(657*3, 16)
        self.conv1658 = DoubleConv(658*3, 16)
        self.conv1659 = DoubleConv(659*3, 16)
        self.conv1660 = DoubleConv(660*3, 16)
        self.conv1661 = DoubleConv(661*3, 16)
        self.conv1662 = DoubleConv(662*3, 16)
        self.conv1663 = DoubleConv(663*3, 16)
        self.conv1664 = DoubleConv(664*3, 16)
        self.conv1665 = DoubleConv(665*3, 16)
        self.conv1666 = DoubleConv(666*3, 16)
        self.conv1667 = DoubleConv(667*3, 16)
        self.conv1668 = DoubleConv(668*3, 16)
        self.conv1669 = DoubleConv(669*3, 16)
        self.conv1670 = DoubleConv(670*3, 16)
        self.conv1671 = DoubleConv(671*3, 16)
        self.conv1672 = DoubleConv(672*3, 16)
        self.conv1673 = DoubleConv(673*3, 16)
        self.conv1674 = DoubleConv(674*3, 16)
        self.conv1675 = DoubleConv(675*3, 16)
        self.conv1676 = DoubleConv(676*3, 16)
        self.conv1677 = DoubleConv(677*3, 16)
        self.conv1678 = DoubleConv(678*3, 16)
        self.conv1679 = DoubleConv(679*3, 16)
        self.conv1680 = DoubleConv(680*3, 16)
        self.conv1681 = DoubleConv(681*3, 16)
        self.conv1682 = DoubleConv(682*3, 16)
        self.conv1683 = DoubleConv(683*3, 16)
        self.conv1684 = DoubleConv(684*3, 16)
        self.conv1685 = DoubleConv(685*3, 16)
        self.conv1686 = DoubleConv(686*3, 16)
        self.conv1687 = DoubleConv(687*3, 16)
        self.conv1688 = DoubleConv(688*3, 16)
        self.conv1689 = DoubleConv(689*3, 16)
        self.conv1690 = DoubleConv(690*3, 16)
        self.conv1691 = DoubleConv(691*3, 16)
        self.conv1692 = DoubleConv(692*3, 16)
        self.conv1693 = DoubleConv(693*3, 16)
        self.conv1694 = DoubleConv(694*3, 16)
        self.conv1695 = DoubleConv(695*3, 16)
        self.conv1696 = DoubleConv(696*3, 16)
        self.conv1697 = DoubleConv(697*3, 16)
        self.conv1698 = DoubleConv(698*3, 16)
        self.conv1699 = DoubleConv(699*3, 16)
        self.conv1700 = DoubleConv(700*3, 16)
        self.conv1701 = DoubleConv(701*3, 16)
        self.conv1702 = DoubleConv(702*3, 16)
        self.conv1703 = DoubleConv(703*3, 16)
        self.conv1704 = DoubleConv(704*3, 16)
        self.conv1705 = DoubleConv(705*3, 16)
        self.conv1706 = DoubleConv(706*3, 16)
        self.conv1707 = DoubleConv(707*3, 16)
        self.conv1708 = DoubleConv(708*3, 16)
        self.conv1709 = DoubleConv(709*3, 16)
        self.conv1710 = DoubleConv(710*3, 16)
        self.conv1711 = DoubleConv(711*3, 16)
        self.conv1712 = DoubleConv(712*3, 16)
        self.conv1713 = DoubleConv(713*3, 16)
        self.conv1714 = DoubleConv(714*3, 16)
        self.conv1715 = DoubleConv(715*3, 16)
        self.conv1716 = DoubleConv(716*3, 16)
        self.conv1717 = DoubleConv(717*3, 16)
        self.conv1718 = DoubleConv(718*3, 16)
        self.conv1719 = DoubleConv(719*3, 16)
        self.conv1720 = DoubleConv(720*3, 16)
        self.conv1721 = DoubleConv(721*3, 16)
        self.conv1722 = DoubleConv(722*3, 16)
        self.conv1723 = DoubleConv(723*3, 16)
        self.conv1724 = DoubleConv(724*3, 16)
        self.conv1725 = DoubleConv(725*3, 16)
        self.conv1726 = DoubleConv(726*3, 16)
        self.conv1727 = DoubleConv(727*3, 16)
        self.conv1728 = DoubleConv(728*3, 16)
        self.conv1729 = DoubleConv(729*3, 16)
        self.conv1730 = DoubleConv(730*3, 16)
        self.conv1731 = DoubleConv(731*3, 16)
        self.conv1732 = DoubleConv(732*3, 16)
        self.conv1733 = DoubleConv(733*3, 16)
        self.conv1734 = DoubleConv(734*3, 16)
        self.conv1735 = DoubleConv(735*3, 16)
        self.conv1736 = DoubleConv(736*3, 16)
        self.conv1737 = DoubleConv(737*3, 16)
        self.conv1738 = DoubleConv(738*3, 16)
        self.conv1739 = DoubleConv(739*3, 16)
        self.conv1740 = DoubleConv(740*3, 16)
        self.conv1741 = DoubleConv(741*3, 16)
        self.conv1742 = DoubleConv(742*3, 16)
        self.conv1743 = DoubleConv(743*3, 16)
        self.conv1744 = DoubleConv(744*3, 16)
        self.conv1745 = DoubleConv(745*3, 16)
        self.conv1746 = DoubleConv(746*3, 16)
        self.conv1747 = DoubleConv(747*3, 16)
        self.conv1748 = DoubleConv(748*3, 16)
        self.conv1749 = DoubleConv(749*3, 16)
        self.conv1750 = DoubleConv(750*3, 16)
        self.conv1751 = DoubleConv(751*3, 16)
        self.conv1752 = DoubleConv(752*3, 16)
        self.conv1753 = DoubleConv(753*3, 16)
        self.conv1754 = DoubleConv(754*3, 16)
        self.conv1755 = DoubleConv(755*3, 16)
        self.conv1756 = DoubleConv(756*3, 16)
        self.conv1757 = DoubleConv(757*3, 16)
        self.conv1758 = DoubleConv(758*3, 16)
        self.conv1759 = DoubleConv(759*3, 16)
        self.conv1760 = DoubleConv(760*3, 16)
        self.conv1761 = DoubleConv(761*3, 16)
        self.conv1762 = DoubleConv(762*3, 16)
        self.conv1763 = DoubleConv(763*3, 16)
        self.conv1764 = DoubleConv(764*3, 16)
        self.conv1765 = DoubleConv(765*3, 16)
        self.conv1766 = DoubleConv(766*3, 16)
        self.conv1767 = DoubleConv(767*3, 16)
        self.conv1768 = DoubleConv(768*3, 16)
        self.conv1769 = DoubleConv(769*3, 16)
        self.conv1770 = DoubleConv(770*3, 16)
        self.conv1771 = DoubleConv(771*3, 16)
        self.conv1772 = DoubleConv(772*3, 16)
        self.conv1773 = DoubleConv(773*3, 16)
        self.conv1774 = DoubleConv(774*3, 16)
        self.conv1775 = DoubleConv(775*3, 16)
        self.conv1776 = DoubleConv(776*3, 16)
        self.conv1777 = DoubleConv(777*3, 16)
        self.conv1778 = DoubleConv(778*3, 16)
        self.conv1779 = DoubleConv(779*3, 16)
        self.conv1780 = DoubleConv(780*3, 16)
        self.conv1781 = DoubleConv(781*3, 16)
        self.conv1782 = DoubleConv(782*3, 16)
        self.conv1783 = DoubleConv(783*3, 16)
        self.conv1784 = DoubleConv(784*3, 16)
        self.conv1785 = DoubleConv(785*3, 16)
        self.conv1786 = DoubleConv(786*3, 16)
        self.conv1787 = DoubleConv(787*3, 16)
        self.conv1788 = DoubleConv(788*3, 16)
        self.conv1789 = DoubleConv(789*3, 16)
        self.conv1790 = DoubleConv(790*3, 16)
        self.conv1791 = DoubleConv(791*3, 16)
        self.conv1792 = DoubleConv(792*3, 16)
        self.conv1793 = DoubleConv(793*3, 16)
        self.conv1794 = DoubleConv(794*3, 16)
        self.conv1795 = DoubleConv(795*3, 16)
        self.conv1796 = DoubleConv(796*3, 16)
        self.conv1797 = DoubleConv(797*3, 16)
        self.conv1798 = DoubleConv(798*3, 16)
        self.conv1799 = DoubleConv(799*3, 16)
        self.conv1800 = DoubleConv(800*3, 16)
        self.conv1801 = DoubleConv(801*3, 16)
        self.conv1802 = DoubleConv(802*3, 16)
        self.conv1803 = DoubleConv(803*3, 16)
        self.conv1804 = DoubleConv(804*3, 16)
        self.conv1805 = DoubleConv(805*3, 16)
        self.conv1806 = DoubleConv(806*3, 16)
        self.conv1807 = DoubleConv(807*3, 16)
        self.conv1808 = DoubleConv(808*3, 16)
        self.conv1809 = DoubleConv(809*3, 16)
        self.conv1810 = DoubleConv(810*3, 16)
        self.conv1811 = DoubleConv(811*3, 16)
        self.conv1812 = DoubleConv(812*3, 16)
        self.conv1813 = DoubleConv(813*3, 16)
        self.conv1814 = DoubleConv(814*3, 16)
        self.conv1815 = DoubleConv(815*3, 16)
        self.conv1816 = DoubleConv(816*3, 16)
        self.conv1817 = DoubleConv(817*3, 16)
        self.conv1818 = DoubleConv(818*3, 16)
        self.conv1819 = DoubleConv(819*3, 16)
        self.conv1820 = DoubleConv(820*3, 16)
        self.conv1821 = DoubleConv(821*3, 16)
        self.conv1822 = DoubleConv(822*3, 16)
        self.conv1823 = DoubleConv(823*3, 16)
        self.conv1824 = DoubleConv(824*3, 16)
        self.conv1825 = DoubleConv(825*3, 16)
        self.conv1826 = DoubleConv(826*3, 16)
        self.conv1827 = DoubleConv(827*3, 16)
        self.conv1828 = DoubleConv(828*3, 16)
        self.conv1829 = DoubleConv(829*3, 16)
        self.conv1830 = DoubleConv(830*3, 16)
        self.conv1831 = DoubleConv(831*3, 16)
        self.conv1832 = DoubleConv(832*3, 16)
        self.conv1833 = DoubleConv(833*3, 16)
        self.conv1834 = DoubleConv(834*3, 16)
        self.conv1835 = DoubleConv(835*3, 16)
        self.conv1836 = DoubleConv(836*3, 16)
        self.conv1837 = DoubleConv(837*3, 16)
        self.conv1838 = DoubleConv(838*3, 16)
        self.conv1839 = DoubleConv(839*3, 16)
        self.conv1840 = DoubleConv(840*3, 16)
        self.conv1841 = DoubleConv(841*3, 16)
        self.conv1842 = DoubleConv(842*3, 16)
        self.conv1843 = DoubleConv(843*3, 16)
        self.conv1844 = DoubleConv(844*3, 16)
        self.conv1845 = DoubleConv(845*3, 16)
        self.conv1846 = DoubleConv(846*3, 16)
        self.conv1847 = DoubleConv(847*3, 16)
        self.conv1848 = DoubleConv(848*3, 16)
        self.conv1849 = DoubleConv(849*3, 16)
        self.conv1850 = DoubleConv(850*3, 16)
        self.conv1851 = DoubleConv(851*3, 16)
        self.conv1852 = DoubleConv(852*3, 16)
        self.conv1853 = DoubleConv(853*3, 16)
        self.conv1854 = DoubleConv(854*3, 16)
        self.conv1855 = DoubleConv(855*3, 16)
        self.conv1856 = DoubleConv(856*3, 16)
        self.conv1857 = DoubleConv(857*3, 16)
        self.conv1858 = DoubleConv(858*3, 16)
        self.conv1859 = DoubleConv(859*3, 16)
        self.conv1860 = DoubleConv(860*3, 16)
        self.conv1861 = DoubleConv(861*3, 16)
        self.conv1862 = DoubleConv(862*3, 16)
        self.conv1863 = DoubleConv(863*3, 16)
        self.conv1864 = DoubleConv(864*3, 16)
        self.conv1865 = DoubleConv(865*3, 16)
        self.conv1866 = DoubleConv(866*3, 16)
        self.conv1867 = DoubleConv(867*3, 16)
        self.conv1868 = DoubleConv(868*3, 16)
        self.conv1869 = DoubleConv(869*3, 16)
        self.conv1870 = DoubleConv(870*3, 16)
        self.conv1871 = DoubleConv(871*3, 16)
        self.conv1872 = DoubleConv(872*3, 16)
        self.conv1873 = DoubleConv(873*3, 16)
        self.conv1874 = DoubleConv(874*3, 16)
        self.conv1875 = DoubleConv(875*3, 16)
        self.conv1876 = DoubleConv(876*3, 16)
        self.conv1877 = DoubleConv(877*3, 16)
        self.conv1878 = DoubleConv(878*3, 16)
        self.conv1879 = DoubleConv(879*3, 16)
        self.conv1880 = DoubleConv(880*3, 16)
        self.conv1881 = DoubleConv(881*3, 16)
        self.conv1882 = DoubleConv(882*3, 16)
        self.conv1883 = DoubleConv(883*3, 16)
        self.conv1884 = DoubleConv(884*3, 16)
        self.conv1885 = DoubleConv(885*3, 16)
        self.conv1886 = DoubleConv(886*3, 16)
        self.conv1887 = DoubleConv(887*3, 16)
        self.conv1888 = DoubleConv(888*3, 16)
        self.conv1889 = DoubleConv(889*3, 16)
        self.conv1890 = DoubleConv(890*3, 16)
        self.conv1891 = DoubleConv(891*3, 16)
        self.conv1892 = DoubleConv(892*3, 16)
        self.conv1893 = DoubleConv(893*3, 16)
        self.conv1894 = DoubleConv(894*3, 16)
        self.conv1895 = DoubleConv(895*3, 16)
        self.conv1896 = DoubleConv(896*3, 16)
        self.conv1897 = DoubleConv(897*3, 16)
        self.conv1898 = DoubleConv(898*3, 16)
        self.conv1899 = DoubleConv(899*3, 16)
        self.conv1900 = DoubleConv(900*3, 16)
        self.conv1901 = DoubleConv(901*3, 16)
        self.conv1902 = DoubleConv(902*3, 16)
        self.conv1903 = DoubleConv(903*3, 16)
        self.conv1904 = DoubleConv(904*3, 16)
        self.conv1905 = DoubleConv(905*3, 16)
        self.conv1906 = DoubleConv(906*3, 16)
        self.conv1907 = DoubleConv(907*3, 16)
        self.conv1908 = DoubleConv(908*3, 16)
        self.conv1909 = DoubleConv(909*3, 16)
        self.conv1910 = DoubleConv(910*3, 16)
        self.conv1911 = DoubleConv(911*3, 16)
        self.conv1912 = DoubleConv(912*3, 16)
        self.conv1913 = DoubleConv(913*3, 16)
        self.conv1914 = DoubleConv(914*3, 16)
        self.conv1915 = DoubleConv(915*3, 16)
        self.conv1916 = DoubleConv(916*3, 16)
        self.conv1917 = DoubleConv(917*3, 16)
        self.conv1918 = DoubleConv(918*3, 16)
        self.conv1919 = DoubleConv(919*3, 16)
        self.conv1920 = DoubleConv(920*3, 16)
        self.conv1921 = DoubleConv(921*3, 16)
        self.conv1922 = DoubleConv(922*3, 16)
        self.conv1923 = DoubleConv(923*3, 16)
        self.conv1924 = DoubleConv(924*3, 16)
        self.conv1925 = DoubleConv(925*3, 16)
        self.conv1926 = DoubleConv(926*3, 16)
        self.conv1927 = DoubleConv(927*3, 16)
        self.conv1928 = DoubleConv(928*3, 16)
        self.conv1929 = DoubleConv(929*3, 16)
        self.conv1930 = DoubleConv(930*3, 16)
        self.conv1931 = DoubleConv(931*3, 16)
        self.conv1932 = DoubleConv(932*3, 16)
        self.conv1933 = DoubleConv(933*3, 16)
        self.conv1934 = DoubleConv(934*3, 16)
        self.conv1935 = DoubleConv(935*3, 16)
        self.conv1936 = DoubleConv(936*3, 16)
        self.conv1937 = DoubleConv(937*3, 16)
        self.conv1938 = DoubleConv(938*3, 16)
        self.conv1939 = DoubleConv(939*3, 16)
        self.conv1940 = DoubleConv(940*3, 16)
        self.conv1941 = DoubleConv(941*3, 16)
        self.conv1942 = DoubleConv(942*3, 16)
        self.conv1943 = DoubleConv(943*3, 16)
        self.conv1944 = DoubleConv(944*3, 16)
        self.conv1945 = DoubleConv(945*3, 16)
        self.conv1946 = DoubleConv(946*3, 16)
        self.conv1947 = DoubleConv(947*3, 16)
        self.conv1948 = DoubleConv(948*3, 16)
        self.conv1949 = DoubleConv(949*3, 16)
        self.conv1950 = DoubleConv(950*3, 16)
        self.conv1951 = DoubleConv(951*3, 16)
        self.conv1952 = DoubleConv(952*3, 16)
        self.conv1953 = DoubleConv(953*3, 16)
        self.conv1954 = DoubleConv(954*3, 16)
        self.conv1955 = DoubleConv(955*3, 16)
        self.conv1956 = DoubleConv(956*3, 16)
        self.conv1957 = DoubleConv(957*3, 16)
        self.conv1958 = DoubleConv(958*3, 16)
        self.conv1959 = DoubleConv(959*3, 16)
        self.conv1960 = DoubleConv(960*3, 16)
        self.conv1961 = DoubleConv(961*3, 16)
        self.conv1962 = DoubleConv(962*3, 16)
        self.conv1963 = DoubleConv(963*3, 16)
        self.conv1964 = DoubleConv(964*3, 16)
        self.conv1965 = DoubleConv(965*3, 16)
        self.conv1966 = DoubleConv(966*3, 16)
        self.conv1967 = DoubleConv(967*3, 16)
        self.conv1968 = DoubleConv(968*3, 16)
        self.conv1969 = DoubleConv(969*3, 16)
        self.conv1970 = DoubleConv(970*3, 16)
        self.conv1971 = DoubleConv(971*3, 16)
        self.conv1972 = DoubleConv(972*3, 16)
        self.conv1973 = DoubleConv(973*3, 16)
        self.conv1974 = DoubleConv(974*3, 16)
        self.conv1975 = DoubleConv(975*3, 16)
        self.conv1976 = DoubleConv(976*3, 16)
        self.conv1977 = DoubleConv(977*3, 16)
        self.conv1978 = DoubleConv(978*3, 16)
        self.conv1979 = DoubleConv(979*3, 16)
        self.conv1980 = DoubleConv(980*3, 16)
        self.conv1981 = DoubleConv(981*3, 16)
        self.conv1982 = DoubleConv(982*3, 16)
        self.conv1983 = DoubleConv(983*3, 16)
        self.conv1984 = DoubleConv(984*3, 16)
        self.conv1985 = DoubleConv(985*3, 16)
        self.conv1986 = DoubleConv(986*3, 16)
        self.conv1987 = DoubleConv(987*3, 16)
        self.conv1988 = DoubleConv(988*3, 16)
        self.conv1989 = DoubleConv(989*3, 16)
        self.conv1990 = DoubleConv(990*3, 16)
        self.conv1991 = DoubleConv(991*3, 16)
        self.conv1992 = DoubleConv(992*3, 16)
        self.conv1993 = DoubleConv(993*3, 16)
        self.conv1994 = DoubleConv(994*3, 16)
        self.conv1995 = DoubleConv(995*3, 16)
        self.conv1996 = DoubleConv(996*3, 16)
        self.conv1997 = DoubleConv(997*3, 16)
        self.conv1998 = DoubleConv(998*3, 16)
        self.conv1999 = DoubleConv(999*3, 16)
        self.conv11000 = DoubleConv(1000*3, 16)
        self.conv11001 = DoubleConv(1001*3, 16)
        self.conv11002 = DoubleConv(1002*3, 16)
        self.conv11003 = DoubleConv(1003*3, 16)
        self.conv11004 = DoubleConv(1004*3, 16)
        self.conv11005 = DoubleConv(1005*3, 16)
        self.conv11006 = DoubleConv(1006*3, 16)
        self.conv11007 = DoubleConv(1007*3, 16)
        self.conv11008 = DoubleConv(1008*3, 16)
        self.conv11009 = DoubleConv(1009*3, 16)
        self.conv11010 = DoubleConv(1010*3, 16)
        self.conv11011 = DoubleConv(1011*3, 16)
        self.conv11012 = DoubleConv(1012*3, 16)
        self.conv11013 = DoubleConv(1013*3, 16)
        self.conv11014 = DoubleConv(1014*3, 16)
        self.conv11015 = DoubleConv(1015*3, 16)
        self.conv11016 = DoubleConv(1016*3, 16)
        self.conv11017 = DoubleConv(1017*3, 16)
        self.conv11018 = DoubleConv(1018*3, 16)
        self.conv11019 = DoubleConv(1019*3, 16)
        self.conv11020 = DoubleConv(1020*3, 16)
        self.conv11021 = DoubleConv(1021*3, 16)
        self.conv11022 = DoubleConv(1022*3, 16)
        self.conv11023 = DoubleConv(1023*3, 16)
        self.conv11024 = DoubleConv(1024*3, 16)
        for bn_layer in [True, False]: 
            self.net1 = NLBlockND(in_channels=256+16*10, mode='concatenate', dimension=2, bn_layer=bn_layer) 
            self.net2 = NLBlockND(in_channels=128+16*10, mode='concatenate', dimension=2, bn_layer=bn_layer) 
            self.net3 = NLBlockND(in_channels=64+16*10, mode='concatenate', dimension=2, bn_layer=bn_layer)
       # self.sobel = SobelOperator(1e-4).cuda(1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(64+16*10, 128)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(128+16*10, 256)
        self.pool5 = nn.MaxPool2d(2)
        self.conv51 = DoubleConv(256+16*10,512)
        self.conv52 = DoubleConv(512, 512)
        self.conv53 = DoubleConv(512, 512)
 
        self.up63 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv63 = DoubleConv(1024, 512)
        self.up62 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv62 = DoubleConv(1024, 512)
        self.up61 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv61 = DoubleConv(512, 256)
        self.up6 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 64)
        self.up8 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv8 = DoubleConv(64, 32)
        self.up9 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv9 = DoubleConv(32, 16)
        self.conv10 = nn.Conv2d(16, out_ch, 1)


    def forward(self, x,B1,C1,D1,num0,num1,num2,B14,C14,D14,num04,num14,num24,B140,C140,D140,num040,num140,num240,B141,C141,D141,num041,num141,num241,B180,C180,D180,num080,num180,num280,B18,C18,D18,num08,num18,num28,B181,C181,D181,num081,num181,num281,B112,C112,D112,num012,num112,num212,B1120,C1120,D1120,num0120,num1120,num2120,B1121,C1121,D1121,num0121,num1121,num2121):
       # hr = GuidedFilter(r=3,eps=1e-8)(x, x)
       # hr_y = self.sobel(hr)
        #pr = self.sobel(x)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
       # print(B1.size())
#        for i in range(1,65):   
#            if num0==i:
#                num=i
#                p33[num]=self.conv126(B1)
        p33={}
        p331={}
        p332={}
        p34={}
        p341={}
        p342={}
        p340={}
        p3401={}
        p3402={}
        p3441={}
        p3411={}
        p3412={}

        p380={}
        p3801={}
        p3802={}
        p38={}
        p381={}
        p382={}
        p3881={}
        p3811={}
        p3812={}
        p312={}
        p3121={}
        p3122={}  
        p3120={}
        p31201={}
        p31202={}
        p312121={}
        p31211={}
        p31212={}                        
#        num=0
#        #for i in range(1,65):  
       # print(B1.size())
        if num0==1:
            p33=self.conv11(B1)
        elif num0==2:
            p33=self.conv12(B1)
        elif num0==3:
            p33=self.conv13(B1)
        elif num0==4:
            p33=self.conv14(B1)
        elif num0==5:
            p33=self.conv15(B1)
        elif num0==6:
            p33=self.conv16(B1)
        elif num0==7:
            p33=self.conv17(B1)
        elif num0==8:
            p33=self.conv18(B1)
        elif num0==9:
            p33=self.conv19(B1)
        elif num0==10:
            p33=self.conv110(B1)
        elif num0==11:
            p33=self.conv111(B1)
        elif num0==12:
            p33=self.conv112(B1)
        elif num0==13:
            p33=self.conv113(B1)
        elif num0==14:
            p33=self.conv114(B1)
        elif num0==15:
            p33=self.conv115(B1)
        elif num0==16:
            p33=self.conv116(B1)
        elif num0==17:
            p33=self.conv117(B1)
        elif num0==18:
            p33=self.conv118(B1)
        elif num0==19:
            p33=self.conv119(B1)
        elif num0==20:
            p33=self.conv120(B1)
        elif num0==21:
            p33=self.conv121(B1)
        elif num0==22:
            p33=self.conv122(B1)
        elif num0==23:
            p33=self.conv123(B1)
        elif num0==24:
            p33=self.conv124(B1)
        elif num0==25:
            p33=self.conv125(B1)
        elif num0==26:
            p33=self.conv126(B1)
        elif num0==27:
            p33=self.conv127(B1)
        elif num0==28:
            p33=self.conv128(B1)
        elif num0==29:
            p33=self.conv129(B1)
        elif num0==30:
            p33=self.conv130(B1)
        elif num0==31:
            p33=self.conv131(B1)
        elif num0==32:
            p33=self.conv132(B1)
        elif num0==33:
            p33=self.conv133(B1)
        elif num0==34:
            p33=self.conv134(B1)
        elif num0==35:
            p33=self.conv135(B1)
        elif num0==36:
            p33=self.conv136(B1)
        elif num0==37:
            p33=self.conv137(B1)
        elif num0==38:
            p33=self.conv138(B1)
        elif num0==39:
            p33=self.conv139(B1)
        elif num0==40:
            p33=self.conv140(B1)
        elif num0==41:
            p33=self.conv141(B1)
        elif num0==42:
            p33=self.conv142(B1)
        elif num0==43:
            p33=self.conv143(B1)
        elif num0==44:
            p33=self.conv144(B1)
        elif num0==45:
            p33=self.conv145(B1)
        elif num0==46:
            p33=self.conv146(B1)
        elif num0==47:
            p33=self.conv147(B1)
        elif num0==48:
            p33=self.conv148(B1)
        elif num0==49:
            p33=self.conv149(B1)
        elif num0==50:
            p33=self.conv150(B1)
        elif num0==51:
            p33=self.conv151(B1)
        elif num0==52:
            p33=self.conv152(B1)
        elif num0==53:
            p33=self.conv153(B1)
        elif num0==54:
            p33=self.conv154(B1)
        elif num0==55:
            p33=self.conv155(B1)
        elif num0==56:
            p33=self.conv156(B1)
        elif num0==57:
            p33=self.conv157(B1)
        elif num0==58:
            p33=self.conv158(B1)
        elif num0==59:
            p33=self.conv159(B1)
        elif num0==60:
            p33=self.conv160(B1)
        elif num0==61:
            p33=self.conv161(B1)
        elif num0==62:
            p33=self.conv162(B1)
        elif num0==63:
            p33=self.conv163(B1)
        elif num0==64:
            p33=self.conv164(B1)
        
        if  num1==1:
            p331=self.conv11(C1)
        elif  num1==2:
            p331=self.conv12(C1)
        elif  num1==3:
            p331=self.conv13(C1)
        elif  num1==4:
            p331=self.conv14(C1)
        elif  num1==5:
            p331=self.conv15(C1)
        elif  num1==6:
            p331=self.conv16(C1)
        elif  num1==7:
            p331=self.conv17(C1)
        elif  num1==8:
            p331=self.conv18(C1)
        elif  num1==9:
            p331=self.conv19(C1)
        elif  num1==10:
            p331=self.conv110(C1)
        elif  num1==11:
            p331=self.conv111(C1)
        elif  num1==12:
            p331=self.conv112(C1)
        elif  num1==13:
            p331=self.conv113(C1)
        elif  num1==14:
            p331=self.conv114(C1)
        elif  num1==15:
            p331=self.conv115(C1)
        elif  num1==16:
            p331=self.conv116(C1)
        elif  num1==17:
            p331=self.conv117(C1)
        elif  num1==18:
            p331=self.conv118(C1)
        elif  num1==19:
            p331=self.conv119(C1)
        elif  num1==20:
            p331=self.conv120(C1)
        elif  num1==21:
            p331=self.conv121(C1)
        elif  num1==22:
            p331=self.conv122(C1)
        elif  num1==23:
            p331=self.conv123(C1)
        elif  num1==24:
            p331=self.conv124(C1)
        elif  num1==25:
            p331=self.conv125(C1)
        elif  num1==26:
            p331=self.conv126(C1)
        elif  num1==27:
            p331=self.conv127(C1)
        elif  num1==28:
            p331=self.conv128(C1)
        elif  num1==29:
            p331=self.conv129(C1)
        elif  num1==30:
            p331=self.conv130(C1)
        elif  num1==31:
            p331=self.conv131(C1)
        elif  num1==32:
            p331=self.conv132(C1)
        elif  num1==33:
            p331=self.conv133(C1)
        elif  num1==34:
            p331=self.conv134(C1)
        elif  num1==35:
            p331=self.conv135(C1)
        elif  num1==36:
            p331=self.conv136(C1)
        elif  num1==37:
            p331=self.conv137(C1)
        elif  num1==38:
            p331=self.conv138(C1)
        elif  num1==39:
            p331=self.conv139(C1)
        elif  num1==40:
            p331=self.conv140(C1)
        elif  num1==41:
            p331=self.conv141(C1)
        elif  num1==42:
            p331=self.conv142(C1)
        elif  num1==43:
            p331=self.conv143(C1)
        elif  num1==44:
            p331=self.conv144(C1)
        elif  num1==45:
            p331=self.conv145(C1)
        elif  num1==46:
            p331=self.conv146(C1)
        elif  num1==47:
            p331=self.conv147(C1)
        elif  num1==48:
            p331=self.conv148(C1)
        elif  num1==49:
            p331=self.conv149(C1)
        elif  num1==50:
            p331=self.conv150(C1)
        elif  num1==51:
            p331=self.conv151(C1)
        elif  num1==52:
            p331=self.conv152(C1)
        elif  num1==53:
            p331=self.conv153(C1)
        elif  num1==54:
            p331=self.conv154(C1)
        elif  num1==55:
            p331=self.conv155(C1)
        elif  num1==56:
            p331=self.conv156(C1)
        elif  num1==57:
            p331=self.conv157(C1)
        elif  num1==58:
            p331=self.conv158(C1)
        elif  num1==59:
            p331=self.conv159(C1)
        elif  num1==60:
            p331=self.conv160(C1)
        elif  num1==61:
            p331=self.conv161(C1)
        elif  num1==62:
            p331=self.conv162(C1)
        elif  num1==63:
            p331=self.conv163(C1)
        elif  num1==64:
            p331=self.conv164(C1)
        elif  num1==65:
            p331=self.conv165(C1)
        elif  num1==66:
            p331=self.conv166(C1)
        elif  num1==67:
            p331=self.conv167(C1)
        elif  num1==68:
            p331=self.conv168(C1)
        elif  num1==69:
            p331=self.conv169(C1)
        elif  num1==70:
            p331=self.conv170(C1)
        elif  num1==71:
            p331=self.conv171(C1)
        elif  num1==72:
            p331=self.conv172(C1)
        elif  num1==73:
            p331=self.conv173(C1)
        elif  num1==74:
            p331=self.conv174(C1)
        elif  num1==75:
            p331=self.conv175(C1)
        elif  num1==76:
            p331=self.conv176(C1)
        elif  num1==77:
            p331=self.conv177(C1)
        elif  num1==78:
            p331=self.conv178(C1)
        elif  num1==79:
            p331=self.conv179(C1)
        elif  num1==80:
            p331=self.conv180(C1)
        elif  num1==81:
            p331=self.conv181(C1)
        elif  num1==82:
            p331=self.conv182(C1)
        elif  num1==83:
            p331=self.conv183(C1)
        elif  num1==84:
            p331=self.conv184(C1)
        elif  num1==85:
            p331=self.conv185(C1)
        elif  num1==86:
            p331=self.conv186(C1)
        elif  num1==87:
            p331=self.conv187(C1)
        elif  num1==88:
            p331=self.conv188(C1)
        elif  num1==89:
            p331=self.conv189(C1)    
        elif  num1==90:
            p331=self.conv190(C1)
        elif  num1==91:
            p331=self.conv191(C1)
        elif  num1==92:
            p331=self.conv192(C1)
        elif  num1==93:
            p331=self.conv193(C1)
        elif  num1==94:
            p331=self.conv194(C1)
        elif  num1==95:
            p331=self.conv195(C1)
        elif  num1==96:
            p331=self.conv196(C1)
        elif  num1==97:
            p331=self.conv197(C1)
        elif  num1==98:
            p331=self.conv198(C1)
        elif  num1==99:
            p331=self.conv199(C1) 
        elif  num1==100:
            p331=self.conv1100(C1)
        elif  num1==101:
            p331=self.conv1101(C1)
        elif  num1==102:
            p331=self.conv1102(C1)
        elif  num1==103:
            p331=self.conv1103(C1)
        elif  num1==104:
            p331=self.conv1104(C1)
        elif  num1==105:
            p331=self.conv1105(C1)
        elif  num1==106:
            p331=self.conv1106(C1)
        elif  num1==107:
            p331=self.conv1107(C1)
        elif  num1==108:
            p331=self.conv1108(C1)
        elif  num1==109:
            p331=self.conv1109(C1)
        elif  num1==110:
            p331=self.conv1110(C1)
        elif  num1==111:
            p331=self.conv1111(C1)
        elif  num1==112:
            p331=self.conv1112(C1)
        elif  num1==113:
            p331=self.conv1113(C1)
        elif  num1==114:
            p331=self.conv1114(C1)
        elif  num1==115:
            p331=self.conv1115(C1)
        elif  num1==116:
            p331=self.conv1116(C1)
        elif  num1==117:
            p331=self.conv1117(C1)
        elif  num1==118:
            p331=self.conv1118(C1)
        elif  num1==119:
            p331=self.conv1119(C1) 
        elif  num1==120:
            p331=self.conv1120(C1)
        elif  num1==121:
            p331=self.conv1121(C1)
        elif  num1==122:
            p331=self.conv1122(C1)
        elif  num1==123:
            p331=self.conv1123(C1)
        elif  num1==124:
            p331=self.conv1124(C1)
        elif  num1==125:
            p331=self.conv1125(C1)
        elif  num1==126:
            p331=self.conv1126(C1)
        elif  num1==127:
            p331=self.conv1127(C1)
        elif  num1==128:
            p331=self.conv1128(C1)
        elif  num1==129:
            p331=self.conv1129(C1) 
        elif  num1==130:
            p331=self.conv1130(C1)
        elif  num1==131:
            p331=self.conv1131(C1)
        elif  num1==132:
            p331=self.conv1132(C1)
        elif  num1==133:
            p331=self.conv1133(C1)
        elif  num1==134:
            p331=self.conv1134(C1)
        elif  num1==135:
            p331=self.conv1135(C1)
        elif  num1==136:
            p331=self.conv1136(C1)
        elif  num1==137:
            p331=self.conv1137(C1)
        elif  num1==138:
            p331=self.conv1138(C1)
        elif  num1==139:
            p331=self.conv1139(C1)
        elif  num1==140:
            p331=self.conv1140(C1)
        elif  num1==141:
            p331=self.conv1141(C1)
        elif  num1==142:
            p331=self.conv1142(C1)
        elif  num1==143:
            p331=self.conv1143(C1)
        elif  num1==144:
            p331=self.conv1144(C1)
        elif  num1==145:
            p331=self.conv1145(C1)
        elif  num1==146:
            p331=self.conv1146(C1)
        elif  num1==147:
            p331=self.conv1147(C1)
        elif  num1==148:
            p331=self.conv1148(C1)
        elif  num1==149:
            p331=self.conv1149(C1) 
        elif  num1==150:
            p331=self.conv1150(C1)
        elif  num1==151:
            p331=self.conv1151(C1)
        elif  num1==152:
            p331=self.conv1152(C1)
        elif  num1==153:
            p331=self.conv1153(C1)
        elif  num1==154:
            p331=self.conv1154(C1)
        elif  num1==155:
            p331=self.conv1155(C1)
        elif  num1==156:
            p331=self.conv1156(C1)
        elif  num1==157:
            p331=self.conv1157(C1)
        elif  num1==158:
            p331=self.conv1158(C1)
        elif  num1==159:
            p331=self.conv1159(C1) 
        elif  num1==160:
            p331=self.conv1160(C1)
        elif  num1==161:
            p331=self.conv1161(C1)
        elif  num1==162:
            p331=self.conv1162(C1)
        elif  num1==163:
            p331=self.conv1163(C1)
        elif  num1==164:
            p331=self.conv1164(C1)
        elif  num1==165:
            p331=self.conv1165(C1)
        elif  num1==166:
            p331=self.conv1166(C1)
        elif  num1==167:
            p331=self.conv1167(C1)
        elif  num1==168:
            p331=self.conv1168(C1)
        elif  num1==169:
            p331=self.conv1169(C1) 
        elif  num1==170:
            p331=self.conv1170(C1)
        elif  num1==171:
            p331=self.conv1171(C1)
        elif  num1==172:
            p331=self.conv1172(C1)
        elif  num1==173:
            p331=self.conv1173(C1)
        elif  num1==174:
            p331=self.conv1174(C1)
        elif  num1==175:
            p331=self.conv1175(C1)
        elif  num1==176:
            p331=self.conv1176(C1)
        elif  num1==177:
            p331=self.conv1177(C1)
        elif  num1==178:
            p331=self.conv1178(C1)
        elif  num1==179:
            p331=self.conv1179(C1)                                                                                              
        elif  num1==180:
            p331=self.conv1180(C1)
        elif  num1==181:
            p331=self.conv1181(C1)
        elif  num1==182:
            p331=self.conv1182(C1)
        elif  num1==183:
            p331=self.conv1183(C1)
        elif  num1==184:
            p331=self.conv1184(C1)
        elif  num1==185:
            p331=self.conv1185(C1)
        elif  num1==186:
            p331=self.conv1186(C1)
        elif  num1==187:
            p331=self.conv1187(C1)
        elif  num1==188:
            p331=self.conv1188(C1)
        elif  num1==189:
            p331=self.conv1189(C1) 
        elif  num1==190:
            p331=self.conv1190(C1)
        elif  num1==191:
            p331=self.conv1191(C1)
        elif  num1==192:
            p331=self.conv1192(C1)
        elif  num1==193:
            p331=self.conv1193(C1)
        elif  num1==194:
            p331=self.conv1194(C1)
        elif  num1==195:
            p331=self.conv1195(C1)
        elif  num1==196:
            p331=self.conv1196(C1)
        elif  num1==197:
            p331=self.conv1197(C1)
        elif  num1==198:
            p331=self.conv1198(C1)
        elif  num1==199:
            p331=self.conv1199(C1)
        elif  num1==200:
            p331=self.conv1200(C1)
        elif  num1==201:
            p331=self.conv1201(C1)
        elif  num1==202:
            p331=self.conv1202(C1)
        elif  num1==203:
            p331=self.conv1203(C1)
        elif  num1==204:
            p331=self.conv1204(C1)
        elif  num1==205:
            p331=self.conv1205(C1)
        elif  num1==206:
            p331=self.conv1206(C1)
        elif  num1==207:
            p331=self.conv1207(C1)
        elif  num1==208:
            p331=self.conv1208(C1)
        elif  num1==209:
            p331=self.conv1209(C1)
        elif  num1==210:
            p331=self.conv1210(C1)
        elif  num1==211:
            p331=self.conv1211(C1)
        elif  num1==212:
            p331=self.conv1212(C1)
        elif  num1==213:
            p331=self.conv1213(C1)
        elif  num1==214:
            p331=self.conv1214(C1)
        elif  num1==215:
            p331=self.conv1215(C1)
        elif  num1==216:
            p331=self.conv1216(C1)
        elif  num1==217:
            p331=self.conv1217(C1)
        elif  num1==218:
            p331=self.conv1218(C1)
        elif  num1==219:
            p331=self.conv1219(C1)
        elif  num1==220:
            p331=self.conv1220(C1)
        elif  num1==221:
            p331=self.conv1221(C1)
        elif  num1==222:
            p331=self.conv1222(C1)
        elif  num1==223:
            p331=self.conv1223(C1)
        elif  num1==224:
            p331=self.conv1224(C1)
        elif  num1==225:
            p331=self.conv1225(C1)
        elif  num1==226:
            p331=self.conv1226(C1)
        elif  num1==227:
            p331=self.conv1227(C1)
        elif  num1==228:
            p331=self.conv1228(C1)
        elif  num1==229:
            p331=self.conv1229(C1)
        elif  num1==230:
            p331=self.conv1230(C1)
        elif  num1==231:
            p331=self.conv1231(C1)
        elif  num1==232:
            p331=self.conv1232(C1)
        elif  num1==233:
            p331=self.conv1233(C1)
        elif  num1==234:
            p331=self.conv1234(C1)
        elif  num1==235:
            p331=self.conv1235(C1)
        elif  num1==236:
            p331=self.conv1236(C1)
        elif  num1==237:
            p331=self.conv1237(C1)
        elif  num1==238:
            p331=self.conv1238(C1)
        elif  num1==239:
            p331=self.conv1239(C1) 
        elif  num1==240:
            p331=self.conv1240(C1)
        elif  num1==241:
            p331=self.conv1241(C1)
        elif  num1==242:
            p331=self.conv1242(C1)
        elif  num1==243:
            p331=self.conv1243(C1)
        elif  num1==244:
            p331=self.conv1244(C1)
        elif  num1==245:
            p331=self.conv1245(C1)
        elif  num1==246:
            p331=self.conv1246(C1)
        elif  num1==247:
            p331=self.conv1247(C1)
        elif  num1==248:
            p331=self.conv1248(C1)
        elif  num1==249:
            p331=self.conv1249(C1)
        elif  num1==250:
            p331=self.conv1250(C1)
        elif  num1==251:
            p331=self.conv1251(C1)
        elif  num1==252:
            p331=self.conv1252(C1)
        elif  num1==253:
            p331=self.conv1253(C1)
        elif  num1==254:
            p331=self.conv1254(C1)
        elif  num1==255:
            p331=self.conv1255(C1)
        elif  num1==256:
            p331=self.conv1256(C1)
            
        if  num2==1:
            p332=self.conv11(D1)
        elif  num2==2:
            p332=self.conv12(D1)
        elif  num2==3:
            p332=self.conv13(D1)
        elif  num2==4:
            p332=self.conv14(D1)
        elif  num2==5:
            p332=self.conv15(D1)
        elif  num2==6:
            p332=self.conv16(D1)
        elif  num2==7:
            p332=self.conv17(D1)
        elif  num2==8:
            p332=self.conv18(D1)
        elif  num2==9:
            p332=self.conv19(D1)
        elif  num2==10:
            p332=self.conv110(D1)
        elif  num2==11:
            p332=self.conv111(D1)
        elif  num2==12:
            p332=self.conv112(D1)
        elif  num2==13:
            p332=self.conv113(D1)
        elif  num2==14:
            p332=self.conv114(D1)
        elif  num2==15:
            p332=self.conv115(D1)
        elif  num2==16:
            p332=self.conv116(D1)
        elif  num2==17:
            p332=self.conv117(D1)
        elif  num2==18:
            p332=self.conv118(D1)
        elif  num2==19:
            p332=self.conv119(D1)
        elif  num2==20:
            p332=self.conv120(D1)
        elif  num2==21:
            p332=self.conv121(D1)
        elif  num2==22:
            p332=self.conv122(D1)
        elif  num2==23:
            p332=self.conv123(D1)
        elif  num2==24:
            p332=self.conv124(D1)
        elif  num2==25:
            p332=self.conv125(D1)
        elif  num2==26:
            p332=self.conv126(D1)
        elif  num2==27:
            p332=self.conv127(D1)
        elif  num2==28:
            p332=self.conv128(D1)
        elif  num2==29:
            p332=self.conv129(D1)
        elif  num2==30:
            p332=self.conv130(D1)
        elif  num2==31:
            p332=self.conv131(D1)
        elif  num2==32:
            p332=self.conv132(D1)
        elif  num2==33:
            p332=self.conv133(D1)
        elif  num2==34:
            p332=self.conv134(D1)
        elif  num2==35:
            p332=self.conv135(D1)
        elif  num2==36:
            p332=self.conv136(D1)
        elif  num2==37:
            p332=self.conv137(D1)
        elif  num2==38:
            p332=self.conv138(D1)
        elif  num2==39:
            p332=self.conv139(D1)
        elif  num2==40:
            p332=self.conv140(D1)
        elif  num2==41:
            p332=self.conv141(D1)
        elif  num2==42:
            p332=self.conv142(D1)
        elif  num2==43:
            p332=self.conv143(D1)
        elif  num2==44:
            p332=self.conv144(D1)
        elif  num2==45:
            p332=self.conv145(D1)
        elif  num2==46:
            p332=self.conv146(D1)
        elif  num2==47:
            p332=self.conv147(D1)
        elif  num2==48:
            p332=self.conv148(D1)
        elif  num2==49:
            p332=self.conv149(D1)
        elif  num2==50:
            p332=self.conv150(D1)
        elif  num2==51:
            p332=self.conv151(D1)
        elif  num2==52:
            p332=self.conv152(D1)
        elif  num2==53:
            p332=self.conv153(D1)
        elif  num2==54:
            p332=self.conv154(D1)
        elif  num2==55:
            p332=self.conv155(D1)
        elif  num2==56:
            p332=self.conv156(D1)
        elif  num2==57:
            p332=self.conv157(D1)
        elif  num2==58:
            p332=self.conv158(D1)
        elif  num2==59:
            p332=self.conv159(D1)
        elif  num2==60:
            p332=self.conv160(D1)
        elif  num2==61:
            p332=self.conv161(D1)
        elif  num2==62:
            p332=self.conv162(D1)
        elif  num2==63:
            p332=self.conv163(D1)
        elif  num2==64:
            p332=self.conv164(D1)
        elif  num2==65:
            p332=self.conv165(D1)
        elif  num2==66:
            p332=self.conv166(D1)
        elif  num2==67:
            p332=self.conv167(D1)
        elif  num2==68:
            p332=self.conv168(D1)
        elif  num2==69:
            p332=self.conv169(D1)
        elif  num2==70:
            p332=self.conv170(D1)
        elif  num2==71:
            p332=self.conv171(D1)
        elif  num2==72:
            p332=self.conv172(D1)
        elif  num2==73:
            p332=self.conv173(D1)
        elif  num2==74:
            p332=self.conv174(D1)
        elif  num2==75:
            p332=self.conv175(D1)
        elif  num2==76:
            p332=self.conv176(D1)
        elif  num2==77:
            p332=self.conv177(D1)
        elif  num2==78:
            p332=self.conv178(D1)
        elif  num2==79:
            p332=self.conv179(D1)
        elif  num2==80:
            p332=self.conv180(D1)
        elif  num2==81:
            p332=self.conv181(D1)
        elif  num2==82:
            p332=self.conv182(D1)
        elif  num2==83:
            p332=self.conv183(D1)
        elif  num2==84:
            p332=self.conv184(D1)
        elif  num2==85:
            p332=self.conv185(D1)
        elif  num2==86:
            p332=self.conv186(D1)
        elif  num2==87:
            p332=self.conv187(D1)
        elif  num2==88:
            p332=self.conv188(D1)
        elif  num2==89:
            p332=self.conv189(D1)    
        elif  num2==90:
            p332=self.conv190(D1)
        elif  num2==91:
            p332=self.conv191(D1)
        elif  num2==92:
            p332=self.conv192(D1)
        elif  num2==93:
            p332=self.conv193(D1)
        elif  num2==94:
            p332=self.conv194(D1)
        elif  num2==95:
            p332=self.conv195(D1)
        elif  num2==96:
            p332=self.conv196(D1)
        elif  num2==97:
            p332=self.conv197(D1)
        elif  num2==98:
            p332=self.conv198(D1)
        elif  num2==99:
            p332=self.conv199(D1) 
        elif  num2==100:
            p332=self.conv1100(D1)
        elif  num2==101:
            p332=self.conv1101(D1)
        elif  num2==102:
            p332=self.conv1102(D1)
        elif  num2==103:
            p332=self.conv1103(D1)
        elif  num2==104:
            p332=self.conv1104(D1)
        elif  num2==105:
            p332=self.conv1105(D1)
        elif  num2==106:
            p332=self.conv1106(D1)
        elif  num2==107:
            p332=self.conv1107(D1)
        elif  num2==108:
            p332=self.conv1108(D1)
        elif  num2==109:
            p332=self.conv1109(D1)
        elif  num2==110:
            p332=self.conv1110(D1)
        elif  num2==111:
            p332=self.conv1111(D1)
        elif  num2==112:
            p332=self.conv1112(D1)
        elif  num2==113:
            p332=self.conv1113(D1)
        elif  num2==114:
            p332=self.conv1114(D1)
        elif  num2==115:
            p332=self.conv1115(D1)
        elif  num2==116:
            p332=self.conv1116(D1)
        elif  num2==117:
            p332=self.conv1117(D1)
        elif  num2==118:
            p332=self.conv1118(D1)
        elif  num2==119:
            p332=self.conv1119(D1) 
        elif  num2==120:
            p332=self.conv1120(D1)
        elif  num2==121:
            p332=self.conv1121(D1)
        elif  num2==122:
            p332=self.conv1122(D1)
        elif  num2==123:
            p332=self.conv1123(D1)
        elif  num2==124:
            p332=self.conv1124(D1)
        elif  num2==125:
            p332=self.conv1125(D1)
        elif  num2==126:
            p332=self.conv1126(D1)
        elif  num2==127:
            p332=self.conv1127(D1)
        elif  num2==128:
            p332=self.conv1128(D1)
        elif  num2==129:
            p332=self.conv1129(D1) 
        elif  num2==130:
            p332=self.conv1130(D1)
        elif  num2==131:
            p332=self.conv1131(D1)
        elif  num2==132:
            p332=self.conv1132(D1)
        elif  num2==133:
            p332=self.conv1133(D1)
        elif  num2==134:
            p332=self.conv1134(D1)
        elif  num2==135:
            p332=self.conv1135(D1)
        elif  num2==136:
            p332=self.conv1136(D1)
        elif  num2==137:
            p332=self.conv1137(D1)
        elif  num2==138:
            p332=self.conv1138(D1)
        elif  num2==139:
            p332=self.conv1139(D1)
        elif  num2==140:
            p332=self.conv1140(D1)
        elif  num2==141:
            p332=self.conv1141(D1)
        elif  num2==142:
            p332=self.conv1142(D1)
        elif  num2==143:
            p332=self.conv1143(D1)
        elif  num2==144:
            p332=self.conv1144(D1)
        elif  num2==145:
            p332=self.conv1145(D1)
        elif  num2==146:
            p332=self.conv1146(D1)
        elif  num2==147:
            p332=self.conv1147(D1)
        elif  num2==148:
            p332=self.conv1148(D1)
        elif  num2==149:
            p332=self.conv1149(D1) 
        elif  num2==150:
            p332=self.conv1150(D1)
        elif  num2==151:
            p332=self.conv1151(D1)
        elif  num2==152:
            p332=self.conv1152(D1)
        elif  num2==153:
            p332=self.conv1153(D1)
        elif  num2==154:
            p332=self.conv1154(D1)
        elif  num2==155:
            p332=self.conv1155(D1)
        elif  num2==156:
            p332=self.conv1156(D1)
        elif  num2==157:
            p332=self.conv1157(D1)
        elif  num2==158:
            p332=self.conv1158(D1)
        elif  num2==159:
            p332=self.conv1159(D1) 
        elif  num2==160:
            p332=self.conv1160(D1)
        elif  num2==161:
            p332=self.conv1161(D1)
        elif  num2==162:
            p332=self.conv1162(D1)
        elif  num2==163:
            p332=self.conv1163(D1)
        elif  num2==164:
            p332=self.conv1164(D1)
        elif  num2==165:
            p332=self.conv1165(D1)
        elif  num2==166:
            p332=self.conv1166(D1)
        elif  num2==167:
            p332=self.conv1167(D1)
        elif  num2==168:
            p332=self.conv1168(D1)
        elif  num2==169:
            p332=self.conv1169(D1) 
        elif  num2==170:
            p332=self.conv1170(D1)
        elif  num2==171:
            p332=self.conv1171(D1)
        elif  num2==172:
            p332=self.conv1172(D1)
        elif  num2==173:
            p332=self.conv1173(D1)
        elif  num2==174:
            p332=self.conv1174(D1)
        elif  num2==175:
            p332=self.conv1175(D1)
        elif  num2==176:
            p332=self.conv1176(D1)
        elif  num2==177:
            p332=self.conv1177(D1)
        elif  num2==178:
            p332=self.conv1178(D1)
        elif  num2==179:
            p332=self.conv1179(D1)                                                                                              
        elif  num2==180:
            p332=self.conv1180(D1)
        elif  num2==181:
            p332=self.conv1181(D1)
        elif  num2==182:
            p332=self.conv1182(D1)
        elif  num2==183:
            p332=self.conv1183(D1)
        elif  num2==184:
            p332=self.conv1184(D1)
        elif  num2==185:
            p332=self.conv1185(D1)
        elif  num2==186:
            p332=self.conv1186(D1)
        elif  num2==187:
            p332=self.conv1187(D1)
        elif  num2==188:
            p332=self.conv1188(D1)
        elif  num2==189:
            p332=self.conv1189(D1) 
        elif  num2==190:
            p332=self.conv1190(D1)
        elif  num2==191:
            p332=self.conv1191(D1)
        elif  num2==192:
            p332=self.conv1192(D1)
        elif  num2==193:
            p332=self.conv1193(D1)
        elif  num2==194:
            p332=self.conv1194(D1)
        elif  num2==195:
            p332=self.conv1195(D1)
        elif  num2==196:
            p332=self.conv1196(D1)
        elif  num2==197:
            p332=self.conv1197(D1)
        elif  num2==198:
            p332=self.conv1198(D1)
        elif  num2==199:
            p332=self.conv1199(D1)
        elif  num2==200:
            p332=self.conv1200(D1)
        elif  num2==201:
            p332=self.conv1201(D1)
        elif  num2==202:
            p332=self.conv1202(D1)
        elif  num2==203:
            p332=self.conv1203(D1)
        elif  num2==204:
            p332=self.conv1204(D1)
        elif  num2==205:
            p332=self.conv1205(D1)
        elif  num2==206:
            p332=self.conv1206(D1)
        elif  num2==207:
            p332=self.conv1207(D1)
        elif  num2==208:
            p332=self.conv1208(D1)
        elif  num2==209:
            p332=self.conv1209(D1)
        elif  num2==210:
            p332=self.conv1210(D1)
        elif  num2==211:
            p332=self.conv1211(D1)
        elif  num2==212:
            p332=self.conv1212(D1)
        elif  num2==213:
            p332=self.conv1213(D1)
        elif  num2==214:
            p332=self.conv1214(D1)
        elif  num2==215:
            p332=self.conv1215(D1)
        elif  num2==216:
            p332=self.conv1216(D1)
        elif  num2==217:
            p332=self.conv1217(D1)
        elif  num2==218:
            p332=self.conv1218(D1)
        elif  num2==219:
            p332=self.conv1219(D1)
        elif  num2==220:
            p332=self.conv1220(D1)
        elif  num2==221:
            p332=self.conv1221(D1)
        elif  num2==222:
            p332=self.conv1222(D1)
        elif  num2==223:
            p332=self.conv1223(D1)
        elif  num2==224:
            p332=self.conv1224(D1)
        elif  num2==225:
            p332=self.conv1225(D1)
        elif  num2==226:
            p332=self.conv1226(D1)
        elif  num2==227:
            p332=self.conv1227(D1)
        elif  num2==228:
            p332=self.conv1228(D1)
        elif  num2==229:
            p332=self.conv1229(D1)
        elif  num2==230:
            p332=self.conv1230(D1)
        elif  num2==231:
            p332=self.conv1231(D1)
        elif  num2==232:
            p332=self.conv1232(D1)
        elif  num2==233:
            p332=self.conv1233(D1)
        elif  num2==234:
            p332=self.conv1234(D1)
        elif  num2==235:
            p332=self.conv1235(D1)
        elif  num2==236:
            p332=self.conv1236(D1)
        elif  num2==237:
            p332=self.conv1237(D1)
        elif  num2==238:
            p332=self.conv1238(D1)
        elif  num2==239:
            p332=self.conv1239(D1) 
        elif  num2==240:
            p332=self.conv1240(D1)
        elif  num2==241:
            p332=self.conv1241(D1)
        elif  num2==242:
            p332=self.conv1242(D1)
        elif  num2==243:
            p332=self.conv1243(D1)
        elif  num2==244:
            p332=self.conv1244(D1)
        elif  num2==245:
            p332=self.conv1245(D1)
        elif  num2==246:
            p332=self.conv1246(D1)
        elif  num2==247:
            p332=self.conv1247(D1)
        elif  num2==248:
            p332=self.conv1248(D1)
        elif  num2==249:
            p332=self.conv1249(D1)
        elif  num2==250:
            p332=self.conv1250(D1)
        elif  num2==251:
            p332=self.conv1251(D1)
        elif  num2==252:
            p332=self.conv1252(D1)
        elif  num2==253:
            p332=self.conv1253(D1)
        elif  num2==254:
            p332=self.conv1254(D1)
        elif  num2==255:
            p332=self.conv1255(D1)
        elif  num2==256:
            p332=self.conv1256(D1)
        elif  num2==257:
            p332=self.conv1257(D1)
        elif  num2==258:
            p332=self.conv1258(D1)
        elif  num2==259:
            p332=self.conv1259(D1)
        elif  num2==260:
            p332=self.conv1260(D1)
        elif  num2==261:
            p332=self.conv1261(D1)
        elif  num2==262:
            p332=self.conv1262(D1)
        elif  num2==263:
            p332=self.conv1263(D1)
        elif  num2==264:
            p332=self.conv1264(D1)
        elif  num2==265:
            p332=self.conv1265(D1)
        elif  num2==266:
            p332=self.conv1266(D1)
        elif  num2==267:
            p332=self.conv1267(D1)
        elif  num2==268:
            p332=self.conv1268(D1)
        elif  num2==269:
            p332=self.conv1269(D1) 
        elif  num2==270:
            p332=self.conv1270(D1)
        elif  num2==271:
            p332=self.conv1271(D1)
        elif  num2==272:
            p332=self.conv1272(D1)
        elif  num2==273:
            p332=self.conv1273(D1)
        elif  num2==274:
            p332=self.conv1274(D1)
        elif  num2==275:
            p332=self.conv1275(D1)
        elif  num2==276:
            p332=self.conv1276(D1)
        elif  num2==277:
            p332=self.conv1277(D1)
        elif  num2==278:
            p332=self.conv1278(D1)
        elif  num2==279:
            p332=self.conv1279(D1)
        elif  num2==280:
            p332=self.conv1280(D1)
        elif  num2==281:
            p332=self.conv1281(D1)
        elif  num2==282:
            p332=self.conv1282(D1)
        elif  num2==283:
            p332=self.conv1283(D1)
        elif  num2==284:
            p332=self.conv1284(D1)
        elif  num2==285:
            p332=self.conv1285(D1)
        elif  num2==286:
            p332=self.conv1286(D1)
        elif  num2==287:
            p332=self.conv1287(D1)
        elif  num2==288:
            p332=self.conv1288(D1)
        elif  num2==289:
            p332=self.conv1289(D1) 
        elif  num2==290:
            p332=self.conv1290(D1)
        elif  num2==291:
            p332=self.conv1291(D1)
        elif  num2==292:
            p332=self.conv1292(D1)
        elif  num2==293:
            p332=self.conv1293(D1)
        elif  num2==294:
            p332=self.conv1294(D1)
        elif  num2==295:
            p332=self.conv1295(D1)
        elif  num2==296:
            p332=self.conv1296(D1)
        elif  num2==297:
            p332=self.conv1297(D1)
        elif  num2==298:
            p332=self.conv1298(D1)
        elif  num2==299:
            p332=self.conv1299(D1)
        elif  num2==300:
            p332=self.conv1300(D1)
        elif  num2==301:
            p332=self.conv1301(D1)
        elif  num2==302:
            p332=self.conv1302(D1)
        elif  num2==303:
            p332=self.conv1303(D1)
        elif  num2==304:
            p332=self.conv1304(D1)
        elif  num2==305:
            p332=self.conv1305(D1)
        elif  num2==306:
            p332=self.conv1306(D1)
        elif  num2==307:
            p332=self.conv1307(D1)
        elif  num2==308:
            p332=self.conv1308(D1)
        elif  num2==309:
            p332=self.conv1309(D1) 
        elif  num2==310:
            p332=self.conv1310(D1)
        elif  num2==311:
            p332=self.conv1311(D1)
        elif  num2==312:
            p332=self.conv1312(D1)
        elif  num2==313:
            p332=self.conv1313(D1)
        elif  num2==314:
            p332=self.conv1314(D1)
        elif  num2==315:
            p332=self.conv1315(D1)
        elif  num2==316:
            p332=self.conv1316(D1)
        elif  num2==317:
            p332=self.conv1317(D1)
        elif  num2==318:
            p332=self.conv1318(D1)
        elif  num2==319:
            p332=self.conv1319(D1)
        elif  num2==320:
            p332=self.conv1320(D1)
        elif  num2==321:
            p332=self.conv1321(D1)
        elif  num2==322:
            p332=self.conv1322(D1)
        elif  num2==323:
            p332=self.conv1323(D1)
        elif  num2==324:
            p332=self.conv1324(D1)
        elif  num2==325:
            p332=self.conv1325(D1)
        elif  num2==326:
            p332=self.conv1326(D1)
        elif  num2==327:
            p332=self.conv1327(D1)
        elif  num2==328:
            p332=self.conv1328(D1)
        elif  num2==329:
            p332=self.conv1329(D1)
        elif  num2==330:
            p332=self.conv1330(D1)
        elif  num2==331:
            p332=self.conv1331(D1)
        elif  num2==332:
            p332=self.conv1332(D1)
        elif  num2==333:
            p332=self.conv1333(D1)
        elif  num2==334:
            p332=self.conv1334(D1)
        elif  num2==335:
            p332=self.conv1335(D1)
        elif  num2==336:
            p332=self.conv1336(D1)
        elif  num2==337:
            p332=self.conv1337(D1)
        elif  num2==338:
            p332=self.conv1338(D1)
        elif  num2==339:
            p332=self.conv1339(D1)
        elif  num2==340:
            p332=self.conv1340(D1)
        elif  num2==341:
            p332=self.conv1341(D1)
        elif  num2==342:
            p332=self.conv1342(D1)
        elif  num2==343:
            p332=self.conv1343(D1)
        elif  num2==344:
            p332=self.conv1344(D1)
        elif  num2==345:
            p332=self.conv1345(D1)
        elif  num2==346:
            p332=self.conv1346(D1)
        elif  num2==347:
            p332=self.conv1347(D1)
        elif  num2==348:
            p332=self.conv1348(D1)
        elif  num2==349:
            p332=self.conv1349(D1)
        elif  num2==350:
            p332=self.conv1350(D1)
        elif  num2==351:
            p332=self.conv1351(D1)
        elif  num2==352:
            p332=self.conv1352(D1)
        elif  num2==353:
            p332=self.conv1335(D1)
        elif  num2==354:
            p332=self.conv1354(D1)
        elif  num2==355:
            p332=self.conv1355(D1)
        elif  num2==356:
            p332=self.conv1356(D1)
        elif  num2==357:
            p332=self.conv1357(D1)
        elif  num2==358:
            p332=self.conv1358(D1)
        elif  num2==359:
            p332=self.conv1359(D1) 
        elif  num2==360:
            p332=self.conv1360(D1)
        elif  num2==361:
            p332=self.conv1361(D1)
        elif  num2==362:
            p332=self.conv1362(D1)
        elif  num2==363:
            p332=self.conv1363(D1)
        elif  num2==364:
            p332=self.conv1364(D1)
        elif  num2==365:
            p332=self.conv1365(D1)
        elif  num2==366:
            p332=self.conv1366(D1)
        elif  num2==367:
            p332=self.conv1367(D1)
        elif  num2==368:
            p332=self.conv1368(D1)
        elif  num2==369:
            p332=self.conv1369(D1) 
        elif  num2==370:
            p332=self.conv1370(D1)
        elif  num2==371:
            p332=self.conv1371(D1)
        elif  num2==372:
            p332=self.conv1372(D1)
        elif  num2==373:
            p332=self.conv1373(D1)
        elif  num2==374:
            p332=self.conv1374(D1)
        elif  num2==375:
            p332=self.conv1375(D1)
        elif  num2==376:
            p332=self.conv1376(D1)
        elif  num2==377:
            p332=self.conv1377(D1)
        elif  num2==378:
            p332=self.conv1378(D1)
        elif  num2==379:
            p332=self.conv1379(D1) 
        elif  num2==380:
            p332=self.conv1380(D1)
        elif  num2==381:
            p332=self.conv1381(D1)
        elif  num2==382:
            p332=self.conv1382(D1)
        elif  num2==383:
            p332=self.conv1383(D1)
        elif  num2==384:
            p332=self.conv1384(D1)
        elif  num2==385:
            p332=self.conv1385(D1)
        elif  num2==386:
            p332=self.conv1386(D1)
        elif  num2==387:
            p332=self.conv1387(D1)
        elif  num2==388:
            p332=self.conv1388(D1)
        elif  num2==389:
            p332=self.conv1389(D1) 
        elif  num2==390:
            p332=self.conv1390(D1)
        elif  num2==391:
            p332=self.conv1391(D1)
        elif  num2==392:
            p332=self.conv1392(D1)
        elif  num2==393:
            p332=self.conv1393(D1)
        elif  num2==394:
            p332=self.conv1394(D1)
        elif  num2==395:
            p332=self.conv1395(D1)
        elif  num2==396:
            p332=self.conv1396(D1)
        elif  num2==397:
            p332=self.conv1397(D1)
        elif  num2==398:
            p332=self.conv1398(D1)
        elif  num2==399:
            p332=self.conv1399(D1)
        elif  num2==400:
            p332=self.conv1400(D1)
        elif  num2==401:
            p332=self.conv1401(D1)
        elif  num2==402:
            p332=self.conv1402(D1)
        elif  num2==403:
            p332=self.conv1403(D1)
        elif  num2==404:
            p332=self.conv1404(D1)
        elif  num2==405:
            p332=self.conv1405(D1)
        elif  num2==406:
            p332=self.conv1406(D1)
        elif  num2==407:
            p332=self.conv1407(D1)
        elif  num2==408:
            p332=self.conv1408(D1)
        elif  num2==409:
            p332=self.conv1409(D1)
        elif  num2==410:
            p332=self.conv1410(D1)
        elif  num2==411:
            p332=self.conv1411(D1)
        elif  num2==412:
            p332=self.conv1412(D1)
        elif  num2==413:
            p332=self.conv1413(D1)
        elif  num2==414:
            p332=self.conv1414(D1)
        elif  num2==415:
            p332=self.conv145(D1)
        elif  num2==416:
            p332=self.conv1416(D1)
        elif  num2==417:
            p332=self.conv1417(D1)
        elif  num2==418:
            p332=self.conv1418(D1)
        elif  num2==419:
            p332=self.conv1419(D1) 
        elif  num2==420:
            p332=self.conv1420(D1)
        elif  num2==421:
            p332=self.conv1421(D1)
        elif  num2==422:
            p332=self.conv1422(D1)
        elif  num2==423:
            p332=self.conv1423(D1)
        elif  num2==424:
            p332=self.conv1424(D1)
        elif  num2==425:
            p332=self.conv1425(D1)
        elif  num2==426:
            p332=self.conv1426(D1)
        elif  num2==427:
            p332=self.conv1427(D1)
        elif  num2==428:
            p332=self.conv1428(D1)
        elif  num2==429:
            p332=self.conv1429(D1) 
        elif  num2==430:
            p332=self.conv1430(D1)
        elif  num2==431:
            p332=self.conv1431(D1)
        elif  num2==432:
            p332=self.conv1432(D1)
        elif  num2==433:
            p332=self.conv1433(D1)
        elif  num2==434:
            p332=self.conv1434(D1)
        elif  num2==435:
            p332=self.conv1435(D1)
        elif  num2==436:
            p332=self.conv1436(D1)
        elif  num2==437:
            p332=self.conv1437(D1)
        elif  num2==438:
            p332=self.conv1438(D1)
        elif  num2==439:
            p332=self.conv1439(D1)
        elif  num2==440:
            p332=self.conv1440(D1)
        elif  num2==441:
            p332=self.conv1441(D1)
        elif  num2==442:
            p332=self.conv1442(D1)
        elif  num2==443:
            p332=self.conv1443(D1)
        elif  num2==444:
            p332=self.conv1444(D1)
        elif  num2==445:
            p332=self.conv1445(D1)
        elif  num2==446:
            p332=self.conv1446(D1)
        elif  num2==447:
            p332=self.conv1447(D1)
        elif  num2==448:
            p332=self.conv1448(D1)
        elif  num2==449:
            p332=self.conv1449(D1)
        elif  num2==450:
            p332=self.conv1450(D1)
        elif  num2==451:
            p332=self.conv1451(D1)
        elif  num2==452:
            p332=self.conv1452(D1)
        elif  num2==453:
            p332=self.conv1453(D1)
        elif  num2==454:
            p332=self.conv1454(D1)
        elif  num2==455:
            p332=self.conv1455(D1)
        elif  num2==456:
            p332=self.conv1456(D1)
        elif  num2==457:
            p332=self.conv1457(D1)
        elif  num2==458:
            p332=self.conv1458(D1)
        elif  num2==459:
            p332=self.conv1459(D1)
        elif  num2==460:
            p332=self.conv1460(D1)
        elif  num2==461:
            p332=self.conv1461(D1)
        elif  num2==462:
            p332=self.conv1462(D1)
        elif  num2==463:
            p332=self.conv1463(D1)
        elif  num2==464:
            p332=self.conv1464(D1)
        elif  num2==465:
            p332=self.conv1465(D1)
        elif  num2==466:
            p332=self.conv1466(D1)
        elif  num2==467:
            p332=self.conv1467(D1)
        elif  num2==468:
            p332=self.conv1468(D1)
        elif  num2==469:
            p332=self.conv1469(D1) 
        elif  num2==470:
            p332=self.conv1470(D1)
        elif  num2==471:
            p332=self.conv1471(D1)
        elif  num2==472:
            p332=self.conv1472(D1)
        elif  num2==473:
            p332=self.conv1473(D1)
        elif  num2==474:
            p332=self.conv1474(D1)
        elif  num2==475:
            p332=self.conv1475(D1)
        elif  num2==476:
            p332=self.conv1476(D1)
        elif  num2==477:
            p332=self.conv1477(D1)
        elif  num2==478:
            p332=self.conv1478(D1)
        elif  num2==479:
            p332=self.conv1479(D1)
        elif  num2==480:
            p332=self.conv1480(D1)
        elif  num2==481:
            p332=self.conv1481(D1)
        elif  num2==482:
            p332=self.conv1482(D1)
        elif  num2==483:
            p332=self.conv1483(D1)
        elif  num2==484:
            p332=self.conv1484(D1)
        elif  num2==485:
            p332=self.conv1485(D1)
        elif  num2==486:
            p332=self.conv1486(D1)
        elif  num2==487:
            p332=self.conv1487(D1)
        elif  num2==488:
            p332=self.conv1488(D1)
        elif  num2==489:
            p332=self.conv1489(D1)
        elif  num2==490:
            p332=self.conv1490(D1)
        elif  num2==491:
            p332=self.conv1491(D1)
        elif  num2==492:
            p332=self.conv1492(D1)
        elif  num2==493:
            p332=self.conv1493(D1)
        elif  num2==494:
            p332=self.conv1494(D1)
        elif  num2==495:
            p332=self.conv1495(D1)
        elif  num2==496:
            p332=self.conv1496(D1)
        elif  num2==497:
            p332=self.conv1497(D1)
        elif  num2==498:
            p332=self.conv1498(D1)
        elif  num2==499:
            p332=self.conv1499(D1)
        elif  num2==500:
            p332=self.conv1500(D1)
        elif  num2==501:
            p332=self.conv1501(D1)
        elif  num2==502:
            p332=self.conv1502(D1)
        elif  num2==503:
            p332=self.conv1503(D1)
        elif  num2==504:
            p332=self.conv1504(D1)
        elif  num2==505:
            p332=self.conv1505(D1)
        elif  num2==506:
            p332=self.conv1506(D1)
        elif  num2==507:
            p332=self.conv1507(D1)
        elif  num2==508:
            p332=self.conv1508(D1)
        elif  num2==509:
            p332=self.conv1509(D1)
        elif  num2==510:
            p332=self.conv1510(D1)
        elif  num2==511:
            p332=self.conv1511(D1)
        elif  num2==512:
            p332=self.conv1512(D1)
        elif  num2==513:
            p332=self.conv1513(D1)
        elif  num2==514:
            p332=self.conv1514(D1)
        elif  num2==515:
            p332=self.conv1515(D1)
        elif  num2==516:
            p332=self.conv1516(D1)
        elif  num2==517:
            p332=self.conv1517(D1)
        elif  num2==518:
            p332=self.conv1518(D1)
        elif  num2==519:
            p332=self.conv1519(D1)
        elif  num2==520:
            p332=self.conv1520(D1)
        elif  num2==521:
            p332=self.conv1521(D1)
        elif  num2==522:
            p332=self.conv1522(D1)
        elif  num2==523:
            p332=self.conv1523(D1)
        elif  num2==524:
            p332=self.conv1524(D1)
        elif  num2==525:
            p332=self.conv1525(D1)
        elif  num2==526:
            p332=self.conv1526(D1)
        elif  num2==527:
            p332=self.conv1527(D1)
        elif  num2==528:
            p332=self.conv1528(D1)
        elif  num2==529:
            p332=self.conv1529(D1)
        elif  num2==530:
            p332=self.conv1530(D1)
        elif  num2==531:
            p332=self.conv1531(D1)
        elif  num2==532:
            p332=self.conv1532(D1)
        elif  num2==533:
            p332=self.conv1533(D1)
        elif  num2==534:
            p332=self.conv1534(D1)
        elif  num2==535:
            p332=self.conv1535(D1)
        elif  num2==536:
            p332=self.conv1536(D1)
        elif  num2==537:
            p332=self.conv1537(D1)
        elif  num2==538:
            p332=self.conv1538(D1)
        elif  num2==539:
            p332=self.conv1539(D1)
        elif  num2==540:
            p332=self.conv1540(D1)
        elif  num2==541:
            p332=self.conv1541(D1)
        elif  num2==542:
            p332=self.conv1542(D1)
        elif  num2==543:
            p332=self.conv1543(D1)
        elif  num2==544:
            p332=self.conv1544(D1)
        elif  num2==545:
            p332=self.conv1545(D1)
        elif  num2==546:
            p332=self.conv1546(D1)
        elif  num2==547:
            p332=self.conv1547(D1)
        elif  num2==548:
            p332=self.conv1548(D1)
        elif  num2==549:
            p332=self.conv1549(D1) 
        elif  num2==550:
            p332=self.conv1550(D1)
        elif  num2==551:
            p332=self.conv1551(D1)
        elif  num2==552:
            p332=self.conv1552(D1)
        elif  num2==553:
            p332=self.conv1553(D1)
        elif  num2==554:
            p332=self.conv1554(D1)
        elif  num2==555:
            p332=self.conv1555(D1)
        elif  num2==556:
            p332=self.conv1556(D1)
        elif  num2==557:
            p332=self.conv1557(D1)
        elif  num2==558:
            p332=self.conv1558(D1)
        elif  num2==559:
            p332=self.conv1559(D1)
        elif  num2==560:
            p332=self.conv1560(D1)
        elif  num2==561:
            p332=self.conv1561(D1)
        elif  num2==562:
            p332=self.conv1562(D1)
        elif  num2==563:
            p332=self.conv1563(D1)
        elif  num2==564:
            p332=self.conv1564(D1)
        elif  num2==565:
            p332=self.conv1565(D1)
        elif  num2==566:
            p332=self.conv1566(D1)
        elif  num2==567:
            p332=self.conv1567(D1)
        elif  num2==568:
            p332=self.conv1568(D1)
        elif  num2==569:
            p332=self.conv1569(D1) 
        elif  num2==570:
            p332=self.conv1570(D1)
        elif  num2==571:
            p332=self.conv1571(D1)
        elif  num2==572:
            p332=self.conv1572(D1)
        elif  num2==573:
            p332=self.conv1573(D1)
        elif  num2==574:
            p332=self.conv1574(D1)
        elif  num2==575:
            p332=self.conv1575(D1)
        elif  num2==576:
            p332=self.conv1576(D1)
        elif  num2==577:
            p332=self.conv1577(D1)
        elif  num2==578:
            p332=self.conv1578(D1)
        elif  num2==579:
            p332=self.conv1579(D1) 
        elif  num2==580:
            p332=self.conv1580(D1)
        elif  num2==581:
            p332=self.conv1581(D1)
        elif  num2==582:
            p332=self.conv1582(D1)
        elif  num2==583:
            p332=self.conv1583(D1)
        elif  num2==584:
            p332=self.conv1584(D1)
        elif  num2==585:
            p332=self.conv1585(D1)
        elif  num2==586:
            p332=self.conv1586(D1)
        elif  num2==587:
            p332=self.conv1587(D1)
        elif  num2==588:
            p332=self.conv1588(D1)
        elif  num2==589:
            p332=self.conv1589(D1)
        elif  num2==590:
            p332=self.conv1590(D1)
        elif  num2==591:
            p332=self.conv1591(D1)
        elif  num2==592:
            p332=self.conv1592(D1)
        elif  num2==593:
            p332=self.conv1593(D1)
        elif  num2==594:
            p332=self.conv1594(D1)
        elif  num2==595:
            p332=self.conv1595(D1)
        elif  num2==596:
            p332=self.conv1596(D1)
        elif  num2==597:
            p332=self.conv1597(D1)
        elif  num2==598:
            p332=self.conv1598(D1)
        elif  num2==599:
            p332=self.conv1599(D1)
        elif  num2==600:
            p332=self.conv1600(D1)
        elif  num2==601:
            p332=self.conv1601(D1)
        elif  num2==602:
            p332=self.conv1602(D1)
        elif  num2==603:
            p332=self.conv1603(D1)
        elif  num2==604:
            p332=self.conv1604(D1)
        elif  num2==605:
            p332=self.conv1605(D1)
        elif  num2==606:
            p332=self.conv1606(D1)
        elif  num2==607:
            p332=self.conv1607(D1)
        elif  num2==608:
            p332=self.conv1608(D1)
        elif  num2==609:
            p332=self.conv1609(D1)                                                                                                                         
        elif  num2==610:
            p332=self.conv1610(D1)
        elif  num2==611:
            p332=self.conv1611(D1)
        elif  num2==612:
            p332=self.conv1612(D1)
        elif  num2==613:
            p332=self.conv1613(D1)
        elif  num2==614:
            p332=self.conv1614(D1)
        elif  num2==615:
            p332=self.conv1615(D1)
        elif  num2==616:
            p332=self.conv1616(D1)
        elif  num2==617:
            p332=self.conv1617(D1)
        elif  num2==618:
            p332=self.conv1618(D1)
        elif  num2==619:
            p332=self.conv1619(D1)                                                                                                                          
        elif  num2==620:
            p332=self.conv1620(D1)
        elif  num2==621:
            p332=self.conv1621(D1)
        elif  num2==622:
            p332=self.conv1622(D1)
        elif  num2==623:
            p332=self.conv1623(D1)
        elif  num2==624:
            p332=self.conv1624(D1)
        elif  num2==625:
            p332=self.conv1625(D1)
        elif  num2==626:
            p332=self.conv1626(D1)
        elif  num2==627:
            p332=self.conv1627(D1)
        elif  num2==628:
            p332=self.conv1628(D1)
        elif  num2==629:
            p332=self.conv1629(D1)  
        elif  num2==630:
            p332=self.conv1630(D1)
        elif  num2==631:
            p332=self.conv1631(D1)
        elif  num2==632:
            p332=self.conv1632(D1)
        elif  num2==633:
            p332=self.conv1633(D1)
        elif  num2==634:
            p332=self.conv1634(D1)
        elif  num2==635:
            p332=self.conv1635(D1)
        elif  num2==636:
            p332=self.conv1636(D1)
        elif  num2==637:
            p332=self.conv1637(D1)
        elif  num2==638:
            p332=self.conv1638(D1)
        elif  num2==639:
            p332=self.conv1639(D1)                                                                                                                          
        elif  num2==640:
            p332=self.conv1640(D1)
        elif  num2==641:
            p332=self.conv1641(D1)
        elif  num2==642:
            p332=self.conv1642(D1)
        elif  num2==643:
            p332=self.conv1643(D1)
        elif  num2==644:
            p332=self.conv1644(D1)
        elif  num2==645:
            p332=self.conv1645(D1)
        elif  num2==646:
            p332=self.conv1646(D1)
        elif  num2==647:
            p332=self.conv1647(D1)
        elif  num2==648:
            p332=self.conv1648(D1)
        elif  num2==649:
            p332=self.conv1649(D1)                                                                                                                          
        elif  num2==650:
            p332=self.conv1650(D1)
        elif  num2==651:
            p332=self.conv1651(D1)
        elif  num2==652:
            p332=self.conv1652(D1)
        elif  num2==653:
            p332=self.conv1653(D1)
        elif  num2==654:
            p332=self.conv1654(D1)
        elif  num2==655:
            p332=self.conv1655(D1)
        elif  num2==656:
            p332=self.conv1656(D1)
        elif  num2==657:
            p332=self.conv1657(D1)
        elif  num2==658:
            p332=self.conv1658(D1)
        elif  num2==659:
            p332=self.conv1659(D1)
        elif  num2==660:
            p332=self.conv1660(D1)
        elif  num2==661:
            p332=self.conv1661(D1)
        elif  num2==662:
            p332=self.conv1662(D1)
        elif  num2==663:
            p332=self.conv1663(D1)
        elif  num2==664:
            p332=self.conv1664(D1)
        elif  num2==665:
            p332=self.conv1665(D1)
        elif  num2==666:
            p332=self.conv1666(D1)
        elif  num2==667:
            p332=self.conv1667(D1)
        elif  num2==668:
            p332=self.conv1668(D1)
        elif  num2==669:
            p332=self.conv1669(D1) 
        elif  num2==670:
            p332=self.conv1670(D1)
        elif  num2==671:
            p332=self.conv1671(D1)
        elif  num2==672:
            p332=self.conv1672(D1)
        elif  num2==673:
            p332=self.conv1673(D1)
        elif  num2==674:
            p332=self.conv1674(D1)
        elif  num2==675:
            p332=self.conv1675(D1)
        elif  num2==676:
            p332=self.conv1676(D1)
        elif  num2==677:
            p332=self.conv1677(D1)
        elif  num2==678:
            p332=self.conv1678(D1)
        elif  num2==679:
            p332=self.conv1679(D1)
        elif  num2==680:
            p332=self.conv1680(D1)
        elif  num2==681:
            p332=self.conv1681(D1)
        elif  num2==682:
            p332=self.conv1682(D1)
        elif  num2==683:
            p332=self.conv1683(D1)
        elif  num2==684:
            p332=self.conv1684(D1)
        elif  num2==685:
            p332=self.conv1685(D1)
        elif  num2==686:
            p332=self.conv1686(D1)
        elif  num2==687:
            p332=self.conv1687(D1)
        elif  num2==688:
            p332=self.conv1688(D1)
        elif  num2==689:
            p332=self.conv1689(D1)
        elif  num2==690:
            p332=self.conv1690(D1)
        elif  num2==691:
            p332=self.conv1691(D1)
        elif  num2==692:
            p332=self.conv1692(D1)
        elif  num2==693:
            p332=self.conv1693(D1)
        elif  num2==694:
            p332=self.conv1694(D1)
        elif  num2==695:
            p332=self.conv1695(D1)
        elif  num2==696:
            p332=self.conv1696(D1)
        elif  num2==697:
            p332=self.conv1697(D1)
        elif  num2==698:
            p332=self.conv1698(D1)
        elif  num2==699:
            p332=self.conv1699(D1)
        elif  num2==700:
            p332=self.conv1700(D1)
        elif  num2==701:
            p332=self.conv1701(D1)
        elif  num2==702:
            p332=self.conv1702(D1)
        elif  num2==703:
            p332=self.conv1703(D1)
        elif  num2==704:
            p332=self.conv1704(D1)
        elif  num2==705:
            p332=self.conv1705(D1)
        elif  num2==706:
            p332=self.conv1706(D1)
        elif  num2==707:
            p332=self.conv1707(D1)
        elif  num2==708:
            p332=self.conv1708(D1)
        elif  num2==709:
            p332=self.conv1709(D1)
        elif  num2==710:
            p332=self.conv1710(D1)
        elif  num2==711:
            p332=self.conv1711(D1)
        elif  num2==712:
            p332=self.conv1712(D1)
        elif  num2==713:
            p332=self.conv1713(D1)
        elif  num2==714:
            p332=self.conv1714(D1)
        elif  num2==715:
            p332=self.conv1715(D1)
        elif  num2==716:
            p332=self.conv1716(D1)
        elif  num2==717:
            p332=self.conv1717(D1)
        elif  num2==718:
            p332=self.conv1718(D1)
        elif  num2==719:
            p332=self.conv1719(D1)
        elif  num2==720:
            p332=self.conv1720(D1)
        elif  num2==721:
            p332=self.conv1721(D1)
        elif  num2==722:
            p332=self.conv1722(D1)
        elif  num2==723:
            p332=self.conv1723(D1)
        elif  num2==724:
            p332=self.conv1724(D1)
        elif  num2==725:
            p332=self.conv1725(D1)
        elif  num2==726:
            p332=self.conv1726(D1)
        elif  num2==727:
            p332=self.conv1727(D1)
        elif  num2==728:
            p332=self.conv1728(D1)
        elif  num2==729:
            p332=self.conv1729(D1)
        elif  num2==730:
            p332=self.conv1730(D1)
        elif  num2==731:
            p332=self.conv1731(D1)
        elif  num2==732:
            p332=self.conv1732(D1)
        elif  num2==733:
            p332=self.conv1733(D1)
        elif  num2==734:
            p332=self.conv1734(D1)
        elif  num2==735:
            p332=self.conv1735(D1)
        elif  num2==736:
            p332=self.conv1736(D1)
        elif  num2==737:
            p332=self.conv1737(D1)
        elif  num2==738:
            p332=self.conv1738(D1)
        elif  num2==739:
            p332=self.conv1739(D1)                                                                                                                          
        elif  num2==740:
            p332=self.conv1740(D1)
        elif  num2==741:
            p332=self.conv1741(D1)
        elif  num2==742:
            p332=self.conv1742(D1)
        elif  num2==743:
            p332=self.conv1743(D1)
        elif  num2==744:
            p332=self.conv1744(D1)
        elif  num2==745:
            p332=self.conv1745(D1)
        elif  num2==746:
            p332=self.conv1746(D1)
        elif  num2==747:
            p332=self.conv1747(D1)
        elif  num2==748:
            p332=self.conv1748(D1)
        elif  num2==749:
            p332=self.conv1749(D1)
        elif  num2==750:
            p332=self.conv1750(D1)
        elif  num2==751:
            p332=self.conv1751(D1)
        elif  num2==752:
            p332=self.conv1752(D1)
        elif  num2==753:
            p332=self.conv1753(D1)
        elif  num2==754:
            p332=self.conv1754(D1)
        elif  num2==755:
            p332=self.conv1755(D1)
        elif  num2==756:
            p332=self.conv1756(D1)
        elif  num2==757:
            p332=self.conv1757(D1)
        elif  num2==758:
            p332=self.conv1758(D1)
        elif  num2==759:
            p332=self.conv1759(D1)
        elif  num2==760:
            p332=self.conv1760(D1)
        elif  num2==761:
            p332=self.conv1761(D1)
        elif  num2==762:
            p332=self.conv1762(D1)
        elif  num2==763:
            p332=self.conv1763(D1)
        elif  num2==764:
            p332=self.conv1764(D1)
        elif  num2==765:
            p332=self.conv1765(D1)
        elif  num2==766:
            p332=self.conv1766(D1)
        elif  num2==767:
            p332=self.conv1767(D1)
        elif  num2==768:
            p332=self.conv1768(D1)
        elif  num2==769:
            p332=self.conv1769(D1) 
        elif  num2==770:
            p332=self.conv1770(D1)
        elif  num2==771:
            p332=self.conv1771(D1)
        elif  num2==772:
            p332=self.conv1772(D1)
        elif  num2==773:
            p332=self.conv1773(D1)
        elif  num2==774:
            p332=self.conv1774(D1)
        elif  num2==775:
            p332=self.conv1775(D1)
        elif  num2==776:
            p332=self.conv1776(D1)
        elif  num2==777:
            p332=self.conv1777(D1)
        elif  num2==778:
            p332=self.conv1778(D1)
        elif  num2==779:
            p332=self.conv1779(D1) 
        elif  num2==780:
            p332=self.conv1780(D1)
        elif  num2==781:
            p332=self.conv1781(D1)
        elif  num2==782:
            p332=self.conv1782(D1)
        elif  num2==783:
            p332=self.conv1783(D1)
        elif  num2==784:
            p332=self.conv1784(D1)
        elif  num2==785:
            p332=self.conv1785(D1)
        elif  num2==786:
            p332=self.conv1786(D1)
        elif  num2==787:
            p332=self.conv1787(D1)
        elif  num2==788:
            p332=self.conv1788(D1)
        elif  num2==789:
            p332=self.conv1789(D1) 
        elif  num2==790:
            p332=self.conv1790(D1)
        elif  num2==791:
            p332=self.conv1791(D1)
        elif  num2==792:
            p332=self.conv1792(D1)
        elif  num2==793:
            p332=self.conv1793(D1)
        elif  num2==794:
            p332=self.conv1794(D1)
        elif  num2==795:
            p332=self.conv1795(D1)
        elif  num2==796:
            p332=self.conv1796(D1)
        elif  num2==797:
            p332=self.conv1797(D1)
        elif  num2==798:
            p332=self.conv1798(D1)
        elif  num2==799:
            p332=self.conv1799(D1) 
        elif  num2==800:
            p332=self.conv1800(D1)
        elif  num2==801:
            p332=self.conv1801(D1)
        elif  num2==802:
            p332=self.conv1802(D1)
        elif  num2==803:
            p332=self.conv1803(D1)
        elif  num2==804:
            p332=self.conv1804(D1)
        elif  num2==805:
            p332=self.conv1805(D1)
        elif  num2==806:
            p332=self.conv1806(D1)
        elif  num2==807:
            p332=self.conv1807(D1)
        elif  num2==808:
            p332=self.conv1808(D1)
        elif  num2==809:
            p332=self.conv1809(D1)
        elif  num2==810:
            p332=self.conv1810(D1)
        elif  num2==811:
            p332=self.conv1811(D1)
        elif  num2==812:
            p332=self.conv1812(D1)
        elif  num2==813:
            p332=self.conv1813(D1)
        elif  num2==814:
            p332=self.conv1814(D1)
        elif  num2==815:
            p332=self.conv1815(D1)
        elif  num2==816:
            p332=self.conv1816(D1)
        elif  num2==817:
            p332=self.conv1817(D1)
        elif  num2==818:
            p332=self.conv1818(D1)
        elif  num2==819:
            p332=self.conv1819(D1)
        elif  num2==820:
            p332=self.conv1820(D1)
        elif  num2==821:
            p332=self.conv1821(D1)
        elif  num2==822:
            p332=self.conv1822(D1)
        elif  num2==823:
            p332=self.conv1823(D1)
        elif  num2==824:
            p332=self.conv1824(D1)
        elif  num2==825:
            p332=self.conv1825(D1)
        elif  num2==826:
            p332=self.conv1826(D1)
        elif  num2==827:
            p332=self.conv1827(D1)
        elif  num2==828:
            p332=self.conv1828(D1)
        elif  num2==829:
            p332=self.conv1829(D1)                                                                                                                          
        elif  num2==830:
            p332=self.conv1830(D1)
        elif  num2==831:
            p332=self.conv1831(D1)
        elif  num2==832:
            p332=self.conv1832(D1)
        elif  num2==833:
            p332=self.conv1833(D1)
        elif  num2==834:
            p332=self.conv1834(D1)
        elif  num2==835:
            p332=self.conv1835(D1)
        elif  num2==836:
            p332=self.conv1836(D1)
        elif  num2==837:
            p332=self.conv1837(D1)
        elif  num2==838:
            p332=self.conv1838(D1)
        elif  num2==839:
            p332=self.conv1839(D1)
        elif  num2==840:
            p332=self.conv1840(D1)
        elif  num2==841:
            p332=self.conv1841(D1)
        elif  num2==842:
            p332=self.conv1842(D1)
        elif  num2==843:
            p332=self.conv1843(D1)
        elif  num2==844:
            p332=self.conv1844(D1)
        elif  num2==845:
            p332=self.conv1845(D1)
        elif  num2==846:
            p332=self.conv1846(D1)
        elif  num2==847:
            p332=self.conv1847(D1)
        elif  num2==848:
            p332=self.conv1848(D1)
        elif  num2==849:
            p332=self.conv1849(D1)
        elif  num2==850:
            p332=self.conv1850(D1)
        elif  num2==851:
            p332=self.conv1851(D1)
        elif  num2==852:
            p332=self.conv1852(D1)
        elif  num2==853:
            p332=self.conv1853(D1)
        elif  num2==854:
            p332=self.conv1854(D1)
        elif  num2==855:
            p332=self.conv1855(D1)
        elif  num2==856:
            p332=self.conv1856(D1)
        elif  num2==857:
            p332=self.conv1857(D1)
        elif  num2==858:
            p332=self.conv1858(D1)
        elif  num2==859:
            p332=self.conv1859(D1)
        elif  num2==860:
            p332=self.conv1860(D1)
        elif  num2==861:
            p332=self.conv1861(D1)
        elif  num2==862:
            p332=self.conv1862(D1)
        elif  num2==863:
            p332=self.conv1863(D1)
        elif  num2==864:
            p332=self.conv1864(D1)
        elif  num2==865:
            p332=self.conv1865(D1)
        elif  num2==866:
            p332=self.conv1866(D1)
        elif  num2==867:
            p332=self.conv1867(D1)
        elif  num2==868:
            p332=self.conv1868(D1)
        elif  num2==869:
            p332=self.conv1869(D1) 
        elif  num2==870:
            p332=self.conv1870(D1)
        elif  num2==871:
            p332=self.conv1871(D1)
        elif  num2==872:
            p332=self.conv1872(D1)
        elif  num2==873:
            p332=self.conv1873(D1)
        elif  num2==874:
            p332=self.conv1874(D1)
        elif  num2==875:
            p332=self.conv1875(D1)
        elif  num2==876:
            p332=self.conv1876(D1)
        elif  num2==877:
            p332=self.conv1877(D1)
        elif  num2==878:
            p332=self.conv1878(D1)
        elif  num2==879:
            p332=self.conv1879(D1)
        elif  num2==880:
            p332=self.conv1880(D1)
        elif  num2==881:
            p332=self.conv1881(D1)
        elif  num2==882:
            p332=self.conv1882(D1)
        elif  num2==883:
            p332=self.conv1883(D1)
        elif  num2==884:
            p332=self.conv1884(D1)
        elif  num2==885:
            p332=self.conv1885(D1)
        elif  num2==886:
            p332=self.conv1886(D1)
        elif  num2==887:
            p332=self.conv1887(D1)
        elif  num2==888:
            p332=self.conv1888(D1)
        elif  num2==889:
            p332=self.conv1889(D1)  
        elif  num2==890:
            p332=self.conv1890(D1)
        elif  num2==891:
            p332=self.conv1891(D1)
        elif  num2==892:
            p332=self.conv1892(D1)
        elif  num2==893:
            p332=self.conv1893(D1)
        elif  num2==894:
            p332=self.conv1894(D1)
        elif  num2==895:
            p332=self.conv1895(D1)
        elif  num2==896:
            p332=self.conv1896(D1)
        elif  num2==897:
            p332=self.conv1897(D1)
        elif  num2==898:
            p332=self.conv1898(D1)
        elif  num2==899:
            p332=self.conv1899(D1)
        elif  num2==900:
            p332=self.conv1900(D1)
        elif  num2==901:
            p332=self.conv1901(D1)
        elif  num2==902:
            p332=self.conv1902(D1)
        elif  num2==903:
            p332=self.conv1903(D1)
        elif  num2==904:
            p332=self.conv1904(D1)
        elif  num2==905:
            p332=self.conv1905(D1)
        elif  num2==906:
            p332=self.conv1906(D1)
        elif  num2==907:
            p332=self.conv1907(D1)
        elif  num2==908:
            p332=self.conv1908(D1)
        elif  num2==909:
            p332=self.conv1909(D1)
        elif  num2==910:
            p332=self.conv1910(D1)
        elif  num2==911:
            p332=self.conv1911(D1)
        elif  num2==912:
            p332=self.conv1912(D1)
        elif  num2==913:
            p332=self.conv1913(D1)
        elif  num2==914:
            p332=self.conv1914(D1)
        elif  num2==915:
            p332=self.conv1915(D1)
        elif  num2==916:
            p332=self.conv1916(D1)
        elif  num2==917:
            p332=self.conv1917(D1)
        elif  num2==918:
            p332=self.conv1918(D1)
        elif  num2==919:
            p332=self.conv1919(D1)
        elif  num2==920:
            p332=self.conv1920(D1)
        elif  num2==921:
            p332=self.conv1921(D1)
        elif  num2==922:
            p332=self.conv1922(D1)
        elif  num2==923:
            p332=self.conv1923(D1)
        elif  num2==924:
            p332=self.conv1924(D1)
        elif  num2==925:
            p332=self.conv1925(D1)
        elif  num2==926:
            p332=self.conv1926(D1)
        elif  num2==927:
            p332=self.conv1927(D1)
        elif  num2==928:
            p332=self.conv1928(D1)
        elif  num2==929:
            p332=self.conv1929(D1)
        elif  num2==930:
            p332=self.conv1930(D1)
        elif  num2==931:
            p332=self.conv1931(D1)
        elif  num2==932:
            p332=self.conv1932(D1)
        elif  num2==933:
            p332=self.conv1933(D1)
        elif  num2==934:
            p332=self.conv1934(D1)
        elif  num2==935:
            p332=self.conv1935(D1)
        elif  num2==936:
            p332=self.conv1936(D1)
        elif  num2==937:
            p332=self.conv1937(D1)
        elif  num2==938:
            p332=self.conv1938(D1)
        elif  num2==939:
            p332=self.conv1939(D1) 
        elif  num2==940:
            p332=self.conv1940(D1)
        elif  num2==941:
            p332=self.conv1941(D1)
        elif  num2==942:
            p332=self.conv1942(D1)
        elif  num2==943:
            p332=self.conv1943(D1)
        elif  num2==944:
            p332=self.conv1944(D1)
        elif  num2==945:
            p332=self.conv1945(D1)
        elif  num2==946:
            p332=self.conv1946(D1)
        elif  num2==947:
            p332=self.conv1947(D1)
        elif  num2==948:
            p332=self.conv1948(D1)
        elif  num2==949:
            p332=self.conv1949(D1)                                                                                                                          
        elif  num2==950:
            p332=self.conv1950(D1)
        elif  num2==951:
            p332=self.conv1951(D1)
        elif  num2==952:
            p332=self.conv1952(D1)
        elif  num2==953:
            p332=self.conv1953(D1)
        elif  num2==954:
            p332=self.conv1954(D1)
        elif  num2==955:
            p332=self.conv1955(D1)
        elif  num2==956:
            p332=self.conv1956(D1)
        elif  num2==957:
            p332=self.conv1957(D1)
        elif  num2==958:
            p332=self.conv1958(D1)
        elif  num2==959:
            p332=self.conv1959(D1)
        elif  num2==960:
            p332=self.conv1960(D1)
        elif  num2==961:
            p332=self.conv1961(D1)
        elif  num2==962:
            p332=self.conv1962(D1)
        elif  num2==963:
            p332=self.conv1963(D1)
        elif  num2==964:
            p332=self.conv1964(D1)
        elif  num2==965:
            p332=self.conv1965(D1)
        elif  num2==966:
            p332=self.conv1966(D1)
        elif  num2==967:
            p332=self.conv1967(D1)
        elif  num2==968:
            p332=self.conv1968(D1)
        elif  num2==969:
            p332=self.conv1969(D1) 
        elif  num2==970:
            p332=self.conv1970(D1)
        elif  num2==971:
            p332=self.conv1971(D1)
        elif  num2==972:
            p332=self.conv1972(D1)
        elif  num2==973:
            p332=self.conv1973(D1)
        elif  num2==974:
            p332=self.conv1974(D1)
        elif  num2==975:
            p332=self.conv1975(D1)
        elif  num2==976:
            p332=self.conv1976(D1)
        elif  num2==977:
            p332=self.conv1977(D1)
        elif  num2==978:
            p332=self.conv1978(D1)
        elif  num2==979:
            p332=self.conv1979(D1) 
        elif  num2==980:
            p332=self.conv1980(D1)
        elif  num2==981:
            p332=self.conv1981(D1)
        elif  num2==982:
            p332=self.conv1982(D1)
        elif  num2==983:
            p332=self.conv1983(D1)
        elif  num2==984:
            p332=self.conv1984(D1)
        elif  num2==985:
            p332=self.conv1985(D1)
        elif  num2==986:
            p332=self.conv1986(D1)
        elif  num2==987:
            p332=self.conv1987(D1)
        elif  num2==988:
            p332=self.conv1988(D1)
        elif  num2==989:
            p332=self.conv1989(D1)
        elif  num2==990:
            p332=self.conv1990(D1)
        elif  num2==991:
            p332=self.conv1991(D1)
        elif  num2==992:
            p332=self.conv1992(D1)
        elif  num2==993:
            p332=self.conv1993(D1)
        elif  num2==994:
            p332=self.conv1994(D1)
        elif  num2==995:
            p332=self.conv1995(D1)
        elif  num2==996:
            p332=self.conv1996(D1)
        elif  num2==997:
            p332=self.conv1997(D1)
        elif  num2==998:
            p332=self.conv1998(D1)
        elif  num2==999:
            p332=self.conv1999(D1) 
        elif  num2==1000:
            p332=self.conv11000(D1)
        elif  num2==1001:
            p332=self.conv11001(D1)
        elif  num2==1002:
            p332=self.conv11002(D1)
        elif  num2==1003:
            p332=self.conv11003(D1)
        elif  num2==1004:
            p332=self.conv11004(D1)
        elif  num2==1005:
            p332=self.conv11005(D1)
        elif  num2==1006:
            p332=self.conv11006(D1)
        elif  num2==1007:
            p332=self.conv11007(D1)
        elif  num2==1008:
            p332=self.conv11008(D1)
        elif  num2==1009:
            p332=self.conv11009(D1) 
        elif  num2==1010:
            p332=self.conv11010(D1)
        elif  num2==1011:
            p332=self.conv11011(D1)
        elif  num2==1012:
            p332=self.conv11012(D1)
        elif  num2==1013:
            p332=self.conv11013(D1)
        elif  num2==1014:
            p332=self.conv11014(D1)
        elif  num2==1015:
            p332=self.conv11015(D1)
        elif  num2==1016:
            p332=self.conv11016(D1)
        elif  num2==1017:
            p332=self.conv11017(D1)
        elif  num2==1018:
            p332=self.conv11018(D1)
        elif  num2==1019:
            p332=self.conv11019(D1)
        elif  num2==1020:
            p332=self.conv11020(D1)
        elif  num2==1021:
            p332=self.conv11021(D1)
        elif  num2==1022:
            p332=self.conv11022(D1)
        elif  num2==1023:
            p332=self.conv11023(D1)
        elif  num2==1024:
            p332=self.conv11024(D1)                                                                                         
         
        if num04==1:
            p34=self.conv11(B14)
        elif num04==2:
            p34=self.conv12(B14)
        elif num04==3:
            p34=self.conv13(B14)
        elif num04==4:
            p34=self.conv14(B14)
        elif num04==5:
            p34=self.conv15(B14)
        elif num04==6:
            p34=self.conv16(B14)
        elif num04==7:
            p34=self.conv17(B14)
        elif num04==8:
            p34=self.conv18(B14)
        elif num04==9:
            p34=self.conv19(B14)
        elif num04==10:
            p34=self.conv110(B14)
        elif num04==11:
            p34=self.conv111(B14)
        elif num04==12:
            p34=self.conv112(B14)
        elif num04==13:
            p34=self.conv113(B14)
        elif num04==14:
            p34=self.conv114(B14)
        elif num04==15:
            p34=self.conv115(B14)
        elif num04==16:
            p34=self.conv116(B14)
        elif num04==17:
            p34=self.conv117(B14)
        elif num04==18:
            p34=self.conv118(B14)
        elif num04==19:
            p34=self.conv119(B14)
        elif num04==20:
            p34=self.conv120(B14)
        elif num04==21:
            p34=self.conv121(B14)
        elif num04==22:
            p34=self.conv122(B14)
        elif num04==23:
            p34=self.conv123(B14)
        elif num04==24:
            p34=self.conv124(B14)
        elif num04==25:
            p34=self.conv125(B14)
        elif num04==26:
            p34=self.conv126(B14)
        elif num04==27:
            p34=self.conv127(B14)
        elif num04==28:
            p34=self.conv128(B14)
        elif num04==29:
            p34=self.conv129(B14)
        elif num04==30:
            p34=self.conv130(B14)
        elif num04==31:
            p34=self.conv131(B14)
        elif num04==32:
            p34=self.conv132(B14)
        elif num04==33:
            p34=self.conv133(B14)
        elif num04==34:
            p34=self.conv134(B14)
        elif num04==35:
            p34=self.conv135(B14)
        elif num04==36:
            p34=self.conv136(B14)
        elif num04==37:
            p34=self.conv137(B14)
        elif num04==38:
            p34=self.conv138(B14)
        elif num04==39:
            p34=self.conv139(B14)
        elif num04==40:
            p34=self.conv140(B14)
        elif num04==41:
            p34=self.conv141(B14)
        elif num04==42:
            p34=self.conv142(B14)
        elif num04==43:
            p34=self.conv143(B14)
        elif num04==44:
            p34=self.conv144(B14)
        elif num04==45:
            p34=self.conv145(B14)
        elif num04==46:
            p34=self.conv146(B14)
        elif num04==47:
            p34=self.conv147(B14)
        elif num04==48:
            p34=self.conv148(B14)
        elif num04==49:
            p34=self.conv149(B14)
        elif num04==50:
            p34=self.conv150(B14)
        elif num04==51:
            p34=self.conv151(B14)
        elif num04==52:
            p34=self.conv152(B14)
        elif num04==53:
            p34=self.conv153(B14)
        elif num04==54:
            p34=self.conv154(B14)
        elif num04==55:
            p34=self.conv155(B14)
        elif num04==56:
            p34=self.conv156(B14)
        elif num04==57:
            p34=self.conv157(B14)
        elif num04==58:
            p34=self.conv158(B14)
        elif num04==59:
            p34=self.conv159(B14)
        elif num04==60:
            p34=self.conv160(B14)
        elif num04==61:
            p34=self.conv161(B14)
        elif num04==62:
            p34=self.conv162(B14)
        elif num04==63:
            p34=self.conv163(B14)
        elif num04==64:
            p34=self.conv164(B14)
        
        if  num14==1:
            p341=self.conv11(C14)
        elif  num14==2:
            p341=self.conv12(C14)
        elif  num14==3:
            p341=self.conv13(C14)
        elif  num14==4:
            p341=self.conv14(C14)
        elif  num14==5:
            p341=self.conv15(C14)
        elif  num14==6:
            p341=self.conv16(C14)
        elif  num14==7:
            p341=self.conv17(C14)
        elif  num14==8:
            p341=self.conv18(C14)
        elif  num14==9:
            p341=self.conv19(C14)
        elif  num14==10:
            p341=self.conv110(C14)
        elif  num14==11:
            p341=self.conv111(C14)
        elif  num14==12:
            p341=self.conv112(C14)
        elif  num14==13:
            p341=self.conv113(C14)
        elif  num14==14:
            p341=self.conv114(C14)
        elif  num14==15:
            p341=self.conv115(C14)
        elif  num14==16:
            p341=self.conv116(C14)
        elif  num14==17:
            p341=self.conv117(C14)
        elif  num14==18:
            p341=self.conv118(C14)
        elif  num14==19:
            p341=self.conv119(C14)
        elif  num14==20:
            p341=self.conv120(C14)
        elif  num14==21:
            p341=self.conv121(C14)
        elif  num14==22:
            p341=self.conv122(C14)
        elif  num14==23:
            p341=self.conv123(C14)
        elif  num14==24:
            p341=self.conv124(C14)
        elif  num14==25:
            p341=self.conv125(C14)
        elif  num14==26:
            p341=self.conv126(C14)
        elif  num14==27:
            p341=self.conv127(C14)
        elif  num14==28:
            p341=self.conv128(C14)
        elif  num14==29:
            p341=self.conv129(C14)
        elif  num14==30:
            p341=self.conv130(C14)
        elif  num14==31:
            p341=self.conv131(C14)
        elif  num14==32:
            p341=self.conv132(C14)
        elif  num14==33:
            p341=self.conv133(C14)
        elif  num14==34:
            p341=self.conv134(C14)
        elif  num14==35:
            p341=self.conv135(C14)
        elif  num14==36:
            p341=self.conv136(C14)
        elif  num14==37:
            p341=self.conv137(C14)
        elif  num14==38:
            p341=self.conv138(C14)
        elif  num14==39:
            p341=self.conv139(C14)
        elif  num14==40:
            p341=self.conv140(C14)
        elif  num14==41:
            p341=self.conv141(C14)
        elif  num14==42:
            p341=self.conv142(C14)
        elif  num14==43:
            p341=self.conv143(C14)
        elif  num14==44:
            p341=self.conv144(C14)
        elif  num14==45:
            p341=self.conv145(C14)
        elif  num14==46:
            p341=self.conv146(C14)
        elif  num14==47:
            p341=self.conv147(C14)
        elif  num14==48:
            p341=self.conv148(C14)
        elif  num14==49:
            p341=self.conv149(C14)
        elif  num14==50:
            p341=self.conv150(C14)
        elif  num14==51:
            p341=self.conv151(C14)
        elif  num14==52:
            p341=self.conv152(C14)
        elif  num14==53:
            p341=self.conv153(C14)
        elif  num14==54:
            p341=self.conv154(C14)
        elif  num14==55:
            p341=self.conv155(C14)
        elif  num14==56:
            p341=self.conv156(C14)
        elif  num14==57:
            p341=self.conv157(C14)
        elif  num14==58:
            p341=self.conv158(C14)
        elif  num14==59:
            p341=self.conv159(C14)
        elif  num14==60:
            p341=self.conv160(C14)
        elif  num14==61:
            p341=self.conv161(C14)
        elif  num14==62:
            p341=self.conv162(C14)
        elif  num14==63:
            p341=self.conv163(C14)
        elif  num14==64:
            p341=self.conv164(C14)
        elif  num14==65:
            p341=self.conv165(C14)
        elif  num14==66:
            p341=self.conv166(C14)
        elif  num14==67:
            p341=self.conv167(C14)
        elif  num14==68:
            p341=self.conv168(C14)
        elif  num14==69:
            p341=self.conv169(C14)
        elif  num14==70:
            p341=self.conv170(C14)
        elif  num14==71:
            p341=self.conv171(C14)
        elif  num14==72:
            p341=self.conv172(C14)
        elif  num14==73:
            p341=self.conv173(C14)
        elif  num14==74:
            p341=self.conv174(C14)
        elif  num14==75:
            p341=self.conv175(C14)
        elif  num14==76:
            p341=self.conv176(C14)
        elif  num14==77:
            p341=self.conv177(C14)
        elif  num14==78:
            p341=self.conv178(C14)
        elif  num14==79:
            p341=self.conv179(C14)
        elif  num14==80:
            p341=self.conv180(C14)
        elif  num14==81:
            p341=self.conv181(C14)
        elif  num14==82:
            p341=self.conv182(C14)
        elif  num14==83:
            p341=self.conv183(C14)
        elif  num14==84:
            p341=self.conv184(C14)
        elif  num14==85:
            p341=self.conv185(C14)
        elif  num14==86:
            p341=self.conv186(C14)
        elif  num14==87:
            p341=self.conv187(C14)
        elif  num14==88:
            p341=self.conv188(C14)
        elif  num14==89:
            p341=self.conv189(C14)    
        elif  num14==90:
            p341=self.conv190(C14)
        elif  num14==91:
            p341=self.conv191(C14)
        elif  num14==92:
            p341=self.conv192(C14)
        elif  num14==93:
            p341=self.conv193(C14)
        elif  num14==94:
            p341=self.conv194(C14)
        elif  num14==95:
            p341=self.conv195(C14)
        elif  num14==96:
            p341=self.conv196(C14)
        elif  num14==97:
            p341=self.conv197(C14)
        elif  num14==98:
            p341=self.conv198(C14)
        elif  num14==99:
            p341=self.conv199(C14) 
        elif  num14==100:
            p341=self.conv1100(C14)
        elif  num14==101:
            p341=self.conv1101(C14)
        elif  num14==102:
            p341=self.conv1102(C14)
        elif  num14==103:
            p341=self.conv1103(C14)
        elif  num14==104:
            p341=self.conv1104(C14)
        elif  num14==105:
            p341=self.conv1105(C14)
        elif  num14==106:
            p341=self.conv1106(C14)
        elif  num14==107:
            p341=self.conv1107(C14)
        elif  num14==108:
            p341=self.conv1108(C14)
        elif  num14==109:
            p341=self.conv1109(C14)
        elif  num14==110:
            p341=self.conv1110(C14)
        elif  num14==111:
            p341=self.conv1111(C14)
        elif  num14==112:
            p341=self.conv1112(C14)
        elif  num14==113:
            p341=self.conv1113(C14)
        elif  num14==114:
            p341=self.conv1114(C14)
        elif  num14==115:
            p341=self.conv1115(C14)
        elif  num14==116:
            p341=self.conv1116(C14)
        elif  num14==117:
            p341=self.conv1117(C14)
        elif  num14==118:
            p341=self.conv1118(C14)
        elif  num14==119:
            p341=self.conv1119(C14) 
        elif  num14==120:
            p341=self.conv1120(C14)
        elif  num14==121:
            p341=self.conv1121(C14)
        elif  num14==122:
            p341=self.conv1122(C14)
        elif  num14==123:
            p341=self.conv1123(C14)
        elif  num14==124:
            p341=self.conv1124(C14)
        elif  num14==125:
            p341=self.conv1125(C14)
        elif  num14==126:
            p341=self.conv1126(C14)
        elif  num14==127:
            p341=self.conv1127(C14)
        elif  num14==128:
            p341=self.conv1128(C14)
        elif  num14==129:
            p341=self.conv1129(C14) 
        elif  num14==130:
            p341=self.conv1130(C14)
        elif  num14==131:
            p341=self.conv1131(C14)
        elif  num14==132:
            p341=self.conv1132(C14)
        elif  num14==133:
            p341=self.conv1133(C14)
        elif  num14==134:
            p341=self.conv1134(C14)
        elif  num14==135:
            p341=self.conv1135(C14)
        elif  num14==136:
            p341=self.conv1136(C14)
        elif  num14==137:
            p341=self.conv1137(C14)
        elif  num14==138:
            p341=self.conv1138(C14)
        elif  num14==139:
            p341=self.conv1139(C14)
        elif  num14==140:
            p341=self.conv1140(C14)
        elif  num14==141:
            p341=self.conv1141(C14)
        elif  num14==142:
            p341=self.conv1142(C14)
        elif  num14==143:
            p341=self.conv1143(C14)
        elif  num14==144:
            p341=self.conv1144(C14)
        elif  num14==145:
            p341=self.conv1145(C14)
        elif  num14==146:
            p341=self.conv1146(C14)
        elif  num14==147:
            p341=self.conv1147(C14)
        elif  num14==148:
            p341=self.conv1148(C14)
        elif  num14==149:
            p341=self.conv1149(C14) 
        elif  num14==150:
            p341=self.conv1150(C14)
        elif  num14==151:
            p341=self.conv1151(C14)
        elif  num14==152:
            p341=self.conv1152(C14)
        elif  num14==153:
            p341=self.conv1153(C14)
        elif  num14==154:
            p341=self.conv1154(C14)
        elif  num14==155:
            p341=self.conv1155(C14)
        elif  num14==156:
            p341=self.conv1156(C14)
        elif  num14==157:
            p341=self.conv1157(C14)
        elif  num14==158:
            p341=self.conv1158(C14)
        elif  num14==159:
            p341=self.conv1159(C14) 
        elif  num14==160:
            p341=self.conv1160(C14)
        elif  num14==161:
            p341=self.conv1161(C14)
        elif  num14==162:
            p341=self.conv1162(C14)
        elif  num14==163:
            p341=self.conv1163(C14)
        elif  num14==164:
            p341=self.conv1164(C14)
        elif  num14==165:
            p341=self.conv1165(C14)
        elif  num14==166:
            p341=self.conv1166(C14)
        elif  num14==167:
            p341=self.conv1167(C14)
        elif  num14==168:
            p341=self.conv1168(C14)
        elif  num14==169:
            p341=self.conv1169(C14) 
        elif  num14==170:
            p341=self.conv1170(C14)
        elif  num14==171:
            p341=self.conv1171(C14)
        elif  num14==172:
            p341=self.conv1172(C14)
        elif  num14==173:
            p341=self.conv1173(C14)
        elif  num14==174:
            p341=self.conv1174(C14)
        elif  num14==175:
            p341=self.conv1175(C14)
        elif  num14==176:
            p341=self.conv1176(C14)
        elif  num14==177:
            p341=self.conv1177(C14)
        elif  num14==178:
            p341=self.conv1178(C14)
        elif  num14==179:
            p341=self.conv1179(C14)                                                                                              
        elif  num14==180:
            p341=self.conv1180(C14)
        elif  num14==181:
            p341=self.conv1181(C14)
        elif  num14==182:
            p341=self.conv1182(C14)
        elif  num14==183:
            p341=self.conv1183(C14)
        elif  num14==184:
            p341=self.conv1184(C14)
        elif  num14==185:
            p341=self.conv1185(C14)
        elif  num14==186:
            p341=self.conv1186(C14)
        elif  num14==187:
            p341=self.conv1187(C14)
        elif  num14==188:
            p341=self.conv1188(C14)
        elif  num14==189:
            p341=self.conv1189(C14) 
        elif  num14==190:
            p341=self.conv1190(C14)
        elif  num14==191:
            p341=self.conv1191(C14)
        elif  num14==192:
            p341=self.conv1192(C14)
        elif  num14==193:
            p341=self.conv1193(C14)
        elif  num14==194:
            p341=self.conv1194(C14)
        elif  num14==195:
            p341=self.conv1195(C14)
        elif  num14==196:
            p341=self.conv1196(C14)
        elif  num14==197:
            p341=self.conv1197(C14)
        elif  num14==198:
            p341=self.conv1198(C14)
        elif  num14==199:
            p341=self.conv1199(C14)
        elif  num14==200:
            p341=self.conv1200(C14)
        elif  num14==201:
            p341=self.conv1201(C14)
        elif  num14==202:
            p341=self.conv1202(C14)
        elif  num14==203:
            p341=self.conv1203(C14)
        elif  num14==204:
            p341=self.conv1204(C14)
        elif  num14==205:
            p341=self.conv1205(C14)
        elif  num14==206:
            p341=self.conv1206(C14)
        elif  num14==207:
            p341=self.conv1207(C14)
        elif  num14==208:
            p341=self.conv1208(C14)
        elif  num14==209:
            p341=self.conv1209(C14)
        elif  num14==210:
            p341=self.conv1210(C14)
        elif  num14==211:
            p341=self.conv1211(C14)
        elif  num14==212:
            p341=self.conv1212(C14)
        elif  num14==213:
            p341=self.conv1213(C14)
        elif  num14==214:
            p341=self.conv1214(C14)
        elif  num14==215:
            p341=self.conv1215(C14)
        elif  num14==216:
            p341=self.conv1216(C14)
        elif  num14==217:
            p341=self.conv1217(C14)
        elif  num14==218:
            p341=self.conv1218(C14)
        elif  num14==219:
            p341=self.conv1219(C14)
        elif  num14==220:
            p341=self.conv1220(C14)
        elif  num14==221:
            p341=self.conv1221(C14)
        elif  num14==222:
            p341=self.conv1222(C14)
        elif  num14==223:
            p341=self.conv1223(C14)
        elif  num14==224:
            p341=self.conv1224(C14)
        elif  num14==225:
            p341=self.conv1225(C14)
        elif  num14==226:
            p341=self.conv1226(C14)
        elif  num14==227:
            p341=self.conv1227(C14)
        elif  num14==228:
            p341=self.conv1228(C14)
        elif  num14==229:
            p341=self.conv1229(C14)
        elif  num14==230:
            p341=self.conv1230(C14)
        elif  num14==231:
            p341=self.conv1231(C14)
        elif  num14==232:
            p341=self.conv1232(C14)
        elif  num14==233:
            p341=self.conv1233(C14)
        elif  num14==234:
            p341=self.conv1234(C14)
        elif  num14==235:
            p341=self.conv1235(C14)
        elif  num14==236:
            p341=self.conv1236(C14)
        elif  num14==237:
            p341=self.conv1237(C14)
        elif  num14==238:
            p341=self.conv1238(C14)
        elif  num14==239:
            p341=self.conv1239(C14) 
        elif  num14==240:
            p341=self.conv1240(C14)
        elif  num14==241:
            p341=self.conv1241(C14)
        elif  num14==242:
            p341=self.conv1242(C14)
        elif  num14==243:
            p341=self.conv1243(C14)
        elif  num14==244:
            p341=self.conv1244(C14)
        elif  num14==245:
            p341=self.conv1245(C14)
        elif  num14==246:
            p341=self.conv1246(C14)
        elif  num14==247:
            p341=self.conv1247(C14)
        elif  num14==248:
            p341=self.conv1248(C14)
        elif  num14==249:
            p341=self.conv1249(C14)
        elif  num14==250:
            p341=self.conv1250(C14)
        elif  num14==251:
            p341=self.conv1251(C14)
        elif  num14==252:
            p341=self.conv1252(C14)
        elif  num14==253:
            p341=self.conv1253(C14)
        elif  num14==254:
            p341=self.conv1254(C14)
        elif  num14==255:
            p341=self.conv1255(C14)
        elif  num14==256:
            p341=self.conv1256(C14)
            
        if  num24==1:
            p342=self.conv11(D14)
        elif  num24==2:
            p342=self.conv12(D14)
        elif  num24==3:
            p342=self.conv13(D14)
        elif  num24==4:
            p342=self.conv14(D14)
        elif  num24==5:
            p342=self.conv15(D14)
        elif  num24==6:
            p342=self.conv16(D14)
        elif  num24==7:
            p342=self.conv17(D14)
        elif  num24==8:
            p342=self.conv18(D14)
        elif  num24==9:
            p342=self.conv19(D14)
        elif  num24==10:
            p342=self.conv110(D14)
        elif  num24==11:
            p342=self.conv111(D14)
        elif  num24==12:
            p342=self.conv112(D14)
        elif  num24==13:
            p342=self.conv113(D14)
        elif  num24==14:
            p342=self.conv114(D14)
        elif  num24==15:
            p342=self.conv115(D14)
        elif  num24==16:
            p342=self.conv116(D14)
        elif  num24==17:
            p342=self.conv117(D14)
        elif  num24==18:
            p342=self.conv118(D14)
        elif  num24==19:
            p342=self.conv119(D14)
        elif  num24==20:
            p342=self.conv120(D14)
        elif  num24==21:
            p342=self.conv121(D14)
        elif  num24==22:
            p342=self.conv122(D14)
        elif  num24==23:
            p342=self.conv123(D14)
        elif  num24==24:
            p342=self.conv124(D14)
        elif  num24==25:
            p342=self.conv125(D14)
        elif  num24==26:
            p342=self.conv126(D14)
        elif  num24==27:
            p342=self.conv127(D14)
        elif  num24==28:
            p342=self.conv128(D14)
        elif  num24==29:
            p342=self.conv129(D14)
        elif  num24==30:
            p342=self.conv130(D14)
        elif  num24==31:
            p342=self.conv131(D14)
        elif  num24==32:
            p342=self.conv132(D14)
        elif  num24==33:
            p342=self.conv133(D14)
        elif  num24==34:
            p342=self.conv134(D14)
        elif  num24==35:
            p342=self.conv135(D14)
        elif  num24==36:
            p342=self.conv136(D14)
        elif  num24==37:
            p342=self.conv137(D14)
        elif  num24==38:
            p342=self.conv138(D14)
        elif  num24==39:
            p342=self.conv139(D14)
        elif  num24==40:
            p342=self.conv140(D14)
        elif  num24==41:
            p342=self.conv141(D14)
        elif  num24==42:
            p342=self.conv142(D14)
        elif  num24==43:
            p342=self.conv143(D14)
        elif  num24==44:
            p342=self.conv144(D14)
        elif  num24==45:
            p342=self.conv145(D14)
        elif  num24==46:
            p342=self.conv146(D14)
        elif  num24==47:
            p342=self.conv147(D14)
        elif  num24==48:
            p342=self.conv148(D14)
        elif  num24==49:
            p342=self.conv149(D14)
        elif  num24==50:
            p342=self.conv150(D14)
        elif  num24==51:
            p342=self.conv151(D14)
        elif  num24==52:
            p342=self.conv152(D14)
        elif  num24==53:
            p342=self.conv153(D14)
        elif  num24==54:
            p342=self.conv154(D14)
        elif  num24==55:
            p342=self.conv155(D14)
        elif  num24==56:
            p342=self.conv156(D14)
        elif  num24==57:
            p342=self.conv157(D14)
        elif  num24==58:
            p342=self.conv158(D14)
        elif  num24==59:
            p342=self.conv159(D14)
        elif  num24==60:
            p342=self.conv160(D14)
        elif  num24==61:
            p342=self.conv161(D14)
        elif  num24==62:
            p342=self.conv162(D14)
        elif  num24==63:
            p342=self.conv163(D14)
        elif  num24==64:
            p342=self.conv164(D14)
        elif  num24==65:
            p342=self.conv165(D14)
        elif  num24==66:
            p342=self.conv166(D14)
        elif  num24==67:
            p342=self.conv167(D14)
        elif  num24==68:
            p342=self.conv168(D14)
        elif  num24==69:
            p342=self.conv169(D14)
        elif  num24==70:
            p342=self.conv170(D14)
        elif  num24==71:
            p342=self.conv171(D14)
        elif  num24==72:
            p342=self.conv172(D14)
        elif  num24==73:
            p342=self.conv173(D14)
        elif  num24==74:
            p342=self.conv174(D14)
        elif  num24==75:
            p342=self.conv175(D14)
        elif  num24==76:
            p342=self.conv176(D14)
        elif  num24==77:
            p342=self.conv177(D14)
        elif  num24==78:
            p342=self.conv178(D14)
        elif  num24==79:
            p342=self.conv179(D14)
        elif  num24==80:
            p342=self.conv180(D14)
        elif  num24==81:
            p342=self.conv181(D14)
        elif  num24==82:
            p342=self.conv182(D14)
        elif  num24==83:
            p342=self.conv183(D14)
        elif  num24==84:
            p342=self.conv184(D14)
        elif  num24==85:
            p342=self.conv185(D14)
        elif  num24==86:
            p342=self.conv186(D14)
        elif  num24==87:
            p342=self.conv187(D14)
        elif  num24==88:
            p342=self.conv188(D14)
        elif  num24==89:
            p342=self.conv189(D14)    
        elif  num24==90:
            p342=self.conv190(D14)
        elif  num24==91:
            p342=self.conv191(D14)
        elif  num24==92:
            p342=self.conv192(D14)
        elif  num24==93:
            p342=self.conv193(D14)
        elif  num24==94:
            p342=self.conv194(D14)
        elif  num24==95:
            p342=self.conv195(D14)
        elif  num24==96:
            p342=self.conv196(D14)
        elif  num24==97:
            p342=self.conv197(D14)
        elif  num24==98:
            p342=self.conv198(D14)
        elif  num24==99:
            p342=self.conv199(D14) 
        elif  num24==100:
            p342=self.conv1100(D14)
        elif  num24==101:
            p342=self.conv1101(D14)
        elif  num24==102:
            p342=self.conv1102(D14)
        elif  num24==103:
            p342=self.conv1103(D14)
        elif  num24==104:
            p342=self.conv1104(D14)
        elif  num24==105:
            p342=self.conv1105(D14)
        elif  num24==106:
            p342=self.conv1106(D14)
        elif  num24==107:
            p342=self.conv1107(D14)
        elif  num24==108:
            p342=self.conv1108(D14)
        elif  num24==109:
            p342=self.conv1109(D14)
        elif  num24==110:
            p342=self.conv1110(D14)
        elif  num24==111:
            p342=self.conv1111(D14)
        elif  num24==112:
            p342=self.conv1112(D14)
        elif  num24==113:
            p342=self.conv1113(D14)
        elif  num24==114:
            p342=self.conv1114(D14)
        elif  num24==115:
            p342=self.conv1115(D14)
        elif  num24==116:
            p342=self.conv1116(D14)
        elif  num24==117:
            p342=self.conv1117(D14)
        elif  num24==118:
            p342=self.conv1118(D14)
        elif  num24==119:
            p342=self.conv1119(D14) 
        elif  num24==120:
            p342=self.conv1120(D14)
        elif  num24==121:
            p342=self.conv1121(D14)
        elif  num24==122:
            p342=self.conv1122(D14)
        elif  num24==123:
            p342=self.conv1123(D14)
        elif  num24==124:
            p342=self.conv1124(D14)
        elif  num24==125:
            p342=self.conv1125(D14)
        elif  num24==126:
            p342=self.conv1126(D14)
        elif  num24==127:
            p342=self.conv1127(D14)
        elif  num24==128:
            p342=self.conv1128(D14)
        elif  num24==129:
            p342=self.conv1129(D14) 
        elif  num24==130:
            p342=self.conv1130(D14)
        elif  num24==131:
            p342=self.conv1131(D14)
        elif  num24==132:
            p342=self.conv1132(D14)
        elif  num24==133:
            p342=self.conv1133(D14)
        elif  num24==134:
            p342=self.conv1134(D14)
        elif  num24==135:
            p342=self.conv1135(D14)
        elif  num24==136:
            p342=self.conv1136(D14)
        elif  num24==137:
            p342=self.conv1137(D14)
        elif  num24==138:
            p342=self.conv1138(D14)
        elif  num24==139:
            p342=self.conv1139(D14)
        elif  num24==140:
            p342=self.conv1140(D14)
        elif  num24==141:
            p342=self.conv1141(D14)
        elif  num24==142:
            p342=self.conv1142(D14)
        elif  num24==143:
            p342=self.conv1143(D14)
        elif  num24==144:
            p342=self.conv1144(D14)
        elif  num24==145:
            p342=self.conv1145(D14)
        elif  num24==146:
            p342=self.conv1146(D14)
        elif  num24==147:
            p342=self.conv1147(D14)
        elif  num24==148:
            p342=self.conv1148(D14)
        elif  num24==149:
            p342=self.conv1149(D14) 
        elif  num24==150:
            p342=self.conv1150(D14)
        elif  num24==151:
            p342=self.conv1151(D14)
        elif  num24==152:
            p342=self.conv1152(D14)
        elif  num24==153:
            p342=self.conv1153(D14)
        elif  num24==154:
            p342=self.conv1154(D14)
        elif  num24==155:
            p342=self.conv1155(D14)
        elif  num24==156:
            p342=self.conv1156(D14)
        elif  num24==157:
            p342=self.conv1157(D14)
        elif  num24==158:
            p342=self.conv1158(D14)
        elif  num24==159:
            p342=self.conv1159(D14) 
        elif  num24==160:
            p342=self.conv1160(D14)
        elif  num24==161:
            p342=self.conv1161(D14)
        elif  num24==162:
            p342=self.conv1162(D14)
        elif  num24==163:
            p342=self.conv1163(D14)
        elif  num24==164:
            p342=self.conv1164(D14)
        elif  num24==165:
            p342=self.conv1165(D14)
        elif  num24==166:
            p342=self.conv1166(D14)
        elif  num24==167:
            p342=self.conv1167(D14)
        elif  num24==168:
            p342=self.conv1168(D14)
        elif  num24==169:
            p342=self.conv1169(D14) 
        elif  num24==170:
            p342=self.conv1170(D14)
        elif  num24==171:
            p342=self.conv1171(D14)
        elif  num24==172:
            p342=self.conv1172(D14)
        elif  num24==173:
            p342=self.conv1173(D14)
        elif  num24==174:
            p342=self.conv1174(D14)
        elif  num24==175:
            p342=self.conv1175(D14)
        elif  num24==176:
            p342=self.conv1176(D14)
        elif  num24==177:
            p342=self.conv1177(D14)
        elif  num24==178:
            p342=self.conv1178(D14)
        elif  num24==179:
            p342=self.conv1179(D14)                                                                                              
        elif  num24==180:
            p342=self.conv1180(D14)
        elif  num24==181:
            p342=self.conv1181(D14)
        elif  num24==182:
            p342=self.conv1182(D14)
        elif  num24==183:
            p342=self.conv1183(D14)
        elif  num24==184:
            p342=self.conv1184(D14)
        elif  num24==185:
            p342=self.conv1185(D14)
        elif  num24==186:
            p342=self.conv1186(D14)
        elif  num24==187:
            p342=self.conv1187(D14)
        elif  num24==188:
            p342=self.conv1188(D14)
        elif  num24==189:
            p342=self.conv1189(D14) 
        elif  num24==190:
            p342=self.conv1190(D14)
        elif  num24==191:
            p342=self.conv1191(D14)
        elif  num24==192:
            p342=self.conv1192(D14)
        elif  num24==193:
            p342=self.conv1193(D14)
        elif  num24==194:
            p342=self.conv1194(D14)
        elif  num24==195:
            p342=self.conv1195(D14)
        elif  num24==196:
            p342=self.conv1196(D14)
        elif  num24==197:
            p342=self.conv1197(D14)
        elif  num24==198:
            p342=self.conv1198(D14)
        elif  num24==199:
            p342=self.conv1199(D14)
        elif  num24==200:
            p342=self.conv1200(D14)
        elif  num24==201:
            p342=self.conv1201(D14)
        elif  num24==202:
            p342=self.conv1202(D14)
        elif  num24==203:
            p342=self.conv1203(D14)
        elif  num24==204:
            p342=self.conv1204(D14)
        elif  num24==205:
            p342=self.conv1205(D14)
        elif  num24==206:
            p342=self.conv1206(D14)
        elif  num24==207:
            p342=self.conv1207(D14)
        elif  num24==208:
            p342=self.conv1208(D14)
        elif  num24==209:
            p342=self.conv1209(D14)
        elif  num24==210:
            p342=self.conv1210(D14)
        elif  num24==211:
            p342=self.conv1211(D14)
        elif  num24==212:
            p342=self.conv1212(D14)
        elif  num24==213:
            p342=self.conv1213(D14)
        elif  num24==214:
            p342=self.conv1214(D14)
        elif  num24==215:
            p342=self.conv1215(D14)
        elif  num24==216:
            p342=self.conv1216(D14)
        elif  num24==217:
            p342=self.conv1217(D14)
        elif  num24==218:
            p342=self.conv1218(D14)
        elif  num24==219:
            p342=self.conv1219(D14)
        elif  num24==220:
            p342=self.conv1220(D14)
        elif  num24==221:
            p342=self.conv1221(D14)
        elif  num24==222:
            p342=self.conv1222(D14)
        elif  num24==223:
            p342=self.conv1223(D14)
        elif  num24==224:
            p342=self.conv1224(D14)
        elif  num24==225:
            p342=self.conv1225(D14)
        elif  num24==226:
            p342=self.conv1226(D14)
        elif  num24==227:
            p342=self.conv1227(D14)
        elif  num24==228:
            p342=self.conv1228(D14)
        elif  num24==229:
            p342=self.conv1229(D14)
        elif  num24==230:
            p342=self.conv1230(D14)
        elif  num24==231:
            p342=self.conv1231(D14)
        elif  num24==232:
            p342=self.conv1232(D14)
        elif  num24==233:
            p342=self.conv1233(D14)
        elif  num24==234:
            p342=self.conv1234(D14)
        elif  num24==235:
            p342=self.conv1235(D14)
        elif  num24==236:
            p342=self.conv1236(D14)
        elif  num24==237:
            p342=self.conv1237(D14)
        elif  num24==238:
            p342=self.conv1238(D14)
        elif  num24==239:
            p342=self.conv1239(D14) 
        elif  num24==240:
            p342=self.conv1240(D14)
        elif  num24==241:
            p342=self.conv1241(D14)
        elif  num24==242:
            p342=self.conv1242(D14)
        elif  num24==243:
            p342=self.conv1243(D14)
        elif  num24==244:
            p342=self.conv1244(D14)
        elif  num24==245:
            p342=self.conv1245(D14)
        elif  num24==246:
            p342=self.conv1246(D14)
        elif  num24==247:
            p342=self.conv1247(D14)
        elif  num24==248:
            p342=self.conv1248(D14)
        elif  num24==249:
            p342=self.conv1249(D14)
        elif  num24==250:
            p342=self.conv1250(D14)
        elif  num24==251:
            p342=self.conv1251(D14)
        elif  num24==252:
            p342=self.conv1252(D14)
        elif  num24==253:
            p342=self.conv1253(D14)
        elif  num24==254:
            p342=self.conv1254(D14)
        elif  num24==255:
            p342=self.conv1255(D14)
        elif  num24==256:
            p342=self.conv1256(D14)
        elif  num24==257:
            p342=self.conv1257(D14)
        elif  num24==258:
            p342=self.conv1258(D14)
        elif  num24==259:
            p342=self.conv1259(D14)
        elif  num24==260:
            p342=self.conv1260(D14)
        elif  num24==261:
            p342=self.conv1261(D14)
        elif  num24==262:
            p342=self.conv1262(D14)
        elif  num24==263:
            p342=self.conv1263(D14)
        elif  num24==264:
            p342=self.conv1264(D14)
        elif  num24==265:
            p342=self.conv1265(D14)
        elif  num24==266:
            p342=self.conv1266(D14)
        elif  num24==267:
            p342=self.conv1267(D14)
        elif  num24==268:
            p342=self.conv1268(D14)
        elif  num24==269:
            p342=self.conv1269(D14) 
        elif  num24==270:
            p342=self.conv1270(D14)
        elif  num24==271:
            p342=self.conv1271(D14)
        elif  num24==272:
            p342=self.conv1272(D14)
        elif  num24==273:
            p342=self.conv1273(D14)
        elif  num24==274:
            p342=self.conv1274(D14)
        elif  num24==275:
            p342=self.conv1275(D14)
        elif  num24==276:
            p342=self.conv1276(D14)
        elif  num24==277:
            p342=self.conv1277(D14)
        elif  num24==278:
            p342=self.conv1278(D14)
        elif  num24==279:
            p342=self.conv1279(D14)
        elif  num24==280:
            p342=self.conv1280(D14)
        elif  num24==281:
            p342=self.conv1281(D14)
        elif  num24==282:
            p342=self.conv1282(D14)
        elif  num24==283:
            p342=self.conv1283(D14)
        elif  num24==284:
            p342=self.conv1284(D14)
        elif  num24==285:
            p342=self.conv1285(D14)
        elif  num24==286:
            p342=self.conv1286(D14)
        elif  num24==287:
            p342=self.conv1287(D14)
        elif  num24==288:
            p342=self.conv1288(D14)
        elif  num24==289:
            p342=self.conv1289(D14) 
        elif  num24==290:
            p342=self.conv1290(D14)
        elif  num24==291:
            p342=self.conv1291(D14)
        elif  num24==292:
            p342=self.conv1292(D14)
        elif  num24==293:
            p342=self.conv1293(D14)
        elif  num24==294:
            p342=self.conv1294(D14)
        elif  num24==295:
            p342=self.conv1295(D14)
        elif  num24==296:
            p342=self.conv1296(D14)
        elif  num24==297:
            p342=self.conv1297(D14)
        elif  num24==298:
            p342=self.conv1298(D14)
        elif  num24==299:
            p342=self.conv1299(D14)
        elif  num24==300:
            p342=self.conv1300(D14)
        elif  num24==301:
            p342=self.conv1301(D14)
        elif  num24==302:
            p342=self.conv1302(D14)
        elif  num24==303:
            p342=self.conv1303(D14)
        elif  num24==304:
            p342=self.conv1304(D14)
        elif  num24==305:
            p342=self.conv1305(D14)
        elif  num24==306:
            p342=self.conv1306(D14)
        elif  num24==307:
            p342=self.conv1307(D14)
        elif  num24==308:
            p342=self.conv1308(D14)
        elif  num24==309:
            p342=self.conv1309(D14) 
        elif  num24==310:
            p342=self.conv1310(D14)
        elif  num24==311:
            p342=self.conv1311(D14)
        elif  num24==312:
            p342=self.conv1312(D14)
        elif  num24==313:
            p342=self.conv1313(D14)
        elif  num24==314:
            p342=self.conv1314(D14)
        elif  num24==315:
            p342=self.conv1315(D14)
        elif  num24==316:
            p342=self.conv1316(D14)
        elif  num24==317:
            p342=self.conv1317(D14)
        elif  num24==318:
            p342=self.conv1318(D14)
        elif  num24==319:
            p342=self.conv1319(D14)
        elif  num24==320:
            p342=self.conv1320(D14)
        elif  num24==321:
            p342=self.conv1321(D14)
        elif  num24==322:
            p342=self.conv1322(D14)
        elif  num24==323:
            p342=self.conv1323(D14)
        elif  num24==324:
            p342=self.conv1324(D14)
        elif  num24==325:
            p342=self.conv1325(D14)
        elif  num24==326:
            p342=self.conv1326(D14)
        elif  num24==327:
            p342=self.conv1327(D14)
        elif  num24==328:
            p342=self.conv1328(D14)
        elif  num24==329:
            p342=self.conv1329(D14)
        elif  num24==330:
            p342=self.conv1330(D14)
        elif  num24==331:
            p342=self.conv1331(D14)
        elif  num24==332:
            p342=self.conv1332(D14)
        elif  num24==333:
            p342=self.conv1333(D14)
        elif  num24==334:
            p342=self.conv1334(D14)
        elif  num24==335:
            p342=self.conv1335(D14)
        elif  num24==336:
            p342=self.conv1336(D14)
        elif  num24==337:
            p342=self.conv1337(D14)
        elif  num24==338:
            p342=self.conv1338(D14)
        elif  num24==339:
            p342=self.conv1339(D14)
        elif  num24==340:
            p342=self.conv1340(D14)
        elif  num24==341:
            p342=self.conv1341(D14)
        elif  num24==342:
            p342=self.conv1342(D14)
        elif  num24==343:
            p342=self.conv1343(D14)
        elif  num24==344:
            p342=self.conv1344(D14)
        elif  num24==345:
            p342=self.conv1345(D14)
        elif  num24==346:
            p342=self.conv1346(D14)
        elif  num24==347:
            p342=self.conv1347(D14)
        elif  num24==348:
            p342=self.conv1348(D14)
        elif  num24==349:
            p342=self.conv1349(D14)
        elif  num24==350:
            p342=self.conv1350(D14)
        elif  num24==351:
            p342=self.conv1351(D14)
        elif  num24==352:
            p342=self.conv1352(D14)
        elif  num24==353:
            p342=self.conv1335(D14)
        elif  num24==354:
            p342=self.conv1354(D14)
        elif  num24==355:
            p342=self.conv1355(D14)
        elif  num24==356:
            p342=self.conv1356(D14)
        elif  num24==357:
            p342=self.conv1357(D14)
        elif  num24==358:
            p342=self.conv1358(D14)
        elif  num24==359:
            p342=self.conv1359(D14) 
        elif  num24==360:
            p342=self.conv1360(D14)
        elif  num24==361:
            p342=self.conv1361(D14)
        elif  num24==362:
            p342=self.conv1362(D14)
        elif  num24==363:
            p342=self.conv1363(D14)
        elif  num24==364:
            p342=self.conv1364(D14)
        elif  num24==365:
            p342=self.conv1365(D14)
        elif  num24==366:
            p342=self.conv1366(D14)
        elif  num24==367:
            p342=self.conv1367(D14)
        elif  num24==368:
            p342=self.conv1368(D14)
        elif  num24==369:
            p342=self.conv1369(D14) 
        elif  num24==370:
            p342=self.conv1370(D14)
        elif  num24==371:
            p342=self.conv1371(D14)
        elif  num24==372:
            p342=self.conv1372(D14)
        elif  num24==373:
            p342=self.conv1373(D14)
        elif  num24==374:
            p342=self.conv1374(D14)
        elif  num24==375:
            p342=self.conv1375(D14)
        elif  num24==376:
            p342=self.conv1376(D14)
        elif  num24==377:
            p342=self.conv1377(D14)
        elif  num24==378:
            p342=self.conv1378(D14)
        elif  num24==379:
            p342=self.conv1379(D14) 
        elif  num24==380:
            p342=self.conv1380(D14)
        elif  num24==381:
            p342=self.conv1381(D14)
        elif  num24==382:
            p342=self.conv1382(D14)
        elif  num24==383:
            p342=self.conv1383(D14)
        elif  num24==384:
            p342=self.conv1384(D14)
        elif  num24==385:
            p342=self.conv1385(D14)
        elif  num24==386:
            p342=self.conv1386(D14)
        elif  num24==387:
            p342=self.conv1387(D14)
        elif  num24==388:
            p342=self.conv1388(D14)
        elif  num24==389:
            p342=self.conv1389(D14) 
        elif  num24==390:
            p342=self.conv1390(D14)
        elif  num24==391:
            p342=self.conv1391(D14)
        elif  num24==392:
            p342=self.conv1392(D14)
        elif  num24==393:
            p342=self.conv1393(D14)
        elif  num24==394:
            p342=self.conv1394(D14)
        elif  num24==395:
            p342=self.conv1395(D14)
        elif  num24==396:
            p342=self.conv1396(D14)
        elif  num24==397:
            p342=self.conv1397(D14)
        elif  num24==398:
            p342=self.conv1398(D14)
        elif  num24==399:
            p342=self.conv1399(D14)
        elif  num24==400:
            p342=self.conv1400(D14)
        elif  num24==401:
            p342=self.conv1401(D14)
        elif  num24==402:
            p342=self.conv1402(D14)
        elif  num24==403:
            p342=self.conv1403(D14)
        elif  num24==404:
            p342=self.conv1404(D14)
        elif  num24==405:
            p342=self.conv1405(D14)
        elif  num24==406:
            p342=self.conv1406(D14)
        elif  num24==407:
            p342=self.conv1407(D14)
        elif  num24==408:
            p342=self.conv1408(D14)
        elif  num24==409:
            p342=self.conv1409(D14)
        elif  num24==410:
            p342=self.conv1410(D14)
        elif  num24==411:
            p342=self.conv1411(D14)
        elif  num24==412:
            p342=self.conv1412(D14)
        elif  num24==413:
            p342=self.conv1413(D14)
        elif  num24==414:
            p342=self.conv1414(D14)
        elif  num24==415:
            p342=self.conv145(D14)
        elif  num24==416:
            p342=self.conv1416(D14)
        elif  num24==417:
            p342=self.conv1417(D14)
        elif  num24==418:
            p342=self.conv1418(D14)
        elif  num24==419:
            p342=self.conv1419(D14) 
        elif  num24==420:
            p342=self.conv1420(D14)
        elif  num24==421:
            p342=self.conv1421(D14)
        elif  num24==422:
            p342=self.conv1422(D14)
        elif  num24==423:
            p342=self.conv1423(D14)
        elif  num24==424:
            p342=self.conv1424(D14)
        elif  num24==425:
            p342=self.conv1425(D14)
        elif  num24==426:
            p342=self.conv1426(D14)
        elif  num24==427:
            p342=self.conv1427(D14)
        elif  num24==428:
            p342=self.conv1428(D14)
        elif  num24==429:
            p342=self.conv1429(D14) 
        elif  num24==430:
            p342=self.conv1430(D14)
        elif  num24==431:
            p342=self.conv1431(D14)
        elif  num24==432:
            p342=self.conv1432(D14)
        elif  num24==433:
            p342=self.conv1433(D14)
        elif  num24==434:
            p342=self.conv1434(D14)
        elif  num24==435:
            p342=self.conv1435(D14)
        elif  num24==436:
            p342=self.conv1436(D14)
        elif  num24==437:
            p342=self.conv1437(D14)
        elif  num24==438:
            p342=self.conv1438(D14)
        elif  num24==439:
            p342=self.conv1439(D14)
        elif  num24==440:
            p342=self.conv1440(D14)
        elif  num24==441:
            p342=self.conv1441(D14)
        elif  num24==442:
            p342=self.conv1442(D14)
        elif  num24==443:
            p342=self.conv1443(D14)
        elif  num24==444:
            p342=self.conv1444(D14)
        elif  num24==445:
            p342=self.conv1445(D14)
        elif  num24==446:
            p342=self.conv1446(D14)
        elif  num24==447:
            p342=self.conv1447(D14)
        elif  num24==448:
            p342=self.conv1448(D14)
        elif  num24==449:
            p342=self.conv1449(D14)
        elif  num24==450:
            p342=self.conv1450(D14)
        elif  num24==451:
            p342=self.conv1451(D14)
        elif  num24==452:
            p342=self.conv1452(D14)
        elif  num24==453:
            p342=self.conv1453(D14)
        elif  num24==454:
            p342=self.conv1454(D14)
        elif  num24==455:
            p342=self.conv1455(D14)
        elif  num24==456:
            p342=self.conv1456(D14)
        elif  num24==457:
            p342=self.conv1457(D14)
        elif  num24==458:
            p342=self.conv1458(D14)
        elif  num24==459:
            p342=self.conv1459(D14)
        elif  num24==460:
            p342=self.conv1460(D14)
        elif  num24==461:
            p342=self.conv1461(D14)
        elif  num24==462:
            p342=self.conv1462(D14)
        elif  num24==463:
            p342=self.conv1463(D14)
        elif  num24==464:
            p342=self.conv1464(D14)
        elif  num24==465:
            p342=self.conv1465(D14)
        elif  num24==466:
            p342=self.conv1466(D14)
        elif  num24==467:
            p342=self.conv1467(D14)
        elif  num24==468:
            p342=self.conv1468(D14)
        elif  num24==469:
            p342=self.conv1469(D14) 
        elif  num24==470:
            p342=self.conv1470(D14)
        elif  num24==471:
            p342=self.conv1471(D14)
        elif  num24==472:
            p342=self.conv1472(D14)
        elif  num24==473:
            p342=self.conv1473(D14)
        elif  num24==474:
            p342=self.conv1474(D14)
        elif  num24==475:
            p342=self.conv1475(D14)
        elif  num24==476:
            p342=self.conv1476(D14)
        elif  num24==477:
            p342=self.conv1477(D14)
        elif  num24==478:
            p342=self.conv1478(D14)
        elif  num24==479:
            p342=self.conv1479(D14)
        elif  num24==480:
            p342=self.conv1480(D14)
        elif  num24==481:
            p342=self.conv1481(D14)
        elif  num24==482:
            p342=self.conv1482(D14)
        elif  num24==483:
            p342=self.conv1483(D14)
        elif  num24==484:
            p342=self.conv1484(D14)
        elif  num24==485:
            p342=self.conv1485(D14)
        elif  num24==486:
            p342=self.conv1486(D14)
        elif  num24==487:
            p342=self.conv1487(D14)
        elif  num24==488:
            p342=self.conv1488(D14)
        elif  num24==489:
            p342=self.conv1489(D14)
        elif  num24==490:
            p342=self.conv1490(D14)
        elif  num24==491:
            p342=self.conv1491(D14)
        elif  num24==492:
            p342=self.conv1492(D14)
        elif  num24==493:
            p342=self.conv1493(D14)
        elif  num24==494:
            p342=self.conv1494(D14)
        elif  num24==495:
            p342=self.conv1495(D14)
        elif  num24==496:
            p342=self.conv1496(D14)
        elif  num24==497:
            p342=self.conv1497(D14)
        elif  num24==498:
            p342=self.conv1498(D14)
        elif  num24==499:
            p342=self.conv1499(D14)
        elif  num24==500:
            p342=self.conv1500(D14)
        elif  num24==501:
            p342=self.conv1501(D14)
        elif  num24==502:
            p342=self.conv1502(D14)
        elif  num24==503:
            p342=self.conv1503(D14)
        elif  num24==504:
            p342=self.conv1504(D14)
        elif  num24==505:
            p342=self.conv1505(D14)
        elif  num24==506:
            p342=self.conv1506(D14)
        elif  num24==507:
            p342=self.conv1507(D14)
        elif  num24==508:
            p342=self.conv1508(D14)
        elif  num24==509:
            p342=self.conv1509(D14)
        elif  num24==510:
            p342=self.conv1510(D14)
        elif  num24==511:
            p342=self.conv1511(D14)
        elif  num24==512:
            p342=self.conv1512(D14)
        elif  num24==513:
            p342=self.conv1513(D14)
        elif  num24==514:
            p342=self.conv1514(D14)
        elif  num24==515:
            p342=self.conv1515(D14)
        elif  num24==516:
            p342=self.conv1516(D14)
        elif  num24==517:
            p342=self.conv1517(D14)
        elif  num24==518:
            p342=self.conv1518(D14)
        elif  num24==519:
            p342=self.conv1519(D14)
        elif  num24==520:
            p342=self.conv1520(D14)
        elif  num24==521:
            p342=self.conv1521(D14)
        elif  num24==522:
            p342=self.conv1522(D14)
        elif  num24==523:
            p342=self.conv1523(D14)
        elif  num24==524:
            p342=self.conv1524(D14)
        elif  num24==525:
            p342=self.conv1525(D14)
        elif  num24==526:
            p342=self.conv1526(D14)
        elif  num24==527:
            p342=self.conv1527(D14)
        elif  num24==528:
            p342=self.conv1528(D14)
        elif  num24==529:
            p342=self.conv1529(D14)
        elif  num24==530:
            p342=self.conv1530(D14)
        elif  num24==531:
            p342=self.conv1531(D14)
        elif  num24==532:
            p342=self.conv1532(D14)
        elif  num24==533:
            p342=self.conv1533(D14)
        elif  num24==534:
            p342=self.conv1534(D14)
        elif  num24==535:
            p342=self.conv1535(D14)
        elif  num24==536:
            p342=self.conv1536(D14)
        elif  num24==537:
            p342=self.conv1537(D14)
        elif  num24==538:
            p342=self.conv1538(D14)
        elif  num24==539:
            p342=self.conv1539(D14)
        elif  num24==540:
            p342=self.conv1540(D14)
        elif  num24==541:
            p342=self.conv1541(D14)
        elif  num24==542:
            p342=self.conv1542(D14)
        elif  num24==543:
            p342=self.conv1543(D14)
        elif  num24==544:
            p342=self.conv1544(D14)
        elif  num24==545:
            p342=self.conv1545(D14)
        elif  num24==546:
            p342=self.conv1546(D14)
        elif  num24==547:
            p342=self.conv1547(D14)
        elif  num24==548:
            p342=self.conv1548(D14)
        elif  num24==549:
            p342=self.conv1549(D14) 
        elif  num24==550:
            p342=self.conv1550(D14)
        elif  num24==551:
            p342=self.conv1551(D14)
        elif  num24==552:
            p342=self.conv1552(D14)
        elif  num24==553:
            p342=self.conv1553(D14)
        elif  num24==554:
            p342=self.conv1554(D14)
        elif  num24==555:
            p342=self.conv1555(D14)
        elif  num24==556:
            p342=self.conv1556(D14)
        elif  num24==557:
            p342=self.conv1557(D14)
        elif  num24==558:
            p342=self.conv1558(D14)
        elif  num24==559:
            p342=self.conv1559(D14)
        elif  num24==560:
            p342=self.conv1560(D14)
        elif  num24==561:
            p342=self.conv1561(D14)
        elif  num24==562:
            p342=self.conv1562(D14)
        elif  num24==563:
            p342=self.conv1563(D14)
        elif  num24==564:
            p342=self.conv1564(D14)
        elif  num24==565:
            p342=self.conv1565(D14)
        elif  num24==566:
            p342=self.conv1566(D14)
        elif  num24==567:
            p342=self.conv1567(D14)
        elif  num24==568:
            p342=self.conv1568(D14)
        elif  num24==569:
            p342=self.conv1569(D14) 
        elif  num24==570:
            p342=self.conv1570(D14)
        elif  num24==571:
            p342=self.conv1571(D14)
        elif  num24==572:
            p342=self.conv1572(D14)
        elif  num24==573:
            p342=self.conv1573(D14)
        elif  num24==574:
            p342=self.conv1574(D14)
        elif  num24==575:
            p342=self.conv1575(D14)
        elif  num24==576:
            p342=self.conv1576(D14)
        elif  num24==577:
            p342=self.conv1577(D14)
        elif  num24==578:
            p342=self.conv1578(D14)
        elif  num24==579:
            p342=self.conv1579(D14) 
        elif  num24==580:
            p342=self.conv1580(D14)
        elif  num24==581:
            p342=self.conv1581(D14)
        elif  num24==582:
            p342=self.conv1582(D14)
        elif  num24==583:
            p342=self.conv1583(D14)
        elif  num24==584:
            p342=self.conv1584(D14)
        elif  num24==585:
            p342=self.conv1585(D14)
        elif  num24==586:
            p342=self.conv1586(D14)
        elif  num24==587:
            p342=self.conv1587(D14)
        elif  num24==588:
            p342=self.conv1588(D14)
        elif  num24==589:
            p342=self.conv1589(D14)
        elif  num24==590:
            p342=self.conv1590(D14)
        elif  num24==591:
            p342=self.conv1591(D14)
        elif  num24==592:
            p342=self.conv1592(D14)
        elif  num24==593:
            p342=self.conv1593(D14)
        elif  num24==594:
            p342=self.conv1594(D14)
        elif  num24==595:
            p342=self.conv1595(D14)
        elif  num24==596:
            p342=self.conv1596(D14)
        elif  num24==597:
            p342=self.conv1597(D14)
        elif  num24==598:
            p342=self.conv1598(D14)
        elif  num24==599:
            p342=self.conv1599(D14)
        elif  num24==600:
            p342=self.conv1600(D14)
        elif  num24==601:
            p342=self.conv1601(D14)
        elif  num24==602:
            p342=self.conv1602(D14)
        elif  num24==603:
            p342=self.conv1603(D14)
        elif  num24==604:
            p342=self.conv1604(D14)
        elif  num24==605:
            p342=self.conv1605(D14)
        elif  num24==606:
            p342=self.conv1606(D14)
        elif  num24==607:
            p342=self.conv1607(D14)
        elif  num24==608:
            p342=self.conv1608(D14)
        elif  num24==609:
            p342=self.conv1609(D14)                                                                                                                         
        elif  num24==610:
            p342=self.conv1610(D14)
        elif  num24==611:
            p342=self.conv1611(D14)
        elif  num24==612:
            p342=self.conv1612(D14)
        elif  num24==613:
            p342=self.conv1613(D14)
        elif  num24==614:
            p342=self.conv1614(D14)
        elif  num24==615:
            p342=self.conv1615(D14)
        elif  num24==616:
            p342=self.conv1616(D14)
        elif  num24==617:
            p342=self.conv1617(D14)
        elif  num24==618:
            p342=self.conv1618(D14)
        elif  num24==619:
            p342=self.conv1619(D14)                                                                                                                          
        elif  num24==620:
            p342=self.conv1620(D14)
        elif  num24==621:
            p342=self.conv1621(D14)
        elif  num24==622:
            p342=self.conv1622(D14)
        elif  num24==623:
            p342=self.conv1623(D14)
        elif  num24==624:
            p342=self.conv1624(D14)
        elif  num24==625:
            p342=self.conv1625(D14)
        elif  num24==626:
            p342=self.conv1626(D14)
        elif  num24==627:
            p342=self.conv1627(D14)
        elif  num24==628:
            p342=self.conv1628(D14)
        elif  num24==629:
            p342=self.conv1629(D14)  
        elif  num24==630:
            p342=self.conv1630(D14)
        elif  num24==631:
            p342=self.conv1631(D14)
        elif  num24==632:
            p342=self.conv1632(D14)
        elif  num24==633:
            p342=self.conv1633(D14)
        elif  num24==634:
            p342=self.conv1634(D14)
        elif  num24==635:
            p342=self.conv1635(D14)
        elif  num24==636:
            p342=self.conv1636(D14)
        elif  num24==637:
            p342=self.conv1637(D14)
        elif  num24==638:
            p342=self.conv1638(D14)
        elif  num24==639:
            p342=self.conv1639(D14)                                                                                                                          
        elif  num24==640:
            p342=self.conv1640(D14)
        elif  num24==641:
            p342=self.conv1641(D14)
        elif  num24==642:
            p342=self.conv1642(D14)
        elif  num24==643:
            p342=self.conv1643(D14)
        elif  num24==644:
            p342=self.conv1644(D14)
        elif  num24==645:
            p342=self.conv1645(D14)
        elif  num24==646:
            p342=self.conv1646(D14)
        elif  num24==647:
            p342=self.conv1647(D14)
        elif  num24==648:
            p342=self.conv1648(D14)
        elif  num24==649:
            p342=self.conv1649(D14)                                                                                                                          
        elif  num24==650:
            p342=self.conv1650(D14)
        elif  num24==651:
            p342=self.conv1651(D14)
        elif  num24==652:
            p342=self.conv1652(D14)
        elif  num24==653:
            p342=self.conv1653(D14)
        elif  num24==654:
            p342=self.conv1654(D14)
        elif  num24==655:
            p342=self.conv1655(D14)
        elif  num24==656:
            p342=self.conv1656(D14)
        elif  num24==657:
            p342=self.conv1657(D14)
        elif  num24==658:
            p342=self.conv1658(D14)
        elif  num24==659:
            p342=self.conv1659(D14)
        elif  num24==660:
            p342=self.conv1660(D14)
        elif  num24==661:
            p342=self.conv1661(D14)
        elif  num24==662:
            p342=self.conv1662(D14)
        elif  num24==663:
            p342=self.conv1663(D14)
        elif  num24==664:
            p342=self.conv1664(D14)
        elif  num24==665:
            p342=self.conv1665(D14)
        elif  num24==666:
            p342=self.conv1666(D14)
        elif  num24==667:
            p342=self.conv1667(D14)
        elif  num24==668:
            p342=self.conv1668(D14)
        elif  num24==669:
            p342=self.conv1669(D14) 
        elif  num24==670:
            p342=self.conv1670(D14)
        elif  num24==671:
            p342=self.conv1671(D14)
        elif  num24==672:
            p342=self.conv1672(D14)
        elif  num24==673:
            p342=self.conv1673(D14)
        elif  num24==674:
            p342=self.conv1674(D14)
        elif  num24==675:
            p342=self.conv1675(D14)
        elif  num24==676:
            p342=self.conv1676(D14)
        elif  num24==677:
            p342=self.conv1677(D14)
        elif  num24==678:
            p342=self.conv1678(D14)
        elif  num24==679:
            p342=self.conv1679(D14)
        elif  num24==680:
            p342=self.conv1680(D14)
        elif  num24==681:
            p342=self.conv1681(D14)
        elif  num24==682:
            p342=self.conv1682(D14)
        elif  num24==683:
            p342=self.conv1683(D14)
        elif  num24==684:
            p342=self.conv1684(D14)
        elif  num24==685:
            p342=self.conv1685(D14)
        elif  num24==686:
            p342=self.conv1686(D14)
        elif  num24==687:
            p342=self.conv1687(D14)
        elif  num24==688:
            p342=self.conv1688(D14)
        elif  num24==689:
            p342=self.conv1689(D14)
        elif  num24==690:
            p342=self.conv1690(D14)
        elif  num24==691:
            p342=self.conv1691(D14)
        elif  num24==692:
            p342=self.conv1692(D14)
        elif  num24==693:
            p342=self.conv1693(D14)
        elif  num24==694:
            p342=self.conv1694(D14)
        elif  num24==695:
            p342=self.conv1695(D14)
        elif  num24==696:
            p342=self.conv1696(D14)
        elif  num24==697:
            p342=self.conv1697(D14)
        elif  num24==698:
            p342=self.conv1698(D14)
        elif  num24==699:
            p342=self.conv1699(D14)
        elif  num24==700:
            p342=self.conv1700(D14)
        elif  num24==701:
            p342=self.conv1701(D14)
        elif  num24==702:
            p342=self.conv1702(D14)
        elif  num24==703:
            p342=self.conv1703(D14)
        elif  num24==704:
            p342=self.conv1704(D14)
        elif  num24==705:
            p342=self.conv1705(D14)
        elif  num24==706:
            p342=self.conv1706(D14)
        elif  num24==707:
            p342=self.conv1707(D14)
        elif  num24==708:
            p342=self.conv1708(D14)
        elif  num24==709:
            p342=self.conv1709(D14)
        elif  num24==710:
            p342=self.conv1710(D14)
        elif  num24==711:
            p342=self.conv1711(D14)
        elif  num24==712:
            p342=self.conv1712(D14)
        elif  num24==713:
            p342=self.conv1713(D14)
        elif  num24==714:
            p342=self.conv1714(D14)
        elif  num24==715:
            p342=self.conv1715(D14)
        elif  num24==716:
            p342=self.conv1716(D14)
        elif  num24==717:
            p342=self.conv1717(D14)
        elif  num24==718:
            p342=self.conv1718(D14)
        elif  num24==719:
            p342=self.conv1719(D14)
        elif  num24==720:
            p342=self.conv1720(D14)
        elif  num24==721:
            p342=self.conv1721(D14)
        elif  num24==722:
            p342=self.conv1722(D14)
        elif  num24==723:
            p342=self.conv1723(D14)
        elif  num24==724:
            p342=self.conv1724(D14)
        elif  num24==725:
            p342=self.conv1725(D14)
        elif  num24==726:
            p342=self.conv1726(D14)
        elif  num24==727:
            p342=self.conv1727(D14)
        elif  num24==728:
            p342=self.conv1728(D14)
        elif  num24==729:
            p342=self.conv1729(D14)
        elif  num24==730:
            p342=self.conv1730(D14)
        elif  num24==731:
            p342=self.conv1731(D14)
        elif  num24==732:
            p342=self.conv1732(D14)
        elif  num24==733:
            p342=self.conv1733(D14)
        elif  num24==734:
            p342=self.conv1734(D14)
        elif  num24==735:
            p342=self.conv1735(D14)
        elif  num24==736:
            p342=self.conv1736(D14)
        elif  num24==737:
            p342=self.conv1737(D14)
        elif  num24==738:
            p342=self.conv1738(D14)
        elif  num24==739:
            p342=self.conv1739(D14)                                                                                                                          
        elif  num24==740:
            p342=self.conv1740(D14)
        elif  num24==741:
            p342=self.conv1741(D14)
        elif  num24==742:
            p342=self.conv1742(D14)
        elif  num24==743:
            p342=self.conv1743(D14)
        elif  num24==744:
            p342=self.conv1744(D14)
        elif  num24==745:
            p342=self.conv1745(D14)
        elif  num24==746:
            p342=self.conv1746(D14)
        elif  num24==747:
            p342=self.conv1747(D14)
        elif  num24==748:
            p342=self.conv1748(D14)
        elif  num24==749:
            p342=self.conv1749(D14)
        elif  num24==750:
            p342=self.conv1750(D14)
        elif  num24==751:
            p342=self.conv1751(D14)
        elif  num24==752:
            p342=self.conv1752(D14)
        elif  num24==753:
            p342=self.conv1753(D14)
        elif  num24==754:
            p342=self.conv1754(D14)
        elif  num24==755:
            p342=self.conv1755(D14)
        elif  num24==756:
            p342=self.conv1756(D14)
        elif  num24==757:
            p342=self.conv1757(D14)
        elif  num24==758:
            p342=self.conv1758(D14)
        elif  num24==759:
            p342=self.conv1759(D14)
        elif  num24==760:
            p342=self.conv1760(D14)
        elif  num24==761:
            p342=self.conv1761(D14)
        elif  num24==762:
            p342=self.conv1762(D14)
        elif  num24==763:
            p342=self.conv1763(D14)
        elif  num24==764:
            p342=self.conv1764(D14)
        elif  num24==765:
            p342=self.conv1765(D14)
        elif  num24==766:
            p342=self.conv1766(D14)
        elif  num24==767:
            p342=self.conv1767(D14)
        elif  num24==768:
            p342=self.conv1768(D14)
        elif  num24==769:
            p342=self.conv1769(D14) 
        elif  num24==770:
            p342=self.conv1770(D14)
        elif  num24==771:
            p342=self.conv1771(D14)
        elif  num24==772:
            p342=self.conv1772(D14)
        elif  num24==773:
            p342=self.conv1773(D14)
        elif  num24==774:
            p342=self.conv1774(D14)
        elif  num24==775:
            p342=self.conv1775(D14)
        elif  num24==776:
            p342=self.conv1776(D14)
        elif  num24==777:
            p342=self.conv1777(D14)
        elif  num24==778:
            p342=self.conv1778(D14)
        elif  num24==779:
            p342=self.conv1779(D14) 
        elif  num24==780:
            p342=self.conv1780(D14)
        elif  num24==781:
            p342=self.conv1781(D14)
        elif  num24==782:
            p342=self.conv1782(D14)
        elif  num24==783:
            p342=self.conv1783(D14)
        elif  num24==784:
            p342=self.conv1784(D14)
        elif  num24==785:
            p342=self.conv1785(D14)
        elif  num24==786:
            p342=self.conv1786(D14)
        elif  num24==787:
            p342=self.conv1787(D14)
        elif  num24==788:
            p342=self.conv1788(D14)
        elif  num24==789:
            p342=self.conv1789(D14) 
        elif  num24==790:
            p342=self.conv1790(D14)
        elif  num24==791:
            p342=self.conv1791(D14)
        elif  num24==792:
            p342=self.conv1792(D14)
        elif  num24==793:
            p342=self.conv1793(D14)
        elif  num24==794:
            p342=self.conv1794(D14)
        elif  num24==795:
            p342=self.conv1795(D14)
        elif  num24==796:
            p342=self.conv1796(D14)
        elif  num24==797:
            p342=self.conv1797(D14)
        elif  num24==798:
            p342=self.conv1798(D14)
        elif  num24==799:
            p342=self.conv1799(D14) 
        elif  num24==800:
            p342=self.conv1800(D14)
        elif  num24==801:
            p342=self.conv1801(D14)
        elif  num24==802:
            p342=self.conv1802(D14)
        elif  num24==803:
            p342=self.conv1803(D14)
        elif  num24==804:
            p342=self.conv1804(D14)
        elif  num24==805:
            p342=self.conv1805(D14)
        elif  num24==806:
            p342=self.conv1806(D14)
        elif  num24==807:
            p342=self.conv1807(D14)
        elif  num24==808:
            p342=self.conv1808(D14)
        elif  num24==809:
            p342=self.conv1809(D14)
        elif  num24==810:
            p342=self.conv1810(D14)
        elif  num24==811:
            p342=self.conv1811(D14)
        elif  num24==812:
            p342=self.conv1812(D14)
        elif  num24==813:
            p342=self.conv1813(D14)
        elif  num24==814:
            p342=self.conv1814(D14)
        elif  num24==815:
            p342=self.conv1815(D14)
        elif  num24==816:
            p342=self.conv1816(D14)
        elif  num24==817:
            p342=self.conv1817(D14)
        elif  num24==818:
            p342=self.conv1818(D14)
        elif  num24==819:
            p342=self.conv1819(D14)
        elif  num24==820:
            p342=self.conv1820(D14)
        elif  num24==821:
            p342=self.conv1821(D14)
        elif  num24==822:
            p342=self.conv1822(D14)
        elif  num24==823:
            p342=self.conv1823(D14)
        elif  num24==824:
            p342=self.conv1824(D14)
        elif  num24==825:
            p342=self.conv1825(D14)
        elif  num24==826:
            p342=self.conv1826(D14)
        elif  num24==827:
            p342=self.conv1827(D14)
        elif  num24==828:
            p342=self.conv1828(D14)
        elif  num24==829:
            p342=self.conv1829(D14)                                                                                                                          
        elif  num24==830:
            p342=self.conv1830(D14)
        elif  num24==831:
            p342=self.conv1831(D14)
        elif  num24==832:
            p342=self.conv1832(D14)
        elif  num24==833:
            p342=self.conv1833(D14)
        elif  num24==834:
            p342=self.conv1834(D14)
        elif  num24==835:
            p342=self.conv1835(D14)
        elif  num24==836:
            p342=self.conv1836(D14)
        elif  num24==837:
            p342=self.conv1837(D14)
        elif  num24==838:
            p342=self.conv1838(D14)
        elif  num24==839:
            p342=self.conv1839(D14)
        elif  num24==840:
            p342=self.conv1840(D14)
        elif  num24==841:
            p342=self.conv1841(D14)
        elif  num24==842:
            p342=self.conv1842(D14)
        elif  num24==843:
            p342=self.conv1843(D14)
        elif  num24==844:
            p342=self.conv1844(D14)
        elif  num24==845:
            p342=self.conv1845(D14)
        elif  num24==846:
            p342=self.conv1846(D14)
        elif  num24==847:
            p342=self.conv1847(D14)
        elif  num24==848:
            p342=self.conv1848(D14)
        elif  num24==849:
            p342=self.conv1849(D14)
        elif  num24==850:
            p342=self.conv1850(D14)
        elif  num24==851:
            p342=self.conv1851(D14)
        elif  num24==852:
            p342=self.conv1852(D14)
        elif  num24==853:
            p342=self.conv1853(D14)
        elif  num24==854:
            p342=self.conv1854(D14)
        elif  num24==855:
            p342=self.conv1855(D14)
        elif  num24==856:
            p342=self.conv1856(D14)
        elif  num24==857:
            p342=self.conv1857(D14)
        elif  num24==858:
            p342=self.conv1858(D14)
        elif  num24==859:
            p342=self.conv1859(D14)
        elif  num24==860:
            p342=self.conv1860(D14)
        elif  num24==861:
            p342=self.conv1861(D14)
        elif  num24==862:
            p342=self.conv1862(D14)
        elif  num24==863:
            p342=self.conv1863(D14)
        elif  num24==864:
            p342=self.conv1864(D14)
        elif  num24==865:
            p342=self.conv1865(D14)
        elif  num24==866:
            p342=self.conv1866(D14)
        elif  num24==867:
            p342=self.conv1867(D14)
        elif  num24==868:
            p342=self.conv1868(D14)
        elif  num24==869:
            p342=self.conv1869(D14) 
        elif  num24==870:
            p342=self.conv1870(D14)
        elif  num24==871:
            p342=self.conv1871(D14)
        elif  num24==872:
            p342=self.conv1872(D14)
        elif  num24==873:
            p342=self.conv1873(D14)
        elif  num24==874:
            p342=self.conv1874(D14)
        elif  num24==875:
            p342=self.conv1875(D14)
        elif  num24==876:
            p342=self.conv1876(D14)
        elif  num24==877:
            p342=self.conv1877(D14)
        elif  num24==878:
            p342=self.conv1878(D14)
        elif  num24==879:
            p342=self.conv1879(D14)
        elif  num24==880:
            p342=self.conv1880(D14)
        elif  num24==881:
            p342=self.conv1881(D14)
        elif  num24==882:
            p342=self.conv1882(D14)
        elif  num24==883:
            p342=self.conv1883(D14)
        elif  num24==884:
            p342=self.conv1884(D14)
        elif  num24==885:
            p342=self.conv1885(D14)
        elif  num24==886:
            p342=self.conv1886(D14)
        elif  num24==887:
            p342=self.conv1887(D14)
        elif  num24==888:
            p342=self.conv1888(D14)
        elif  num24==889:
            p342=self.conv1889(D14)  
        elif  num24==890:
            p342=self.conv1890(D14)
        elif  num24==891:
            p342=self.conv1891(D14)
        elif  num24==892:
            p342=self.conv1892(D14)
        elif  num24==893:
            p342=self.conv1893(D14)
        elif  num24==894:
            p342=self.conv1894(D14)
        elif  num24==895:
            p342=self.conv1895(D14)
        elif  num24==896:
            p342=self.conv1896(D14)
        elif  num24==897:
            p342=self.conv1897(D14)
        elif  num24==898:
            p342=self.conv1898(D14)
        elif  num24==899:
            p342=self.conv1899(D14)
        elif  num24==900:
            p342=self.conv1900(D14)
        elif  num24==901:
            p342=self.conv1901(D14)
        elif  num24==902:
            p342=self.conv1902(D14)
        elif  num24==903:
            p342=self.conv1903(D14)
        elif  num24==904:
            p342=self.conv1904(D14)
        elif  num24==905:
            p342=self.conv1905(D14)
        elif  num24==906:
            p342=self.conv1906(D14)
        elif  num24==907:
            p342=self.conv1907(D14)
        elif  num24==908:
            p342=self.conv1908(D14)
        elif  num24==909:
            p342=self.conv1909(D14)
        elif  num24==910:
            p342=self.conv1910(D14)
        elif  num24==911:
            p342=self.conv1911(D14)
        elif  num24==912:
            p342=self.conv1912(D14)
        elif  num24==913:
            p342=self.conv1913(D14)
        elif  num24==914:
            p342=self.conv1914(D14)
        elif  num24==915:
            p342=self.conv1915(D14)
        elif  num24==916:
            p342=self.conv1916(D14)
        elif  num24==917:
            p342=self.conv1917(D14)
        elif  num24==918:
            p342=self.conv1918(D14)
        elif  num24==919:
            p342=self.conv1919(D14)
        elif  num24==920:
            p342=self.conv1920(D14)
        elif  num24==921:
            p342=self.conv1921(D14)
        elif  num24==922:
            p342=self.conv1922(D14)
        elif  num24==923:
            p342=self.conv1923(D14)
        elif  num24==924:
            p342=self.conv1924(D14)
        elif  num24==925:
            p342=self.conv1925(D14)
        elif  num24==926:
            p342=self.conv1926(D14)
        elif  num24==927:
            p342=self.conv1927(D14)
        elif  num24==928:
            p342=self.conv1928(D14)
        elif  num24==929:
            p342=self.conv1929(D14)
        elif  num24==930:
            p342=self.conv1930(D14)
        elif  num24==931:
            p342=self.conv1931(D14)
        elif  num24==932:
            p342=self.conv1932(D14)
        elif  num24==933:
            p342=self.conv1933(D14)
        elif  num24==934:
            p342=self.conv1934(D14)
        elif  num24==935:
            p342=self.conv1935(D14)
        elif  num24==936:
            p342=self.conv1936(D14)
        elif  num24==937:
            p342=self.conv1937(D14)
        elif  num24==938:
            p342=self.conv1938(D14)
        elif  num24==939:
            p342=self.conv1939(D14) 
        elif  num24==940:
            p342=self.conv1940(D14)
        elif  num24==941:
            p342=self.conv1941(D14)
        elif  num24==942:
            p342=self.conv1942(D14)
        elif  num24==943:
            p342=self.conv1943(D14)
        elif  num24==944:
            p342=self.conv1944(D14)
        elif  num24==945:
            p342=self.conv1945(D14)
        elif  num24==946:
            p342=self.conv1946(D14)
        elif  num24==947:
            p342=self.conv1947(D14)
        elif  num24==948:
            p342=self.conv1948(D14)
        elif  num24==949:
            p342=self.conv1949(D14)                                                                                                                          
        elif  num24==950:
            p342=self.conv1950(D14)
        elif  num24==951:
            p342=self.conv1951(D14)
        elif  num24==952:
            p342=self.conv1952(D14)
        elif  num24==953:
            p342=self.conv1953(D14)
        elif  num24==954:
            p342=self.conv1954(D14)
        elif  num24==955:
            p342=self.conv1955(D14)
        elif  num24==956:
            p342=self.conv1956(D14)
        elif  num24==957:
            p342=self.conv1957(D14)
        elif  num24==958:
            p342=self.conv1958(D14)
        elif  num24==959:
            p342=self.conv1959(D14)
        elif  num24==960:
            p342=self.conv1960(D14)
        elif  num24==961:
            p342=self.conv1961(D14)
        elif  num24==962:
            p342=self.conv1962(D14)
        elif  num24==963:
            p342=self.conv1963(D14)
        elif  num24==964:
            p342=self.conv1964(D14)
        elif  num24==965:
            p342=self.conv1965(D14)
        elif  num24==966:
            p342=self.conv1966(D14)
        elif  num24==967:
            p342=self.conv1967(D14)
        elif  num24==968:
            p342=self.conv1968(D14)
        elif  num24==969:
            p342=self.conv1969(D14) 
        elif  num24==970:
            p342=self.conv1970(D14)
        elif  num24==971:
            p342=self.conv1971(D14)
        elif  num24==972:
            p342=self.conv1972(D14)
        elif  num24==973:
            p342=self.conv1973(D14)
        elif  num24==974:
            p342=self.conv1974(D14)
        elif  num24==975:
            p342=self.conv1975(D14)
        elif  num24==976:
            p342=self.conv1976(D14)
        elif  num24==977:
            p342=self.conv1977(D14)
        elif  num24==978:
            p342=self.conv1978(D14)
        elif  num24==979:
            p342=self.conv1979(D14) 
        elif  num24==980:
            p342=self.conv1980(D14)
        elif  num24==981:
            p342=self.conv1981(D14)
        elif  num24==982:
            p342=self.conv1982(D14)
        elif  num24==983:
            p342=self.conv1983(D14)
        elif  num24==984:
            p342=self.conv1984(D14)
        elif  num24==985:
            p342=self.conv1985(D14)
        elif  num24==986:
            p342=self.conv1986(D14)
        elif  num24==987:
            p342=self.conv1987(D14)
        elif  num24==988:
            p342=self.conv1988(D14)
        elif  num24==989:
            p342=self.conv1989(D14)
        elif  num24==990:
            p342=self.conv1990(D14)
        elif  num24==991:
            p342=self.conv1991(D14)
        elif  num24==992:
            p342=self.conv1992(D14)
        elif  num24==993:
            p342=self.conv1993(D14)
        elif  num24==994:
            p342=self.conv1994(D14)
        elif  num24==995:
            p342=self.conv1995(D14)
        elif  num24==996:
            p342=self.conv1996(D14)
        elif  num24==997:
            p342=self.conv1997(D14)
        elif  num24==998:
            p342=self.conv1998(D14)
        elif  num24==999:
            p342=self.conv1999(D14) 
        elif  num24==1000:
            p342=self.conv11000(D14)
        elif  num24==1001:
            p342=self.conv11001(D14)
        elif  num24==1002:
            p342=self.conv11002(D14)
        elif  num24==1003:
            p342=self.conv11003(D14)
        elif  num24==1004:
            p342=self.conv11004(D14)
        elif  num24==1005:
            p342=self.conv11005(D14)
        elif  num24==1006:
            p342=self.conv11006(D14)
        elif  num24==1007:
            p342=self.conv11007(D14)
        elif  num24==1008:
            p342=self.conv11008(D14)
        elif  num24==1009:
            p342=self.conv11009(D14) 
        elif  num24==1010:
            p342=self.conv11010(D14)
        elif  num24==1011:
            p342=self.conv11011(D14)
        elif  num24==1012:
            p342=self.conv11012(D14)
        elif  num24==1013:
            p342=self.conv11013(D14)
        elif  num24==1014:
            p342=self.conv11014(D14)
        elif  num24==1015:
            p342=self.conv11015(D14)
        elif  num24==1016:
            p342=self.conv11016(D14)
        elif  num24==1017:
            p342=self.conv11017(D14)
        elif  num24==1018:
            p342=self.conv11018(D14)
        elif  num24==1019:
            p342=self.conv11019(D14)
        elif  num24==1020:
            p342=self.conv11020(D14)
        elif  num24==1021:
            p342=self.conv11021(D14)
        elif  num24==1022:
            p342=self.conv11022(D14)
        elif  num24==1023:
            p342=self.conv11023(D14)
        elif  num24==1024:
            p342=self.conv11024(D14) 
        
        if num040==1:
            p340=self.conv11(B140)
        elif num040==2:
            p340=self.conv12(B140)
        elif num040==3:
            p340=self.conv13(B140)
        elif num040==4:
            p340=self.conv14(B140)
        elif num040==5:
            p340=self.conv15(B140)
        elif num040==6:
            p340=self.conv16(B140)
        elif num040==7:
            p340=self.conv17(B140)
        elif num040==8:
            p340=self.conv18(B140)
        elif num040==9:
            p340=self.conv19(B140)
        elif num040==10:
            p340=self.conv110(B140)
        elif num040==11:
            p340=self.conv111(B140)
        elif num040==12:
            p340=self.conv112(B140)
        elif num040==13:
            p340=self.conv113(B140)
        elif num040==14:
            p340=self.conv114(B140)
        elif num040==15:
            p340=self.conv115(B140)
        elif num040==16:
            p340=self.conv116(B140)
        elif num040==17:
            p340=self.conv117(B140)
        elif num040==18:
            p340=self.conv118(B140)
        elif num040==19:
            p340=self.conv119(B140)
        elif num040==20:
            p340=self.conv120(B140)
        elif num040==21:
            p340=self.conv121(B140)
        elif num040==22:
            p340=self.conv122(B140)
        elif num040==23:
            p340=self.conv123(B140)
        elif num040==24:
            p340=self.conv124(B140)
        elif num040==25:
            p340=self.conv125(B140)
        elif num040==26:
            p340=self.conv126(B140)
        elif num040==27:
            p340=self.conv127(B140)
        elif num040==28:
            p340=self.conv128(B140)
        elif num040==29:
            p340=self.conv129(B140)
        elif num040==30:
            p340=self.conv130(B140)
        elif num040==31:
            p340=self.conv131(B140)
        elif num040==32:
            p340=self.conv132(B140)
        elif num040==33:
            p340=self.conv133(B140)
        elif num040==34:
            p340=self.conv134(B140)
        elif num040==35:
            p340=self.conv135(B140)
        elif num040==36:
            p340=self.conv136(B140)
        elif num040==37:
            p340=self.conv137(B140)
        elif num040==38:
            p340=self.conv138(B140)
        elif num040==39:
            p340=self.conv139(B140)
        elif num040==40:
            p340=self.conv140(B140)
        elif num040==41:
            p340=self.conv141(B140)
        elif num040==42:
            p340=self.conv142(B140)
        elif num040==43:
            p340=self.conv143(B140)
        elif num040==44:
            p340=self.conv144(B140)
        elif num040==45:
            p340=self.conv145(B140)
        elif num040==46:
            p340=self.conv146(B140)
        elif num040==47:
            p340=self.conv147(B140)
        elif num040==48:
            p340=self.conv148(B140)
        elif num040==49:
            p340=self.conv149(B140)
        elif num040==50:
            p340=self.conv150(B140)
        elif num040==51:
            p340=self.conv151(B140)
        elif num040==52:
            p340=self.conv152(B140)
        elif num040==53:
            p340=self.conv153(B140)
        elif num040==54:
            p340=self.conv154(B140)
        elif num040==55:
            p340=self.conv155(B140)
        elif num040==56:
            p340=self.conv156(B140)
        elif num040==57:
            p340=self.conv157(B140)
        elif num040==58:
            p340=self.conv158(B140)
        elif num040==59:
            p340=self.conv159(B140)
        elif num040==60:
            p340=self.conv160(B140)
        elif num040==61:
            p340=self.conv161(B140)
        elif num040==62:
            p340=self.conv162(B140)
        elif num040==63:
            p340=self.conv163(B140)
        elif num040==64:
            p340=self.conv164(B140)
        
        if  num140==1:
            p3401=self.conv11(C140)
        elif  num140==2:
            p3401=self.conv12(C140)
        elif  num140==3:
            p3401=self.conv13(C140)
        elif  num140==4:
            p3401=self.conv14(C140)
        elif  num140==5:
            p3401=self.conv15(C140)
        elif  num140==6:
            p3401=self.conv16(C140)
        elif  num140==7:
            p3401=self.conv17(C140)
        elif  num140==8:
            p3401=self.conv18(C140)
        elif  num140==9:
            p3401=self.conv19(C140)
        elif  num140==10:
            p3401=self.conv110(C140)
        elif  num140==11:
            p3401=self.conv111(C140)
        elif  num140==12:
            p3401=self.conv112(C140)
        elif  num140==13:
            p3401=self.conv113(C140)
        elif  num140==14:
            p3401=self.conv114(C140)
        elif  num140==15:
            p3401=self.conv115(C140)
        elif  num140==16:
            p3401=self.conv116(C140)
        elif  num140==17:
            p3401=self.conv117(C140)
        elif  num140==18:
            p3401=self.conv118(C140)
        elif  num140==19:
            p3401=self.conv119(C140)
        elif  num140==20:
            p3401=self.conv120(C140)
        elif  num140==21:
            p3401=self.conv121(C140)
        elif  num140==22:
            p3401=self.conv122(C140)
        elif  num140==23:
            p3401=self.conv123(C140)
        elif  num140==24:
            p3401=self.conv124(C140)
        elif  num140==25:
            p3401=self.conv125(C140)
        elif  num140==26:
            p3401=self.conv126(C140)
        elif  num140==27:
            p3401=self.conv127(C140)
        elif  num140==28:
            p3401=self.conv128(C140)
        elif  num140==29:
            p3401=self.conv129(C140)
        elif  num140==30:
            p3401=self.conv130(C140)
        elif  num140==31:
            p3401=self.conv131(C140)
        elif  num140==32:
            p3401=self.conv132(C140)
        elif  num140==33:
            p3401=self.conv133(C140)
        elif  num140==34:
            p3401=self.conv134(C140)
        elif  num140==35:
            p3401=self.conv135(C140)
        elif  num140==36:
            p3401=self.conv136(C140)
        elif  num140==37:
            p3401=self.conv137(C140)
        elif  num140==38:
            p3401=self.conv138(C140)
        elif  num140==39:
            p3401=self.conv139(C140)
        elif  num140==40:
            p3401=self.conv140(C140)
        elif  num140==41:
            p3401=self.conv141(C140)
        elif  num140==42:
            p3401=self.conv142(C140)
        elif  num140==43:
            p3401=self.conv143(C140)
        elif  num140==44:
            p3401=self.conv144(C140)
        elif  num140==45:
            p3401=self.conv145(C140)
        elif  num140==46:
            p3401=self.conv146(C140)
        elif  num140==47:
            p3401=self.conv147(C140)
        elif  num140==48:
            p3401=self.conv148(C140)
        elif  num140==49:
            p3401=self.conv149(C140)
        elif  num140==50:
            p3401=self.conv150(C140)
        elif  num140==51:
            p3401=self.conv151(C140)
        elif  num140==52:
            p3401=self.conv152(C140)
        elif  num140==53:
            p3401=self.conv153(C140)
        elif  num140==54:
            p3401=self.conv154(C140)
        elif  num140==55:
            p3401=self.conv155(C140)
        elif  num140==56:
            p3401=self.conv156(C140)
        elif  num140==57:
            p3401=self.conv157(C140)
        elif  num140==58:
            p3401=self.conv158(C140)
        elif  num140==59:
            p3401=self.conv159(C140)
        elif  num140==60:
            p3401=self.conv160(C140)
        elif  num140==61:
            p3401=self.conv161(C140)
        elif  num140==62:
            p3401=self.conv162(C140)
        elif  num140==63:
            p3401=self.conv163(C140)
        elif  num140==64:
            p3401=self.conv164(C140)
        elif  num140==65:
            p3401=self.conv165(C140)
        elif  num140==66:
            p3401=self.conv166(C140)
        elif  num140==67:
            p3401=self.conv167(C140)
        elif  num140==68:
            p3401=self.conv168(C140)
        elif  num140==69:
            p3401=self.conv169(C140)
        elif  num140==70:
            p3401=self.conv170(C140)
        elif  num140==71:
            p3401=self.conv171(C140)
        elif  num140==72:
            p3401=self.conv172(C140)
        elif  num140==73:
            p3401=self.conv173(C140)
        elif  num140==74:
            p3401=self.conv174(C140)
        elif  num140==75:
            p3401=self.conv175(C140)
        elif  num140==76:
            p3401=self.conv176(C140)
        elif  num140==77:
            p3401=self.conv177(C140)
        elif  num140==78:
            p3401=self.conv178(C140)
        elif  num140==79:
            p3401=self.conv179(C140)
        elif  num140==80:
            p3401=self.conv180(C140)
        elif  num140==81:
            p3401=self.conv181(C140)
        elif  num140==82:
            p3401=self.conv182(C140)
        elif  num140==83:
            p3401=self.conv183(C140)
        elif  num140==84:
            p3401=self.conv184(C140)
        elif  num140==85:
            p3401=self.conv185(C140)
        elif  num140==86:
            p3401=self.conv186(C140)
        elif  num140==87:
            p3401=self.conv187(C140)
        elif  num140==88:
            p3401=self.conv188(C140)
        elif  num140==89:
            p3401=self.conv189(C140)    
        elif  num140==90:
            p3401=self.conv190(C140)
        elif  num140==91:
            p3401=self.conv191(C140)
        elif  num140==92:
            p3401=self.conv192(C140)
        elif  num140==93:
            p3401=self.conv193(C140)
        elif  num140==94:
            p3401=self.conv194(C140)
        elif  num140==95:
            p3401=self.conv195(C140)
        elif  num140==96:
            p3401=self.conv196(C140)
        elif  num140==97:
            p3401=self.conv197(C140)
        elif  num140==98:
            p3401=self.conv198(C140)
        elif  num140==99:
            p3401=self.conv199(C140) 
        elif  num140==100:
            p3401=self.conv1100(C140)
        elif  num140==101:
            p3401=self.conv1101(C140)
        elif  num140==102:
            p3401=self.conv1102(C140)
        elif  num140==103:
            p3401=self.conv1103(C140)
        elif  num140==104:
            p3401=self.conv1104(C140)
        elif  num140==105:
            p3401=self.conv1105(C140)
        elif  num140==106:
            p3401=self.conv1106(C140)
        elif  num140==107:
            p3401=self.conv1107(C140)
        elif  num140==108:
            p3401=self.conv1108(C140)
        elif  num140==109:
            p3401=self.conv1109(C140)
        elif  num140==110:
            p3401=self.conv1110(C140)
        elif  num140==111:
            p3401=self.conv1111(C140)
        elif  num140==112:
            p3401=self.conv1112(C140)
        elif  num140==113:
            p3401=self.conv1113(C140)
        elif  num140==114:
            p3401=self.conv1114(C140)
        elif  num140==115:
            p3401=self.conv1115(C140)
        elif  num140==116:
            p3401=self.conv1116(C140)
        elif  num140==117:
            p3401=self.conv1117(C140)
        elif  num140==118:
            p3401=self.conv1118(C140)
        elif  num140==119:
            p3401=self.conv1119(C140) 
        elif  num140==120:
            p3401=self.conv1120(C140)
        elif  num140==121:
            p3401=self.conv1121(C140)
        elif  num140==122:
            p3401=self.conv1122(C140)
        elif  num140==123:
            p3401=self.conv1123(C140)
        elif  num140==124:
            p3401=self.conv1124(C140)
        elif  num140==125:
            p3401=self.conv1125(C140)
        elif  num140==126:
            p3401=self.conv1126(C140)
        elif  num140==127:
            p3401=self.conv1127(C140)
        elif  num140==128:
            p3401=self.conv1128(C140)
        elif  num140==129:
            p3401=self.conv1129(C140) 
        elif  num140==130:
            p3401=self.conv1130(C140)
        elif  num140==131:
            p3401=self.conv1131(C140)
        elif  num140==132:
            p3401=self.conv1132(C140)
        elif  num140==133:
            p3401=self.conv1133(C140)
        elif  num140==134:
            p3401=self.conv1134(C140)
        elif  num140==135:
            p3401=self.conv1135(C140)
        elif  num140==136:
            p3401=self.conv1136(C140)
        elif  num140==137:
            p3401=self.conv1137(C140)
        elif  num140==138:
            p3401=self.conv1138(C140)
        elif  num140==139:
            p3401=self.conv1139(C140)
        elif  num140==140:
            p3401=self.conv1140(C140)
        elif  num140==141:
            p3401=self.conv1141(C140)
        elif  num140==142:
            p3401=self.conv1142(C140)
        elif  num140==143:
            p3401=self.conv1143(C140)
        elif  num140==144:
            p3401=self.conv1144(C140)
        elif  num140==145:
            p3401=self.conv1145(C140)
        elif  num140==146:
            p3401=self.conv1146(C140)
        elif  num140==147:
            p3401=self.conv1147(C140)
        elif  num140==148:
            p3401=self.conv1148(C140)
        elif  num140==149:
            p3401=self.conv1149(C140) 
        elif  num140==150:
            p3401=self.conv1150(C140)
        elif  num140==151:
            p3401=self.conv1151(C140)
        elif  num140==152:
            p3401=self.conv1152(C140)
        elif  num140==153:
            p3401=self.conv1153(C140)
        elif  num140==154:
            p3401=self.conv1154(C140)
        elif  num140==155:
            p3401=self.conv1155(C140)
        elif  num140==156:
            p3401=self.conv1156(C140)
        elif  num140==157:
            p3401=self.conv1157(C140)
        elif  num140==158:
            p3401=self.conv1158(C140)
        elif  num140==159:
            p3401=self.conv1159(C140) 
        elif  num140==160:
            p3401=self.conv1160(C140)
        elif  num140==161:
            p3401=self.conv1161(C140)
        elif  num140==162:
            p3401=self.conv1162(C140)
        elif  num140==163:
            p3401=self.conv1163(C140)
        elif  num140==164:
            p3401=self.conv1164(C140)
        elif  num140==165:
            p3401=self.conv1165(C140)
        elif  num140==166:
            p3401=self.conv1166(C140)
        elif  num140==167:
            p3401=self.conv1167(C140)
        elif  num140==168:
            p3401=self.conv1168(C140)
        elif  num140==169:
            p3401=self.conv1169(C140) 
        elif  num140==170:
            p3401=self.conv1170(C140)
        elif  num140==171:
            p3401=self.conv1171(C140)
        elif  num140==172:
            p3401=self.conv1172(C140)
        elif  num140==173:
            p3401=self.conv1173(C140)
        elif  num140==174:
            p3401=self.conv1174(C140)
        elif  num140==175:
            p3401=self.conv1175(C140)
        elif  num140==176:
            p3401=self.conv1176(C140)
        elif  num140==177:
            p3401=self.conv1177(C140)
        elif  num140==178:
            p3401=self.conv1178(C140)
        elif  num140==179:
            p3401=self.conv1179(C140)                                                                                              
        elif  num140==180:
            p3401=self.conv1180(C140)
        elif  num140==181:
            p3401=self.conv1181(C140)
        elif  num140==182:
            p3401=self.conv1182(C140)
        elif  num140==183:
            p3401=self.conv1183(C140)
        elif  num140==184:
            p3401=self.conv1184(C140)
        elif  num140==185:
            p3401=self.conv1185(C140)
        elif  num140==186:
            p3401=self.conv1186(C140)
        elif  num140==187:
            p3401=self.conv1187(C140)
        elif  num140==188:
            p3401=self.conv1188(C140)
        elif  num140==189:
            p3401=self.conv1189(C140) 
        elif  num140==190:
            p3401=self.conv1190(C140)
        elif  num140==191:
            p3401=self.conv1191(C140)
        elif  num140==192:
            p3401=self.conv1192(C140)
        elif  num140==193:
            p3401=self.conv1193(C140)
        elif  num140==194:
            p3401=self.conv1194(C140)
        elif  num140==195:
            p3401=self.conv1195(C140)
        elif  num140==196:
            p3401=self.conv1196(C140)
        elif  num140==197:
            p3401=self.conv1197(C140)
        elif  num140==198:
            p3401=self.conv1198(C140)
        elif  num140==199:
            p3401=self.conv1199(C140)
        elif  num140==200:
            p3401=self.conv1200(C140)
        elif  num140==201:
            p3401=self.conv1201(C140)
        elif  num140==202:
            p3401=self.conv1202(C140)
        elif  num140==203:
            p3401=self.conv1203(C140)
        elif  num140==204:
            p3401=self.conv1204(C140)
        elif  num140==205:
            p3401=self.conv1205(C140)
        elif  num140==206:
            p3401=self.conv1206(C140)
        elif  num140==207:
            p3401=self.conv1207(C140)
        elif  num140==208:
            p3401=self.conv1208(C140)
        elif  num140==209:
            p3401=self.conv1209(C140)
        elif  num140==210:
            p3401=self.conv1210(C140)
        elif  num140==211:
            p3401=self.conv1211(C140)
        elif  num140==212:
            p3401=self.conv1212(C140)
        elif  num140==213:
            p3401=self.conv1213(C140)
        elif  num140==214:
            p3401=self.conv1214(C140)
        elif  num140==215:
            p3401=self.conv1215(C140)
        elif  num140==216:
            p3401=self.conv1216(C140)
        elif  num140==217:
            p3401=self.conv1217(C140)
        elif  num140==218:
            p3401=self.conv1218(C140)
        elif  num140==219:
            p3401=self.conv1219(C140)
        elif  num140==220:
            p3401=self.conv1220(C140)
        elif  num140==221:
            p3401=self.conv1221(C140)
        elif  num140==222:
            p3401=self.conv1222(C140)
        elif  num140==223:
            p3401=self.conv1223(C140)
        elif  num140==224:
            p3401=self.conv1224(C140)
        elif  num140==225:
            p3401=self.conv1225(C140)
        elif  num140==226:
            p3401=self.conv1226(C140)
        elif  num140==227:
            p3401=self.conv1227(C140)
        elif  num140==228:
            p3401=self.conv1228(C140)
        elif  num140==229:
            p3401=self.conv1229(C140)
        elif  num140==230:
            p3401=self.conv1230(C140)
        elif  num140==231:
            p3401=self.conv1231(C140)
        elif  num140==232:
            p3401=self.conv1232(C140)
        elif  num140==233:
            p3401=self.conv1233(C140)
        elif  num140==234:
            p3401=self.conv1234(C140)
        elif  num140==235:
            p3401=self.conv1235(C140)
        elif  num140==236:
            p3401=self.conv1236(C140)
        elif  num140==237:
            p3401=self.conv1237(C140)
        elif  num140==238:
            p3401=self.conv1238(C140)
        elif  num140==239:
            p3401=self.conv1239(C140) 
        elif  num140==240:
            p3401=self.conv1240(C140)
        elif  num140==241:
            p3401=self.conv1241(C140)
        elif  num140==242:
            p3401=self.conv1242(C140)
        elif  num140==243:
            p3401=self.conv1243(C140)
        elif  num140==244:
            p3401=self.conv1244(C140)
        elif  num140==245:
            p3401=self.conv1245(C140)
        elif  num140==246:
            p3401=self.conv1246(C140)
        elif  num140==247:
            p3401=self.conv1247(C140)
        elif  num140==248:
            p3401=self.conv1248(C140)
        elif  num140==249:
            p3401=self.conv1249(C140)
        elif  num140==250:
            p3401=self.conv1250(C140)
        elif  num140==251:
            p3401=self.conv1251(C140)
        elif  num140==252:
            p3401=self.conv1252(C140)
        elif  num140==253:
            p3401=self.conv1253(C140)
        elif  num140==254:
            p3401=self.conv1254(C140)
        elif  num140==255:
            p3401=self.conv1255(C140)
        elif  num140==256:
            p3401=self.conv1256(C140)
            
        if  num240==1:
            p3402=self.conv11(D140)
        elif  num240==2:
            p3402=self.conv12(D140)
        elif  num240==3:
            p3402=self.conv13(D140)
        elif  num240==4:
            p3402=self.conv14(D140)
        elif  num240==5:
            p3402=self.conv15(D140)
        elif  num240==6:
            p3402=self.conv16(D140)
        elif  num240==7:
            p3402=self.conv17(D140)
        elif  num240==8:
            p3402=self.conv18(D140)
        elif  num240==9:
            p3402=self.conv19(D140)
        elif  num240==10:
            p3402=self.conv110(D140)
        elif  num240==11:
            p3402=self.conv111(D140)
        elif  num240==12:
            p3402=self.conv112(D140)
        elif  num240==13:
            p3402=self.conv113(D140)
        elif  num240==14:
            p3402=self.conv114(D140)
        elif  num240==15:
            p3402=self.conv115(D140)
        elif  num240==16:
            p3402=self.conv116(D140)
        elif  num240==17:
            p3402=self.conv117(D140)
        elif  num240==18:
            p3402=self.conv118(D140)
        elif  num240==19:
            p3402=self.conv119(D140)
        elif  num240==20:
            p3402=self.conv120(D140)
        elif  num240==21:
            p3402=self.conv121(D140)
        elif  num240==22:
            p3402=self.conv122(D140)
        elif  num240==23:
            p3402=self.conv123(D140)
        elif  num240==24:
            p3402=self.conv124(D140)
        elif  num240==25:
            p3402=self.conv125(D140)
        elif  num240==26:
            p3402=self.conv126(D140)
        elif  num240==27:
            p3402=self.conv127(D140)
        elif  num240==28:
            p3402=self.conv128(D140)
        elif  num240==29:
            p3402=self.conv129(D140)
        elif  num240==30:
            p3402=self.conv130(D140)
        elif  num240==31:
            p3402=self.conv131(D140)
        elif  num240==32:
            p3402=self.conv132(D140)
        elif  num240==33:
            p3402=self.conv133(D140)
        elif  num240==34:
            p3402=self.conv134(D140)
        elif  num240==35:
            p3402=self.conv135(D140)
        elif  num240==36:
            p3402=self.conv136(D140)
        elif  num240==37:
            p3402=self.conv137(D140)
        elif  num240==38:
            p3402=self.conv138(D140)
        elif  num240==39:
            p3402=self.conv139(D140)
        elif  num240==40:
            p3402=self.conv140(D140)
        elif  num240==41:
            p3402=self.conv141(D140)
        elif  num240==42:
            p3402=self.conv142(D140)
        elif  num240==43:
            p3402=self.conv143(D140)
        elif  num240==44:
            p3402=self.conv144(D140)
        elif  num240==45:
            p3402=self.conv145(D140)
        elif  num240==46:
            p3402=self.conv146(D140)
        elif  num240==47:
            p3402=self.conv147(D140)
        elif  num240==48:
            p3402=self.conv148(D140)
        elif  num240==49:
            p3402=self.conv149(D140)
        elif  num240==50:
            p3402=self.conv150(D140)
        elif  num240==51:
            p3402=self.conv151(D140)
        elif  num240==52:
            p3402=self.conv152(D140)
        elif  num240==53:
            p3402=self.conv153(D140)
        elif  num240==54:
            p3402=self.conv154(D140)
        elif  num240==55:
            p3402=self.conv155(D140)
        elif  num240==56:
            p3402=self.conv156(D140)
        elif  num240==57:
            p3402=self.conv157(D140)
        elif  num240==58:
            p3402=self.conv158(D140)
        elif  num240==59:
            p3402=self.conv159(D140)
        elif  num240==60:
            p3402=self.conv160(D140)
        elif  num240==61:
            p3402=self.conv161(D140)
        elif  num240==62:
            p3402=self.conv162(D140)
        elif  num240==63:
            p3402=self.conv163(D140)
        elif  num240==64:
            p3402=self.conv164(D140)
        elif  num240==65:
            p3402=self.conv165(D140)
        elif  num240==66:
            p3402=self.conv166(D140)
        elif  num240==67:
            p3402=self.conv167(D140)
        elif  num240==68:
            p3402=self.conv168(D140)
        elif  num240==69:
            p3402=self.conv169(D140)
        elif  num240==70:
            p3402=self.conv170(D140)
        elif  num240==71:
            p3402=self.conv171(D140)
        elif  num240==72:
            p3402=self.conv172(D140)
        elif  num240==73:
            p3402=self.conv173(D140)
        elif  num240==74:
            p3402=self.conv174(D140)
        elif  num240==75:
            p3402=self.conv175(D140)
        elif  num240==76:
            p3402=self.conv176(D140)
        elif  num240==77:
            p3402=self.conv177(D140)
        elif  num240==78:
            p3402=self.conv178(D140)
        elif  num240==79:
            p3402=self.conv179(D140)
        elif  num240==80:
            p3402=self.conv180(D140)
        elif  num240==81:
            p3402=self.conv181(D140)
        elif  num240==82:
            p3402=self.conv182(D140)
        elif  num240==83:
            p3402=self.conv183(D140)
        elif  num240==84:
            p3402=self.conv184(D140)
        elif  num240==85:
            p3402=self.conv185(D140)
        elif  num240==86:
            p3402=self.conv186(D140)
        elif  num240==87:
            p3402=self.conv187(D140)
        elif  num240==88:
            p3402=self.conv188(D140)
        elif  num240==89:
            p3402=self.conv189(D140)    
        elif  num240==90:
            p3402=self.conv190(D140)
        elif  num240==91:
            p3402=self.conv191(D140)
        elif  num240==92:
            p3402=self.conv192(D140)
        elif  num240==93:
            p3402=self.conv193(D140)
        elif  num240==94:
            p3402=self.conv194(D140)
        elif  num240==95:
            p3402=self.conv195(D140)
        elif  num240==96:
            p3402=self.conv196(D140)
        elif  num240==97:
            p3402=self.conv197(D140)
        elif  num240==98:
            p3402=self.conv198(D140)
        elif  num240==99:
            p3402=self.conv199(D140) 
        elif  num240==100:
            p3402=self.conv1100(D140)
        elif  num240==101:
            p3402=self.conv1101(D140)
        elif  num240==102:
            p3402=self.conv1102(D140)
        elif  num240==103:
            p3402=self.conv1103(D140)
        elif  num240==104:
            p3402=self.conv1104(D140)
        elif  num240==105:
            p3402=self.conv1105(D140)
        elif  num240==106:
            p3402=self.conv1106(D140)
        elif  num240==107:
            p3402=self.conv1107(D140)
        elif  num240==108:
            p3402=self.conv1108(D140)
        elif  num240==109:
            p3402=self.conv1109(D140)
        elif  num240==110:
            p3402=self.conv1110(D140)
        elif  num240==111:
            p3402=self.conv1111(D140)
        elif  num240==112:
            p3402=self.conv1112(D140)
        elif  num240==113:
            p3402=self.conv1113(D140)
        elif  num240==114:
            p3402=self.conv1114(D140)
        elif  num240==115:
            p3402=self.conv1115(D140)
        elif  num240==116:
            p3402=self.conv1116(D140)
        elif  num240==117:
            p3402=self.conv1117(D140)
        elif  num240==118:
            p3402=self.conv1118(D140)
        elif  num240==119:
            p3402=self.conv1119(D140) 
        elif  num240==120:
            p3402=self.conv1120(D140)
        elif  num240==121:
            p3402=self.conv1121(D140)
        elif  num240==122:
            p3402=self.conv1122(D140)
        elif  num240==123:
            p3402=self.conv1123(D140)
        elif  num240==124:
            p3402=self.conv1124(D140)
        elif  num240==125:
            p3402=self.conv1125(D140)
        elif  num240==126:
            p3402=self.conv1126(D140)
        elif  num240==127:
            p3402=self.conv1127(D140)
        elif  num240==128:
            p3402=self.conv1128(D140)
        elif  num240==129:
            p3402=self.conv1129(D140) 
        elif  num240==130:
            p3402=self.conv1130(D140)
        elif  num240==131:
            p3402=self.conv1131(D140)
        elif  num240==132:
            p3402=self.conv1132(D140)
        elif  num240==133:
            p3402=self.conv1133(D140)
        elif  num240==134:
            p3402=self.conv1134(D140)
        elif  num240==135:
            p3402=self.conv1135(D140)
        elif  num240==136:
            p3402=self.conv1136(D140)
        elif  num240==137:
            p3402=self.conv1137(D140)
        elif  num240==138:
            p3402=self.conv1138(D140)
        elif  num240==139:
            p3402=self.conv1139(D140)
        elif  num240==140:
            p3402=self.conv1140(D140)
        elif  num240==141:
            p3402=self.conv1141(D140)
        elif  num240==142:
            p3402=self.conv1142(D140)
        elif  num240==143:
            p3402=self.conv1143(D140)
        elif  num240==144:
            p3402=self.conv1144(D140)
        elif  num240==145:
            p3402=self.conv1145(D140)
        elif  num240==146:
            p3402=self.conv1146(D140)
        elif  num240==147:
            p3402=self.conv1147(D140)
        elif  num240==148:
            p3402=self.conv1148(D140)
        elif  num240==149:
            p3402=self.conv1149(D140) 
        elif  num240==150:
            p3402=self.conv1150(D140)
        elif  num240==151:
            p3402=self.conv1151(D140)
        elif  num240==152:
            p3402=self.conv1152(D140)
        elif  num240==153:
            p3402=self.conv1153(D140)
        elif  num240==154:
            p3402=self.conv1154(D140)
        elif  num240==155:
            p3402=self.conv1155(D140)
        elif  num240==156:
            p3402=self.conv1156(D140)
        elif  num240==157:
            p3402=self.conv1157(D140)
        elif  num240==158:
            p3402=self.conv1158(D140)
        elif  num240==159:
            p3402=self.conv1159(D140) 
        elif  num240==160:
            p3402=self.conv1160(D140)
        elif  num240==161:
            p3402=self.conv1161(D140)
        elif  num240==162:
            p3402=self.conv1162(D140)
        elif  num240==163:
            p3402=self.conv1163(D140)
        elif  num240==164:
            p3402=self.conv1164(D140)
        elif  num240==165:
            p3402=self.conv1165(D140)
        elif  num240==166:
            p3402=self.conv1166(D140)
        elif  num240==167:
            p3402=self.conv1167(D140)
        elif  num240==168:
            p3402=self.conv1168(D140)
        elif  num240==169:
            p3402=self.conv1169(D140) 
        elif  num240==170:
            p3402=self.conv1170(D140)
        elif  num240==171:
            p3402=self.conv1171(D140)
        elif  num240==172:
            p3402=self.conv1172(D140)
        elif  num240==173:
            p3402=self.conv1173(D140)
        elif  num240==174:
            p3402=self.conv1174(D140)
        elif  num240==175:
            p3402=self.conv1175(D140)
        elif  num240==176:
            p3402=self.conv1176(D140)
        elif  num240==177:
            p3402=self.conv1177(D140)
        elif  num240==178:
            p3402=self.conv1178(D140)
        elif  num240==179:
            p3402=self.conv1179(D140)                                                                                              
        elif  num240==180:
            p3402=self.conv1180(D140)
        elif  num240==181:
            p3402=self.conv1181(D140)
        elif  num240==182:
            p3402=self.conv1182(D140)
        elif  num240==183:
            p3402=self.conv1183(D140)
        elif  num240==184:
            p3402=self.conv1184(D140)
        elif  num240==185:
            p3402=self.conv1185(D140)
        elif  num240==186:
            p3402=self.conv1186(D140)
        elif  num240==187:
            p3402=self.conv1187(D140)
        elif  num240==188:
            p3402=self.conv1188(D140)
        elif  num240==189:
            p3402=self.conv1189(D140) 
        elif  num240==190:
            p3402=self.conv1190(D140)
        elif  num240==191:
            p3402=self.conv1191(D140)
        elif  num240==192:
            p3402=self.conv1192(D140)
        elif  num240==193:
            p3402=self.conv1193(D140)
        elif  num240==194:
            p3402=self.conv1194(D140)
        elif  num240==195:
            p3402=self.conv1195(D140)
        elif  num240==196:
            p3402=self.conv1196(D140)
        elif  num240==197:
            p3402=self.conv1197(D140)
        elif  num240==198:
            p3402=self.conv1198(D140)
        elif  num240==199:
            p3402=self.conv1199(D140)
        elif  num240==200:
            p3402=self.conv1200(D140)
        elif  num240==201:
            p3402=self.conv1201(D140)
        elif  num240==202:
            p3402=self.conv1202(D140)
        elif  num240==203:
            p3402=self.conv1203(D140)
        elif  num240==204:
            p3402=self.conv1204(D140)
        elif  num240==205:
            p3402=self.conv1205(D140)
        elif  num240==206:
            p3402=self.conv1206(D140)
        elif  num240==207:
            p3402=self.conv1207(D140)
        elif  num240==208:
            p3402=self.conv1208(D140)
        elif  num240==209:
            p3402=self.conv1209(D140)
        elif  num240==210:
            p3402=self.conv1210(D140)
        elif  num240==211:
            p3402=self.conv1211(D140)
        elif  num240==212:
            p3402=self.conv1212(D140)
        elif  num240==213:
            p3402=self.conv1213(D140)
        elif  num240==214:
            p3402=self.conv1214(D140)
        elif  num240==215:
            p3402=self.conv1215(D140)
        elif  num240==216:
            p3402=self.conv1216(D140)
        elif  num240==217:
            p3402=self.conv1217(D140)
        elif  num240==218:
            p3402=self.conv1218(D140)
        elif  num240==219:
            p3402=self.conv1219(D140)
        elif  num240==220:
            p3402=self.conv1220(D140)
        elif  num240==221:
            p3402=self.conv1221(D140)
        elif  num240==222:
            p3402=self.conv1222(D140)
        elif  num240==223:
            p3402=self.conv1223(D140)
        elif  num240==224:
            p3402=self.conv1224(D140)
        elif  num240==225:
            p3402=self.conv1225(D140)
        elif  num240==226:
            p3402=self.conv1226(D140)
        elif  num240==227:
            p3402=self.conv1227(D140)
        elif  num240==228:
            p3402=self.conv1228(D140)
        elif  num240==229:
            p3402=self.conv1229(D140)
        elif  num240==230:
            p3402=self.conv1230(D140)
        elif  num240==231:
            p3402=self.conv1231(D140)
        elif  num240==232:
            p3402=self.conv1232(D140)
        elif  num240==233:
            p3402=self.conv1233(D140)
        elif  num240==234:
            p3402=self.conv1234(D140)
        elif  num240==235:
            p3402=self.conv1235(D140)
        elif  num240==236:
            p3402=self.conv1236(D140)
        elif  num240==237:
            p3402=self.conv1237(D140)
        elif  num240==238:
            p3402=self.conv1238(D140)
        elif  num240==239:
            p3402=self.conv1239(D140) 
        elif  num240==240:
            p3402=self.conv1240(D140)
        elif  num240==241:
            p3402=self.conv1241(D140)
        elif  num240==242:
            p3402=self.conv1242(D140)
        elif  num240==243:
            p3402=self.conv1243(D140)
        elif  num240==244:
            p3402=self.conv1244(D140)
        elif  num240==245:
            p3402=self.conv1245(D140)
        elif  num240==246:
            p3402=self.conv1246(D140)
        elif  num240==247:
            p3402=self.conv1247(D140)
        elif  num240==248:
            p3402=self.conv1248(D140)
        elif  num240==249:
            p3402=self.conv1249(D140)
        elif  num240==250:
            p3402=self.conv1250(D140)
        elif  num240==251:
            p3402=self.conv1251(D140)
        elif  num240==252:
            p3402=self.conv1252(D140)
        elif  num240==253:
            p3402=self.conv1253(D140)
        elif  num240==254:
            p3402=self.conv1254(D140)
        elif  num240==255:
            p3402=self.conv1255(D140)
        elif  num240==256:
            p3402=self.conv1256(D140)
        elif  num240==257:
            p3402=self.conv1257(D140)
        elif  num240==258:
            p3402=self.conv1258(D140)
        elif  num240==259:
            p3402=self.conv1259(D140)
        elif  num240==260:
            p3402=self.conv1260(D140)
        elif  num240==261:
            p3402=self.conv1261(D140)
        elif  num240==262:
            p3402=self.conv1262(D140)
        elif  num240==263:
            p3402=self.conv1263(D140)
        elif  num240==264:
            p3402=self.conv1264(D140)
        elif  num240==265:
            p3402=self.conv1265(D140)
        elif  num240==266:
            p3402=self.conv1266(D140)
        elif  num240==267:
            p3402=self.conv1267(D140)
        elif  num240==268:
            p3402=self.conv1268(D140)
        elif  num240==269:
            p3402=self.conv1269(D140) 
        elif  num240==270:
            p3402=self.conv1270(D140)
        elif  num240==271:
            p3402=self.conv1271(D140)
        elif  num240==272:
            p3402=self.conv1272(D140)
        elif  num240==273:
            p3402=self.conv1273(D140)
        elif  num240==274:
            p3402=self.conv1274(D140)
        elif  num240==275:
            p3402=self.conv1275(D140)
        elif  num240==276:
            p3402=self.conv1276(D140)
        elif  num240==277:
            p3402=self.conv1277(D140)
        elif  num240==278:
            p3402=self.conv1278(D140)
        elif  num240==279:
            p3402=self.conv1279(D140)
        elif  num240==280:
            p3402=self.conv1280(D140)
        elif  num240==281:
            p3402=self.conv1281(D140)
        elif  num240==282:
            p3402=self.conv1282(D140)
        elif  num240==283:
            p3402=self.conv1283(D140)
        elif  num240==284:
            p3402=self.conv1284(D140)
        elif  num240==285:
            p3402=self.conv1285(D140)
        elif  num240==286:
            p3402=self.conv1286(D140)
        elif  num240==287:
            p3402=self.conv1287(D140)
        elif  num240==288:
            p3402=self.conv1288(D140)
        elif  num240==289:
            p3402=self.conv1289(D140) 
        elif  num240==290:
            p3402=self.conv1290(D140)
        elif  num240==291:
            p3402=self.conv1291(D140)
        elif  num240==292:
            p3402=self.conv1292(D140)
        elif  num240==293:
            p3402=self.conv1293(D140)
        elif  num240==294:
            p3402=self.conv1294(D140)
        elif  num240==295:
            p3402=self.conv1295(D140)
        elif  num240==296:
            p3402=self.conv1296(D140)
        elif  num240==297:
            p3402=self.conv1297(D140)
        elif  num240==298:
            p3402=self.conv1298(D140)
        elif  num240==299:
            p3402=self.conv1299(D140)
        elif  num240==300:
            p3402=self.conv1300(D140)
        elif  num240==301:
            p3402=self.conv1301(D140)
        elif  num240==302:
            p3402=self.conv1302(D140)
        elif  num240==303:
            p3402=self.conv1303(D140)
        elif  num240==304:
            p3402=self.conv1304(D140)
        elif  num240==305:
            p3402=self.conv1305(D140)
        elif  num240==306:
            p3402=self.conv1306(D140)
        elif  num240==307:
            p3402=self.conv1307(D140)
        elif  num240==308:
            p3402=self.conv1308(D140)
        elif  num240==309:
            p3402=self.conv1309(D140) 
        elif  num240==310:
            p3402=self.conv1310(D140)
        elif  num240==311:
            p3402=self.conv1311(D140)
        elif  num240==312:
            p3402=self.conv1312(D140)
        elif  num240==313:
            p3402=self.conv1313(D140)
        elif  num240==314:
            p3402=self.conv1314(D140)
        elif  num240==315:
            p3402=self.conv1315(D140)
        elif  num240==316:
            p3402=self.conv1316(D140)
        elif  num240==317:
            p3402=self.conv1317(D140)
        elif  num240==318:
            p3402=self.conv1318(D140)
        elif  num240==319:
            p3402=self.conv1319(D140)
        elif  num240==320:
            p3402=self.conv1320(D140)
        elif  num240==321:
            p3402=self.conv1321(D140)
        elif  num240==322:
            p3402=self.conv1322(D140)
        elif  num240==323:
            p3402=self.conv1323(D140)
        elif  num240==324:
            p3402=self.conv1324(D140)
        elif  num240==325:
            p3402=self.conv1325(D140)
        elif  num240==326:
            p3402=self.conv1326(D140)
        elif  num240==327:
            p3402=self.conv1327(D140)
        elif  num240==328:
            p3402=self.conv1328(D140)
        elif  num240==329:
            p3402=self.conv1329(D140)
        elif  num240==330:
            p3402=self.conv1330(D140)
        elif  num240==331:
            p3402=self.conv1331(D140)
        elif  num240==332:
            p3402=self.conv1332(D140)
        elif  num240==333:
            p3402=self.conv1333(D140)
        elif  num240==334:
            p3402=self.conv1334(D140)
        elif  num240==335:
            p3402=self.conv1335(D140)
        elif  num240==336:
            p3402=self.conv1336(D140)
        elif  num240==337:
            p3402=self.conv1337(D140)
        elif  num240==338:
            p3402=self.conv1338(D140)
        elif  num240==339:
            p3402=self.conv1339(D140)
        elif  num240==340:
            p3402=self.conv1340(D140)
        elif  num240==341:
            p3402=self.conv1341(D140)
        elif  num240==342:
            p3402=self.conv1342(D140)
        elif  num240==343:
            p3402=self.conv1343(D140)
        elif  num240==344:
            p3402=self.conv1344(D140)
        elif  num240==345:
            p3402=self.conv1345(D140)
        elif  num240==346:
            p3402=self.conv1346(D140)
        elif  num240==347:
            p3402=self.conv1347(D140)
        elif  num240==348:
            p3402=self.conv1348(D140)
        elif  num240==349:
            p3402=self.conv1349(D140)
        elif  num240==350:
            p3402=self.conv1350(D140)
        elif  num240==351:
            p3402=self.conv1351(D140)
        elif  num240==352:
            p3402=self.conv1352(D140)
        elif  num240==353:
            p3402=self.conv1335(D140)
        elif  num240==354:
            p3402=self.conv1354(D140)
        elif  num240==355:
            p3402=self.conv1355(D140)
        elif  num240==356:
            p3402=self.conv1356(D140)
        elif  num240==357:
            p3402=self.conv1357(D140)
        elif  num240==358:
            p3402=self.conv1358(D140)
        elif  num240==359:
            p3402=self.conv1359(D140) 
        elif  num240==360:
            p3402=self.conv1360(D140)
        elif  num240==361:
            p3402=self.conv1361(D140)
        elif  num240==362:
            p3402=self.conv1362(D140)
        elif  num240==363:
            p3402=self.conv1363(D140)
        elif  num240==364:
            p3402=self.conv1364(D140)
        elif  num240==365:
            p3402=self.conv1365(D140)
        elif  num240==366:
            p3402=self.conv1366(D140)
        elif  num240==367:
            p3402=self.conv1367(D140)
        elif  num240==368:
            p3402=self.conv1368(D140)
        elif  num240==369:
            p3402=self.conv1369(D140) 
        elif  num240==370:
            p3402=self.conv1370(D140)
        elif  num240==371:
            p3402=self.conv1371(D140)
        elif  num240==372:
            p3402=self.conv1372(D140)
        elif  num240==373:
            p3402=self.conv1373(D140)
        elif  num240==374:
            p3402=self.conv1374(D140)
        elif  num240==375:
            p3402=self.conv1375(D140)
        elif  num240==376:
            p3402=self.conv1376(D140)
        elif  num240==377:
            p3402=self.conv1377(D140)
        elif  num240==378:
            p3402=self.conv1378(D140)
        elif  num240==379:
            p3402=self.conv1379(D140) 
        elif  num240==380:
            p3402=self.conv1380(D140)
        elif  num240==381:
            p3402=self.conv1381(D140)
        elif  num240==382:
            p3402=self.conv1382(D140)
        elif  num240==383:
            p3402=self.conv1383(D140)
        elif  num240==384:
            p3402=self.conv1384(D140)
        elif  num240==385:
            p3402=self.conv1385(D140)
        elif  num240==386:
            p3402=self.conv1386(D140)
        elif  num240==387:
            p3402=self.conv1387(D140)
        elif  num240==388:
            p3402=self.conv1388(D140)
        elif  num240==389:
            p3402=self.conv1389(D140) 
        elif  num240==390:
            p3402=self.conv1390(D140)
        elif  num240==391:
            p3402=self.conv1391(D140)
        elif  num240==392:
            p3402=self.conv1392(D140)
        elif  num240==393:
            p3402=self.conv1393(D140)
        elif  num240==394:
            p3402=self.conv1394(D140)
        elif  num240==395:
            p3402=self.conv1395(D140)
        elif  num240==396:
            p3402=self.conv1396(D140)
        elif  num240==397:
            p3402=self.conv1397(D140)
        elif  num240==398:
            p3402=self.conv1398(D140)
        elif  num240==399:
            p3402=self.conv1399(D140)
        elif  num240==400:
            p3402=self.conv1400(D140)
        elif  num240==401:
            p3402=self.conv1401(D140)
        elif  num240==402:
            p3402=self.conv1402(D140)
        elif  num240==403:
            p3402=self.conv1403(D140)
        elif  num240==404:
            p3402=self.conv1404(D140)
        elif  num240==405:
            p3402=self.conv1405(D140)
        elif  num240==406:
            p3402=self.conv1406(D140)
        elif  num240==407:
            p3402=self.conv1407(D140)
        elif  num240==408:
            p3402=self.conv1408(D140)
        elif  num240==409:
            p3402=self.conv1409(D140)
        elif  num240==410:
            p3402=self.conv1410(D140)
        elif  num240==411:
            p3402=self.conv1411(D140)
        elif  num240==412:
            p3402=self.conv1412(D140)
        elif  num240==413:
            p3402=self.conv1413(D140)
        elif  num240==414:
            p3402=self.conv1414(D140)
        elif  num240==415:
            p3402=self.conv145(D140)
        elif  num240==416:
            p3402=self.conv1416(D140)
        elif  num240==417:
            p3402=self.conv1417(D140)
        elif  num240==418:
            p3402=self.conv1418(D140)
        elif  num240==419:
            p3402=self.conv1419(D140) 
        elif  num240==420:
            p3402=self.conv1420(D140)
        elif  num240==421:
            p3402=self.conv1421(D140)
        elif  num240==422:
            p3402=self.conv1422(D140)
        elif  num240==423:
            p3402=self.conv1423(D140)
        elif  num240==424:
            p3402=self.conv1424(D140)
        elif  num240==425:
            p3402=self.conv1425(D140)
        elif  num240==426:
            p3402=self.conv1426(D140)
        elif  num240==427:
            p3402=self.conv1427(D140)
        elif  num240==428:
            p3402=self.conv1428(D140)
        elif  num240==429:
            p3402=self.conv1429(D140) 
        elif  num240==430:
            p3402=self.conv1430(D140)
        elif  num240==431:
            p3402=self.conv1431(D140)
        elif  num240==432:
            p3402=self.conv1432(D140)
        elif  num240==433:
            p3402=self.conv1433(D140)
        elif  num240==434:
            p3402=self.conv1434(D140)
        elif  num240==435:
            p3402=self.conv1435(D140)
        elif  num240==436:
            p3402=self.conv1436(D140)
        elif  num240==437:
            p3402=self.conv1437(D140)
        elif  num240==438:
            p3402=self.conv1438(D140)
        elif  num240==439:
            p3402=self.conv1439(D140)
        elif  num240==440:
            p3402=self.conv1440(D140)
        elif  num240==441:
            p3402=self.conv1441(D140)
        elif  num240==442:
            p3402=self.conv1442(D140)
        elif  num240==443:
            p3402=self.conv1443(D140)
        elif  num240==444:
            p3402=self.conv1444(D140)
        elif  num240==445:
            p3402=self.conv1445(D140)
        elif  num240==446:
            p3402=self.conv1446(D140)
        elif  num240==447:
            p3402=self.conv1447(D140)
        elif  num240==448:
            p3402=self.conv1448(D140)
        elif  num240==449:
            p3402=self.conv1449(D140)
        elif  num240==450:
            p3402=self.conv1450(D140)
        elif  num240==451:
            p3402=self.conv1451(D140)
        elif  num240==452:
            p3402=self.conv1452(D140)
        elif  num240==453:
            p3402=self.conv1453(D140)
        elif  num240==454:
            p3402=self.conv1454(D140)
        elif  num240==455:
            p3402=self.conv1455(D140)
        elif  num240==456:
            p3402=self.conv1456(D140)
        elif  num240==457:
            p3402=self.conv1457(D140)
        elif  num240==458:
            p3402=self.conv1458(D140)
        elif  num240==459:
            p3402=self.conv1459(D140)
        elif  num240==460:
            p3402=self.conv1460(D140)
        elif  num240==461:
            p3402=self.conv1461(D140)
        elif  num240==462:
            p3402=self.conv1462(D140)
        elif  num240==463:
            p3402=self.conv1463(D140)
        elif  num240==464:
            p3402=self.conv1464(D140)
        elif  num240==465:
            p3402=self.conv1465(D140)
        elif  num240==466:
            p3402=self.conv1466(D140)
        elif  num240==467:
            p3402=self.conv1467(D140)
        elif  num240==468:
            p3402=self.conv1468(D140)
        elif  num240==469:
            p3402=self.conv1469(D140) 
        elif  num240==470:
            p3402=self.conv1470(D140)
        elif  num240==471:
            p3402=self.conv1471(D140)
        elif  num240==472:
            p3402=self.conv1472(D140)
        elif  num240==473:
            p3402=self.conv1473(D140)
        elif  num240==474:
            p3402=self.conv1474(D140)
        elif  num240==475:
            p3402=self.conv1475(D140)
        elif  num240==476:
            p3402=self.conv1476(D140)
        elif  num240==477:
            p3402=self.conv1477(D140)
        elif  num240==478:
            p3402=self.conv1478(D140)
        elif  num240==479:
            p3402=self.conv1479(D140)
        elif  num240==480:
            p3402=self.conv1480(D140)
        elif  num240==481:
            p3402=self.conv1481(D140)
        elif  num240==482:
            p3402=self.conv1482(D140)
        elif  num240==483:
            p3402=self.conv1483(D140)
        elif  num240==484:
            p3402=self.conv1484(D140)
        elif  num240==485:
            p3402=self.conv1485(D140)
        elif  num240==486:
            p3402=self.conv1486(D140)
        elif  num240==487:
            p3402=self.conv1487(D140)
        elif  num240==488:
            p3402=self.conv1488(D140)
        elif  num240==489:
            p3402=self.conv1489(D140)
        elif  num240==490:
            p3402=self.conv1490(D140)
        elif  num240==491:
            p3402=self.conv1491(D140)
        elif  num240==492:
            p3402=self.conv1492(D140)
        elif  num240==493:
            p3402=self.conv1493(D140)
        elif  num240==494:
            p3402=self.conv1494(D140)
        elif  num240==495:
            p3402=self.conv1495(D140)
        elif  num240==496:
            p3402=self.conv1496(D140)
        elif  num240==497:
            p3402=self.conv1497(D140)
        elif  num240==498:
            p3402=self.conv1498(D140)
        elif  num240==499:
            p3402=self.conv1499(D140)
        elif  num240==500:
            p3402=self.conv1500(D140)
        elif  num240==501:
            p3402=self.conv1501(D140)
        elif  num240==502:
            p3402=self.conv1502(D140)
        elif  num240==503:
            p3402=self.conv1503(D140)
        elif  num240==504:
            p3402=self.conv1504(D140)
        elif  num240==505:
            p3402=self.conv1505(D140)
        elif  num240==506:
            p3402=self.conv1506(D140)
        elif  num240==507:
            p3402=self.conv1507(D140)
        elif  num240==508:
            p3402=self.conv1508(D140)
        elif  num240==509:
            p3402=self.conv1509(D140)
        elif  num240==510:
            p3402=self.conv1510(D140)
        elif  num240==511:
            p3402=self.conv1511(D140)
        elif  num240==512:
            p3402=self.conv1512(D140)
        elif  num240==513:
            p3402=self.conv1513(D140)
        elif  num240==514:
            p3402=self.conv1514(D140)
        elif  num240==515:
            p3402=self.conv1515(D140)
        elif  num240==516:
            p3402=self.conv1516(D140)
        elif  num240==517:
            p3402=self.conv1517(D140)
        elif  num240==518:
            p3402=self.conv1518(D140)
        elif  num240==519:
            p3402=self.conv1519(D140)
        elif  num240==520:
            p3402=self.conv1520(D140)
        elif  num240==521:
            p3402=self.conv1521(D140)
        elif  num240==522:
            p3402=self.conv1522(D140)
        elif  num240==523:
            p3402=self.conv1523(D140)
        elif  num240==524:
            p3402=self.conv1524(D140)
        elif  num240==525:
            p3402=self.conv1525(D140)
        elif  num240==526:
            p3402=self.conv1526(D140)
        elif  num240==527:
            p3402=self.conv1527(D140)
        elif  num240==528:
            p3402=self.conv1528(D140)
        elif  num240==529:
            p3402=self.conv1529(D140)
        elif  num240==530:
            p3402=self.conv1530(D140)
        elif  num240==531:
            p3402=self.conv1531(D140)
        elif  num240==532:
            p3402=self.conv1532(D140)
        elif  num240==533:
            p3402=self.conv1533(D140)
        elif  num240==534:
            p3402=self.conv1534(D140)
        elif  num240==535:
            p3402=self.conv1535(D140)
        elif  num240==536:
            p3402=self.conv1536(D140)
        elif  num240==537:
            p3402=self.conv1537(D140)
        elif  num240==538:
            p3402=self.conv1538(D140)
        elif  num240==539:
            p3402=self.conv1539(D140)
        elif  num240==540:
            p3402=self.conv1540(D140)
        elif  num240==541:
            p3402=self.conv1541(D140)
        elif  num240==542:
            p3402=self.conv1542(D140)
        elif  num240==543:
            p3402=self.conv1543(D140)
        elif  num240==544:
            p3402=self.conv1544(D140)
        elif  num240==545:
            p3402=self.conv1545(D140)
        elif  num240==546:
            p3402=self.conv1546(D140)
        elif  num240==547:
            p3402=self.conv1547(D140)
        elif  num240==548:
            p3402=self.conv1548(D140)
        elif  num240==549:
            p3402=self.conv1549(D140) 
        elif  num240==550:
            p3402=self.conv1550(D140)
        elif  num240==551:
            p3402=self.conv1551(D140)
        elif  num240==552:
            p3402=self.conv1552(D140)
        elif  num240==553:
            p3402=self.conv1553(D140)
        elif  num240==554:
            p3402=self.conv1554(D140)
        elif  num240==555:
            p3402=self.conv1555(D140)
        elif  num240==556:
            p3402=self.conv1556(D140)
        elif  num240==557:
            p3402=self.conv1557(D140)
        elif  num240==558:
            p3402=self.conv1558(D140)
        elif  num240==559:
            p3402=self.conv1559(D140)
        elif  num240==560:
            p3402=self.conv1560(D140)
        elif  num240==561:
            p3402=self.conv1561(D140)
        elif  num240==562:
            p3402=self.conv1562(D140)
        elif  num240==563:
            p3402=self.conv1563(D140)
        elif  num240==564:
            p3402=self.conv1564(D140)
        elif  num240==565:
            p3402=self.conv1565(D140)
        elif  num240==566:
            p3402=self.conv1566(D140)
        elif  num240==567:
            p3402=self.conv1567(D140)
        elif  num240==568:
            p3402=self.conv1568(D140)
        elif  num240==569:
            p3402=self.conv1569(D140) 
        elif  num240==570:
            p3402=self.conv1570(D140)
        elif  num240==571:
            p3402=self.conv1571(D140)
        elif  num240==572:
            p3402=self.conv1572(D140)
        elif  num240==573:
            p3402=self.conv1573(D140)
        elif  num240==574:
            p3402=self.conv1574(D140)
        elif  num240==575:
            p3402=self.conv1575(D140)
        elif  num240==576:
            p3402=self.conv1576(D140)
        elif  num240==577:
            p3402=self.conv1577(D140)
        elif  num240==578:
            p3402=self.conv1578(D140)
        elif  num240==579:
            p3402=self.conv1579(D140) 
        elif  num240==580:
            p3402=self.conv1580(D140)
        elif  num240==581:
            p3402=self.conv1581(D140)
        elif  num240==582:
            p3402=self.conv1582(D140)
        elif  num240==583:
            p3402=self.conv1583(D140)
        elif  num240==584:
            p3402=self.conv1584(D140)
        elif  num240==585:
            p3402=self.conv1585(D140)
        elif  num240==586:
            p3402=self.conv1586(D140)
        elif  num240==587:
            p3402=self.conv1587(D140)
        elif  num240==588:
            p3402=self.conv1588(D140)
        elif  num240==589:
            p3402=self.conv1589(D140)
        elif  num240==590:
            p3402=self.conv1590(D140)
        elif  num240==591:
            p3402=self.conv1591(D140)
        elif  num240==592:
            p3402=self.conv1592(D140)
        elif  num240==593:
            p3402=self.conv1593(D140)
        elif  num240==594:
            p3402=self.conv1594(D140)
        elif  num240==595:
            p3402=self.conv1595(D140)
        elif  num240==596:
            p3402=self.conv1596(D140)
        elif  num240==597:
            p3402=self.conv1597(D140)
        elif  num240==598:
            p3402=self.conv1598(D140)
        elif  num240==599:
            p3402=self.conv1599(D140)
        elif  num240==600:
            p3402=self.conv1600(D140)
        elif  num240==601:
            p3402=self.conv1601(D140)
        elif  num240==602:
            p3402=self.conv1602(D140)
        elif  num240==603:
            p3402=self.conv1603(D140)
        elif  num240==604:
            p3402=self.conv1604(D140)
        elif  num240==605:
            p3402=self.conv1605(D140)
        elif  num240==606:
            p3402=self.conv1606(D140)
        elif  num240==607:
            p3402=self.conv1607(D140)
        elif  num240==608:
            p3402=self.conv1608(D140)
        elif  num240==609:
            p3402=self.conv1609(D140)                                                                                                                         
        elif  num240==610:
            p3402=self.conv1610(D140)
        elif  num240==611:
            p3402=self.conv1611(D140)
        elif  num240==612:
            p3402=self.conv1612(D140)
        elif  num240==613:
            p3402=self.conv1613(D140)
        elif  num240==614:
            p3402=self.conv1614(D140)
        elif  num240==615:
            p3402=self.conv1615(D140)
        elif  num240==616:
            p3402=self.conv1616(D140)
        elif  num240==617:
            p3402=self.conv1617(D140)
        elif  num240==618:
            p3402=self.conv1618(D140)
        elif  num240==619:
            p3402=self.conv1619(D140)                                                                                                                          
        elif  num240==620:
            p3402=self.conv1620(D140)
        elif  num240==621:
            p3402=self.conv1621(D140)
        elif  num240==622:
            p3402=self.conv1622(D140)
        elif  num240==623:
            p3402=self.conv1623(D140)
        elif  num240==624:
            p3402=self.conv1624(D140)
        elif  num240==625:
            p3402=self.conv1625(D140)
        elif  num240==626:
            p3402=self.conv1626(D140)
        elif  num240==627:
            p3402=self.conv1627(D140)
        elif  num240==628:
            p3402=self.conv1628(D140)
        elif  num240==629:
            p3402=self.conv1629(D140)  
        elif  num240==630:
            p3402=self.conv1630(D140)
        elif  num240==631:
            p3402=self.conv1631(D140)
        elif  num240==632:
            p3402=self.conv1632(D140)
        elif  num240==633:
            p3402=self.conv1633(D140)
        elif  num240==634:
            p3402=self.conv1634(D140)
        elif  num240==635:
            p3402=self.conv1635(D140)
        elif  num240==636:
            p3402=self.conv1636(D140)
        elif  num240==637:
            p3402=self.conv1637(D140)
        elif  num240==638:
            p3402=self.conv1638(D140)
        elif  num240==639:
            p3402=self.conv1639(D140)                                                                                                                          
        elif  num240==640:
            p3402=self.conv1640(D140)
        elif  num240==641:
            p3402=self.conv1641(D140)
        elif  num240==642:
            p3402=self.conv1642(D140)
        elif  num240==643:
            p3402=self.conv1643(D140)
        elif  num240==644:
            p3402=self.conv1644(D140)
        elif  num240==645:
            p3402=self.conv1645(D140)
        elif  num240==646:
            p3402=self.conv1646(D140)
        elif  num240==647:
            p3402=self.conv1647(D140)
        elif  num240==648:
            p3402=self.conv1648(D140)
        elif  num240==649:
            p3402=self.conv1649(D140)                                                                                                                          
        elif  num240==650:
            p3402=self.conv1650(D140)
        elif  num240==651:
            p3402=self.conv1651(D140)
        elif  num240==652:
            p3402=self.conv1652(D140)
        elif  num240==653:
            p3402=self.conv1653(D140)
        elif  num240==654:
            p3402=self.conv1654(D140)
        elif  num240==655:
            p3402=self.conv1655(D140)
        elif  num240==656:
            p3402=self.conv1656(D140)
        elif  num240==657:
            p3402=self.conv1657(D140)
        elif  num240==658:
            p3402=self.conv1658(D140)
        elif  num240==659:
            p3402=self.conv1659(D140)
        elif  num240==660:
            p3402=self.conv1660(D140)
        elif  num240==661:
            p3402=self.conv1661(D140)
        elif  num240==662:
            p3402=self.conv1662(D140)
        elif  num240==663:
            p3402=self.conv1663(D140)
        elif  num240==664:
            p3402=self.conv1664(D140)
        elif  num240==665:
            p3402=self.conv1665(D140)
        elif  num240==666:
            p3402=self.conv1666(D140)
        elif  num240==667:
            p3402=self.conv1667(D140)
        elif  num240==668:
            p3402=self.conv1668(D140)
        elif  num240==669:
            p3402=self.conv1669(D140) 
        elif  num240==670:
            p3402=self.conv1670(D140)
        elif  num240==671:
            p3402=self.conv1671(D140)
        elif  num240==672:
            p3402=self.conv1672(D140)
        elif  num240==673:
            p3402=self.conv1673(D140)
        elif  num240==674:
            p3402=self.conv1674(D140)
        elif  num240==675:
            p3402=self.conv1675(D140)
        elif  num240==676:
            p3402=self.conv1676(D140)
        elif  num240==677:
            p3402=self.conv1677(D140)
        elif  num240==678:
            p3402=self.conv1678(D140)
        elif  num240==679:
            p3402=self.conv1679(D140)
        elif  num240==680:
            p3402=self.conv1680(D140)
        elif  num240==681:
            p3402=self.conv1681(D140)
        elif  num240==682:
            p3402=self.conv1682(D140)
        elif  num240==683:
            p3402=self.conv1683(D140)
        elif  num240==684:
            p3402=self.conv1684(D140)
        elif  num240==685:
            p3402=self.conv1685(D140)
        elif  num240==686:
            p3402=self.conv1686(D140)
        elif  num240==687:
            p3402=self.conv1687(D140)
        elif  num240==688:
            p3402=self.conv1688(D140)
        elif  num240==689:
            p3402=self.conv1689(D140)
        elif  num240==690:
            p3402=self.conv1690(D140)
        elif  num240==691:
            p3402=self.conv1691(D140)
        elif  num240==692:
            p3402=self.conv1692(D140)
        elif  num240==693:
            p3402=self.conv1693(D140)
        elif  num240==694:
            p3402=self.conv1694(D140)
        elif  num240==695:
            p3402=self.conv1695(D140)
        elif  num240==696:
            p3402=self.conv1696(D140)
        elif  num240==697:
            p3402=self.conv1697(D140)
        elif  num240==698:
            p3402=self.conv1698(D140)
        elif  num240==699:
            p3402=self.conv1699(D140)
        elif  num240==700:
            p3402=self.conv1700(D140)
        elif  num240==701:
            p3402=self.conv1701(D140)
        elif  num240==702:
            p3402=self.conv1702(D140)
        elif  num240==703:
            p3402=self.conv1703(D140)
        elif  num240==704:
            p3402=self.conv1704(D140)
        elif  num240==705:
            p3402=self.conv1705(D140)
        elif  num240==706:
            p3402=self.conv1706(D140)
        elif  num240==707:
            p3402=self.conv1707(D140)
        elif  num240==708:
            p3402=self.conv1708(D140)
        elif  num240==709:
            p3402=self.conv1709(D140)
        elif  num240==710:
            p3402=self.conv1710(D140)
        elif  num240==711:
            p3402=self.conv1711(D140)
        elif  num240==712:
            p3402=self.conv1712(D140)
        elif  num240==713:
            p3402=self.conv1713(D140)
        elif  num240==714:
            p3402=self.conv1714(D140)
        elif  num240==715:
            p3402=self.conv1715(D140)
        elif  num240==716:
            p3402=self.conv1716(D140)
        elif  num240==717:
            p3402=self.conv1717(D140)
        elif  num240==718:
            p3402=self.conv1718(D140)
        elif  num240==719:
            p3402=self.conv1719(D140)
        elif  num240==720:
            p3402=self.conv1720(D140)
        elif  num240==721:
            p3402=self.conv1721(D140)
        elif  num240==722:
            p3402=self.conv1722(D140)
        elif  num240==723:
            p3402=self.conv1723(D140)
        elif  num240==724:
            p3402=self.conv1724(D140)
        elif  num240==725:
            p3402=self.conv1725(D140)
        elif  num240==726:
            p3402=self.conv1726(D140)
        elif  num240==727:
            p3402=self.conv1727(D140)
        elif  num240==728:
            p3402=self.conv1728(D140)
        elif  num240==729:
            p3402=self.conv1729(D140)
        elif  num240==730:
            p3402=self.conv1730(D140)
        elif  num240==731:
            p3402=self.conv1731(D140)
        elif  num240==732:
            p3402=self.conv1732(D140)
        elif  num240==733:
            p3402=self.conv1733(D140)
        elif  num240==734:
            p3402=self.conv1734(D140)
        elif  num240==735:
            p3402=self.conv1735(D140)
        elif  num240==736:
            p3402=self.conv1736(D140)
        elif  num240==737:
            p3402=self.conv1737(D140)
        elif  num240==738:
            p3402=self.conv1738(D140)
        elif  num240==739:
            p3402=self.conv1739(D140)                                                                                                                          
        elif  num240==740:
            p3402=self.conv1740(D140)
        elif  num240==741:
            p3402=self.conv1741(D140)
        elif  num240==742:
            p3402=self.conv1742(D140)
        elif  num240==743:
            p3402=self.conv1743(D140)
        elif  num240==744:
            p3402=self.conv1744(D140)
        elif  num240==745:
            p3402=self.conv1745(D140)
        elif  num240==746:
            p3402=self.conv1746(D140)
        elif  num240==747:
            p3402=self.conv1747(D140)
        elif  num240==748:
            p3402=self.conv1748(D140)
        elif  num240==749:
            p3402=self.conv1749(D140)
        elif  num240==750:
            p3402=self.conv1750(D140)
        elif  num240==751:
            p3402=self.conv1751(D140)
        elif  num240==752:
            p3402=self.conv1752(D140)
        elif  num240==753:
            p3402=self.conv1753(D140)
        elif  num240==754:
            p3402=self.conv1754(D140)
        elif  num240==755:
            p3402=self.conv1755(D140)
        elif  num240==756:
            p3402=self.conv1756(D140)
        elif  num240==757:
            p3402=self.conv1757(D140)
        elif  num240==758:
            p3402=self.conv1758(D140)
        elif  num240==759:
            p3402=self.conv1759(D140)
        elif  num240==760:
            p3402=self.conv1760(D140)
        elif  num240==761:
            p3402=self.conv1761(D140)
        elif  num240==762:
            p3402=self.conv1762(D140)
        elif  num240==763:
            p3402=self.conv1763(D140)
        elif  num240==764:
            p3402=self.conv1764(D140)
        elif  num240==765:
            p3402=self.conv1765(D140)
        elif  num240==766:
            p3402=self.conv1766(D140)
        elif  num240==767:
            p3402=self.conv1767(D140)
        elif  num240==768:
            p3402=self.conv1768(D140)
        elif  num240==769:
            p3402=self.conv1769(D140) 
        elif  num240==770:
            p3402=self.conv1770(D140)
        elif  num240==771:
            p3402=self.conv1771(D140)
        elif  num240==772:
            p3402=self.conv1772(D140)
        elif  num240==773:
            p3402=self.conv1773(D140)
        elif  num240==774:
            p3402=self.conv1774(D140)
        elif  num240==775:
            p3402=self.conv1775(D140)
        elif  num240==776:
            p3402=self.conv1776(D140)
        elif  num240==777:
            p3402=self.conv1777(D140)
        elif  num240==778:
            p3402=self.conv1778(D140)
        elif  num240==779:
            p3402=self.conv1779(D140) 
        elif  num240==780:
            p3402=self.conv1780(D140)
        elif  num240==781:
            p3402=self.conv1781(D140)
        elif  num240==782:
            p3402=self.conv1782(D140)
        elif  num240==783:
            p3402=self.conv1783(D140)
        elif  num240==784:
            p3402=self.conv1784(D140)
        elif  num240==785:
            p3402=self.conv1785(D140)
        elif  num240==786:
            p3402=self.conv1786(D140)
        elif  num240==787:
            p3402=self.conv1787(D140)
        elif  num240==788:
            p3402=self.conv1788(D140)
        elif  num240==789:
            p3402=self.conv1789(D140) 
        elif  num240==790:
            p3402=self.conv1790(D140)
        elif  num240==791:
            p3402=self.conv1791(D140)
        elif  num240==792:
            p3402=self.conv1792(D140)
        elif  num240==793:
            p3402=self.conv1793(D140)
        elif  num240==794:
            p3402=self.conv1794(D140)
        elif  num240==795:
            p3402=self.conv1795(D140)
        elif  num240==796:
            p3402=self.conv1796(D140)
        elif  num240==797:
            p3402=self.conv1797(D140)
        elif  num240==798:
            p3402=self.conv1798(D140)
        elif  num240==799:
            p3402=self.conv1799(D140) 
        elif  num240==800:
            p3402=self.conv1800(D140)
        elif  num240==801:
            p3402=self.conv1801(D140)
        elif  num240==802:
            p3402=self.conv1802(D140)
        elif  num240==803:
            p3402=self.conv1803(D140)
        elif  num240==804:
            p3402=self.conv1804(D140)
        elif  num240==805:
            p3402=self.conv1805(D140)
        elif  num240==806:
            p3402=self.conv1806(D140)
        elif  num240==807:
            p3402=self.conv1807(D140)
        elif  num240==808:
            p3402=self.conv1808(D140)
        elif  num240==809:
            p3402=self.conv1809(D140)
        elif  num240==810:
            p3402=self.conv1810(D140)
        elif  num240==811:
            p3402=self.conv1811(D140)
        elif  num240==812:
            p3402=self.conv1812(D140)
        elif  num240==813:
            p3402=self.conv1813(D140)
        elif  num240==814:
            p3402=self.conv1814(D140)
        elif  num240==815:
            p3402=self.conv1815(D140)
        elif  num240==816:
            p3402=self.conv1816(D140)
        elif  num240==817:
            p3402=self.conv1817(D140)
        elif  num240==818:
            p3402=self.conv1818(D140)
        elif  num240==819:
            p3402=self.conv1819(D140)
        elif  num240==820:
            p3402=self.conv1820(D140)
        elif  num240==821:
            p3402=self.conv1821(D140)
        elif  num240==822:
            p3402=self.conv1822(D140)
        elif  num240==823:
            p3402=self.conv1823(D140)
        elif  num240==824:
            p3402=self.conv1824(D140)
        elif  num240==825:
            p3402=self.conv1825(D140)
        elif  num240==826:
            p3402=self.conv1826(D140)
        elif  num240==827:
            p3402=self.conv1827(D140)
        elif  num240==828:
            p3402=self.conv1828(D140)
        elif  num240==829:
            p3402=self.conv1829(D140)                                                                                                                          
        elif  num240==830:
            p3402=self.conv1830(D140)
        elif  num240==831:
            p3402=self.conv1831(D140)
        elif  num240==832:
            p3402=self.conv1832(D140)
        elif  num240==833:
            p3402=self.conv1833(D140)
        elif  num240==834:
            p3402=self.conv1834(D140)
        elif  num240==835:
            p3402=self.conv1835(D140)
        elif  num240==836:
            p3402=self.conv1836(D140)
        elif  num240==837:
            p3402=self.conv1837(D140)
        elif  num240==838:
            p3402=self.conv1838(D140)
        elif  num240==839:
            p3402=self.conv1839(D140)
        elif  num240==840:
            p3402=self.conv1840(D140)
        elif  num240==841:
            p3402=self.conv1841(D140)
        elif  num240==842:
            p3402=self.conv1842(D140)
        elif  num240==843:
            p3402=self.conv1843(D140)
        elif  num240==844:
            p3402=self.conv1844(D140)
        elif  num240==845:
            p3402=self.conv1845(D140)
        elif  num240==846:
            p3402=self.conv1846(D140)
        elif  num240==847:
            p3402=self.conv1847(D140)
        elif  num240==848:
            p3402=self.conv1848(D140)
        elif  num240==849:
            p3402=self.conv1849(D140)
        elif  num240==850:
            p3402=self.conv1850(D140)
        elif  num240==851:
            p3402=self.conv1851(D140)
        elif  num240==852:
            p3402=self.conv1852(D140)
        elif  num240==853:
            p3402=self.conv1853(D140)
        elif  num240==854:
            p3402=self.conv1854(D140)
        elif  num240==855:
            p3402=self.conv1855(D140)
        elif  num240==856:
            p3402=self.conv1856(D140)
        elif  num240==857:
            p3402=self.conv1857(D140)
        elif  num240==858:
            p3402=self.conv1858(D140)
        elif  num240==859:
            p3402=self.conv1859(D140)
        elif  num240==860:
            p3402=self.conv1860(D140)
        elif  num240==861:
            p3402=self.conv1861(D140)
        elif  num240==862:
            p3402=self.conv1862(D140)
        elif  num240==863:
            p3402=self.conv1863(D140)
        elif  num240==864:
            p3402=self.conv1864(D140)
        elif  num240==865:
            p3402=self.conv1865(D140)
        elif  num240==866:
            p3402=self.conv1866(D140)
        elif  num240==867:
            p3402=self.conv1867(D140)
        elif  num240==868:
            p3402=self.conv1868(D140)
        elif  num240==869:
            p3402=self.conv1869(D140) 
        elif  num240==870:
            p3402=self.conv1870(D140)
        elif  num240==871:
            p3402=self.conv1871(D140)
        elif  num240==872:
            p3402=self.conv1872(D140)
        elif  num240==873:
            p3402=self.conv1873(D140)
        elif  num240==874:
            p3402=self.conv1874(D140)
        elif  num240==875:
            p3402=self.conv1875(D140)
        elif  num240==876:
            p3402=self.conv1876(D140)
        elif  num240==877:
            p3402=self.conv1877(D140)
        elif  num240==878:
            p3402=self.conv1878(D140)
        elif  num240==879:
            p3402=self.conv1879(D140)
        elif  num240==880:
            p3402=self.conv1880(D140)
        elif  num240==881:
            p3402=self.conv1881(D140)
        elif  num240==882:
            p3402=self.conv1882(D140)
        elif  num240==883:
            p3402=self.conv1883(D140)
        elif  num240==884:
            p3402=self.conv1884(D140)
        elif  num240==885:
            p3402=self.conv1885(D140)
        elif  num240==886:
            p3402=self.conv1886(D140)
        elif  num240==887:
            p3402=self.conv1887(D140)
        elif  num240==888:
            p3402=self.conv1888(D140)
        elif  num240==889:
            p3402=self.conv1889(D140)  
        elif  num240==890:
            p3402=self.conv1890(D140)
        elif  num240==891:
            p3402=self.conv1891(D140)
        elif  num240==892:
            p3402=self.conv1892(D140)
        elif  num240==893:
            p3402=self.conv1893(D140)
        elif  num240==894:
            p3402=self.conv1894(D140)
        elif  num240==895:
            p3402=self.conv1895(D140)
        elif  num240==896:
            p3402=self.conv1896(D140)
        elif  num240==897:
            p3402=self.conv1897(D140)
        elif  num240==898:
            p3402=self.conv1898(D140)
        elif  num240==899:
            p3402=self.conv1899(D140)
        elif  num240==900:
            p3402=self.conv1900(D140)
        elif  num240==901:
            p3402=self.conv1901(D140)
        elif  num240==902:
            p3402=self.conv1902(D140)
        elif  num240==903:
            p3402=self.conv1903(D140)
        elif  num240==904:
            p3402=self.conv1904(D140)
        elif  num240==905:
            p3402=self.conv1905(D140)
        elif  num240==906:
            p3402=self.conv1906(D140)
        elif  num240==907:
            p3402=self.conv1907(D140)
        elif  num240==908:
            p3402=self.conv1908(D140)
        elif  num240==909:
            p3402=self.conv1909(D140)
        elif  num240==910:
            p3402=self.conv1910(D140)
        elif  num240==911:
            p3402=self.conv1911(D140)
        elif  num240==912:
            p3402=self.conv1912(D140)
        elif  num240==913:
            p3402=self.conv1913(D140)
        elif  num240==914:
            p3402=self.conv1914(D140)
        elif  num240==915:
            p3402=self.conv1915(D140)
        elif  num240==916:
            p3402=self.conv1916(D140)
        elif  num240==917:
            p3402=self.conv1917(D140)
        elif  num240==918:
            p3402=self.conv1918(D140)
        elif  num240==919:
            p3402=self.conv1919(D140)
        elif  num240==920:
            p3402=self.conv1920(D140)
        elif  num240==921:
            p3402=self.conv1921(D140)
        elif  num240==922:
            p3402=self.conv1922(D140)
        elif  num240==923:
            p3402=self.conv1923(D140)
        elif  num240==924:
            p3402=self.conv1924(D140)
        elif  num240==925:
            p3402=self.conv1925(D140)
        elif  num240==926:
            p3402=self.conv1926(D140)
        elif  num240==927:
            p3402=self.conv1927(D140)
        elif  num240==928:
            p3402=self.conv1928(D140)
        elif  num240==929:
            p3402=self.conv1929(D140)
        elif  num240==930:
            p3402=self.conv1930(D140)
        elif  num240==931:
            p3402=self.conv1931(D140)
        elif  num240==932:
            p3402=self.conv1932(D140)
        elif  num240==933:
            p3402=self.conv1933(D140)
        elif  num240==934:
            p3402=self.conv1934(D140)
        elif  num240==935:
            p3402=self.conv1935(D140)
        elif  num240==936:
            p3402=self.conv1936(D140)
        elif  num240==937:
            p3402=self.conv1937(D140)
        elif  num240==938:
            p3402=self.conv1938(D140)
        elif  num240==939:
            p3402=self.conv1939(D140) 
        elif  num240==940:
            p3402=self.conv1940(D140)
        elif  num240==941:
            p3402=self.conv1941(D140)
        elif  num240==942:
            p3402=self.conv1942(D140)
        elif  num240==943:
            p3402=self.conv1943(D140)
        elif  num240==944:
            p3402=self.conv1944(D140)
        elif  num240==945:
            p3402=self.conv1945(D140)
        elif  num240==946:
            p3402=self.conv1946(D140)
        elif  num240==947:
            p3402=self.conv1947(D140)
        elif  num240==948:
            p3402=self.conv1948(D140)
        elif  num240==949:
            p3402=self.conv1949(D140)                                                                                                                          
        elif  num240==950:
            p3402=self.conv1950(D140)
        elif  num240==951:
            p3402=self.conv1951(D140)
        elif  num240==952:
            p3402=self.conv1952(D140)
        elif  num240==953:
            p3402=self.conv1953(D140)
        elif  num240==954:
            p3402=self.conv1954(D140)
        elif  num240==955:
            p3402=self.conv1955(D140)
        elif  num240==956:
            p3402=self.conv1956(D140)
        elif  num240==957:
            p3402=self.conv1957(D140)
        elif  num240==958:
            p3402=self.conv1958(D140)
        elif  num240==959:
            p3402=self.conv1959(D140)
        elif  num240==960:
            p3402=self.conv1960(D140)
        elif  num240==961:
            p3402=self.conv1961(D140)
        elif  num240==962:
            p3402=self.conv1962(D140)
        elif  num240==963:
            p3402=self.conv1963(D140)
        elif  num240==964:
            p3402=self.conv1964(D140)
        elif  num240==965:
            p3402=self.conv1965(D140)
        elif  num240==966:
            p3402=self.conv1966(D140)
        elif  num240==967:
            p3402=self.conv1967(D140)
        elif  num240==968:
            p3402=self.conv1968(D140)
        elif  num240==969:
            p3402=self.conv1969(D140) 
        elif  num240==970:
            p3402=self.conv1970(D140)
        elif  num240==971:
            p3402=self.conv1971(D140)
        elif  num240==972:
            p3402=self.conv1972(D140)
        elif  num240==973:
            p3402=self.conv1973(D140)
        elif  num240==974:
            p3402=self.conv1974(D140)
        elif  num240==975:
            p3402=self.conv1975(D140)
        elif  num240==976:
            p3402=self.conv1976(D140)
        elif  num240==977:
            p3402=self.conv1977(D140)
        elif  num240==978:
            p3402=self.conv1978(D140)
        elif  num240==979:
            p3402=self.conv1979(D140) 
        elif  num240==980:
            p3402=self.conv1980(D140)
        elif  num240==981:
            p3402=self.conv1981(D140)
        elif  num240==982:
            p3402=self.conv1982(D140)
        elif  num240==983:
            p3402=self.conv1983(D140)
        elif  num240==984:
            p3402=self.conv1984(D140)
        elif  num240==985:
            p3402=self.conv1985(D140)
        elif  num240==986:
            p3402=self.conv1986(D140)
        elif  num240==987:
            p3402=self.conv1987(D140)
        elif  num240==988:
            p3402=self.conv1988(D140)
        elif  num240==989:
            p3402=self.conv1989(D140)
        elif  num240==990:
            p3402=self.conv1990(D140)
        elif  num240==991:
            p3402=self.conv1991(D140)
        elif  num240==992:
            p3402=self.conv1992(D140)
        elif  num240==993:
            p3402=self.conv1993(D140)
        elif  num240==994:
            p3402=self.conv1994(D140)
        elif  num240==995:
            p3402=self.conv1995(D140)
        elif  num240==996:
            p3402=self.conv1996(D140)
        elif  num240==997:
            p3402=self.conv1997(D140)
        elif  num240==998:
            p3402=self.conv1998(D140)
        elif  num240==999:
            p3402=self.conv1999(D140) 
        elif  num240==1000:
            p3402=self.conv11000(D140)
        elif  num240==1001:
            p3402=self.conv11001(D140)
        elif  num240==1002:
            p3402=self.conv11002(D140)
        elif  num240==1003:
            p3402=self.conv11003(D140)
        elif  num240==1004:
            p3402=self.conv11004(D140)
        elif  num240==1005:
            p3402=self.conv11005(D140)
        elif  num240==1006:
            p3402=self.conv11006(D140)
        elif  num240==1007:
            p3402=self.conv11007(D140)
        elif  num240==1008:
            p3402=self.conv11008(D140)
        elif  num240==1009:
            p3402=self.conv11009(D140) 
        elif  num240==1010:
            p3402=self.conv11010(D140)
        elif  num240==1011:
            p3402=self.conv11011(D140)
        elif  num240==1012:
            p3402=self.conv11012(D140)
        elif  num240==1013:
            p3402=self.conv11013(D140)
        elif  num240==1014:
            p3402=self.conv11014(D140)
        elif  num240==1015:
            p3402=self.conv11015(D140)
        elif  num240==1016:
            p3402=self.conv11016(D140)
        elif  num240==1017:
            p3402=self.conv11017(D140)
        elif  num240==1018:
            p3402=self.conv11018(D140)
        elif  num240==1019:
            p3402=self.conv11019(D140)
        elif  num240==1020:
            p3402=self.conv11020(D140)
        elif  num240==1021:
            p3402=self.conv11021(D140)
        elif  num240==1022:
            p3402=self.conv11022(D140)
        elif  num240==1023:
            p3402=self.conv11023(D140)
        elif  num240==1024:
            p3402=self.conv11024(D140) 
            
        if num041==1:
            p3441=self.conv11(B141)
        elif num041==2:
            p3441=self.conv12(B141)
        elif num041==3:
            p3441=self.conv13(B141)
        elif num041==4:
            p3441=self.conv14(B141)
        elif num041==5:
            p3441=self.conv15(B141)
        elif num041==6:
            p3441=self.conv16(B141)
        elif num041==7:
            p3441=self.conv17(B141)
        elif num041==8:
            p3441=self.conv18(B141)
        elif num041==9:
            p3441=self.conv19(B141)
        elif num041==10:
            p3441=self.conv110(B141)
        elif num041==11:
            p3441=self.conv111(B141)
        elif num041==12:
            p3441=self.conv112(B141)
        elif num041==13:
            p3441=self.conv113(B141)
        elif num041==14:
            p3441=self.conv114(B141)
        elif num041==15:
            p3441=self.conv115(B141)
        elif num041==16:
            p3441=self.conv116(B141)
        elif num041==17:
            p3441=self.conv117(B141)
        elif num041==18:
            p3441=self.conv118(B141)
        elif num041==19:
            p3441=self.conv119(B141)
        elif num041==20:
            p3441=self.conv120(B141)
        elif num041==21:
            p3441=self.conv121(B141)
        elif num041==22:
            p3441=self.conv122(B141)
        elif num041==23:
            p3441=self.conv123(B141)
        elif num041==24:
            p3441=self.conv124(B141)
        elif num041==25:
            p3441=self.conv125(B141)
        elif num041==26:
            p3441=self.conv126(B141)
        elif num041==27:
            p3441=self.conv127(B141)
        elif num041==28:
            p3441=self.conv128(B141)
        elif num041==29:
            p3441=self.conv129(B141)
        elif num041==30:
            p3441=self.conv130(B141)
        elif num041==31:
            p3441=self.conv131(B141)
        elif num041==32:
            p3441=self.conv132(B141)
        elif num041==33:
            p3441=self.conv133(B141)
        elif num041==34:
            p3441=self.conv134(B141)
        elif num041==35:
            p3441=self.conv135(B141)
        elif num041==36:
            p3441=self.conv136(B141)
        elif num041==37:
            p3441=self.conv137(B141)
        elif num041==38:
            p3441=self.conv138(B141)
        elif num041==39:
            p3441=self.conv139(B141)
        elif num041==40:
            p3441=self.conv140(B141)
        elif num041==41:
            p3441=self.conv141(B141)
        elif num041==42:
            p3441=self.conv142(B141)
        elif num041==43:
            p3441=self.conv143(B141)
        elif num041==44:
            p3441=self.conv144(B141)
        elif num041==45:
            p3441=self.conv145(B141)
        elif num041==46:
            p3441=self.conv146(B141)
        elif num041==47:
            p3441=self.conv147(B141)
        elif num041==48:
            p3441=self.conv148(B141)
        elif num041==49:
            p3441=self.conv149(B141)
        elif num041==50:
            p3441=self.conv150(B141)
        elif num041==51:
            p3441=self.conv151(B141)
        elif num041==52:
            p3441=self.conv152(B141)
        elif num041==53:
            p3441=self.conv153(B141)
        elif num041==54:
            p3441=self.conv154(B141)
        elif num041==55:
            p3441=self.conv155(B141)
        elif num041==56:
            p3441=self.conv156(B141)
        elif num041==57:
            p3441=self.conv157(B141)
        elif num041==58:
            p3441=self.conv158(B141)
        elif num041==59:
            p3441=self.conv159(B141)
        elif num041==60:
            p3441=self.conv160(B141)
        elif num041==61:
            p3441=self.conv161(B141)
        elif num041==62:
            p3441=self.conv162(B141)
        elif num041==63:
            p3441=self.conv163(B141)
        elif num041==64:
            p3441=self.conv164(B141)
        
        if  num141==1:
            p3411=self.conv11(C141)
        elif  num141==2:
            p3411=self.conv12(C141)
        elif  num141==3:
            p3411=self.conv13(C141)
        elif  num141==4:
            p3411=self.conv14(C141)
        elif  num141==5:
            p3411=self.conv15(C141)
        elif  num141==6:
            p3411=self.conv16(C141)
        elif  num141==7:
            p3411=self.conv17(C141)
        elif  num141==8:
            p3411=self.conv18(C141)
        elif  num141==9:
            p3411=self.conv19(C141)
        elif  num141==10:
            p3411=self.conv110(C141)
        elif  num141==11:
            p3411=self.conv111(C141)
        elif  num141==12:
            p3411=self.conv112(C141)
        elif  num141==13:
            p3411=self.conv113(C141)
        elif  num141==14:
            p3411=self.conv114(C141)
        elif  num141==15:
            p3411=self.conv115(C141)
        elif  num141==16:
            p3411=self.conv116(C141)
        elif  num141==17:
            p3411=self.conv117(C141)
        elif  num141==18:
            p3411=self.conv118(C141)
        elif  num141==19:
            p3411=self.conv119(C141)
        elif  num141==20:
            p3411=self.conv120(C141)
        elif  num141==21:
            p3411=self.conv121(C141)
        elif  num141==22:
            p3411=self.conv122(C141)
        elif  num141==23:
            p3411=self.conv123(C141)
        elif  num141==24:
            p3411=self.conv124(C141)
        elif  num141==25:
            p3411=self.conv125(C141)
        elif  num141==26:
            p3411=self.conv126(C141)
        elif  num141==27:
            p3411=self.conv127(C141)
        elif  num141==28:
            p3411=self.conv128(C141)
        elif  num141==29:
            p3411=self.conv129(C141)
        elif  num141==30:
            p3411=self.conv130(C141)
        elif  num141==31:
            p3411=self.conv131(C141)
        elif  num141==32:
            p3411=self.conv132(C141)
        elif  num141==33:
            p3411=self.conv133(C141)
        elif  num141==34:
            p3411=self.conv134(C141)
        elif  num141==35:
            p3411=self.conv135(C141)
        elif  num141==36:
            p3411=self.conv136(C141)
        elif  num141==37:
            p3411=self.conv137(C141)
        elif  num141==38:
            p3411=self.conv138(C141)
        elif  num141==39:
            p3411=self.conv139(C141)
        elif  num141==40:
            p3411=self.conv140(C141)
        elif  num141==41:
            p3411=self.conv141(C141)
        elif  num141==42:
            p3411=self.conv142(C141)
        elif  num141==43:
            p3411=self.conv143(C141)
        elif  num141==44:
            p3411=self.conv144(C141)
        elif  num141==45:
            p3411=self.conv145(C141)
        elif  num141==46:
            p3411=self.conv146(C141)
        elif  num141==47:
            p3411=self.conv147(C141)
        elif  num141==48:
            p3411=self.conv148(C141)
        elif  num141==49:
            p3411=self.conv149(C141)
        elif  num141==50:
            p3411=self.conv150(C141)
        elif  num141==51:
            p3411=self.conv151(C141)
        elif  num141==52:
            p3411=self.conv152(C141)
        elif  num141==53:
            p3411=self.conv153(C141)
        elif  num141==54:
            p3411=self.conv154(C141)
        elif  num141==55:
            p3411=self.conv155(C141)
        elif  num141==56:
            p3411=self.conv156(C141)
        elif  num141==57:
            p3411=self.conv157(C141)
        elif  num141==58:
            p3411=self.conv158(C141)
        elif  num141==59:
            p3411=self.conv159(C141)
        elif  num141==60:
            p3411=self.conv160(C141)
        elif  num141==61:
            p3411=self.conv161(C141)
        elif  num141==62:
            p3411=self.conv162(C141)
        elif  num141==63:
            p3411=self.conv163(C141)
        elif  num141==64:
            p3411=self.conv164(C141)
        elif  num141==65:
            p3411=self.conv165(C141)
        elif  num141==66:
            p3411=self.conv166(C141)
        elif  num141==67:
            p3411=self.conv167(C141)
        elif  num141==68:
            p3411=self.conv168(C141)
        elif  num141==69:
            p3411=self.conv169(C141)
        elif  num141==70:
            p3411=self.conv170(C141)
        elif  num141==71:
            p3411=self.conv171(C141)
        elif  num141==72:
            p3411=self.conv172(C141)
        elif  num141==73:
            p3411=self.conv173(C141)
        elif  num141==74:
            p3411=self.conv174(C141)
        elif  num141==75:
            p3411=self.conv175(C141)
        elif  num141==76:
            p3411=self.conv176(C141)
        elif  num141==77:
            p3411=self.conv177(C141)
        elif  num141==78:
            p3411=self.conv178(C141)
        elif  num141==79:
            p3411=self.conv179(C141)
        elif  num141==80:
            p3411=self.conv180(C141)
        elif  num141==81:
            p3411=self.conv181(C141)
        elif  num141==82:
            p3411=self.conv182(C141)
        elif  num141==83:
            p3411=self.conv183(C141)
        elif  num141==84:
            p3411=self.conv184(C141)
        elif  num141==85:
            p3411=self.conv185(C141)
        elif  num141==86:
            p3411=self.conv186(C141)
        elif  num141==87:
            p3411=self.conv187(C141)
        elif  num141==88:
            p3411=self.conv188(C141)
        elif  num141==89:
            p3411=self.conv189(C141)    
        elif  num141==90:
            p3411=self.conv190(C141)
        elif  num141==91:
            p3411=self.conv191(C141)
        elif  num141==92:
            p3411=self.conv192(C141)
        elif  num141==93:
            p3411=self.conv193(C141)
        elif  num141==94:
            p3411=self.conv194(C141)
        elif  num141==95:
            p3411=self.conv195(C141)
        elif  num141==96:
            p3411=self.conv196(C141)
        elif  num141==97:
            p3411=self.conv197(C141)
        elif  num141==98:
            p3411=self.conv198(C141)
        elif  num141==99:
            p3411=self.conv199(C141) 
        elif  num141==100:
            p3411=self.conv1100(C141)
        elif  num141==101:
            p3411=self.conv1101(C141)
        elif  num141==102:
            p3411=self.conv1102(C141)
        elif  num141==103:
            p3411=self.conv1103(C141)
        elif  num141==104:
            p3411=self.conv1104(C141)
        elif  num141==105:
            p3411=self.conv1105(C141)
        elif  num141==106:
            p3411=self.conv1106(C141)
        elif  num141==107:
            p3411=self.conv1107(C141)
        elif  num141==108:
            p3411=self.conv1108(C141)
        elif  num141==109:
            p3411=self.conv1109(C141)
        elif  num141==110:
            p3411=self.conv1110(C141)
        elif  num141==111:
            p3411=self.conv1111(C141)
        elif  num141==112:
            p3411=self.conv1112(C141)
        elif  num141==113:
            p3411=self.conv1113(C141)
        elif  num141==114:
            p3411=self.conv1114(C141)
        elif  num141==115:
            p3411=self.conv1115(C141)
        elif  num141==116:
            p3411=self.conv1116(C141)
        elif  num141==117:
            p3411=self.conv1117(C141)
        elif  num141==118:
            p3411=self.conv1118(C141)
        elif  num141==119:
            p3411=self.conv1119(C141) 
        elif  num141==120:
            p3411=self.conv1120(C141)
        elif  num141==121:
            p3411=self.conv1121(C141)
        elif  num141==122:
            p3411=self.conv1122(C141)
        elif  num141==123:
            p3411=self.conv1123(C141)
        elif  num141==124:
            p3411=self.conv1124(C141)
        elif  num141==125:
            p3411=self.conv1125(C141)
        elif  num141==126:
            p3411=self.conv1126(C141)
        elif  num141==127:
            p3411=self.conv1127(C141)
        elif  num141==128:
            p3411=self.conv1128(C141)
        elif  num141==129:
            p3411=self.conv1129(C141) 
        elif  num141==130:
            p3411=self.conv1130(C141)
        elif  num141==131:
            p3411=self.conv1131(C141)
        elif  num141==132:
            p3411=self.conv1132(C141)
        elif  num141==133:
            p3411=self.conv1133(C141)
        elif  num141==134:
            p3411=self.conv1134(C141)
        elif  num141==135:
            p3411=self.conv1135(C141)
        elif  num141==136:
            p3411=self.conv1136(C141)
        elif  num141==137:
            p3411=self.conv1137(C141)
        elif  num141==138:
            p3411=self.conv1138(C141)
        elif  num141==139:
            p3411=self.conv1139(C141)
        elif  num141==140:
            p3411=self.conv1140(C141)
        elif  num141==141:
            p3411=self.conv1141(C141)
        elif  num141==142:
            p3411=self.conv1142(C141)
        elif  num141==143:
            p3411=self.conv1143(C141)
        elif  num141==144:
            p3411=self.conv1144(C141)
        elif  num141==145:
            p3411=self.conv1145(C141)
        elif  num141==146:
            p3411=self.conv1146(C141)
        elif  num141==147:
            p3411=self.conv1147(C141)
        elif  num141==148:
            p3411=self.conv1148(C141)
        elif  num141==149:
            p3411=self.conv1149(C141) 
        elif  num141==150:
            p3411=self.conv1150(C141)
        elif  num141==151:
            p3411=self.conv1151(C141)
        elif  num141==152:
            p3411=self.conv1152(C141)
        elif  num141==153:
            p3411=self.conv1153(C141)
        elif  num141==154:
            p3411=self.conv1154(C141)
        elif  num141==155:
            p3411=self.conv1155(C141)
        elif  num141==156:
            p3411=self.conv1156(C141)
        elif  num141==157:
            p3411=self.conv1157(C141)
        elif  num141==158:
            p3411=self.conv1158(C141)
        elif  num141==159:
            p3411=self.conv1159(C141) 
        elif  num141==160:
            p3411=self.conv1160(C141)
        elif  num141==161:
            p3411=self.conv1161(C141)
        elif  num141==162:
            p3411=self.conv1162(C141)
        elif  num141==163:
            p3411=self.conv1163(C141)
        elif  num141==164:
            p3411=self.conv1164(C141)
        elif  num141==165:
            p3411=self.conv1165(C141)
        elif  num141==166:
            p3411=self.conv1166(C141)
        elif  num141==167:
            p3411=self.conv1167(C141)
        elif  num141==168:
            p3411=self.conv1168(C141)
        elif  num141==169:
            p3411=self.conv1169(C141) 
        elif  num141==170:
            p3411=self.conv1170(C141)
        elif  num141==171:
            p3411=self.conv1171(C141)
        elif  num141==172:
            p3411=self.conv1172(C141)
        elif  num141==173:
            p3411=self.conv1173(C141)
        elif  num141==174:
            p3411=self.conv1174(C141)
        elif  num141==175:
            p3411=self.conv1175(C141)
        elif  num141==176:
            p3411=self.conv1176(C141)
        elif  num141==177:
            p3411=self.conv1177(C141)
        elif  num141==178:
            p3411=self.conv1178(C141)
        elif  num141==179:
            p3411=self.conv1179(C141)                                                                                              
        elif  num141==180:
            p3411=self.conv1180(C141)
        elif  num141==181:
            p3411=self.conv1181(C141)
        elif  num141==182:
            p3411=self.conv1182(C141)
        elif  num141==183:
            p3411=self.conv1183(C141)
        elif  num141==184:
            p3411=self.conv1184(C141)
        elif  num141==185:
            p3411=self.conv1185(C141)
        elif  num141==186:
            p3411=self.conv1186(C141)
        elif  num141==187:
            p3411=self.conv1187(C141)
        elif  num141==188:
            p3411=self.conv1188(C141)
        elif  num141==189:
            p3411=self.conv1189(C141) 
        elif  num141==190:
            p3411=self.conv1190(C141)
        elif  num141==191:
            p3411=self.conv1191(C141)
        elif  num141==192:
            p3411=self.conv1192(C141)
        elif  num141==193:
            p3411=self.conv1193(C141)
        elif  num141==194:
            p3411=self.conv1194(C141)
        elif  num141==195:
            p3411=self.conv1195(C141)
        elif  num141==196:
            p3411=self.conv1196(C141)
        elif  num141==197:
            p3411=self.conv1197(C141)
        elif  num141==198:
            p3411=self.conv1198(C141)
        elif  num141==199:
            p3411=self.conv1199(C141)
        elif  num141==200:
            p3411=self.conv1200(C141)
        elif  num141==201:
            p3411=self.conv1201(C141)
        elif  num141==202:
            p3411=self.conv1202(C141)
        elif  num141==203:
            p3411=self.conv1203(C141)
        elif  num141==204:
            p3411=self.conv1204(C141)
        elif  num141==205:
            p3411=self.conv1205(C141)
        elif  num141==206:
            p3411=self.conv1206(C141)
        elif  num141==207:
            p3411=self.conv1207(C141)
        elif  num141==208:
            p3411=self.conv1208(C141)
        elif  num141==209:
            p3411=self.conv1209(C141)
        elif  num141==210:
            p3411=self.conv1210(C141)
        elif  num141==211:
            p3411=self.conv1211(C141)
        elif  num141==212:
            p3411=self.conv1212(C141)
        elif  num141==213:
            p3411=self.conv1213(C141)
        elif  num141==214:
            p3411=self.conv1214(C141)
        elif  num141==215:
            p3411=self.conv1215(C141)
        elif  num141==216:
            p3411=self.conv1216(C141)
        elif  num141==217:
            p3411=self.conv1217(C141)
        elif  num141==218:
            p3411=self.conv1218(C141)
        elif  num141==219:
            p3411=self.conv1219(C141)
        elif  num141==220:
            p3411=self.conv1220(C141)
        elif  num141==221:
            p3411=self.conv1221(C141)
        elif  num141==222:
            p3411=self.conv1222(C141)
        elif  num141==223:
            p3411=self.conv1223(C141)
        elif  num141==224:
            p3411=self.conv1224(C141)
        elif  num141==225:
            p3411=self.conv1225(C141)
        elif  num141==226:
            p3411=self.conv1226(C141)
        elif  num141==227:
            p3411=self.conv1227(C141)
        elif  num141==228:
            p3411=self.conv1228(C141)
        elif  num141==229:
            p3411=self.conv1229(C141)
        elif  num141==230:
            p3411=self.conv1230(C141)
        elif  num141==231:
            p3411=self.conv1231(C141)
        elif  num141==232:
            p3411=self.conv1232(C141)
        elif  num141==233:
            p3411=self.conv1233(C141)
        elif  num141==234:
            p3411=self.conv1234(C141)
        elif  num141==235:
            p3411=self.conv1235(C141)
        elif  num141==236:
            p3411=self.conv1236(C141)
        elif  num141==237:
            p3411=self.conv1237(C141)
        elif  num141==238:
            p3411=self.conv1238(C141)
        elif  num141==239:
            p3411=self.conv1239(C141) 
        elif  num141==240:
            p3411=self.conv1240(C141)
        elif  num141==241:
            p3411=self.conv1241(C141)
        elif  num141==242:
            p3411=self.conv1242(C141)
        elif  num141==243:
            p3411=self.conv1243(C141)
        elif  num141==244:
            p3411=self.conv1244(C141)
        elif  num141==245:
            p3411=self.conv1245(C141)
        elif  num141==246:
            p3411=self.conv1246(C141)
        elif  num141==247:
            p3411=self.conv1247(C141)
        elif  num141==248:
            p3411=self.conv1248(C141)
        elif  num141==249:
            p3411=self.conv1249(C141)
        elif  num141==250:
            p3411=self.conv1250(C141)
        elif  num141==251:
            p3411=self.conv1251(C141)
        elif  num141==252:
            p3411=self.conv1252(C141)
        elif  num141==253:
            p3411=self.conv1253(C141)
        elif  num141==254:
            p3411=self.conv1254(C141)
        elif  num141==255:
            p3411=self.conv1255(C141)
        elif  num141==256:
            p3411=self.conv1256(C141)
            
        if  num241==1:
            p3412=self.conv11(D141)
        elif  num241==2:
            p3412=self.conv12(D141)
        elif  num241==3:
            p3412=self.conv13(D141)
        elif  num241==4:
            p3412=self.conv14(D141)
        elif  num241==5:
            p3412=self.conv15(D141)
        elif  num241==6:
            p3412=self.conv16(D141)
        elif  num241==7:
            p3412=self.conv17(D141)
        elif  num241==8:
            p3412=self.conv18(D141)
        elif  num241==9:
            p3412=self.conv19(D141)
        elif  num241==10:
            p3412=self.conv110(D141)
        elif  num241==11:
            p3412=self.conv111(D141)
        elif  num241==12:
            p3412=self.conv112(D141)
        elif  num241==13:
            p3412=self.conv113(D141)
        elif  num241==14:
            p3412=self.conv114(D141)
        elif  num241==15:
            p3412=self.conv115(D141)
        elif  num241==16:
            p3412=self.conv116(D141)
        elif  num241==17:
            p3412=self.conv117(D141)
        elif  num241==18:
            p3412=self.conv118(D141)
        elif  num241==19:
            p3412=self.conv119(D141)
        elif  num241==20:
            p3412=self.conv120(D141)
        elif  num241==21:
            p3412=self.conv121(D141)
        elif  num241==22:
            p3412=self.conv122(D141)
        elif  num241==23:
            p3412=self.conv123(D141)
        elif  num241==24:
            p3412=self.conv124(D141)
        elif  num241==25:
            p3412=self.conv125(D141)
        elif  num241==26:
            p3412=self.conv126(D141)
        elif  num241==27:
            p3412=self.conv127(D141)
        elif  num241==28:
            p3412=self.conv128(D141)
        elif  num241==29:
            p3412=self.conv129(D141)
        elif  num241==30:
            p3412=self.conv130(D141)
        elif  num241==31:
            p3412=self.conv131(D141)
        elif  num241==32:
            p3412=self.conv132(D141)
        elif  num241==33:
            p3412=self.conv133(D141)
        elif  num241==34:
            p3412=self.conv134(D141)
        elif  num241==35:
            p3412=self.conv135(D141)
        elif  num241==36:
            p3412=self.conv136(D141)
        elif  num241==37:
            p3412=self.conv137(D141)
        elif  num241==38:
            p3412=self.conv138(D141)
        elif  num241==39:
            p3412=self.conv139(D141)
        elif  num241==40:
            p3412=self.conv140(D141)
        elif  num241==41:
            p3412=self.conv141(D141)
        elif  num241==42:
            p3412=self.conv142(D141)
        elif  num241==43:
            p3412=self.conv143(D141)
        elif  num241==44:
            p3412=self.conv144(D141)
        elif  num241==45:
            p3412=self.conv145(D141)
        elif  num241==46:
            p3412=self.conv146(D141)
        elif  num241==47:
            p3412=self.conv147(D141)
        elif  num241==48:
            p3412=self.conv148(D141)
        elif  num241==49:
            p3412=self.conv149(D141)
        elif  num241==50:
            p3412=self.conv150(D141)
        elif  num241==51:
            p3412=self.conv151(D141)
        elif  num241==52:
            p3412=self.conv152(D141)
        elif  num241==53:
            p3412=self.conv153(D141)
        elif  num241==54:
            p3412=self.conv154(D141)
        elif  num241==55:
            p3412=self.conv155(D141)
        elif  num241==56:
            p3412=self.conv156(D141)
        elif  num241==57:
            p3412=self.conv157(D141)
        elif  num241==58:
            p3412=self.conv158(D141)
        elif  num241==59:
            p3412=self.conv159(D141)
        elif  num241==60:
            p3412=self.conv160(D141)
        elif  num241==61:
            p3412=self.conv161(D141)
        elif  num241==62:
            p3412=self.conv162(D141)
        elif  num241==63:
            p3412=self.conv163(D141)
        elif  num241==64:
            p3412=self.conv164(D141)
        elif  num241==65:
            p3412=self.conv165(D141)
        elif  num241==66:
            p3412=self.conv166(D141)
        elif  num241==67:
            p3412=self.conv167(D141)
        elif  num241==68:
            p3412=self.conv168(D141)
        elif  num241==69:
            p3412=self.conv169(D141)
        elif  num241==70:
            p3412=self.conv170(D141)
        elif  num241==71:
            p3412=self.conv171(D141)
        elif  num241==72:
            p3412=self.conv172(D141)
        elif  num241==73:
            p3412=self.conv173(D141)
        elif  num241==74:
            p3412=self.conv174(D141)
        elif  num241==75:
            p3412=self.conv175(D141)
        elif  num241==76:
            p3412=self.conv176(D141)
        elif  num241==77:
            p3412=self.conv177(D141)
        elif  num241==78:
            p3412=self.conv178(D141)
        elif  num241==79:
            p3412=self.conv179(D141)
        elif  num241==80:
            p3412=self.conv180(D141)
        elif  num241==81:
            p3412=self.conv181(D141)
        elif  num241==82:
            p3412=self.conv182(D141)
        elif  num241==83:
            p3412=self.conv183(D141)
        elif  num241==84:
            p3412=self.conv184(D141)
        elif  num241==85:
            p3412=self.conv185(D141)
        elif  num241==86:
            p3412=self.conv186(D141)
        elif  num241==87:
            p3412=self.conv187(D141)
        elif  num241==88:
            p3412=self.conv188(D141)
        elif  num241==89:
            p3412=self.conv189(D141)    
        elif  num241==90:
            p3412=self.conv190(D141)
        elif  num241==91:
            p3412=self.conv191(D141)
        elif  num241==92:
            p3412=self.conv192(D141)
        elif  num241==93:
            p3412=self.conv193(D141)
        elif  num241==94:
            p3412=self.conv194(D141)
        elif  num241==95:
            p3412=self.conv195(D141)
        elif  num241==96:
            p3412=self.conv196(D141)
        elif  num241==97:
            p3412=self.conv197(D141)
        elif  num241==98:
            p3412=self.conv198(D141)
        elif  num241==99:
            p3412=self.conv199(D141) 
        elif  num241==100:
            p3412=self.conv1100(D141)
        elif  num241==101:
            p3412=self.conv1101(D141)
        elif  num241==102:
            p3412=self.conv1102(D141)
        elif  num241==103:
            p3412=self.conv1103(D141)
        elif  num241==104:
            p3412=self.conv1104(D141)
        elif  num241==105:
            p3412=self.conv1105(D141)
        elif  num241==106:
            p3412=self.conv1106(D141)
        elif  num241==107:
            p3412=self.conv1107(D141)
        elif  num241==108:
            p3412=self.conv1108(D141)
        elif  num241==109:
            p3412=self.conv1109(D141)
        elif  num241==110:
            p3412=self.conv1110(D141)
        elif  num241==111:
            p3412=self.conv1111(D141)
        elif  num241==112:
            p3412=self.conv1112(D141)
        elif  num241==113:
            p3412=self.conv1113(D141)
        elif  num241==114:
            p3412=self.conv1114(D141)
        elif  num241==115:
            p3412=self.conv1115(D141)
        elif  num241==116:
            p3412=self.conv1116(D141)
        elif  num241==117:
            p3412=self.conv1117(D141)
        elif  num241==118:
            p3412=self.conv1118(D141)
        elif  num241==119:
            p3412=self.conv1119(D141) 
        elif  num241==120:
            p3412=self.conv1120(D141)
        elif  num241==121:
            p3412=self.conv1121(D141)
        elif  num241==122:
            p3412=self.conv1122(D141)
        elif  num241==123:
            p3412=self.conv1123(D141)
        elif  num241==124:
            p3412=self.conv1124(D141)
        elif  num241==125:
            p3412=self.conv1125(D141)
        elif  num241==126:
            p3412=self.conv1126(D141)
        elif  num241==127:
            p3412=self.conv1127(D141)
        elif  num241==128:
            p3412=self.conv1128(D141)
        elif  num241==129:
            p3412=self.conv1129(D141) 
        elif  num241==130:
            p3412=self.conv1130(D141)
        elif  num241==131:
            p3412=self.conv1131(D141)
        elif  num241==132:
            p3412=self.conv1132(D141)
        elif  num241==133:
            p3412=self.conv1133(D141)
        elif  num241==134:
            p3412=self.conv1134(D141)
        elif  num241==135:
            p3412=self.conv1135(D141)
        elif  num241==136:
            p3412=self.conv1136(D141)
        elif  num241==137:
            p3412=self.conv1137(D141)
        elif  num241==138:
            p3412=self.conv1138(D141)
        elif  num241==139:
            p3412=self.conv1139(D141)
        elif  num241==140:
            p3412=self.conv1140(D141)
        elif  num241==141:
            p3412=self.conv1141(D141)
        elif  num241==142:
            p3412=self.conv1142(D141)
        elif  num241==143:
            p3412=self.conv1143(D141)
        elif  num241==144:
            p3412=self.conv1144(D141)
        elif  num241==145:
            p3412=self.conv1145(D141)
        elif  num241==146:
            p3412=self.conv1146(D141)
        elif  num241==147:
            p3412=self.conv1147(D141)
        elif  num241==148:
            p3412=self.conv1148(D141)
        elif  num241==149:
            p3412=self.conv1149(D141) 
        elif  num241==150:
            p3412=self.conv1150(D141)
        elif  num241==151:
            p3412=self.conv1151(D141)
        elif  num241==152:
            p3412=self.conv1152(D141)
        elif  num241==153:
            p3412=self.conv1153(D141)
        elif  num241==154:
            p3412=self.conv1154(D141)
        elif  num241==155:
            p3412=self.conv1155(D141)
        elif  num241==156:
            p3412=self.conv1156(D141)
        elif  num241==157:
            p3412=self.conv1157(D141)
        elif  num241==158:
            p3412=self.conv1158(D141)
        elif  num241==159:
            p3412=self.conv1159(D141) 
        elif  num241==160:
            p3412=self.conv1160(D141)
        elif  num241==161:
            p3412=self.conv1161(D141)
        elif  num241==162:
            p3412=self.conv1162(D141)
        elif  num241==163:
            p3412=self.conv1163(D141)
        elif  num241==164:
            p3412=self.conv1164(D141)
        elif  num241==165:
            p3412=self.conv1165(D141)
        elif  num241==166:
            p3412=self.conv1166(D141)
        elif  num241==167:
            p3412=self.conv1167(D141)
        elif  num241==168:
            p3412=self.conv1168(D141)
        elif  num241==169:
            p3412=self.conv1169(D141) 
        elif  num241==170:
            p3412=self.conv1170(D141)
        elif  num241==171:
            p3412=self.conv1171(D141)
        elif  num241==172:
            p3412=self.conv1172(D141)
        elif  num241==173:
            p3412=self.conv1173(D141)
        elif  num241==174:
            p3412=self.conv1174(D141)
        elif  num241==175:
            p3412=self.conv1175(D141)
        elif  num241==176:
            p3412=self.conv1176(D141)
        elif  num241==177:
            p3412=self.conv1177(D141)
        elif  num241==178:
            p3412=self.conv1178(D141)
        elif  num241==179:
            p3412=self.conv1179(D141)                                                                                              
        elif  num241==180:
            p3412=self.conv1180(D141)
        elif  num241==181:
            p3412=self.conv1181(D141)
        elif  num241==182:
            p3412=self.conv1182(D141)
        elif  num241==183:
            p3412=self.conv1183(D141)
        elif  num241==184:
            p3412=self.conv1184(D141)
        elif  num241==185:
            p3412=self.conv1185(D141)
        elif  num241==186:
            p3412=self.conv1186(D141)
        elif  num241==187:
            p3412=self.conv1187(D141)
        elif  num241==188:
            p3412=self.conv1188(D141)
        elif  num241==189:
            p3412=self.conv1189(D141) 
        elif  num241==190:
            p3412=self.conv1190(D141)
        elif  num241==191:
            p3412=self.conv1191(D141)
        elif  num241==192:
            p3412=self.conv1192(D141)
        elif  num241==193:
            p3412=self.conv1193(D141)
        elif  num241==194:
            p3412=self.conv1194(D141)
        elif  num241==195:
            p3412=self.conv1195(D141)
        elif  num241==196:
            p3412=self.conv1196(D141)
        elif  num241==197:
            p3412=self.conv1197(D141)
        elif  num241==198:
            p3412=self.conv1198(D141)
        elif  num241==199:
            p3412=self.conv1199(D141)
        elif  num241==200:
            p3412=self.conv1200(D141)
        elif  num241==201:
            p3412=self.conv1201(D141)
        elif  num241==202:
            p3412=self.conv1202(D141)
        elif  num241==203:
            p3412=self.conv1203(D141)
        elif  num241==204:
            p3412=self.conv1204(D141)
        elif  num241==205:
            p3412=self.conv1205(D141)
        elif  num241==206:
            p3412=self.conv1206(D141)
        elif  num241==207:
            p3412=self.conv1207(D141)
        elif  num241==208:
            p3412=self.conv1208(D141)
        elif  num241==209:
            p3412=self.conv1209(D141)
        elif  num241==210:
            p3412=self.conv1210(D141)
        elif  num241==211:
            p3412=self.conv1211(D141)
        elif  num241==212:
            p3412=self.conv1212(D141)
        elif  num241==213:
            p3412=self.conv1213(D141)
        elif  num241==214:
            p3412=self.conv1214(D141)
        elif  num241==215:
            p3412=self.conv1215(D141)
        elif  num241==216:
            p3412=self.conv1216(D141)
        elif  num241==217:
            p3412=self.conv1217(D141)
        elif  num241==218:
            p3412=self.conv1218(D141)
        elif  num241==219:
            p3412=self.conv1219(D141)
        elif  num241==220:
            p3412=self.conv1220(D141)
        elif  num241==221:
            p3412=self.conv1221(D141)
        elif  num241==222:
            p3412=self.conv1222(D141)
        elif  num241==223:
            p3412=self.conv1223(D141)
        elif  num241==224:
            p3412=self.conv1224(D141)
        elif  num241==225:
            p3412=self.conv1225(D141)
        elif  num241==226:
            p3412=self.conv1226(D141)
        elif  num241==227:
            p3412=self.conv1227(D141)
        elif  num241==228:
            p3412=self.conv1228(D141)
        elif  num241==229:
            p3412=self.conv1229(D141)
        elif  num241==230:
            p3412=self.conv1230(D141)
        elif  num241==231:
            p3412=self.conv1231(D141)
        elif  num241==232:
            p3412=self.conv1232(D141)
        elif  num241==233:
            p3412=self.conv1233(D141)
        elif  num241==234:
            p3412=self.conv1234(D141)
        elif  num241==235:
            p3412=self.conv1235(D141)
        elif  num241==236:
            p3412=self.conv1236(D141)
        elif  num241==237:
            p3412=self.conv1237(D141)
        elif  num241==238:
            p3412=self.conv1238(D141)
        elif  num241==239:
            p3412=self.conv1239(D141) 
        elif  num241==240:
            p3412=self.conv1240(D141)
        elif  num241==241:
            p3412=self.conv1241(D141)
        elif  num241==242:
            p3412=self.conv1242(D141)
        elif  num241==243:
            p3412=self.conv1243(D141)
        elif  num241==244:
            p3412=self.conv1244(D141)
        elif  num241==245:
            p3412=self.conv1245(D141)
        elif  num241==246:
            p3412=self.conv1246(D141)
        elif  num241==247:
            p3412=self.conv1247(D141)
        elif  num241==248:
            p3412=self.conv1248(D141)
        elif  num241==249:
            p3412=self.conv1249(D141)
        elif  num241==250:
            p3412=self.conv1250(D141)
        elif  num241==251:
            p3412=self.conv1251(D141)
        elif  num241==252:
            p3412=self.conv1252(D141)
        elif  num241==253:
            p3412=self.conv1253(D141)
        elif  num241==254:
            p3412=self.conv1254(D141)
        elif  num241==255:
            p3412=self.conv1255(D141)
        elif  num241==256:
            p3412=self.conv1256(D141)
        elif  num241==257:
            p3412=self.conv1257(D141)
        elif  num241==258:
            p3412=self.conv1258(D141)
        elif  num241==259:
            p3412=self.conv1259(D141)
        elif  num241==260:
            p3412=self.conv1260(D141)
        elif  num241==261:
            p3412=self.conv1261(D141)
        elif  num241==262:
            p3412=self.conv1262(D141)
        elif  num241==263:
            p3412=self.conv1263(D141)
        elif  num241==264:
            p3412=self.conv1264(D141)
        elif  num241==265:
            p3412=self.conv1265(D141)
        elif  num241==266:
            p3412=self.conv1266(D141)
        elif  num241==267:
            p3412=self.conv1267(D141)
        elif  num241==268:
            p3412=self.conv1268(D141)
        elif  num241==269:
            p3412=self.conv1269(D141) 
        elif  num241==270:
            p3412=self.conv1270(D141)
        elif  num241==271:
            p3412=self.conv1271(D141)
        elif  num241==272:
            p3412=self.conv1272(D141)
        elif  num241==273:
            p3412=self.conv1273(D141)
        elif  num241==274:
            p3412=self.conv1274(D141)
        elif  num241==275:
            p3412=self.conv1275(D141)
        elif  num241==276:
            p3412=self.conv1276(D141)
        elif  num241==277:
            p3412=self.conv1277(D141)
        elif  num241==278:
            p3412=self.conv1278(D141)
        elif  num241==279:
            p3412=self.conv1279(D141)
        elif  num241==280:
            p3412=self.conv1280(D141)
        elif  num241==281:
            p3412=self.conv1281(D141)
        elif  num241==282:
            p3412=self.conv1282(D141)
        elif  num241==283:
            p3412=self.conv1283(D141)
        elif  num241==284:
            p3412=self.conv1284(D141)
        elif  num241==285:
            p3412=self.conv1285(D141)
        elif  num241==286:
            p3412=self.conv1286(D141)
        elif  num241==287:
            p3412=self.conv1287(D141)
        elif  num241==288:
            p3412=self.conv1288(D141)
        elif  num241==289:
            p3412=self.conv1289(D141) 
        elif  num241==290:
            p3412=self.conv1290(D141)
        elif  num241==291:
            p3412=self.conv1291(D141)
        elif  num241==292:
            p3412=self.conv1292(D141)
        elif  num241==293:
            p3412=self.conv1293(D141)
        elif  num241==294:
            p3412=self.conv1294(D141)
        elif  num241==295:
            p3412=self.conv1295(D141)
        elif  num241==296:
            p3412=self.conv1296(D141)
        elif  num241==297:
            p3412=self.conv1297(D141)
        elif  num241==298:
            p3412=self.conv1298(D141)
        elif  num241==299:
            p3412=self.conv1299(D141)
        elif  num241==300:
            p3412=self.conv1300(D141)
        elif  num241==301:
            p3412=self.conv1301(D141)
        elif  num241==302:
            p3412=self.conv1302(D141)
        elif  num241==303:
            p3412=self.conv1303(D141)
        elif  num241==304:
            p3412=self.conv1304(D141)
        elif  num241==305:
            p3412=self.conv1305(D141)
        elif  num241==306:
            p3412=self.conv1306(D141)
        elif  num241==307:
            p3412=self.conv1307(D141)
        elif  num241==308:
            p3412=self.conv1308(D141)
        elif  num241==309:
            p3412=self.conv1309(D141) 
        elif  num241==310:
            p3412=self.conv1310(D141)
        elif  num241==311:
            p3412=self.conv1311(D141)
        elif  num241==312:
            p3412=self.conv1312(D141)
        elif  num241==313:
            p3412=self.conv1313(D141)
        elif  num241==314:
            p3412=self.conv1314(D141)
        elif  num241==315:
            p3412=self.conv1315(D141)
        elif  num241==316:
            p3412=self.conv1316(D141)
        elif  num241==317:
            p3412=self.conv1317(D141)
        elif  num241==318:
            p3412=self.conv1318(D141)
        elif  num241==319:
            p3412=self.conv1319(D141)
        elif  num241==320:
            p3412=self.conv1320(D141)
        elif  num241==321:
            p3412=self.conv1321(D141)
        elif  num241==322:
            p3412=self.conv1322(D141)
        elif  num241==323:
            p3412=self.conv1323(D141)
        elif  num241==324:
            p3412=self.conv1324(D141)
        elif  num241==325:
            p3412=self.conv1325(D141)
        elif  num241==326:
            p3412=self.conv1326(D141)
        elif  num241==327:
            p3412=self.conv1327(D141)
        elif  num241==328:
            p3412=self.conv1328(D141)
        elif  num241==329:
            p3412=self.conv1329(D141)
        elif  num241==330:
            p3412=self.conv1330(D141)
        elif  num241==331:
            p3412=self.conv1331(D141)
        elif  num241==332:
            p3412=self.conv1332(D141)
        elif  num241==333:
            p3412=self.conv1333(D141)
        elif  num241==334:
            p3412=self.conv1334(D141)
        elif  num241==335:
            p3412=self.conv1335(D141)
        elif  num241==336:
            p3412=self.conv1336(D141)
        elif  num241==337:
            p3412=self.conv1337(D141)
        elif  num241==338:
            p3412=self.conv1338(D141)
        elif  num241==339:
            p3412=self.conv1339(D141)
        elif  num241==340:
            p3412=self.conv1340(D141)
        elif  num241==341:
            p3412=self.conv1341(D141)
        elif  num241==342:
            p3412=self.conv1342(D141)
        elif  num241==343:
            p3412=self.conv1343(D141)
        elif  num241==344:
            p3412=self.conv1344(D141)
        elif  num241==345:
            p3412=self.conv1345(D141)
        elif  num241==346:
            p3412=self.conv1346(D141)
        elif  num241==347:
            p3412=self.conv1347(D141)
        elif  num241==348:
            p3412=self.conv1348(D141)
        elif  num241==349:
            p3412=self.conv1349(D141)
        elif  num241==350:
            p3412=self.conv1350(D141)
        elif  num241==351:
            p3412=self.conv1351(D141)
        elif  num241==352:
            p3412=self.conv1352(D141)
        elif  num241==353:
            p3412=self.conv1335(D141)
        elif  num241==354:
            p3412=self.conv1354(D141)
        elif  num241==355:
            p3412=self.conv1355(D141)
        elif  num241==356:
            p3412=self.conv1356(D141)
        elif  num241==357:
            p3412=self.conv1357(D141)
        elif  num241==358:
            p3412=self.conv1358(D141)
        elif  num241==359:
            p3412=self.conv1359(D141) 
        elif  num241==360:
            p3412=self.conv1360(D141)
        elif  num241==361:
            p3412=self.conv1361(D141)
        elif  num241==362:
            p3412=self.conv1362(D141)
        elif  num241==363:
            p3412=self.conv1363(D141)
        elif  num241==364:
            p3412=self.conv1364(D141)
        elif  num241==365:
            p3412=self.conv1365(D141)
        elif  num241==366:
            p3412=self.conv1366(D141)
        elif  num241==367:
            p3412=self.conv1367(D141)
        elif  num241==368:
            p3412=self.conv1368(D141)
        elif  num241==369:
            p3412=self.conv1369(D141) 
        elif  num241==370:
            p3412=self.conv1370(D141)
        elif  num241==371:
            p3412=self.conv1371(D141)
        elif  num241==372:
            p3412=self.conv1372(D141)
        elif  num241==373:
            p3412=self.conv1373(D141)
        elif  num241==374:
            p3412=self.conv1374(D141)
        elif  num241==375:
            p3412=self.conv1375(D141)
        elif  num241==376:
            p3412=self.conv1376(D141)
        elif  num241==377:
            p3412=self.conv1377(D141)
        elif  num241==378:
            p3412=self.conv1378(D141)
        elif  num241==379:
            p3412=self.conv1379(D141) 
        elif  num241==380:
            p3412=self.conv1380(D141)
        elif  num241==381:
            p3412=self.conv1381(D141)
        elif  num241==382:
            p3412=self.conv1382(D141)
        elif  num241==383:
            p3412=self.conv1383(D141)
        elif  num241==384:
            p3412=self.conv1384(D141)
        elif  num241==385:
            p3412=self.conv1385(D141)
        elif  num241==386:
            p3412=self.conv1386(D141)
        elif  num241==387:
            p3412=self.conv1387(D141)
        elif  num241==388:
            p3412=self.conv1388(D141)
        elif  num241==389:
            p3412=self.conv1389(D141) 
        elif  num241==390:
            p3412=self.conv1390(D141)
        elif  num241==391:
            p3412=self.conv1391(D141)
        elif  num241==392:
            p3412=self.conv1392(D141)
        elif  num241==393:
            p3412=self.conv1393(D141)
        elif  num241==394:
            p3412=self.conv1394(D141)
        elif  num241==395:
            p3412=self.conv1395(D141)
        elif  num241==396:
            p3412=self.conv1396(D141)
        elif  num241==397:
            p3412=self.conv1397(D141)
        elif  num241==398:
            p3412=self.conv1398(D141)
        elif  num241==399:
            p3412=self.conv1399(D141)
        elif  num241==400:
            p3412=self.conv1400(D141)
        elif  num241==401:
            p3412=self.conv1401(D141)
        elif  num241==402:
            p3412=self.conv1402(D141)
        elif  num241==403:
            p3412=self.conv1403(D141)
        elif  num241==404:
            p3412=self.conv1404(D141)
        elif  num241==405:
            p3412=self.conv1405(D141)
        elif  num241==406:
            p3412=self.conv1406(D141)
        elif  num241==407:
            p3412=self.conv1407(D141)
        elif  num241==408:
            p3412=self.conv1408(D141)
        elif  num241==409:
            p3412=self.conv1409(D141)
        elif  num241==410:
            p3412=self.conv1410(D141)
        elif  num241==411:
            p3412=self.conv1411(D141)
        elif  num241==412:
            p3412=self.conv1412(D141)
        elif  num241==413:
            p3412=self.conv1413(D141)
        elif  num241==414:
            p3412=self.conv1414(D141)
        elif  num241==415:
            p3412=self.conv145(D141)
        elif  num241==416:
            p3412=self.conv1416(D141)
        elif  num241==417:
            p3412=self.conv1417(D141)
        elif  num241==418:
            p3412=self.conv1418(D141)
        elif  num241==419:
            p3412=self.conv1419(D141) 
        elif  num241==420:
            p3412=self.conv1420(D141)
        elif  num241==421:
            p3412=self.conv1421(D141)
        elif  num241==422:
            p3412=self.conv1422(D141)
        elif  num241==423:
            p3412=self.conv1423(D141)
        elif  num241==424:
            p3412=self.conv1424(D141)
        elif  num241==425:
            p3412=self.conv1425(D141)
        elif  num241==426:
            p3412=self.conv1426(D141)
        elif  num241==427:
            p3412=self.conv1427(D141)
        elif  num241==428:
            p3412=self.conv1428(D141)
        elif  num241==429:
            p3412=self.conv1429(D141) 
        elif  num241==430:
            p3412=self.conv1430(D141)
        elif  num241==431:
            p3412=self.conv1431(D141)
        elif  num241==432:
            p3412=self.conv1432(D141)
        elif  num241==433:
            p3412=self.conv1433(D141)
        elif  num241==434:
            p3412=self.conv1434(D141)
        elif  num241==435:
            p3412=self.conv1435(D141)
        elif  num241==436:
            p3412=self.conv1436(D141)
        elif  num241==437:
            p3412=self.conv1437(D141)
        elif  num241==438:
            p3412=self.conv1438(D141)
        elif  num241==439:
            p3412=self.conv1439(D141)
        elif  num241==440:
            p3412=self.conv1440(D141)
        elif  num241==441:
            p3412=self.conv1441(D141)
        elif  num241==442:
            p3412=self.conv1442(D141)
        elif  num241==443:
            p3412=self.conv1443(D141)
        elif  num241==444:
            p3412=self.conv1444(D141)
        elif  num241==445:
            p3412=self.conv1445(D141)
        elif  num241==446:
            p3412=self.conv1446(D141)
        elif  num241==447:
            p3412=self.conv1447(D141)
        elif  num241==448:
            p3412=self.conv1448(D141)
        elif  num241==449:
            p3412=self.conv1449(D141)
        elif  num241==450:
            p3412=self.conv1450(D141)
        elif  num241==451:
            p3412=self.conv1451(D141)
        elif  num241==452:
            p3412=self.conv1452(D141)
        elif  num241==453:
            p3412=self.conv1453(D141)
        elif  num241==454:
            p3412=self.conv1454(D141)
        elif  num241==455:
            p3412=self.conv1455(D141)
        elif  num241==456:
            p3412=self.conv1456(D141)
        elif  num241==457:
            p3412=self.conv1457(D141)
        elif  num241==458:
            p3412=self.conv1458(D141)
        elif  num241==459:
            p3412=self.conv1459(D141)
        elif  num241==460:
            p3412=self.conv1460(D141)
        elif  num241==461:
            p3412=self.conv1461(D141)
        elif  num241==462:
            p3412=self.conv1462(D141)
        elif  num241==463:
            p3412=self.conv1463(D141)
        elif  num241==464:
            p3412=self.conv1464(D141)
        elif  num241==465:
            p3412=self.conv1465(D141)
        elif  num241==466:
            p3412=self.conv1466(D141)
        elif  num241==467:
            p3412=self.conv1467(D141)
        elif  num241==468:
            p3412=self.conv1468(D141)
        elif  num241==469:
            p3412=self.conv1469(D141) 
        elif  num241==470:
            p3412=self.conv1470(D141)
        elif  num241==471:
            p3412=self.conv1471(D141)
        elif  num241==472:
            p3412=self.conv1472(D141)
        elif  num241==473:
            p3412=self.conv1473(D141)
        elif  num241==474:
            p3412=self.conv1474(D141)
        elif  num241==475:
            p3412=self.conv1475(D141)
        elif  num241==476:
            p3412=self.conv1476(D141)
        elif  num241==477:
            p3412=self.conv1477(D141)
        elif  num241==478:
            p3412=self.conv1478(D141)
        elif  num241==479:
            p3412=self.conv1479(D141)
        elif  num241==480:
            p3412=self.conv1480(D141)
        elif  num241==481:
            p3412=self.conv1481(D141)
        elif  num241==482:
            p3412=self.conv1482(D141)
        elif  num241==483:
            p3412=self.conv1483(D141)
        elif  num241==484:
            p3412=self.conv1484(D141)
        elif  num241==485:
            p3412=self.conv1485(D141)
        elif  num241==486:
            p3412=self.conv1486(D141)
        elif  num241==487:
            p3412=self.conv1487(D141)
        elif  num241==488:
            p3412=self.conv1488(D141)
        elif  num241==489:
            p3412=self.conv1489(D141)
        elif  num241==490:
            p3412=self.conv1490(D141)
        elif  num241==491:
            p3412=self.conv1491(D141)
        elif  num241==492:
            p3412=self.conv1492(D141)
        elif  num241==493:
            p3412=self.conv1493(D141)
        elif  num241==494:
            p3412=self.conv1494(D141)
        elif  num241==495:
            p3412=self.conv1495(D141)
        elif  num241==496:
            p3412=self.conv1496(D141)
        elif  num241==497:
            p3412=self.conv1497(D141)
        elif  num241==498:
            p3412=self.conv1498(D141)
        elif  num241==499:
            p3412=self.conv1499(D141)
        elif  num241==500:
            p3412=self.conv1500(D141)
        elif  num241==501:
            p3412=self.conv1501(D141)
        elif  num241==502:
            p3412=self.conv1502(D141)
        elif  num241==503:
            p3412=self.conv1503(D141)
        elif  num241==504:
            p3412=self.conv1504(D141)
        elif  num241==505:
            p3412=self.conv1505(D141)
        elif  num241==506:
            p3412=self.conv1506(D141)
        elif  num241==507:
            p3412=self.conv1507(D141)
        elif  num241==508:
            p3412=self.conv1508(D141)
        elif  num241==509:
            p3412=self.conv1509(D141)
        elif  num241==510:
            p3412=self.conv1510(D141)
        elif  num241==511:
            p3412=self.conv1511(D141)
        elif  num241==512:
            p3412=self.conv1512(D141)
        elif  num241==513:
            p3412=self.conv1513(D141)
        elif  num241==514:
            p3412=self.conv1514(D141)
        elif  num241==515:
            p3412=self.conv1515(D141)
        elif  num241==516:
            p3412=self.conv1516(D141)
        elif  num241==517:
            p3412=self.conv1517(D141)
        elif  num241==518:
            p3412=self.conv1518(D141)
        elif  num241==519:
            p3412=self.conv1519(D141)
        elif  num241==520:
            p3412=self.conv1520(D141)
        elif  num241==521:
            p3412=self.conv1521(D141)
        elif  num241==522:
            p3412=self.conv1522(D141)
        elif  num241==523:
            p3412=self.conv1523(D141)
        elif  num241==524:
            p3412=self.conv1524(D141)
        elif  num241==525:
            p3412=self.conv1525(D141)
        elif  num241==526:
            p3412=self.conv1526(D141)
        elif  num241==527:
            p3412=self.conv1527(D141)
        elif  num241==528:
            p3412=self.conv1528(D141)
        elif  num241==529:
            p3412=self.conv1529(D141)
        elif  num241==530:
            p3412=self.conv1530(D141)
        elif  num241==531:
            p3412=self.conv1531(D141)
        elif  num241==532:
            p3412=self.conv1532(D141)
        elif  num241==533:
            p3412=self.conv1533(D141)
        elif  num241==534:
            p3412=self.conv1534(D141)
        elif  num241==535:
            p3412=self.conv1535(D141)
        elif  num241==536:
            p3412=self.conv1536(D141)
        elif  num241==537:
            p3412=self.conv1537(D141)
        elif  num241==538:
            p3412=self.conv1538(D141)
        elif  num241==539:
            p3412=self.conv1539(D141)
        elif  num241==540:
            p3412=self.conv1540(D141)
        elif  num241==541:
            p3412=self.conv1541(D141)
        elif  num241==542:
            p3412=self.conv1542(D141)
        elif  num241==543:
            p3412=self.conv1543(D141)
        elif  num241==544:
            p3412=self.conv1544(D141)
        elif  num241==545:
            p3412=self.conv1545(D141)
        elif  num241==546:
            p3412=self.conv1546(D141)
        elif  num241==547:
            p3412=self.conv1547(D141)
        elif  num241==548:
            p3412=self.conv1548(D141)
        elif  num241==549:
            p3412=self.conv1549(D141) 
        elif  num241==550:
            p3412=self.conv1550(D141)
        elif  num241==551:
            p3412=self.conv1551(D141)
        elif  num241==552:
            p3412=self.conv1552(D141)
        elif  num241==553:
            p3412=self.conv1553(D141)
        elif  num241==554:
            p3412=self.conv1554(D141)
        elif  num241==555:
            p3412=self.conv1555(D141)
        elif  num241==556:
            p3412=self.conv1556(D141)
        elif  num241==557:
            p3412=self.conv1557(D141)
        elif  num241==558:
            p3412=self.conv1558(D141)
        elif  num241==559:
            p3412=self.conv1559(D141)
        elif  num241==560:
            p3412=self.conv1560(D141)
        elif  num241==561:
            p3412=self.conv1561(D141)
        elif  num241==562:
            p3412=self.conv1562(D141)
        elif  num241==563:
            p3412=self.conv1563(D141)
        elif  num241==564:
            p3412=self.conv1564(D141)
        elif  num241==565:
            p3412=self.conv1565(D141)
        elif  num241==566:
            p3412=self.conv1566(D141)
        elif  num241==567:
            p3412=self.conv1567(D141)
        elif  num241==568:
            p3412=self.conv1568(D141)
        elif  num241==569:
            p3412=self.conv1569(D141) 
        elif  num241==570:
            p3412=self.conv1570(D141)
        elif  num241==571:
            p3412=self.conv1571(D141)
        elif  num241==572:
            p3412=self.conv1572(D141)
        elif  num241==573:
            p3412=self.conv1573(D141)
        elif  num241==574:
            p3412=self.conv1574(D141)
        elif  num241==575:
            p3412=self.conv1575(D141)
        elif  num241==576:
            p3412=self.conv1576(D141)
        elif  num241==577:
            p3412=self.conv1577(D141)
        elif  num241==578:
            p3412=self.conv1578(D141)
        elif  num241==579:
            p3412=self.conv1579(D141) 
        elif  num241==580:
            p3412=self.conv1580(D141)
        elif  num241==581:
            p3412=self.conv1581(D141)
        elif  num241==582:
            p3412=self.conv1582(D141)
        elif  num241==583:
            p3412=self.conv1583(D141)
        elif  num241==584:
            p3412=self.conv1584(D141)
        elif  num241==585:
            p3412=self.conv1585(D141)
        elif  num241==586:
            p3412=self.conv1586(D141)
        elif  num241==587:
            p3412=self.conv1587(D141)
        elif  num241==588:
            p3412=self.conv1588(D141)
        elif  num241==589:
            p3412=self.conv1589(D141)
        elif  num241==590:
            p3412=self.conv1590(D141)
        elif  num241==591:
            p3412=self.conv1591(D141)
        elif  num241==592:
            p3412=self.conv1592(D141)
        elif  num241==593:
            p3412=self.conv1593(D141)
        elif  num241==594:
            p3412=self.conv1594(D141)
        elif  num241==595:
            p3412=self.conv1595(D141)
        elif  num241==596:
            p3412=self.conv1596(D141)
        elif  num241==597:
            p3412=self.conv1597(D141)
        elif  num241==598:
            p3412=self.conv1598(D141)
        elif  num241==599:
            p3412=self.conv1599(D141)
        elif  num241==600:
            p3412=self.conv1600(D141)
        elif  num241==601:
            p3412=self.conv1601(D141)
        elif  num241==602:
            p3412=self.conv1602(D141)
        elif  num241==603:
            p3412=self.conv1603(D141)
        elif  num241==604:
            p3412=self.conv1604(D141)
        elif  num241==605:
            p3412=self.conv1605(D141)
        elif  num241==606:
            p3412=self.conv1606(D141)
        elif  num241==607:
            p3412=self.conv1607(D141)
        elif  num241==608:
            p3412=self.conv1608(D141)
        elif  num241==609:
            p3412=self.conv1609(D141)                                                                                                                         
        elif  num241==610:
            p3412=self.conv1610(D141)
        elif  num241==611:
            p3412=self.conv1611(D141)
        elif  num241==612:
            p3412=self.conv1612(D141)
        elif  num241==613:
            p3412=self.conv1613(D141)
        elif  num241==614:
            p3412=self.conv1614(D141)
        elif  num241==615:
            p3412=self.conv1615(D141)
        elif  num241==616:
            p3412=self.conv1616(D141)
        elif  num241==617:
            p3412=self.conv1617(D141)
        elif  num241==618:
            p3412=self.conv1618(D141)
        elif  num241==619:
            p3412=self.conv1619(D141)                                                                                                                          
        elif  num241==620:
            p3412=self.conv1620(D141)
        elif  num241==621:
            p3412=self.conv1621(D141)
        elif  num241==622:
            p3412=self.conv1622(D141)
        elif  num241==623:
            p3412=self.conv1623(D141)
        elif  num241==624:
            p3412=self.conv1624(D141)
        elif  num241==625:
            p3412=self.conv1625(D141)
        elif  num241==626:
            p3412=self.conv1626(D141)
        elif  num241==627:
            p3412=self.conv1627(D141)
        elif  num241==628:
            p3412=self.conv1628(D141)
        elif  num241==629:
            p3412=self.conv1629(D141)  
        elif  num241==630:
            p3412=self.conv1630(D141)
        elif  num241==631:
            p3412=self.conv1631(D141)
        elif  num241==632:
            p3412=self.conv1632(D141)
        elif  num241==633:
            p3412=self.conv1633(D141)
        elif  num241==634:
            p3412=self.conv1634(D141)
        elif  num241==635:
            p3412=self.conv1635(D141)
        elif  num241==636:
            p3412=self.conv1636(D141)
        elif  num241==637:
            p3412=self.conv1637(D141)
        elif  num241==638:
            p3412=self.conv1638(D141)
        elif  num241==639:
            p3412=self.conv1639(D141)                                                                                                                          
        elif  num241==640:
            p3412=self.conv1640(D141)
        elif  num241==641:
            p3412=self.conv1641(D141)
        elif  num241==642:
            p3412=self.conv1642(D141)
        elif  num241==643:
            p3412=self.conv1643(D141)
        elif  num241==644:
            p3412=self.conv1644(D141)
        elif  num241==645:
            p3412=self.conv1645(D141)
        elif  num241==646:
            p3412=self.conv1646(D141)
        elif  num241==647:
            p3412=self.conv1647(D141)
        elif  num241==648:
            p3412=self.conv1648(D141)
        elif  num241==649:
            p3412=self.conv1649(D141)                                                                                                                          
        elif  num241==650:
            p3412=self.conv1650(D141)
        elif  num241==651:
            p3412=self.conv1651(D141)
        elif  num241==652:
            p3412=self.conv1652(D141)
        elif  num241==653:
            p3412=self.conv1653(D141)
        elif  num241==654:
            p3412=self.conv1654(D141)
        elif  num241==655:
            p3412=self.conv1655(D141)
        elif  num241==656:
            p3412=self.conv1656(D141)
        elif  num241==657:
            p3412=self.conv1657(D141)
        elif  num241==658:
            p3412=self.conv1658(D141)
        elif  num241==659:
            p3412=self.conv1659(D141)
        elif  num241==660:
            p3412=self.conv1660(D141)
        elif  num241==661:
            p3412=self.conv1661(D141)
        elif  num241==662:
            p3412=self.conv1662(D141)
        elif  num241==663:
            p3412=self.conv1663(D141)
        elif  num241==664:
            p3412=self.conv1664(D141)
        elif  num241==665:
            p3412=self.conv1665(D141)
        elif  num241==666:
            p3412=self.conv1666(D141)
        elif  num241==667:
            p3412=self.conv1667(D141)
        elif  num241==668:
            p3412=self.conv1668(D141)
        elif  num241==669:
            p3412=self.conv1669(D141) 
        elif  num241==670:
            p3412=self.conv1670(D141)
        elif  num241==671:
            p3412=self.conv1671(D141)
        elif  num241==672:
            p3412=self.conv1672(D141)
        elif  num241==673:
            p3412=self.conv1673(D141)
        elif  num241==674:
            p3412=self.conv1674(D141)
        elif  num241==675:
            p3412=self.conv1675(D141)
        elif  num241==676:
            p3412=self.conv1676(D141)
        elif  num241==677:
            p3412=self.conv1677(D141)
        elif  num241==678:
            p3412=self.conv1678(D141)
        elif  num241==679:
            p3412=self.conv1679(D141)
        elif  num241==680:
            p3412=self.conv1680(D141)
        elif  num241==681:
            p3412=self.conv1681(D141)
        elif  num241==682:
            p3412=self.conv1682(D141)
        elif  num241==683:
            p3412=self.conv1683(D141)
        elif  num241==684:
            p3412=self.conv1684(D141)
        elif  num241==685:
            p3412=self.conv1685(D141)
        elif  num241==686:
            p3412=self.conv1686(D141)
        elif  num241==687:
            p3412=self.conv1687(D141)
        elif  num241==688:
            p3412=self.conv1688(D141)
        elif  num241==689:
            p3412=self.conv1689(D141)
        elif  num241==690:
            p3412=self.conv1690(D141)
        elif  num241==691:
            p3412=self.conv1691(D141)
        elif  num241==692:
            p3412=self.conv1692(D141)
        elif  num241==693:
            p3412=self.conv1693(D141)
        elif  num241==694:
            p3412=self.conv1694(D141)
        elif  num241==695:
            p3412=self.conv1695(D141)
        elif  num241==696:
            p3412=self.conv1696(D141)
        elif  num241==697:
            p3412=self.conv1697(D141)
        elif  num241==698:
            p3412=self.conv1698(D141)
        elif  num241==699:
            p3412=self.conv1699(D141)
        elif  num241==700:
            p3412=self.conv1700(D141)
        elif  num241==701:
            p3412=self.conv1701(D141)
        elif  num241==702:
            p3412=self.conv1702(D141)
        elif  num241==703:
            p3412=self.conv1703(D141)
        elif  num241==704:
            p3412=self.conv1704(D141)
        elif  num241==705:
            p3412=self.conv1705(D141)
        elif  num241==706:
            p3412=self.conv1706(D141)
        elif  num241==707:
            p3412=self.conv1707(D141)
        elif  num241==708:
            p3412=self.conv1708(D141)
        elif  num241==709:
            p3412=self.conv1709(D141)
        elif  num241==710:
            p3412=self.conv1710(D141)
        elif  num241==711:
            p3412=self.conv1711(D141)
        elif  num241==712:
            p3412=self.conv1712(D141)
        elif  num241==713:
            p3412=self.conv1713(D141)
        elif  num241==714:
            p3412=self.conv1714(D141)
        elif  num241==715:
            p3412=self.conv1715(D141)
        elif  num241==716:
            p3412=self.conv1716(D141)
        elif  num241==717:
            p3412=self.conv1717(D141)
        elif  num241==718:
            p3412=self.conv1718(D141)
        elif  num241==719:
            p3412=self.conv1719(D141)
        elif  num241==720:
            p3412=self.conv1720(D141)
        elif  num241==721:
            p3412=self.conv1721(D141)
        elif  num241==722:
            p3412=self.conv1722(D141)
        elif  num241==723:
            p3412=self.conv1723(D141)
        elif  num241==724:
            p3412=self.conv1724(D141)
        elif  num241==725:
            p3412=self.conv1725(D141)
        elif  num241==726:
            p3412=self.conv1726(D141)
        elif  num241==727:
            p3412=self.conv1727(D141)
        elif  num241==728:
            p3412=self.conv1728(D141)
        elif  num241==729:
            p3412=self.conv1729(D141)
        elif  num241==730:
            p3412=self.conv1730(D141)
        elif  num241==731:
            p3412=self.conv1731(D141)
        elif  num241==732:
            p3412=self.conv1732(D141)
        elif  num241==733:
            p3412=self.conv1733(D141)
        elif  num241==734:
            p3412=self.conv1734(D141)
        elif  num241==735:
            p3412=self.conv1735(D141)
        elif  num241==736:
            p3412=self.conv1736(D141)
        elif  num241==737:
            p3412=self.conv1737(D141)
        elif  num241==738:
            p3412=self.conv1738(D141)
        elif  num241==739:
            p3412=self.conv1739(D141)                                                                                                                          
        elif  num241==740:
            p3412=self.conv1740(D141)
        elif  num241==741:
            p3412=self.conv1741(D141)
        elif  num241==742:
            p3412=self.conv1742(D141)
        elif  num241==743:
            p3412=self.conv1743(D141)
        elif  num241==744:
            p3412=self.conv1744(D141)
        elif  num241==745:
            p3412=self.conv1745(D141)
        elif  num241==746:
            p3412=self.conv1746(D141)
        elif  num241==747:
            p3412=self.conv1747(D141)
        elif  num241==748:
            p3412=self.conv1748(D141)
        elif  num241==749:
            p3412=self.conv1749(D141)
        elif  num241==750:
            p3412=self.conv1750(D141)
        elif  num241==751:
            p3412=self.conv1751(D141)
        elif  num241==752:
            p3412=self.conv1752(D141)
        elif  num241==753:
            p3412=self.conv1753(D141)
        elif  num241==754:
            p3412=self.conv1754(D141)
        elif  num241==755:
            p3412=self.conv1755(D141)
        elif  num241==756:
            p3412=self.conv1756(D141)
        elif  num241==757:
            p3412=self.conv1757(D141)
        elif  num241==758:
            p3412=self.conv1758(D141)
        elif  num241==759:
            p3412=self.conv1759(D141)
        elif  num241==760:
            p3412=self.conv1760(D141)
        elif  num241==761:
            p3412=self.conv1761(D141)
        elif  num241==762:
            p3412=self.conv1762(D141)
        elif  num241==763:
            p3412=self.conv1763(D141)
        elif  num241==764:
            p3412=self.conv1764(D141)
        elif  num241==765:
            p3412=self.conv1765(D141)
        elif  num241==766:
            p3412=self.conv1766(D141)
        elif  num241==767:
            p3412=self.conv1767(D141)
        elif  num241==768:
            p3412=self.conv1768(D141)
        elif  num241==769:
            p3412=self.conv1769(D141) 
        elif  num241==770:
            p3412=self.conv1770(D141)
        elif  num241==771:
            p3412=self.conv1771(D141)
        elif  num241==772:
            p3412=self.conv1772(D141)
        elif  num241==773:
            p3412=self.conv1773(D141)
        elif  num241==774:
            p3412=self.conv1774(D141)
        elif  num241==775:
            p3412=self.conv1775(D141)
        elif  num241==776:
            p3412=self.conv1776(D141)
        elif  num241==777:
            p3412=self.conv1777(D141)
        elif  num241==778:
            p3412=self.conv1778(D141)
        elif  num241==779:
            p3412=self.conv1779(D141) 
        elif  num241==780:
            p3412=self.conv1780(D141)
        elif  num241==781:
            p3412=self.conv1781(D141)
        elif  num241==782:
            p3412=self.conv1782(D141)
        elif  num241==783:
            p3412=self.conv1783(D141)
        elif  num241==784:
            p3412=self.conv1784(D141)
        elif  num241==785:
            p3412=self.conv1785(D141)
        elif  num241==786:
            p3412=self.conv1786(D141)
        elif  num241==787:
            p3412=self.conv1787(D141)
        elif  num241==788:
            p3412=self.conv1788(D141)
        elif  num241==789:
            p3412=self.conv1789(D141) 
        elif  num241==790:
            p3412=self.conv1790(D141)
        elif  num241==791:
            p3412=self.conv1791(D141)
        elif  num241==792:
            p3412=self.conv1792(D141)
        elif  num241==793:
            p3412=self.conv1793(D141)
        elif  num241==794:
            p3412=self.conv1794(D141)
        elif  num241==795:
            p3412=self.conv1795(D141)
        elif  num241==796:
            p3412=self.conv1796(D141)
        elif  num241==797:
            p3412=self.conv1797(D141)
        elif  num241==798:
            p3412=self.conv1798(D141)
        elif  num241==799:
            p3412=self.conv1799(D141) 
        elif  num241==800:
            p3412=self.conv1800(D141)
        elif  num241==801:
            p3412=self.conv1801(D141)
        elif  num241==802:
            p3412=self.conv1802(D141)
        elif  num241==803:
            p3412=self.conv1803(D141)
        elif  num241==804:
            p3412=self.conv1804(D141)
        elif  num241==805:
            p3412=self.conv1805(D141)
        elif  num241==806:
            p3412=self.conv1806(D141)
        elif  num241==807:
            p3412=self.conv1807(D141)
        elif  num241==808:
            p3412=self.conv1808(D141)
        elif  num241==809:
            p3412=self.conv1809(D141)
        elif  num241==810:
            p3412=self.conv1810(D141)
        elif  num241==811:
            p3412=self.conv1811(D141)
        elif  num241==812:
            p3412=self.conv1812(D141)
        elif  num241==813:
            p3412=self.conv1813(D141)
        elif  num241==814:
            p3412=self.conv1814(D141)
        elif  num241==815:
            p3412=self.conv1815(D141)
        elif  num241==816:
            p3412=self.conv1816(D141)
        elif  num241==817:
            p3412=self.conv1817(D141)
        elif  num241==818:
            p3412=self.conv1818(D141)
        elif  num241==819:
            p3412=self.conv1819(D141)
        elif  num241==820:
            p3412=self.conv1820(D141)
        elif  num241==821:
            p3412=self.conv1821(D141)
        elif  num241==822:
            p3412=self.conv1822(D141)
        elif  num241==823:
            p3412=self.conv1823(D141)
        elif  num241==824:
            p3412=self.conv1824(D141)
        elif  num241==825:
            p3412=self.conv1825(D141)
        elif  num241==826:
            p3412=self.conv1826(D141)
        elif  num241==827:
            p3412=self.conv1827(D141)
        elif  num241==828:
            p3412=self.conv1828(D141)
        elif  num241==829:
            p3412=self.conv1829(D141)                                                                                                                          
        elif  num241==830:
            p3412=self.conv1830(D141)
        elif  num241==831:
            p3412=self.conv1831(D141)
        elif  num241==832:
            p3412=self.conv1832(D141)
        elif  num241==833:
            p3412=self.conv1833(D141)
        elif  num241==834:
            p3412=self.conv1834(D141)
        elif  num241==835:
            p3412=self.conv1835(D141)
        elif  num241==836:
            p3412=self.conv1836(D141)
        elif  num241==837:
            p3412=self.conv1837(D141)
        elif  num241==838:
            p3412=self.conv1838(D141)
        elif  num241==839:
            p3412=self.conv1839(D141)
        elif  num241==840:
            p3412=self.conv1840(D141)
        elif  num241==841:
            p3412=self.conv1841(D141)
        elif  num241==842:
            p3412=self.conv1842(D141)
        elif  num241==843:
            p3412=self.conv1843(D141)
        elif  num241==844:
            p3412=self.conv1844(D141)
        elif  num241==845:
            p3412=self.conv1845(D141)
        elif  num241==846:
            p3412=self.conv1846(D141)
        elif  num241==847:
            p3412=self.conv1847(D141)
        elif  num241==848:
            p3412=self.conv1848(D141)
        elif  num241==849:
            p3412=self.conv1849(D141)
        elif  num241==850:
            p3412=self.conv1850(D141)
        elif  num241==851:
            p3412=self.conv1851(D141)
        elif  num241==852:
            p3412=self.conv1852(D141)
        elif  num241==853:
            p3412=self.conv1853(D141)
        elif  num241==854:
            p3412=self.conv1854(D141)
        elif  num241==855:
            p3412=self.conv1855(D141)
        elif  num241==856:
            p3412=self.conv1856(D141)
        elif  num241==857:
            p3412=self.conv1857(D141)
        elif  num241==858:
            p3412=self.conv1858(D141)
        elif  num241==859:
            p3412=self.conv1859(D141)
        elif  num241==860:
            p3412=self.conv1860(D141)
        elif  num241==861:
            p3412=self.conv1861(D141)
        elif  num241==862:
            p3412=self.conv1862(D141)
        elif  num241==863:
            p3412=self.conv1863(D141)
        elif  num241==864:
            p3412=self.conv1864(D141)
        elif  num241==865:
            p3412=self.conv1865(D141)
        elif  num241==866:
            p3412=self.conv1866(D141)
        elif  num241==867:
            p3412=self.conv1867(D141)
        elif  num241==868:
            p3412=self.conv1868(D141)
        elif  num241==869:
            p3412=self.conv1869(D141) 
        elif  num241==870:
            p3412=self.conv1870(D141)
        elif  num241==871:
            p3412=self.conv1871(D141)
        elif  num241==872:
            p3412=self.conv1872(D141)
        elif  num241==873:
            p3412=self.conv1873(D141)
        elif  num241==874:
            p3412=self.conv1874(D141)
        elif  num241==875:
            p3412=self.conv1875(D141)
        elif  num241==876:
            p3412=self.conv1876(D141)
        elif  num241==877:
            p3412=self.conv1877(D141)
        elif  num241==878:
            p3412=self.conv1878(D141)
        elif  num241==879:
            p3412=self.conv1879(D141)
        elif  num241==880:
            p3412=self.conv1880(D141)
        elif  num241==881:
            p3412=self.conv1881(D141)
        elif  num241==882:
            p3412=self.conv1882(D141)
        elif  num241==883:
            p3412=self.conv1883(D141)
        elif  num241==884:
            p3412=self.conv1884(D141)
        elif  num241==885:
            p3412=self.conv1885(D141)
        elif  num241==886:
            p3412=self.conv1886(D141)
        elif  num241==887:
            p3412=self.conv1887(D141)
        elif  num241==888:
            p3412=self.conv1888(D141)
        elif  num241==889:
            p3412=self.conv1889(D141)  
        elif  num241==890:
            p3412=self.conv1890(D141)
        elif  num241==891:
            p3412=self.conv1891(D141)
        elif  num241==892:
            p3412=self.conv1892(D141)
        elif  num241==893:
            p3412=self.conv1893(D141)
        elif  num241==894:
            p3412=self.conv1894(D141)
        elif  num241==895:
            p3412=self.conv1895(D141)
        elif  num241==896:
            p3412=self.conv1896(D141)
        elif  num241==897:
            p3412=self.conv1897(D141)
        elif  num241==898:
            p3412=self.conv1898(D141)
        elif  num241==899:
            p3412=self.conv1899(D141)
        elif  num241==900:
            p3412=self.conv1900(D141)
        elif  num241==901:
            p3412=self.conv1901(D141)
        elif  num241==902:
            p3412=self.conv1902(D141)
        elif  num241==903:
            p3412=self.conv1903(D141)
        elif  num241==904:
            p3412=self.conv1904(D141)
        elif  num241==905:
            p3412=self.conv1905(D141)
        elif  num241==906:
            p3412=self.conv1906(D141)
        elif  num241==907:
            p3412=self.conv1907(D141)
        elif  num241==908:
            p3412=self.conv1908(D141)
        elif  num241==909:
            p3412=self.conv1909(D141)
        elif  num241==910:
            p3412=self.conv1910(D141)
        elif  num241==911:
            p3412=self.conv1911(D141)
        elif  num241==912:
            p3412=self.conv1912(D141)
        elif  num241==913:
            p3412=self.conv1913(D141)
        elif  num241==914:
            p3412=self.conv1914(D141)
        elif  num241==915:
            p3412=self.conv1915(D141)
        elif  num241==916:
            p3412=self.conv1916(D141)
        elif  num241==917:
            p3412=self.conv1917(D141)
        elif  num241==918:
            p3412=self.conv1918(D141)
        elif  num241==919:
            p3412=self.conv1919(D141)
        elif  num241==920:
            p3412=self.conv1920(D141)
        elif  num241==921:
            p3412=self.conv1921(D141)
        elif  num241==922:
            p3412=self.conv1922(D141)
        elif  num241==923:
            p3412=self.conv1923(D141)
        elif  num241==924:
            p3412=self.conv1924(D141)
        elif  num241==925:
            p3412=self.conv1925(D141)
        elif  num241==926:
            p3412=self.conv1926(D141)
        elif  num241==927:
            p3412=self.conv1927(D141)
        elif  num241==928:
            p3412=self.conv1928(D141)
        elif  num241==929:
            p3412=self.conv1929(D141)
        elif  num241==930:
            p3412=self.conv1930(D141)
        elif  num241==931:
            p3412=self.conv1931(D141)
        elif  num241==932:
            p3412=self.conv1932(D141)
        elif  num241==933:
            p3412=self.conv1933(D141)
        elif  num241==934:
            p3412=self.conv1934(D141)
        elif  num241==935:
            p3412=self.conv1935(D141)
        elif  num241==936:
            p3412=self.conv1936(D141)
        elif  num241==937:
            p3412=self.conv1937(D141)
        elif  num241==938:
            p3412=self.conv1938(D141)
        elif  num241==939:
            p3412=self.conv1939(D141) 
        elif  num241==940:
            p3412=self.conv1940(D141)
        elif  num241==941:
            p3412=self.conv1941(D141)
        elif  num241==942:
            p3412=self.conv1942(D141)
        elif  num241==943:
            p3412=self.conv1943(D141)
        elif  num241==944:
            p3412=self.conv1944(D141)
        elif  num241==945:
            p3412=self.conv1945(D141)
        elif  num241==946:
            p3412=self.conv1946(D141)
        elif  num241==947:
            p3412=self.conv1947(D141)
        elif  num241==948:
            p3412=self.conv1948(D141)
        elif  num241==949:
            p3412=self.conv1949(D141)                                                                                                                          
        elif  num241==950:
            p3412=self.conv1950(D141)
        elif  num241==951:
            p3412=self.conv1951(D141)
        elif  num241==952:
            p3412=self.conv1952(D141)
        elif  num241==953:
            p3412=self.conv1953(D141)
        elif  num241==954:
            p3412=self.conv1954(D141)
        elif  num241==955:
            p3412=self.conv1955(D141)
        elif  num241==956:
            p3412=self.conv1956(D141)
        elif  num241==957:
            p3412=self.conv1957(D141)
        elif  num241==958:
            p3412=self.conv1958(D141)
        elif  num241==959:
            p3412=self.conv1959(D141)
        elif  num241==960:
            p3412=self.conv1960(D141)
        elif  num241==961:
            p3412=self.conv1961(D141)
        elif  num241==962:
            p3412=self.conv1962(D141)
        elif  num241==963:
            p3412=self.conv1963(D141)
        elif  num241==964:
            p3412=self.conv1964(D141)
        elif  num241==965:
            p3412=self.conv1965(D141)
        elif  num241==966:
            p3412=self.conv1966(D141)
        elif  num241==967:
            p3412=self.conv1967(D141)
        elif  num241==968:
            p3412=self.conv1968(D141)
        elif  num241==969:
            p3412=self.conv1969(D141) 
        elif  num241==970:
            p3412=self.conv1970(D141)
        elif  num241==971:
            p3412=self.conv1971(D141)
        elif  num241==972:
            p3412=self.conv1972(D141)
        elif  num241==973:
            p3412=self.conv1973(D141)
        elif  num241==974:
            p3412=self.conv1974(D141)
        elif  num241==975:
            p3412=self.conv1975(D141)
        elif  num241==976:
            p3412=self.conv1976(D141)
        elif  num241==977:
            p3412=self.conv1977(D141)
        elif  num241==978:
            p3412=self.conv1978(D141)
        elif  num241==979:
            p3412=self.conv1979(D141) 
        elif  num241==980:
            p3412=self.conv1980(D141)
        elif  num241==981:
            p3412=self.conv1981(D141)
        elif  num241==982:
            p3412=self.conv1982(D141)
        elif  num241==983:
            p3412=self.conv1983(D141)
        elif  num241==984:
            p3412=self.conv1984(D141)
        elif  num241==985:
            p3412=self.conv1985(D141)
        elif  num241==986:
            p3412=self.conv1986(D141)
        elif  num241==987:
            p3412=self.conv1987(D141)
        elif  num241==988:
            p3412=self.conv1988(D141)
        elif  num241==989:
            p3412=self.conv1989(D141)
        elif  num241==990:
            p3412=self.conv1990(D141)
        elif  num241==991:
            p3412=self.conv1991(D141)
        elif  num241==992:
            p3412=self.conv1992(D141)
        elif  num241==993:
            p3412=self.conv1993(D141)
        elif  num241==994:
            p3412=self.conv1994(D141)
        elif  num241==995:
            p3412=self.conv1995(D141)
        elif  num241==996:
            p3412=self.conv1996(D141)
        elif  num241==997:
            p3412=self.conv1997(D141)
        elif  num241==998:
            p3412=self.conv1998(D141)
        elif  num241==999:
            p3412=self.conv1999(D141) 
        elif  num241==1000:
            p3412=self.conv11000(D141)
        elif  num241==1001:
            p3412=self.conv11001(D141)
        elif  num241==1002:
            p3412=self.conv11002(D141)
        elif  num241==1003:
            p3412=self.conv11003(D141)
        elif  num241==1004:
            p3412=self.conv11004(D141)
        elif  num241==1005:
            p3412=self.conv11005(D141)
        elif  num241==1006:
            p3412=self.conv11006(D141)
        elif  num241==1007:
            p3412=self.conv11007(D141)
        elif  num241==1008:
            p3412=self.conv11008(D141)
        elif  num241==1009:
            p3412=self.conv11009(D141) 
        elif  num241==1010:
            p3412=self.conv11010(D141)
        elif  num241==1011:
            p3412=self.conv11011(D141)
        elif  num241==1012:
            p3412=self.conv11012(D141)
        elif  num241==1013:
            p3412=self.conv11013(D141)
        elif  num241==1014:
            p3412=self.conv11014(D141)
        elif  num241==1015:
            p3412=self.conv11015(D141)
        elif  num241==1016:
            p3412=self.conv11016(D141)
        elif  num241==1017:
            p3412=self.conv11017(D141)
        elif  num241==1018:
            p3412=self.conv11018(D141)
        elif  num241==1019:
            p3412=self.conv11019(D141)
        elif  num241==1020:
            p3412=self.conv11020(D141)
        elif  num241==1021:
            p3412=self.conv11021(D141)
        elif  num241==1022:
            p3412=self.conv11022(D141)
        elif  num241==1023:
            p3412=self.conv11023(D141)
        elif  num241==1024:
            p3412=self.conv11024(D141) 
            
        if num080==1:
            p380=self.conv11(B180)
        elif num080==2:
            p380=self.conv12(B180)
        elif num080==3:
            p380=self.conv13(B180)
        elif num080==4:
            p380=self.conv14(B180)
        elif num080==5:
            p380=self.conv15(B180)
        elif num080==6:
            p380=self.conv16(B180)
        elif num080==7:
            p380=self.conv17(B180)
        elif num080==8:
            p380=self.conv18(B180)
        elif num080==9:
            p380=self.conv19(B180)
        elif num080==10:
            p380=self.conv110(B180)
        elif num080==11:
            p380=self.conv111(B180)
        elif num080==12:
            p380=self.conv112(B180)
        elif num080==13:
            p380=self.conv113(B180)
        elif num080==14:
            p380=self.conv114(B180)
        elif num080==15:
            p380=self.conv115(B180)
        elif num080==16:
            p380=self.conv116(B180)
        elif num080==17:
            p380=self.conv117(B180)
        elif num080==18:
            p380=self.conv118(B180)
        elif num080==19:
            p380=self.conv119(B180)
        elif num080==20:
            p380=self.conv120(B180)
        elif num080==21:
            p380=self.conv121(B180)
        elif num080==22:
            p380=self.conv122(B180)
        elif num080==23:
            p380=self.conv123(B180)
        elif num080==24:
            p380=self.conv124(B180)
        elif num080==25:
            p380=self.conv125(B180)
        elif num080==26:
            p380=self.conv126(B180)
        elif num080==27:
            p380=self.conv127(B180)
        elif num080==28:
            p380=self.conv128(B180)
        elif num080==29:
            p380=self.conv129(B180)
        elif num080==30:
            p380=self.conv130(B180)
        elif num080==31:
            p380=self.conv131(B180)
        elif num080==32:
            p380=self.conv132(B180)
        elif num080==33:
            p380=self.conv133(B180)
        elif num080==34:
            p380=self.conv134(B180)
        elif num080==35:
            p380=self.conv135(B180)
        elif num080==36:
            p380=self.conv136(B180)
        elif num080==37:
            p380=self.conv137(B180)
        elif num080==38:
            p380=self.conv138(B180)
        elif num080==39:
            p380=self.conv139(B180)
        elif num080==40:
            p380=self.conv140(B180)
        elif num080==41:
            p380=self.conv141(B180)
        elif num080==42:
            p380=self.conv142(B180)
        elif num080==43:
            p380=self.conv143(B180)
        elif num080==44:
            p380=self.conv144(B180)
        elif num080==45:
            p380=self.conv145(B180)
        elif num080==46:
            p380=self.conv146(B180)
        elif num080==47:
            p380=self.conv147(B180)
        elif num080==48:
            p380=self.conv148(B180)
        elif num080==49:
            p380=self.conv149(B180)
        elif num080==50:
            p380=self.conv150(B180)
        elif num080==51:
            p380=self.conv151(B180)
        elif num080==52:
            p380=self.conv152(B180)
        elif num080==53:
            p380=self.conv153(B180)
        elif num080==54:
            p380=self.conv154(B180)
        elif num080==55:
            p380=self.conv155(B180)
        elif num080==56:
            p380=self.conv156(B180)
        elif num080==57:
            p380=self.conv157(B180)
        elif num080==58:
            p380=self.conv158(B180)
        elif num080==59:
            p380=self.conv159(B180)
        elif num080==60:
            p380=self.conv160(B180)
        elif num080==61:
            p380=self.conv161(B180)
        elif num080==62:
            p380=self.conv162(B180)
        elif num080==63:
            p380=self.conv163(B180)
        elif num080==64:
            p380=self.conv164(B180)
        
        if  num180==1:
            p3801=self.conv11(C180)
        elif  num180==2:
            p3801=self.conv12(C180)
        elif  num180==3:
            p3801=self.conv13(C180)
        elif  num180==4:
            p3801=self.conv14(C180)
        elif  num180==5:
            p3801=self.conv15(C180)
        elif  num180==6:
            p3801=self.conv16(C180)
        elif  num180==7:
            p3801=self.conv17(C180)
        elif  num180==8:
            p3801=self.conv18(C180)
        elif  num180==9:
            p3801=self.conv19(C180)
        elif  num180==10:
            p3801=self.conv110(C180)
        elif  num180==11:
            p3801=self.conv111(C180)
        elif  num180==12:
            p3801=self.conv112(C180)
        elif  num180==13:
            p3801=self.conv113(C180)
        elif  num180==14:
            p3801=self.conv114(C180)
        elif  num180==15:
            p3801=self.conv115(C180)
        elif  num180==16:
            p3801=self.conv116(C180)
        elif  num180==17:
            p3801=self.conv117(C180)
        elif  num180==18:
            p3801=self.conv118(C180)
        elif  num180==19:
            p3801=self.conv119(C180)
        elif  num180==20:
            p3801=self.conv120(C180)
        elif  num180==21:
            p3801=self.conv121(C180)
        elif  num180==22:
            p3801=self.conv122(C180)
        elif  num180==23:
            p3801=self.conv123(C180)
        elif  num180==24:
            p3801=self.conv124(C180)
        elif  num180==25:
            p3801=self.conv125(C180)
        elif  num180==26:
            p3801=self.conv126(C180)
        elif  num180==27:
            p3801=self.conv127(C180)
        elif  num180==28:
            p3801=self.conv128(C180)
        elif  num180==29:
            p3801=self.conv129(C180)
        elif  num180==30:
            p3801=self.conv130(C180)
        elif  num180==31:
            p3801=self.conv131(C180)
        elif  num180==32:
            p3801=self.conv132(C180)
        elif  num180==33:
            p3801=self.conv133(C180)
        elif  num180==34:
            p3801=self.conv134(C180)
        elif  num180==35:
            p3801=self.conv135(C180)
        elif  num180==36:
            p3801=self.conv136(C180)
        elif  num180==37:
            p3801=self.conv137(C180)
        elif  num180==38:
            p3801=self.conv138(C180)
        elif  num180==39:
            p3801=self.conv139(C180)
        elif  num180==40:
            p3801=self.conv140(C180)
        elif  num180==41:
            p3801=self.conv141(C180)
        elif  num180==42:
            p3801=self.conv142(C180)
        elif  num180==43:
            p3801=self.conv143(C180)
        elif  num180==44:
            p3801=self.conv144(C180)
        elif  num180==45:
            p3801=self.conv145(C180)
        elif  num180==46:
            p3801=self.conv146(C180)
        elif  num180==47:
            p3801=self.conv147(C180)
        elif  num180==48:
            p3801=self.conv148(C180)
        elif  num180==49:
            p3801=self.conv149(C180)
        elif  num180==50:
            p3801=self.conv150(C180)
        elif  num180==51:
            p3801=self.conv151(C180)
        elif  num180==52:
            p3801=self.conv152(C180)
        elif  num180==53:
            p3801=self.conv153(C180)
        elif  num180==54:
            p3801=self.conv154(C180)
        elif  num180==55:
            p3801=self.conv155(C180)
        elif  num180==56:
            p3801=self.conv156(C180)
        elif  num180==57:
            p3801=self.conv157(C180)
        elif  num180==58:
            p3801=self.conv158(C180)
        elif  num180==59:
            p3801=self.conv159(C180)
        elif  num180==60:
            p3801=self.conv160(C180)
        elif  num180==61:
            p3801=self.conv161(C180)
        elif  num180==62:
            p3801=self.conv162(C180)
        elif  num180==63:
            p3801=self.conv163(C180)
        elif  num180==64:
            p3801=self.conv164(C180)
        elif  num180==65:
            p3801=self.conv165(C180)
        elif  num180==66:
            p3801=self.conv166(C180)
        elif  num180==67:
            p3801=self.conv167(C180)
        elif  num180==68:
            p3801=self.conv168(C180)
        elif  num180==69:
            p3801=self.conv169(C180)
        elif  num180==70:
            p3801=self.conv170(C180)
        elif  num180==71:
            p3801=self.conv171(C180)
        elif  num180==72:
            p3801=self.conv172(C180)
        elif  num180==73:
            p3801=self.conv173(C180)
        elif  num180==74:
            p3801=self.conv174(C180)
        elif  num180==75:
            p3801=self.conv175(C180)
        elif  num180==76:
            p3801=self.conv176(C180)
        elif  num180==77:
            p3801=self.conv177(C180)
        elif  num180==78:
            p3801=self.conv178(C180)
        elif  num180==79:
            p3801=self.conv179(C180)
        elif  num180==80:
            p3801=self.conv180(C180)
        elif  num180==81:
            p3801=self.conv181(C180)
        elif  num180==82:
            p3801=self.conv182(C180)
        elif  num180==83:
            p3801=self.conv183(C180)
        elif  num180==84:
            p3801=self.conv184(C180)
        elif  num180==85:
            p3801=self.conv185(C180)
        elif  num180==86:
            p3801=self.conv186(C180)
        elif  num180==87:
            p3801=self.conv187(C180)
        elif  num180==88:
            p3801=self.conv188(C180)
        elif  num180==89:
            p3801=self.conv189(C180)    
        elif  num180==90:
            p3801=self.conv190(C180)
        elif  num180==91:
            p3801=self.conv191(C180)
        elif  num180==92:
            p3801=self.conv192(C180)
        elif  num180==93:
            p3801=self.conv193(C180)
        elif  num180==94:
            p3801=self.conv194(C180)
        elif  num180==95:
            p3801=self.conv195(C180)
        elif  num180==96:
            p3801=self.conv196(C180)
        elif  num180==97:
            p3801=self.conv197(C180)
        elif  num180==98:
            p3801=self.conv198(C180)
        elif  num180==99:
            p3801=self.conv199(C180) 
        elif  num180==100:
            p3801=self.conv1100(C180)
        elif  num180==101:
            p3801=self.conv1101(C180)
        elif  num180==102:
            p3801=self.conv1102(C180)
        elif  num180==103:
            p3801=self.conv1103(C180)
        elif  num180==104:
            p3801=self.conv1104(C180)
        elif  num180==105:
            p3801=self.conv1105(C180)
        elif  num180==106:
            p3801=self.conv1106(C180)
        elif  num180==107:
            p3801=self.conv1107(C180)
        elif  num180==108:
            p3801=self.conv1108(C180)
        elif  num180==109:
            p3801=self.conv1109(C180)
        elif  num180==110:
            p3801=self.conv1110(C180)
        elif  num180==111:
            p3801=self.conv1111(C180)
        elif  num180==112:
            p3801=self.conv1112(C180)
        elif  num180==113:
            p3801=self.conv1113(C180)
        elif  num180==114:
            p3801=self.conv1114(C180)
        elif  num180==115:
            p3801=self.conv1115(C180)
        elif  num180==116:
            p3801=self.conv1116(C180)
        elif  num180==117:
            p3801=self.conv1117(C180)
        elif  num180==118:
            p3801=self.conv1118(C180)
        elif  num180==119:
            p3801=self.conv1119(C180) 
        elif  num180==120:
            p3801=self.conv1120(C180)
        elif  num180==121:
            p3801=self.conv1121(C180)
        elif  num180==122:
            p3801=self.conv1122(C180)
        elif  num180==123:
            p3801=self.conv1123(C180)
        elif  num180==124:
            p3801=self.conv1124(C180)
        elif  num180==125:
            p3801=self.conv1125(C180)
        elif  num180==126:
            p3801=self.conv1126(C180)
        elif  num180==127:
            p3801=self.conv1127(C180)
        elif  num180==128:
            p3801=self.conv1128(C180)
        elif  num180==129:
            p3801=self.conv1129(C180) 
        elif  num180==130:
            p3801=self.conv1130(C180)
        elif  num180==131:
            p3801=self.conv1131(C180)
        elif  num180==132:
            p3801=self.conv1132(C180)
        elif  num180==133:
            p3801=self.conv1133(C180)
        elif  num180==134:
            p3801=self.conv1134(C180)
        elif  num180==135:
            p3801=self.conv1135(C180)
        elif  num180==136:
            p3801=self.conv1136(C180)
        elif  num180==137:
            p3801=self.conv1137(C180)
        elif  num180==138:
            p3801=self.conv1138(C180)
        elif  num180==139:
            p3801=self.conv1139(C180)
        elif  num180==140:
            p3801=self.conv1140(C180)
        elif  num180==141:
            p3801=self.conv1141(C180)
        elif  num180==142:
            p3801=self.conv1142(C180)
        elif  num180==143:
            p3801=self.conv1143(C180)
        elif  num180==144:
            p3801=self.conv1144(C180)
        elif  num180==145:
            p3801=self.conv1145(C180)
        elif  num180==146:
            p3801=self.conv1146(C180)
        elif  num180==147:
            p3801=self.conv1147(C180)
        elif  num180==148:
            p3801=self.conv1148(C180)
        elif  num180==149:
            p3801=self.conv1149(C180) 
        elif  num180==150:
            p3801=self.conv1150(C180)
        elif  num180==151:
            p3801=self.conv1151(C180)
        elif  num180==152:
            p3801=self.conv1152(C180)
        elif  num180==153:
            p3801=self.conv1153(C180)
        elif  num180==154:
            p3801=self.conv1154(C180)
        elif  num180==155:
            p3801=self.conv1155(C180)
        elif  num180==156:
            p3801=self.conv1156(C180)
        elif  num180==157:
            p3801=self.conv1157(C180)
        elif  num180==158:
            p3801=self.conv1158(C180)
        elif  num180==159:
            p3801=self.conv1159(C180) 
        elif  num180==160:
            p3801=self.conv1160(C180)
        elif  num180==161:
            p3801=self.conv1161(C180)
        elif  num180==162:
            p3801=self.conv1162(C180)
        elif  num180==163:
            p3801=self.conv1163(C180)
        elif  num180==164:
            p3801=self.conv1164(C180)
        elif  num180==165:
            p3801=self.conv1165(C180)
        elif  num180==166:
            p3801=self.conv1166(C180)
        elif  num180==167:
            p3801=self.conv1167(C180)
        elif  num180==168:
            p3801=self.conv1168(C180)
        elif  num180==169:
            p3801=self.conv1169(C180) 
        elif  num180==170:
            p3801=self.conv1170(C180)
        elif  num180==171:
            p3801=self.conv1171(C180)
        elif  num180==172:
            p3801=self.conv1172(C180)
        elif  num180==173:
            p3801=self.conv1173(C180)
        elif  num180==174:
            p3801=self.conv1174(C180)
        elif  num180==175:
            p3801=self.conv1175(C180)
        elif  num180==176:
            p3801=self.conv1176(C180)
        elif  num180==177:
            p3801=self.conv1177(C180)
        elif  num180==178:
            p3801=self.conv1178(C180)
        elif  num180==179:
            p3801=self.conv1179(C180)                                                                                              
        elif  num180==180:
            p3801=self.conv1180(C180)
        elif  num180==181:
            p3801=self.conv1181(C180)
        elif  num180==182:
            p3801=self.conv1182(C180)
        elif  num180==183:
            p3801=self.conv1183(C180)
        elif  num180==184:
            p3801=self.conv1184(C180)
        elif  num180==185:
            p3801=self.conv1185(C180)
        elif  num180==186:
            p3801=self.conv1186(C180)
        elif  num180==187:
            p3801=self.conv1187(C180)
        elif  num180==188:
            p3801=self.conv1188(C180)
        elif  num180==189:
            p3801=self.conv1189(C180) 
        elif  num180==190:
            p3801=self.conv1190(C180)
        elif  num180==191:
            p3801=self.conv1191(C180)
        elif  num180==192:
            p3801=self.conv1192(C180)
        elif  num180==193:
            p3801=self.conv1193(C180)
        elif  num180==194:
            p3801=self.conv1194(C180)
        elif  num180==195:
            p3801=self.conv1195(C180)
        elif  num180==196:
            p3801=self.conv1196(C180)
        elif  num180==197:
            p3801=self.conv1197(C180)
        elif  num180==198:
            p3801=self.conv1198(C180)
        elif  num180==199:
            p3801=self.conv1199(C180)
        elif  num180==200:
            p3801=self.conv1200(C180)
        elif  num180==201:
            p3801=self.conv1201(C180)
        elif  num180==202:
            p3801=self.conv1202(C180)
        elif  num180==203:
            p3801=self.conv1203(C180)
        elif  num180==204:
            p3801=self.conv1204(C180)
        elif  num180==205:
            p3801=self.conv1205(C180)
        elif  num180==206:
            p3801=self.conv1206(C180)
        elif  num180==207:
            p3801=self.conv1207(C180)
        elif  num180==208:
            p3801=self.conv1208(C180)
        elif  num180==209:
            p3801=self.conv1209(C180)
        elif  num180==210:
            p3801=self.conv1210(C180)
        elif  num180==211:
            p3801=self.conv1211(C180)
        elif  num180==212:
            p3801=self.conv1212(C180)
        elif  num180==213:
            p3801=self.conv1213(C180)
        elif  num180==214:
            p3801=self.conv1214(C180)
        elif  num180==215:
            p3801=self.conv1215(C180)
        elif  num180==216:
            p3801=self.conv1216(C180)
        elif  num180==217:
            p3801=self.conv1217(C180)
        elif  num180==218:
            p3801=self.conv1218(C180)
        elif  num180==219:
            p3801=self.conv1219(C180)
        elif  num180==220:
            p3801=self.conv1220(C180)
        elif  num180==221:
            p3801=self.conv1221(C180)
        elif  num180==222:
            p3801=self.conv1222(C180)
        elif  num180==223:
            p3801=self.conv1223(C180)
        elif  num180==224:
            p3801=self.conv1224(C180)
        elif  num180==225:
            p3801=self.conv1225(C180)
        elif  num180==226:
            p3801=self.conv1226(C180)
        elif  num180==227:
            p3801=self.conv1227(C180)
        elif  num180==228:
            p3801=self.conv1228(C180)
        elif  num180==229:
            p3801=self.conv1229(C180)
        elif  num180==230:
            p3801=self.conv1230(C180)
        elif  num180==231:
            p3801=self.conv1231(C180)
        elif  num180==232:
            p3801=self.conv1232(C180)
        elif  num180==233:
            p3801=self.conv1233(C180)
        elif  num180==234:
            p3801=self.conv1234(C180)
        elif  num180==235:
            p3801=self.conv1235(C180)
        elif  num180==236:
            p3801=self.conv1236(C180)
        elif  num180==237:
            p3801=self.conv1237(C180)
        elif  num180==238:
            p3801=self.conv1238(C180)
        elif  num180==239:
            p3801=self.conv1239(C180) 
        elif  num180==240:
            p3801=self.conv1240(C180)
        elif  num180==241:
            p3801=self.conv1241(C180)
        elif  num180==242:
            p3801=self.conv1242(C180)
        elif  num180==243:
            p3801=self.conv1243(C180)
        elif  num180==244:
            p3801=self.conv1244(C180)
        elif  num180==245:
            p3801=self.conv1245(C180)
        elif  num180==246:
            p3801=self.conv1246(C180)
        elif  num180==247:
            p3801=self.conv1247(C180)
        elif  num180==248:
            p3801=self.conv1248(C180)
        elif  num180==249:
            p3801=self.conv1249(C180)
        elif  num180==250:
            p3801=self.conv1250(C180)
        elif  num180==251:
            p3801=self.conv1251(C180)
        elif  num180==252:
            p3801=self.conv1252(C180)
        elif  num180==253:
            p3801=self.conv1253(C180)
        elif  num180==254:
            p3801=self.conv1254(C180)
        elif  num180==255:
            p3801=self.conv1255(C180)
        elif  num180==256:
            p3801=self.conv1256(C180)
            
        if  num280==1:
            p3802=self.conv11(D180)
        elif  num280==2:
            p3802=self.conv12(D180)
        elif  num280==3:
            p3802=self.conv13(D180)
        elif  num280==4:
            p3802=self.conv14(D180)
        elif  num280==5:
            p3802=self.conv15(D180)
        elif  num280==6:
            p3802=self.conv16(D180)
        elif  num280==7:
            p3802=self.conv17(D180)
        elif  num280==8:
            p3802=self.conv18(D180)
        elif  num280==9:
            p3802=self.conv19(D180)
        elif  num280==10:
            p3802=self.conv110(D180)
        elif  num280==11:
            p3802=self.conv111(D180)
        elif  num280==12:
            p3802=self.conv112(D180)
        elif  num280==13:
            p3802=self.conv113(D180)
        elif  num280==14:
            p3802=self.conv114(D180)
        elif  num280==15:
            p3802=self.conv115(D180)
        elif  num280==16:
            p3802=self.conv116(D180)
        elif  num280==17:
            p3802=self.conv117(D180)
        elif  num280==18:
            p3802=self.conv118(D180)
        elif  num280==19:
            p3802=self.conv119(D180)
        elif  num280==20:
            p3802=self.conv120(D180)
        elif  num280==21:
            p3802=self.conv121(D180)
        elif  num280==22:
            p3802=self.conv122(D180)
        elif  num280==23:
            p3802=self.conv123(D180)
        elif  num280==24:
            p3802=self.conv124(D180)
        elif  num280==25:
            p3802=self.conv125(D180)
        elif  num280==26:
            p3802=self.conv126(D180)
        elif  num280==27:
            p3802=self.conv127(D180)
        elif  num280==28:
            p3802=self.conv128(D180)
        elif  num280==29:
            p3802=self.conv129(D180)
        elif  num280==30:
            p3802=self.conv130(D180)
        elif  num280==31:
            p3802=self.conv131(D180)
        elif  num280==32:
            p3802=self.conv132(D180)
        elif  num280==33:
            p3802=self.conv133(D180)
        elif  num280==34:
            p3802=self.conv134(D180)
        elif  num280==35:
            p3802=self.conv135(D180)
        elif  num280==36:
            p3802=self.conv136(D180)
        elif  num280==37:
            p3802=self.conv137(D180)
        elif  num280==38:
            p3802=self.conv138(D180)
        elif  num280==39:
            p3802=self.conv139(D180)
        elif  num280==40:
            p3802=self.conv140(D180)
        elif  num280==41:
            p3802=self.conv141(D180)
        elif  num280==42:
            p3802=self.conv142(D180)
        elif  num280==43:
            p3802=self.conv143(D180)
        elif  num280==44:
            p3802=self.conv144(D180)
        elif  num280==45:
            p3802=self.conv145(D180)
        elif  num280==46:
            p3802=self.conv146(D180)
        elif  num280==47:
            p3802=self.conv147(D180)
        elif  num280==48:
            p3802=self.conv148(D180)
        elif  num280==49:
            p3802=self.conv149(D180)
        elif  num280==50:
            p3802=self.conv150(D180)
        elif  num280==51:
            p3802=self.conv151(D180)
        elif  num280==52:
            p3802=self.conv152(D180)
        elif  num280==53:
            p3802=self.conv153(D180)
        elif  num280==54:
            p3802=self.conv154(D180)
        elif  num280==55:
            p3802=self.conv155(D180)
        elif  num280==56:
            p3802=self.conv156(D180)
        elif  num280==57:
            p3802=self.conv157(D180)
        elif  num280==58:
            p3802=self.conv158(D180)
        elif  num280==59:
            p3802=self.conv159(D180)
        elif  num280==60:
            p3802=self.conv160(D180)
        elif  num280==61:
            p3802=self.conv161(D180)
        elif  num280==62:
            p3802=self.conv162(D180)
        elif  num280==63:
            p3802=self.conv163(D180)
        elif  num280==64:
            p3802=self.conv164(D180)
        elif  num280==65:
            p3802=self.conv165(D180)
        elif  num280==66:
            p3802=self.conv166(D180)
        elif  num280==67:
            p3802=self.conv167(D180)
        elif  num280==68:
            p3802=self.conv168(D180)
        elif  num280==69:
            p3802=self.conv169(D180)
        elif  num280==70:
            p3802=self.conv170(D180)
        elif  num280==71:
            p3802=self.conv171(D180)
        elif  num280==72:
            p3802=self.conv172(D180)
        elif  num280==73:
            p3802=self.conv173(D180)
        elif  num280==74:
            p3802=self.conv174(D180)
        elif  num280==75:
            p3802=self.conv175(D180)
        elif  num280==76:
            p3802=self.conv176(D180)
        elif  num280==77:
            p3802=self.conv177(D180)
        elif  num280==78:
            p3802=self.conv178(D180)
        elif  num280==79:
            p3802=self.conv179(D180)
        elif  num280==80:
            p3802=self.conv180(D180)
        elif  num280==81:
            p3802=self.conv181(D180)
        elif  num280==82:
            p3802=self.conv182(D180)
        elif  num280==83:
            p3802=self.conv183(D180)
        elif  num280==84:
            p3802=self.conv184(D180)
        elif  num280==85:
            p3802=self.conv185(D180)
        elif  num280==86:
            p3802=self.conv186(D180)
        elif  num280==87:
            p3802=self.conv187(D180)
        elif  num280==88:
            p3802=self.conv188(D180)
        elif  num280==89:
            p3802=self.conv189(D180)    
        elif  num280==90:
            p3802=self.conv190(D180)
        elif  num280==91:
            p3802=self.conv191(D180)
        elif  num280==92:
            p3802=self.conv192(D180)
        elif  num280==93:
            p3802=self.conv193(D180)
        elif  num280==94:
            p3802=self.conv194(D180)
        elif  num280==95:
            p3802=self.conv195(D180)
        elif  num280==96:
            p3802=self.conv196(D180)
        elif  num280==97:
            p3802=self.conv197(D180)
        elif  num280==98:
            p3802=self.conv198(D180)
        elif  num280==99:
            p3802=self.conv199(D180) 
        elif  num280==100:
            p3802=self.conv1100(D180)
        elif  num280==101:
            p3802=self.conv1101(D180)
        elif  num280==102:
            p3802=self.conv1102(D180)
        elif  num280==103:
            p3802=self.conv1103(D180)
        elif  num280==104:
            p3802=self.conv1104(D180)
        elif  num280==105:
            p3802=self.conv1105(D180)
        elif  num280==106:
            p3802=self.conv1106(D180)
        elif  num280==107:
            p3802=self.conv1107(D180)
        elif  num280==108:
            p3802=self.conv1108(D180)
        elif  num280==109:
            p3802=self.conv1109(D180)
        elif  num280==110:
            p3802=self.conv1110(D180)
        elif  num280==111:
            p3802=self.conv1111(D180)
        elif  num280==112:
            p3802=self.conv1112(D180)
        elif  num280==113:
            p3802=self.conv1113(D180)
        elif  num280==114:
            p3802=self.conv1114(D180)
        elif  num280==115:
            p3802=self.conv1115(D180)
        elif  num280==116:
            p3802=self.conv1116(D180)
        elif  num280==117:
            p3802=self.conv1117(D180)
        elif  num280==118:
            p3802=self.conv1118(D180)
        elif  num280==119:
            p3802=self.conv1119(D180) 
        elif  num280==120:
            p3802=self.conv1120(D180)
        elif  num280==121:
            p3802=self.conv1121(D180)
        elif  num280==122:
            p3802=self.conv1122(D180)
        elif  num280==123:
            p3802=self.conv1123(D180)
        elif  num280==124:
            p3802=self.conv1124(D180)
        elif  num280==125:
            p3802=self.conv1125(D180)
        elif  num280==126:
            p3802=self.conv1126(D180)
        elif  num280==127:
            p3802=self.conv1127(D180)
        elif  num280==128:
            p3802=self.conv1128(D180)
        elif  num280==129:
            p3802=self.conv1129(D180) 
        elif  num280==130:
            p3802=self.conv1130(D180)
        elif  num280==131:
            p3802=self.conv1131(D180)
        elif  num280==132:
            p3802=self.conv1132(D180)
        elif  num280==133:
            p3802=self.conv1133(D180)
        elif  num280==134:
            p3802=self.conv1134(D180)
        elif  num280==135:
            p3802=self.conv1135(D180)
        elif  num280==136:
            p3802=self.conv1136(D180)
        elif  num280==137:
            p3802=self.conv1137(D180)
        elif  num280==138:
            p3802=self.conv1138(D180)
        elif  num280==139:
            p3802=self.conv1139(D180)
        elif  num280==140:
            p3802=self.conv1140(D180)
        elif  num280==141:
            p3802=self.conv1141(D180)
        elif  num280==142:
            p3802=self.conv1142(D180)
        elif  num280==143:
            p3802=self.conv1143(D180)
        elif  num280==144:
            p3802=self.conv1144(D180)
        elif  num280==145:
            p3802=self.conv1145(D180)
        elif  num280==146:
            p3802=self.conv1146(D180)
        elif  num280==147:
            p3802=self.conv1147(D180)
        elif  num280==148:
            p3802=self.conv1148(D180)
        elif  num280==149:
            p3802=self.conv1149(D180) 
        elif  num280==150:
            p3802=self.conv1150(D180)
        elif  num280==151:
            p3802=self.conv1151(D180)
        elif  num280==152:
            p3802=self.conv1152(D180)
        elif  num280==153:
            p3802=self.conv1153(D180)
        elif  num280==154:
            p3802=self.conv1154(D180)
        elif  num280==155:
            p3802=self.conv1155(D180)
        elif  num280==156:
            p3802=self.conv1156(D180)
        elif  num280==157:
            p3802=self.conv1157(D180)
        elif  num280==158:
            p3802=self.conv1158(D180)
        elif  num280==159:
            p3802=self.conv1159(D180) 
        elif  num280==160:
            p3802=self.conv1160(D180)
        elif  num280==161:
            p3802=self.conv1161(D180)
        elif  num280==162:
            p3802=self.conv1162(D180)
        elif  num280==163:
            p3802=self.conv1163(D180)
        elif  num280==164:
            p3802=self.conv1164(D180)
        elif  num280==165:
            p3802=self.conv1165(D180)
        elif  num280==166:
            p3802=self.conv1166(D180)
        elif  num280==167:
            p3802=self.conv1167(D180)
        elif  num280==168:
            p3802=self.conv1168(D180)
        elif  num280==169:
            p3802=self.conv1169(D180) 
        elif  num280==170:
            p3802=self.conv1170(D180)
        elif  num280==171:
            p3802=self.conv1171(D180)
        elif  num280==172:
            p3802=self.conv1172(D180)
        elif  num280==173:
            p3802=self.conv1173(D180)
        elif  num280==174:
            p3802=self.conv1174(D180)
        elif  num280==175:
            p3802=self.conv1175(D180)
        elif  num280==176:
            p3802=self.conv1176(D180)
        elif  num280==177:
            p3802=self.conv1177(D180)
        elif  num280==178:
            p3802=self.conv1178(D180)
        elif  num280==179:
            p3802=self.conv1179(D180)                                                                                              
        elif  num280==180:
            p3802=self.conv1180(D180)
        elif  num280==181:
            p3802=self.conv1181(D180)
        elif  num280==182:
            p3802=self.conv1182(D180)
        elif  num280==183:
            p3802=self.conv1183(D180)
        elif  num280==184:
            p3802=self.conv1184(D180)
        elif  num280==185:
            p3802=self.conv1185(D180)
        elif  num280==186:
            p3802=self.conv1186(D180)
        elif  num280==187:
            p3802=self.conv1187(D180)
        elif  num280==188:
            p3802=self.conv1188(D180)
        elif  num280==189:
            p3802=self.conv1189(D180) 
        elif  num280==190:
            p3802=self.conv1190(D180)
        elif  num280==191:
            p3802=self.conv1191(D180)
        elif  num280==192:
            p3802=self.conv1192(D180)
        elif  num280==193:
            p3802=self.conv1193(D180)
        elif  num280==194:
            p3802=self.conv1194(D180)
        elif  num280==195:
            p3802=self.conv1195(D180)
        elif  num280==196:
            p3802=self.conv1196(D180)
        elif  num280==197:
            p3802=self.conv1197(D180)
        elif  num280==198:
            p3802=self.conv1198(D180)
        elif  num280==199:
            p3802=self.conv1199(D180)
        elif  num280==200:
            p3802=self.conv1200(D180)
        elif  num280==201:
            p3802=self.conv1201(D180)
        elif  num280==202:
            p3802=self.conv1202(D180)
        elif  num280==203:
            p3802=self.conv1203(D180)
        elif  num280==204:
            p3802=self.conv1204(D180)
        elif  num280==205:
            p3802=self.conv1205(D180)
        elif  num280==206:
            p3802=self.conv1206(D180)
        elif  num280==207:
            p3802=self.conv1207(D180)
        elif  num280==208:
            p3802=self.conv1208(D180)
        elif  num280==209:
            p3802=self.conv1209(D180)
        elif  num280==210:
            p3802=self.conv1210(D180)
        elif  num280==211:
            p3802=self.conv1211(D180)
        elif  num280==212:
            p3802=self.conv1212(D180)
        elif  num280==213:
            p3802=self.conv1213(D180)
        elif  num280==214:
            p3802=self.conv1214(D180)
        elif  num280==215:
            p3802=self.conv1215(D180)
        elif  num280==216:
            p3802=self.conv1216(D180)
        elif  num280==217:
            p3802=self.conv1217(D180)
        elif  num280==218:
            p3802=self.conv1218(D180)
        elif  num280==219:
            p3802=self.conv1219(D180)
        elif  num280==220:
            p3802=self.conv1220(D180)
        elif  num280==221:
            p3802=self.conv1221(D180)
        elif  num280==222:
            p3802=self.conv1222(D180)
        elif  num280==223:
            p3802=self.conv1223(D180)
        elif  num280==224:
            p3802=self.conv1224(D180)
        elif  num280==225:
            p3802=self.conv1225(D180)
        elif  num280==226:
            p3802=self.conv1226(D180)
        elif  num280==227:
            p3802=self.conv1227(D180)
        elif  num280==228:
            p3802=self.conv1228(D180)
        elif  num280==229:
            p3802=self.conv1229(D180)
        elif  num280==230:
            p3802=self.conv1230(D180)
        elif  num280==231:
            p3802=self.conv1231(D180)
        elif  num280==232:
            p3802=self.conv1232(D180)
        elif  num280==233:
            p3802=self.conv1233(D180)
        elif  num280==234:
            p3802=self.conv1234(D180)
        elif  num280==235:
            p3802=self.conv1235(D180)
        elif  num280==236:
            p3802=self.conv1236(D180)
        elif  num280==237:
            p3802=self.conv1237(D180)
        elif  num280==238:
            p3802=self.conv1238(D180)
        elif  num280==239:
            p3802=self.conv1239(D180) 
        elif  num280==240:
            p3802=self.conv1240(D180)
        elif  num280==241:
            p3802=self.conv1241(D180)
        elif  num280==242:
            p3802=self.conv1242(D180)
        elif  num280==243:
            p3802=self.conv1243(D180)
        elif  num280==244:
            p3802=self.conv1244(D180)
        elif  num280==245:
            p3802=self.conv1245(D180)
        elif  num280==246:
            p3802=self.conv1246(D180)
        elif  num280==247:
            p3802=self.conv1247(D180)
        elif  num280==248:
            p3802=self.conv1248(D180)
        elif  num280==249:
            p3802=self.conv1249(D180)
        elif  num280==250:
            p3802=self.conv1250(D180)
        elif  num280==251:
            p3802=self.conv1251(D180)
        elif  num280==252:
            p3802=self.conv1252(D180)
        elif  num280==253:
            p3802=self.conv1253(D180)
        elif  num280==254:
            p3802=self.conv1254(D180)
        elif  num280==255:
            p3802=self.conv1255(D180)
        elif  num280==256:
            p3802=self.conv1256(D180)
        elif  num280==257:
            p3802=self.conv1257(D180)
        elif  num280==258:
            p3802=self.conv1258(D180)
        elif  num280==259:
            p3802=self.conv1259(D180)
        elif  num280==260:
            p3802=self.conv1260(D180)
        elif  num280==261:
            p3802=self.conv1261(D180)
        elif  num280==262:
            p3802=self.conv1262(D180)
        elif  num280==263:
            p3802=self.conv1263(D180)
        elif  num280==264:
            p3802=self.conv1264(D180)
        elif  num280==265:
            p3802=self.conv1265(D180)
        elif  num280==266:
            p3802=self.conv1266(D180)
        elif  num280==267:
            p3802=self.conv1267(D180)
        elif  num280==268:
            p3802=self.conv1268(D180)
        elif  num280==269:
            p3802=self.conv1269(D180) 
        elif  num280==270:
            p3802=self.conv1270(D180)
        elif  num280==271:
            p3802=self.conv1271(D180)
        elif  num280==272:
            p3802=self.conv1272(D180)
        elif  num280==273:
            p3802=self.conv1273(D180)
        elif  num280==274:
            p3802=self.conv1274(D180)
        elif  num280==275:
            p3802=self.conv1275(D180)
        elif  num280==276:
            p3802=self.conv1276(D180)
        elif  num280==277:
            p3802=self.conv1277(D180)
        elif  num280==278:
            p3802=self.conv1278(D180)
        elif  num280==279:
            p3802=self.conv1279(D180)
        elif  num280==280:
            p3802=self.conv1280(D180)
        elif  num280==281:
            p3802=self.conv1281(D180)
        elif  num280==282:
            p3802=self.conv1282(D180)
        elif  num280==283:
            p3802=self.conv1283(D180)
        elif  num280==284:
            p3802=self.conv1284(D180)
        elif  num280==285:
            p3802=self.conv1285(D180)
        elif  num280==286:
            p3802=self.conv1286(D180)
        elif  num280==287:
            p3802=self.conv1287(D180)
        elif  num280==288:
            p3802=self.conv1288(D180)
        elif  num280==289:
            p3802=self.conv1289(D180) 
        elif  num280==290:
            p3802=self.conv1290(D180)
        elif  num280==291:
            p3802=self.conv1291(D180)
        elif  num280==292:
            p3802=self.conv1292(D180)
        elif  num280==293:
            p3802=self.conv1293(D180)
        elif  num280==294:
            p3802=self.conv1294(D180)
        elif  num280==295:
            p3802=self.conv1295(D180)
        elif  num280==296:
            p3802=self.conv1296(D180)
        elif  num280==297:
            p3802=self.conv1297(D180)
        elif  num280==298:
            p3802=self.conv1298(D180)
        elif  num280==299:
            p3802=self.conv1299(D180)
        elif  num280==300:
            p3802=self.conv1300(D180)
        elif  num280==301:
            p3802=self.conv1301(D180)
        elif  num280==302:
            p3802=self.conv1302(D180)
        elif  num280==303:
            p3802=self.conv1303(D180)
        elif  num280==304:
            p3802=self.conv1304(D180)
        elif  num280==305:
            p3802=self.conv1305(D180)
        elif  num280==306:
            p3802=self.conv1306(D180)
        elif  num280==307:
            p3802=self.conv1307(D180)
        elif  num280==308:
            p3802=self.conv1308(D180)
        elif  num280==309:
            p3802=self.conv1309(D180) 
        elif  num280==310:
            p3802=self.conv1310(D180)
        elif  num280==311:
            p3802=self.conv1311(D180)
        elif  num280==312:
            p3802=self.conv1312(D180)
        elif  num280==313:
            p3802=self.conv1313(D180)
        elif  num280==314:
            p3802=self.conv1314(D180)
        elif  num280==315:
            p3802=self.conv1315(D180)
        elif  num280==316:
            p3802=self.conv1316(D180)
        elif  num280==317:
            p3802=self.conv1317(D180)
        elif  num280==318:
            p3802=self.conv1318(D180)
        elif  num280==319:
            p3802=self.conv1319(D180)
        elif  num280==320:
            p3802=self.conv1320(D180)
        elif  num280==321:
            p3802=self.conv1321(D180)
        elif  num280==322:
            p3802=self.conv1322(D180)
        elif  num280==323:
            p3802=self.conv1323(D180)
        elif  num280==324:
            p3802=self.conv1324(D180)
        elif  num280==325:
            p3802=self.conv1325(D180)
        elif  num280==326:
            p3802=self.conv1326(D180)
        elif  num280==327:
            p3802=self.conv1327(D180)
        elif  num280==328:
            p3802=self.conv1328(D180)
        elif  num280==329:
            p3802=self.conv1329(D180)
        elif  num280==330:
            p3802=self.conv1330(D180)
        elif  num280==331:
            p3802=self.conv1331(D180)
        elif  num280==332:
            p3802=self.conv1332(D180)
        elif  num280==333:
            p3802=self.conv1333(D180)
        elif  num280==334:
            p3802=self.conv1334(D180)
        elif  num280==335:
            p3802=self.conv1335(D180)
        elif  num280==336:
            p3802=self.conv1336(D180)
        elif  num280==337:
            p3802=self.conv1337(D180)
        elif  num280==338:
            p3802=self.conv1338(D180)
        elif  num280==339:
            p3802=self.conv1339(D180)
        elif  num280==340:
            p3802=self.conv1340(D180)
        elif  num280==341:
            p3802=self.conv1341(D180)
        elif  num280==342:
            p3802=self.conv1342(D180)
        elif  num280==343:
            p3802=self.conv1343(D180)
        elif  num280==344:
            p3802=self.conv1344(D180)
        elif  num280==345:
            p3802=self.conv1345(D180)
        elif  num280==346:
            p3802=self.conv1346(D180)
        elif  num280==347:
            p3802=self.conv1347(D180)
        elif  num280==348:
            p3802=self.conv1348(D180)
        elif  num280==349:
            p3802=self.conv1349(D180)
        elif  num280==350:
            p3802=self.conv1350(D180)
        elif  num280==351:
            p3802=self.conv1351(D180)
        elif  num280==352:
            p3802=self.conv1352(D180)
        elif  num280==353:
            p3802=self.conv1335(D180)
        elif  num280==354:
            p3802=self.conv1354(D180)
        elif  num280==355:
            p3802=self.conv1355(D180)
        elif  num280==356:
            p3802=self.conv1356(D180)
        elif  num280==357:
            p3802=self.conv1357(D180)
        elif  num280==358:
            p3802=self.conv1358(D180)
        elif  num280==359:
            p3802=self.conv1359(D180) 
        elif  num280==360:
            p3802=self.conv1360(D180)
        elif  num280==361:
            p3802=self.conv1361(D180)
        elif  num280==362:
            p3802=self.conv1362(D180)
        elif  num280==363:
            p3802=self.conv1363(D180)
        elif  num280==364:
            p3802=self.conv1364(D180)
        elif  num280==365:
            p3802=self.conv1365(D180)
        elif  num280==366:
            p3802=self.conv1366(D180)
        elif  num280==367:
            p3802=self.conv1367(D180)
        elif  num280==368:
            p3802=self.conv1368(D180)
        elif  num280==369:
            p3802=self.conv1369(D180) 
        elif  num280==370:
            p3802=self.conv1370(D180)
        elif  num280==371:
            p3802=self.conv1371(D180)
        elif  num280==372:
            p3802=self.conv1372(D180)
        elif  num280==373:
            p3802=self.conv1373(D180)
        elif  num280==374:
            p3802=self.conv1374(D180)
        elif  num280==375:
            p3802=self.conv1375(D180)
        elif  num280==376:
            p3802=self.conv1376(D180)
        elif  num280==377:
            p3802=self.conv1377(D180)
        elif  num280==378:
            p3802=self.conv1378(D180)
        elif  num280==379:
            p3802=self.conv1379(D180) 
        elif  num280==380:
            p3802=self.conv1380(D180)
        elif  num280==381:
            p3802=self.conv1381(D180)
        elif  num280==382:
            p3802=self.conv1382(D180)
        elif  num280==383:
            p3802=self.conv1383(D180)
        elif  num280==384:
            p3802=self.conv1384(D180)
        elif  num280==385:
            p3802=self.conv1385(D180)
        elif  num280==386:
            p3802=self.conv1386(D180)
        elif  num280==387:
            p3802=self.conv1387(D180)
        elif  num280==388:
            p3802=self.conv1388(D180)
        elif  num280==389:
            p3802=self.conv1389(D180) 
        elif  num280==390:
            p3802=self.conv1390(D180)
        elif  num280==391:
            p3802=self.conv1391(D180)
        elif  num280==392:
            p3802=self.conv1392(D180)
        elif  num280==393:
            p3802=self.conv1393(D180)
        elif  num280==394:
            p3802=self.conv1394(D180)
        elif  num280==395:
            p3802=self.conv1395(D180)
        elif  num280==396:
            p3802=self.conv1396(D180)
        elif  num280==397:
            p3802=self.conv1397(D180)
        elif  num280==398:
            p3802=self.conv1398(D180)
        elif  num280==399:
            p3802=self.conv1399(D180)
        elif  num280==400:
            p3802=self.conv1400(D180)
        elif  num280==401:
            p3802=self.conv1401(D180)
        elif  num280==402:
            p3802=self.conv1402(D180)
        elif  num280==403:
            p3802=self.conv1403(D180)
        elif  num280==404:
            p3802=self.conv1404(D180)
        elif  num280==405:
            p3802=self.conv1405(D180)
        elif  num280==406:
            p3802=self.conv1406(D180)
        elif  num280==407:
            p3802=self.conv1407(D180)
        elif  num280==408:
            p3802=self.conv1408(D180)
        elif  num280==409:
            p3802=self.conv1409(D180)
        elif  num280==410:
            p3802=self.conv1410(D180)
        elif  num280==411:
            p3802=self.conv1411(D180)
        elif  num280==412:
            p3802=self.conv1412(D180)
        elif  num280==413:
            p3802=self.conv1413(D180)
        elif  num280==414:
            p3802=self.conv1414(D180)
        elif  num280==415:
            p3802=self.conv145(D180)
        elif  num280==416:
            p3802=self.conv1416(D180)
        elif  num280==417:
            p3802=self.conv1417(D180)
        elif  num280==418:
            p3802=self.conv1418(D180)
        elif  num280==419:
            p3802=self.conv1419(D180) 
        elif  num280==420:
            p3802=self.conv1420(D180)
        elif  num280==421:
            p3802=self.conv1421(D180)
        elif  num280==422:
            p3802=self.conv1422(D180)
        elif  num280==423:
            p3802=self.conv1423(D180)
        elif  num280==424:
            p3802=self.conv1424(D180)
        elif  num280==425:
            p3802=self.conv1425(D180)
        elif  num280==426:
            p3802=self.conv1426(D180)
        elif  num280==427:
            p3802=self.conv1427(D180)
        elif  num280==428:
            p3802=self.conv1428(D180)
        elif  num280==429:
            p3802=self.conv1429(D180) 
        elif  num280==430:
            p3802=self.conv1430(D180)
        elif  num280==431:
            p3802=self.conv1431(D180)
        elif  num280==432:
            p3802=self.conv1432(D180)
        elif  num280==433:
            p3802=self.conv1433(D180)
        elif  num280==434:
            p3802=self.conv1434(D180)
        elif  num280==435:
            p3802=self.conv1435(D180)
        elif  num280==436:
            p3802=self.conv1436(D180)
        elif  num280==437:
            p3802=self.conv1437(D180)
        elif  num280==438:
            p3802=self.conv1438(D180)
        elif  num280==439:
            p3802=self.conv1439(D180)
        elif  num280==440:
            p3802=self.conv1440(D180)
        elif  num280==441:
            p3802=self.conv1441(D180)
        elif  num280==442:
            p3802=self.conv1442(D180)
        elif  num280==443:
            p3802=self.conv1443(D180)
        elif  num280==444:
            p3802=self.conv1444(D180)
        elif  num280==445:
            p3802=self.conv1445(D180)
        elif  num280==446:
            p3802=self.conv1446(D180)
        elif  num280==447:
            p3802=self.conv1447(D180)
        elif  num280==448:
            p3802=self.conv1448(D180)
        elif  num280==449:
            p3802=self.conv1449(D180)
        elif  num280==450:
            p3802=self.conv1450(D180)
        elif  num280==451:
            p3802=self.conv1451(D180)
        elif  num280==452:
            p3802=self.conv1452(D180)
        elif  num280==453:
            p3802=self.conv1453(D180)
        elif  num280==454:
            p3802=self.conv1454(D180)
        elif  num280==455:
            p3802=self.conv1455(D180)
        elif  num280==456:
            p3802=self.conv1456(D180)
        elif  num280==457:
            p3802=self.conv1457(D180)
        elif  num280==458:
            p3802=self.conv1458(D180)
        elif  num280==459:
            p3802=self.conv1459(D180)
        elif  num280==460:
            p3802=self.conv1460(D180)
        elif  num280==461:
            p3802=self.conv1461(D180)
        elif  num280==462:
            p3802=self.conv1462(D180)
        elif  num280==463:
            p3802=self.conv1463(D180)
        elif  num280==464:
            p3802=self.conv1464(D180)
        elif  num280==465:
            p3802=self.conv1465(D180)
        elif  num280==466:
            p3802=self.conv1466(D180)
        elif  num280==467:
            p3802=self.conv1467(D180)
        elif  num280==468:
            p3802=self.conv1468(D180)
        elif  num280==469:
            p3802=self.conv1469(D180) 
        elif  num280==470:
            p3802=self.conv1470(D180)
        elif  num280==471:
            p3802=self.conv1471(D180)
        elif  num280==472:
            p3802=self.conv1472(D180)
        elif  num280==473:
            p3802=self.conv1473(D180)
        elif  num280==474:
            p3802=self.conv1474(D180)
        elif  num280==475:
            p3802=self.conv1475(D180)
        elif  num280==476:
            p3802=self.conv1476(D180)
        elif  num280==477:
            p3802=self.conv1477(D180)
        elif  num280==478:
            p3802=self.conv1478(D180)
        elif  num280==479:
            p3802=self.conv1479(D180)
        elif  num280==480:
            p3802=self.conv1480(D180)
        elif  num280==481:
            p3802=self.conv1481(D180)
        elif  num280==482:
            p3802=self.conv1482(D180)
        elif  num280==483:
            p3802=self.conv1483(D180)
        elif  num280==484:
            p3802=self.conv1484(D180)
        elif  num280==485:
            p3802=self.conv1485(D180)
        elif  num280==486:
            p3802=self.conv1486(D180)
        elif  num280==487:
            p3802=self.conv1487(D180)
        elif  num280==488:
            p3802=self.conv1488(D180)
        elif  num280==489:
            p3802=self.conv1489(D180)
        elif  num280==490:
            p3802=self.conv1490(D180)
        elif  num280==491:
            p3802=self.conv1491(D180)
        elif  num280==492:
            p3802=self.conv1492(D180)
        elif  num280==493:
            p3802=self.conv1493(D180)
        elif  num280==494:
            p3802=self.conv1494(D180)
        elif  num280==495:
            p3802=self.conv1495(D180)
        elif  num280==496:
            p3802=self.conv1496(D180)
        elif  num280==497:
            p3802=self.conv1497(D180)
        elif  num280==498:
            p3802=self.conv1498(D180)
        elif  num280==499:
            p3802=self.conv1499(D180)
        elif  num280==500:
            p3802=self.conv1500(D180)
        elif  num280==501:
            p3802=self.conv1501(D180)
        elif  num280==502:
            p3802=self.conv1502(D180)
        elif  num280==503:
            p3802=self.conv1503(D180)
        elif  num280==504:
            p3802=self.conv1504(D180)
        elif  num280==505:
            p3802=self.conv1505(D180)
        elif  num280==506:
            p3802=self.conv1506(D180)
        elif  num280==507:
            p3802=self.conv1507(D180)
        elif  num280==508:
            p3802=self.conv1508(D180)
        elif  num280==509:
            p3802=self.conv1509(D180)
        elif  num280==510:
            p3802=self.conv1510(D180)
        elif  num280==511:
            p3802=self.conv1511(D180)
        elif  num280==512:
            p3802=self.conv1512(D180)
        elif  num280==513:
            p3802=self.conv1513(D180)
        elif  num280==514:
            p3802=self.conv1514(D180)
        elif  num280==515:
            p3802=self.conv1515(D180)
        elif  num280==516:
            p3802=self.conv1516(D180)
        elif  num280==517:
            p3802=self.conv1517(D180)
        elif  num280==518:
            p3802=self.conv1518(D180)
        elif  num280==519:
            p3802=self.conv1519(D180)
        elif  num280==520:
            p3802=self.conv1520(D180)
        elif  num280==521:
            p3802=self.conv1521(D180)
        elif  num280==522:
            p3802=self.conv1522(D180)
        elif  num280==523:
            p3802=self.conv1523(D180)
        elif  num280==524:
            p3802=self.conv1524(D180)
        elif  num280==525:
            p3802=self.conv1525(D180)
        elif  num280==526:
            p3802=self.conv1526(D180)
        elif  num280==527:
            p3802=self.conv1527(D180)
        elif  num280==528:
            p3802=self.conv1528(D180)
        elif  num280==529:
            p3802=self.conv1529(D180)
        elif  num280==530:
            p3802=self.conv1530(D180)
        elif  num280==531:
            p3802=self.conv1531(D180)
        elif  num280==532:
            p3802=self.conv1532(D180)
        elif  num280==533:
            p3802=self.conv1533(D180)
        elif  num280==534:
            p3802=self.conv1534(D180)
        elif  num280==535:
            p3802=self.conv1535(D180)
        elif  num280==536:
            p3802=self.conv1536(D180)
        elif  num280==537:
            p3802=self.conv1537(D180)
        elif  num280==538:
            p3802=self.conv1538(D180)
        elif  num280==539:
            p3802=self.conv1539(D180)
        elif  num280==540:
            p3802=self.conv1540(D180)
        elif  num280==541:
            p3802=self.conv1541(D180)
        elif  num280==542:
            p3802=self.conv1542(D180)
        elif  num280==543:
            p3802=self.conv1543(D180)
        elif  num280==544:
            p3802=self.conv1544(D180)
        elif  num280==545:
            p3802=self.conv1545(D180)
        elif  num280==546:
            p3802=self.conv1546(D180)
        elif  num280==547:
            p3802=self.conv1547(D180)
        elif  num280==548:
            p3802=self.conv1548(D180)
        elif  num280==549:
            p3802=self.conv1549(D180) 
        elif  num280==550:
            p3802=self.conv1550(D180)
        elif  num280==551:
            p3802=self.conv1551(D180)
        elif  num280==552:
            p3802=self.conv1552(D180)
        elif  num280==553:
            p3802=self.conv1553(D180)
        elif  num280==554:
            p3802=self.conv1554(D180)
        elif  num280==555:
            p3802=self.conv1555(D180)
        elif  num280==556:
            p3802=self.conv1556(D180)
        elif  num280==557:
            p3802=self.conv1557(D180)
        elif  num280==558:
            p3802=self.conv1558(D180)
        elif  num280==559:
            p3802=self.conv1559(D180)
        elif  num280==560:
            p3802=self.conv1560(D180)
        elif  num280==561:
            p3802=self.conv1561(D180)
        elif  num280==562:
            p3802=self.conv1562(D180)
        elif  num280==563:
            p3802=self.conv1563(D180)
        elif  num280==564:
            p3802=self.conv1564(D180)
        elif  num280==565:
            p3802=self.conv1565(D180)
        elif  num280==566:
            p3802=self.conv1566(D180)
        elif  num280==567:
            p3802=self.conv1567(D180)
        elif  num280==568:
            p3802=self.conv1568(D180)
        elif  num280==569:
            p3802=self.conv1569(D180) 
        elif  num280==570:
            p3802=self.conv1570(D180)
        elif  num280==571:
            p3802=self.conv1571(D180)
        elif  num280==572:
            p3802=self.conv1572(D180)
        elif  num280==573:
            p3802=self.conv1573(D180)
        elif  num280==574:
            p3802=self.conv1574(D180)
        elif  num280==575:
            p3802=self.conv1575(D180)
        elif  num280==576:
            p3802=self.conv1576(D180)
        elif  num280==577:
            p3802=self.conv1577(D180)
        elif  num280==578:
            p3802=self.conv1578(D180)
        elif  num280==579:
            p3802=self.conv1579(D180) 
        elif  num280==580:
            p3802=self.conv1580(D180)
        elif  num280==581:
            p3802=self.conv1581(D180)
        elif  num280==582:
            p3802=self.conv1582(D180)
        elif  num280==583:
            p3802=self.conv1583(D180)
        elif  num280==584:
            p3802=self.conv1584(D180)
        elif  num280==585:
            p3802=self.conv1585(D180)
        elif  num280==586:
            p3802=self.conv1586(D180)
        elif  num280==587:
            p3802=self.conv1587(D180)
        elif  num280==588:
            p3802=self.conv1588(D180)
        elif  num280==589:
            p3802=self.conv1589(D180)
        elif  num280==590:
            p3802=self.conv1590(D180)
        elif  num280==591:
            p3802=self.conv1591(D180)
        elif  num280==592:
            p3802=self.conv1592(D180)
        elif  num280==593:
            p3802=self.conv1593(D180)
        elif  num280==594:
            p3802=self.conv1594(D180)
        elif  num280==595:
            p3802=self.conv1595(D180)
        elif  num280==596:
            p3802=self.conv1596(D180)
        elif  num280==597:
            p3802=self.conv1597(D180)
        elif  num280==598:
            p3802=self.conv1598(D180)
        elif  num280==599:
            p3802=self.conv1599(D180)
        elif  num280==600:
            p3802=self.conv1600(D180)
        elif  num280==601:
            p3802=self.conv1601(D180)
        elif  num280==602:
            p3802=self.conv1602(D180)
        elif  num280==603:
            p3802=self.conv1603(D180)
        elif  num280==604:
            p3802=self.conv1604(D180)
        elif  num280==605:
            p3802=self.conv1605(D180)
        elif  num280==606:
            p3802=self.conv1606(D180)
        elif  num280==607:
            p3802=self.conv1607(D180)
        elif  num280==608:
            p3802=self.conv1608(D180)
        elif  num280==609:
            p3802=self.conv1609(D180)                                                                                                                         
        elif  num280==610:
            p3802=self.conv1610(D180)
        elif  num280==611:
            p3802=self.conv1611(D180)
        elif  num280==612:
            p3802=self.conv1612(D180)
        elif  num280==613:
            p3802=self.conv1613(D180)
        elif  num280==614:
            p3802=self.conv1614(D180)
        elif  num280==615:
            p3802=self.conv1615(D180)
        elif  num280==616:
            p3802=self.conv1616(D180)
        elif  num280==617:
            p3802=self.conv1617(D180)
        elif  num280==618:
            p3802=self.conv1618(D180)
        elif  num280==619:
            p3802=self.conv1619(D180)                                                                                                                          
        elif  num280==620:
            p3802=self.conv1620(D180)
        elif  num280==621:
            p3802=self.conv1621(D180)
        elif  num280==622:
            p3802=self.conv1622(D180)
        elif  num280==623:
            p3802=self.conv1623(D180)
        elif  num280==624:
            p3802=self.conv1624(D180)
        elif  num280==625:
            p3802=self.conv1625(D180)
        elif  num280==626:
            p3802=self.conv1626(D180)
        elif  num280==627:
            p3802=self.conv1627(D180)
        elif  num280==628:
            p3802=self.conv1628(D180)
        elif  num280==629:
            p3802=self.conv1629(D180)  
        elif  num280==630:
            p3802=self.conv1630(D180)
        elif  num280==631:
            p3802=self.conv1631(D180)
        elif  num280==632:
            p3802=self.conv1632(D180)
        elif  num280==633:
            p3802=self.conv1633(D180)
        elif  num280==634:
            p3802=self.conv1634(D180)
        elif  num280==635:
            p3802=self.conv1635(D180)
        elif  num280==636:
            p3802=self.conv1636(D180)
        elif  num280==637:
            p3802=self.conv1637(D180)
        elif  num280==638:
            p3802=self.conv1638(D180)
        elif  num280==639:
            p3802=self.conv1639(D180)                                                                                                                          
        elif  num280==640:
            p3802=self.conv1640(D180)
        elif  num280==641:
            p3802=self.conv1641(D180)
        elif  num280==642:
            p3802=self.conv1642(D180)
        elif  num280==643:
            p3802=self.conv1643(D180)
        elif  num280==644:
            p3802=self.conv1644(D180)
        elif  num280==645:
            p3802=self.conv1645(D180)
        elif  num280==646:
            p3802=self.conv1646(D180)
        elif  num280==647:
            p3802=self.conv1647(D180)
        elif  num280==648:
            p3802=self.conv1648(D180)
        elif  num280==649:
            p3802=self.conv1649(D180)                                                                                                                          
        elif  num280==650:
            p3802=self.conv1650(D180)
        elif  num280==651:
            p3802=self.conv1651(D180)
        elif  num280==652:
            p3802=self.conv1652(D180)
        elif  num280==653:
            p3802=self.conv1653(D180)
        elif  num280==654:
            p3802=self.conv1654(D180)
        elif  num280==655:
            p3802=self.conv1655(D180)
        elif  num280==656:
            p3802=self.conv1656(D180)
        elif  num280==657:
            p3802=self.conv1657(D180)
        elif  num280==658:
            p3802=self.conv1658(D180)
        elif  num280==659:
            p3802=self.conv1659(D180)
        elif  num280==660:
            p3802=self.conv1660(D180)
        elif  num280==661:
            p3802=self.conv1661(D180)
        elif  num280==662:
            p3802=self.conv1662(D180)
        elif  num280==663:
            p3802=self.conv1663(D180)
        elif  num280==664:
            p3802=self.conv1664(D180)
        elif  num280==665:
            p3802=self.conv1665(D180)
        elif  num280==666:
            p3802=self.conv1666(D180)
        elif  num280==667:
            p3802=self.conv1667(D180)
        elif  num280==668:
            p3802=self.conv1668(D180)
        elif  num280==669:
            p3802=self.conv1669(D180) 
        elif  num280==670:
            p3802=self.conv1670(D180)
        elif  num280==671:
            p3802=self.conv1671(D180)
        elif  num280==672:
            p3802=self.conv1672(D180)
        elif  num280==673:
            p3802=self.conv1673(D180)
        elif  num280==674:
            p3802=self.conv1674(D180)
        elif  num280==675:
            p3802=self.conv1675(D180)
        elif  num280==676:
            p3802=self.conv1676(D180)
        elif  num280==677:
            p3802=self.conv1677(D180)
        elif  num280==678:
            p3802=self.conv1678(D180)
        elif  num280==679:
            p3802=self.conv1679(D180)
        elif  num280==680:
            p3802=self.conv1680(D180)
        elif  num280==681:
            p3802=self.conv1681(D180)
        elif  num280==682:
            p3802=self.conv1682(D180)
        elif  num280==683:
            p3802=self.conv1683(D180)
        elif  num280==684:
            p3802=self.conv1684(D180)
        elif  num280==685:
            p3802=self.conv1685(D180)
        elif  num280==686:
            p3802=self.conv1686(D180)
        elif  num280==687:
            p3802=self.conv1687(D180)
        elif  num280==688:
            p3802=self.conv1688(D180)
        elif  num280==689:
            p3802=self.conv1689(D180)
        elif  num280==690:
            p3802=self.conv1690(D180)
        elif  num280==691:
            p3802=self.conv1691(D180)
        elif  num280==692:
            p3802=self.conv1692(D180)
        elif  num280==693:
            p3802=self.conv1693(D180)
        elif  num280==694:
            p3802=self.conv1694(D180)
        elif  num280==695:
            p3802=self.conv1695(D180)
        elif  num280==696:
            p3802=self.conv1696(D180)
        elif  num280==697:
            p3802=self.conv1697(D180)
        elif  num280==698:
            p3802=self.conv1698(D180)
        elif  num280==699:
            p3802=self.conv1699(D180)
        elif  num280==700:
            p3802=self.conv1700(D180)
        elif  num280==701:
            p3802=self.conv1701(D180)
        elif  num280==702:
            p3802=self.conv1702(D180)
        elif  num280==703:
            p3802=self.conv1703(D180)
        elif  num280==704:
            p3802=self.conv1704(D180)
        elif  num280==705:
            p3802=self.conv1705(D180)
        elif  num280==706:
            p3802=self.conv1706(D180)
        elif  num280==707:
            p3802=self.conv1707(D180)
        elif  num280==708:
            p3802=self.conv1708(D180)
        elif  num280==709:
            p3802=self.conv1709(D180)
        elif  num280==710:
            p3802=self.conv1710(D180)
        elif  num280==711:
            p3802=self.conv1711(D180)
        elif  num280==712:
            p3802=self.conv1712(D180)
        elif  num280==713:
            p3802=self.conv1713(D180)
        elif  num280==714:
            p3802=self.conv1714(D180)
        elif  num280==715:
            p3802=self.conv1715(D180)
        elif  num280==716:
            p3802=self.conv1716(D180)
        elif  num280==717:
            p3802=self.conv1717(D180)
        elif  num280==718:
            p3802=self.conv1718(D180)
        elif  num280==719:
            p3802=self.conv1719(D180)
        elif  num280==720:
            p3802=self.conv1720(D180)
        elif  num280==721:
            p3802=self.conv1721(D180)
        elif  num280==722:
            p3802=self.conv1722(D180)
        elif  num280==723:
            p3802=self.conv1723(D180)
        elif  num280==724:
            p3802=self.conv1724(D180)
        elif  num280==725:
            p3802=self.conv1725(D180)
        elif  num280==726:
            p3802=self.conv1726(D180)
        elif  num280==727:
            p3802=self.conv1727(D180)
        elif  num280==728:
            p3802=self.conv1728(D180)
        elif  num280==729:
            p3802=self.conv1729(D180)
        elif  num280==730:
            p3802=self.conv1730(D180)
        elif  num280==731:
            p3802=self.conv1731(D180)
        elif  num280==732:
            p3802=self.conv1732(D180)
        elif  num280==733:
            p3802=self.conv1733(D180)
        elif  num280==734:
            p3802=self.conv1734(D180)
        elif  num280==735:
            p3802=self.conv1735(D180)
        elif  num280==736:
            p3802=self.conv1736(D180)
        elif  num280==737:
            p3802=self.conv1737(D180)
        elif  num280==738:
            p3802=self.conv1738(D180)
        elif  num280==739:
            p3802=self.conv1739(D180)                                                                                                                          
        elif  num280==740:
            p3802=self.conv1740(D180)
        elif  num280==741:
            p3802=self.conv1741(D180)
        elif  num280==742:
            p3802=self.conv1742(D180)
        elif  num280==743:
            p3802=self.conv1743(D180)
        elif  num280==744:
            p3802=self.conv1744(D180)
        elif  num280==745:
            p3802=self.conv1745(D180)
        elif  num280==746:
            p3802=self.conv1746(D180)
        elif  num280==747:
            p3802=self.conv1747(D180)
        elif  num280==748:
            p3802=self.conv1748(D180)
        elif  num280==749:
            p3802=self.conv1749(D180)
        elif  num280==750:
            p3802=self.conv1750(D180)
        elif  num280==751:
            p3802=self.conv1751(D180)
        elif  num280==752:
            p3802=self.conv1752(D180)
        elif  num280==753:
            p3802=self.conv1753(D180)
        elif  num280==754:
            p3802=self.conv1754(D180)
        elif  num280==755:
            p3802=self.conv1755(D180)
        elif  num280==756:
            p3802=self.conv1756(D180)
        elif  num280==757:
            p3802=self.conv1757(D180)
        elif  num280==758:
            p3802=self.conv1758(D180)
        elif  num280==759:
            p3802=self.conv1759(D180)
        elif  num280==760:
            p3802=self.conv1760(D180)
        elif  num280==761:
            p3802=self.conv1761(D180)
        elif  num280==762:
            p3802=self.conv1762(D180)
        elif  num280==763:
            p3802=self.conv1763(D180)
        elif  num280==764:
            p3802=self.conv1764(D180)
        elif  num280==765:
            p3802=self.conv1765(D180)
        elif  num280==766:
            p3802=self.conv1766(D180)
        elif  num280==767:
            p3802=self.conv1767(D180)
        elif  num280==768:
            p3802=self.conv1768(D180)
        elif  num280==769:
            p3802=self.conv1769(D180) 
        elif  num280==770:
            p3802=self.conv1770(D180)
        elif  num280==771:
            p3802=self.conv1771(D180)
        elif  num280==772:
            p3802=self.conv1772(D180)
        elif  num280==773:
            p3802=self.conv1773(D180)
        elif  num280==774:
            p3802=self.conv1774(D180)
        elif  num280==775:
            p3802=self.conv1775(D180)
        elif  num280==776:
            p3802=self.conv1776(D180)
        elif  num280==777:
            p3802=self.conv1777(D180)
        elif  num280==778:
            p3802=self.conv1778(D180)
        elif  num280==779:
            p3802=self.conv1779(D180) 
        elif  num280==780:
            p3802=self.conv1780(D180)
        elif  num280==781:
            p3802=self.conv1781(D180)
        elif  num280==782:
            p3802=self.conv1782(D180)
        elif  num280==783:
            p3802=self.conv1783(D180)
        elif  num280==784:
            p3802=self.conv1784(D180)
        elif  num280==785:
            p3802=self.conv1785(D180)
        elif  num280==786:
            p3802=self.conv1786(D180)
        elif  num280==787:
            p3802=self.conv1787(D180)
        elif  num280==788:
            p3802=self.conv1788(D180)
        elif  num280==789:
            p3802=self.conv1789(D180) 
        elif  num280==790:
            p3802=self.conv1790(D180)
        elif  num280==791:
            p3802=self.conv1791(D180)
        elif  num280==792:
            p3802=self.conv1792(D180)
        elif  num280==793:
            p3802=self.conv1793(D180)
        elif  num280==794:
            p3802=self.conv1794(D180)
        elif  num280==795:
            p3802=self.conv1795(D180)
        elif  num280==796:
            p3802=self.conv1796(D180)
        elif  num280==797:
            p3802=self.conv1797(D180)
        elif  num280==798:
            p3802=self.conv1798(D180)
        elif  num280==799:
            p3802=self.conv1799(D180) 
        elif  num280==800:
            p3802=self.conv1800(D180)
        elif  num280==801:
            p3802=self.conv1801(D180)
        elif  num280==802:
            p3802=self.conv1802(D180)
        elif  num280==803:
            p3802=self.conv1803(D180)
        elif  num280==804:
            p3802=self.conv1804(D180)
        elif  num280==805:
            p3802=self.conv1805(D180)
        elif  num280==806:
            p3802=self.conv1806(D180)
        elif  num280==807:
            p3802=self.conv1807(D180)
        elif  num280==808:
            p3802=self.conv1808(D180)
        elif  num280==809:
            p3802=self.conv1809(D180)
        elif  num280==810:
            p3802=self.conv1810(D180)
        elif  num280==811:
            p3802=self.conv1811(D180)
        elif  num280==812:
            p3802=self.conv1812(D180)
        elif  num280==813:
            p3802=self.conv1813(D180)
        elif  num280==814:
            p3802=self.conv1814(D180)
        elif  num280==815:
            p3802=self.conv1815(D180)
        elif  num280==816:
            p3802=self.conv1816(D180)
        elif  num280==817:
            p3802=self.conv1817(D180)
        elif  num280==818:
            p3802=self.conv1818(D180)
        elif  num280==819:
            p3802=self.conv1819(D180)
        elif  num280==820:
            p3802=self.conv1820(D180)
        elif  num280==821:
            p3802=self.conv1821(D180)
        elif  num280==822:
            p3802=self.conv1822(D180)
        elif  num280==823:
            p3802=self.conv1823(D180)
        elif  num280==824:
            p3802=self.conv1824(D180)
        elif  num280==825:
            p3802=self.conv1825(D180)
        elif  num280==826:
            p3802=self.conv1826(D180)
        elif  num280==827:
            p3802=self.conv1827(D180)
        elif  num280==828:
            p3802=self.conv1828(D180)
        elif  num280==829:
            p3802=self.conv1829(D180)                                                                                                                          
        elif  num280==830:
            p3802=self.conv1830(D180)
        elif  num280==831:
            p3802=self.conv1831(D180)
        elif  num280==832:
            p3802=self.conv1832(D180)
        elif  num280==833:
            p3802=self.conv1833(D180)
        elif  num280==834:
            p3802=self.conv1834(D180)
        elif  num280==835:
            p3802=self.conv1835(D180)
        elif  num280==836:
            p3802=self.conv1836(D180)
        elif  num280==837:
            p3802=self.conv1837(D180)
        elif  num280==838:
            p3802=self.conv1838(D180)
        elif  num280==839:
            p3802=self.conv1839(D180)
        elif  num280==840:
            p3802=self.conv1840(D180)
        elif  num280==841:
            p3802=self.conv1841(D180)
        elif  num280==842:
            p3802=self.conv1842(D180)
        elif  num280==843:
            p3802=self.conv1843(D180)
        elif  num280==844:
            p3802=self.conv1844(D180)
        elif  num280==845:
            p3802=self.conv1845(D180)
        elif  num280==846:
            p3802=self.conv1846(D180)
        elif  num280==847:
            p3802=self.conv1847(D180)
        elif  num280==848:
            p3802=self.conv1848(D180)
        elif  num280==849:
            p3802=self.conv1849(D180)
        elif  num280==850:
            p3802=self.conv1850(D180)
        elif  num280==851:
            p3802=self.conv1851(D180)
        elif  num280==852:
            p3802=self.conv1852(D180)
        elif  num280==853:
            p3802=self.conv1853(D180)
        elif  num280==854:
            p3802=self.conv1854(D180)
        elif  num280==855:
            p3802=self.conv1855(D180)
        elif  num280==856:
            p3802=self.conv1856(D180)
        elif  num280==857:
            p3802=self.conv1857(D180)
        elif  num280==858:
            p3802=self.conv1858(D180)
        elif  num280==859:
            p3802=self.conv1859(D180)
        elif  num280==860:
            p3802=self.conv1860(D180)
        elif  num280==861:
            p3802=self.conv1861(D180)
        elif  num280==862:
            p3802=self.conv1862(D180)
        elif  num280==863:
            p3802=self.conv1863(D180)
        elif  num280==864:
            p3802=self.conv1864(D180)
        elif  num280==865:
            p3802=self.conv1865(D180)
        elif  num280==866:
            p3802=self.conv1866(D180)
        elif  num280==867:
            p3802=self.conv1867(D180)
        elif  num280==868:
            p3802=self.conv1868(D180)
        elif  num280==869:
            p3802=self.conv1869(D180) 
        elif  num280==870:
            p3802=self.conv1870(D180)
        elif  num280==871:
            p3802=self.conv1871(D180)
        elif  num280==872:
            p3802=self.conv1872(D180)
        elif  num280==873:
            p3802=self.conv1873(D180)
        elif  num280==874:
            p3802=self.conv1874(D180)
        elif  num280==875:
            p3802=self.conv1875(D180)
        elif  num280==876:
            p3802=self.conv1876(D180)
        elif  num280==877:
            p3802=self.conv1877(D180)
        elif  num280==878:
            p3802=self.conv1878(D180)
        elif  num280==879:
            p3802=self.conv1879(D180)
        elif  num280==880:
            p3802=self.conv1880(D180)
        elif  num280==881:
            p3802=self.conv1881(D180)
        elif  num280==882:
            p3802=self.conv1882(D180)
        elif  num280==883:
            p3802=self.conv1883(D180)
        elif  num280==884:
            p3802=self.conv1884(D180)
        elif  num280==885:
            p3802=self.conv1885(D180)
        elif  num280==886:
            p3802=self.conv1886(D180)
        elif  num280==887:
            p3802=self.conv1887(D180)
        elif  num280==888:
            p3802=self.conv1888(D180)
        elif  num280==889:
            p3802=self.conv1889(D180)  
        elif  num280==890:
            p3802=self.conv1890(D180)
        elif  num280==891:
            p3802=self.conv1891(D180)
        elif  num280==892:
            p3802=self.conv1892(D180)
        elif  num280==893:
            p3802=self.conv1893(D180)
        elif  num280==894:
            p3802=self.conv1894(D180)
        elif  num280==895:
            p3802=self.conv1895(D180)
        elif  num280==896:
            p3802=self.conv1896(D180)
        elif  num280==897:
            p3802=self.conv1897(D180)
        elif  num280==898:
            p3802=self.conv1898(D180)
        elif  num280==899:
            p3802=self.conv1899(D180)
        elif  num280==900:
            p3802=self.conv1900(D180)
        elif  num280==901:
            p3802=self.conv1901(D180)
        elif  num280==902:
            p3802=self.conv1902(D180)
        elif  num280==903:
            p3802=self.conv1903(D180)
        elif  num280==904:
            p3802=self.conv1904(D180)
        elif  num280==905:
            p3802=self.conv1905(D180)
        elif  num280==906:
            p3802=self.conv1906(D180)
        elif  num280==907:
            p3802=self.conv1907(D180)
        elif  num280==908:
            p3802=self.conv1908(D180)
        elif  num280==909:
            p3802=self.conv1909(D180)
        elif  num280==910:
            p3802=self.conv1910(D180)
        elif  num280==911:
            p3802=self.conv1911(D180)
        elif  num280==912:
            p3802=self.conv1912(D180)
        elif  num280==913:
            p3802=self.conv1913(D180)
        elif  num280==914:
            p3802=self.conv1914(D180)
        elif  num280==915:
            p3802=self.conv1915(D180)
        elif  num280==916:
            p3802=self.conv1916(D180)
        elif  num280==917:
            p3802=self.conv1917(D180)
        elif  num280==918:
            p3802=self.conv1918(D180)
        elif  num280==919:
            p3802=self.conv1919(D180)
        elif  num280==920:
            p3802=self.conv1920(D180)
        elif  num280==921:
            p3802=self.conv1921(D180)
        elif  num280==922:
            p3802=self.conv1922(D180)
        elif  num280==923:
            p3802=self.conv1923(D180)
        elif  num280==924:
            p3802=self.conv1924(D180)
        elif  num280==925:
            p3802=self.conv1925(D180)
        elif  num280==926:
            p3802=self.conv1926(D180)
        elif  num280==927:
            p3802=self.conv1927(D180)
        elif  num280==928:
            p3802=self.conv1928(D180)
        elif  num280==929:
            p3802=self.conv1929(D180)
        elif  num280==930:
            p3802=self.conv1930(D180)
        elif  num280==931:
            p3802=self.conv1931(D180)
        elif  num280==932:
            p3802=self.conv1932(D180)
        elif  num280==933:
            p3802=self.conv1933(D180)
        elif  num280==934:
            p3802=self.conv1934(D180)
        elif  num280==935:
            p3802=self.conv1935(D180)
        elif  num280==936:
            p3802=self.conv1936(D180)
        elif  num280==937:
            p3802=self.conv1937(D180)
        elif  num280==938:
            p3802=self.conv1938(D180)
        elif  num280==939:
            p3802=self.conv1939(D180) 
        elif  num280==940:
            p3802=self.conv1940(D180)
        elif  num280==941:
            p3802=self.conv1941(D180)
        elif  num280==942:
            p3802=self.conv1942(D180)
        elif  num280==943:
            p3802=self.conv1943(D180)
        elif  num280==944:
            p3802=self.conv1944(D180)
        elif  num280==945:
            p3802=self.conv1945(D180)
        elif  num280==946:
            p3802=self.conv1946(D180)
        elif  num280==947:
            p3802=self.conv1947(D180)
        elif  num280==948:
            p3802=self.conv1948(D180)
        elif  num280==949:
            p3802=self.conv1949(D180)                                                                                                                          
        elif  num280==950:
            p3802=self.conv1950(D180)
        elif  num280==951:
            p3802=self.conv1951(D180)
        elif  num280==952:
            p3802=self.conv1952(D180)
        elif  num280==953:
            p3802=self.conv1953(D180)
        elif  num280==954:
            p3802=self.conv1954(D180)
        elif  num280==955:
            p3802=self.conv1955(D180)
        elif  num280==956:
            p3802=self.conv1956(D180)
        elif  num280==957:
            p3802=self.conv1957(D180)
        elif  num280==958:
            p3802=self.conv1958(D180)
        elif  num280==959:
            p3802=self.conv1959(D180)
        elif  num280==960:
            p3802=self.conv1960(D180)
        elif  num280==961:
            p3802=self.conv1961(D180)
        elif  num280==962:
            p3802=self.conv1962(D180)
        elif  num280==963:
            p3802=self.conv1963(D180)
        elif  num280==964:
            p3802=self.conv1964(D180)
        elif  num280==965:
            p3802=self.conv1965(D180)
        elif  num280==966:
            p3802=self.conv1966(D180)
        elif  num280==967:
            p3802=self.conv1967(D180)
        elif  num280==968:
            p3802=self.conv1968(D180)
        elif  num280==969:
            p3802=self.conv1969(D180) 
        elif  num280==970:
            p3802=self.conv1970(D180)
        elif  num280==971:
            p3802=self.conv1971(D180)
        elif  num280==972:
            p3802=self.conv1972(D180)
        elif  num280==973:
            p3802=self.conv1973(D180)
        elif  num280==974:
            p3802=self.conv1974(D180)
        elif  num280==975:
            p3802=self.conv1975(D180)
        elif  num280==976:
            p3802=self.conv1976(D180)
        elif  num280==977:
            p3802=self.conv1977(D180)
        elif  num280==978:
            p3802=self.conv1978(D180)
        elif  num280==979:
            p3802=self.conv1979(D180) 
        elif  num280==980:
            p3802=self.conv1980(D180)
        elif  num280==981:
            p3802=self.conv1981(D180)
        elif  num280==982:
            p3802=self.conv1982(D180)
        elif  num280==983:
            p3802=self.conv1983(D180)
        elif  num280==984:
            p3802=self.conv1984(D180)
        elif  num280==985:
            p3802=self.conv1985(D180)
        elif  num280==986:
            p3802=self.conv1986(D180)
        elif  num280==987:
            p3802=self.conv1987(D180)
        elif  num280==988:
            p3802=self.conv1988(D180)
        elif  num280==989:
            p3802=self.conv1989(D180)
        elif  num280==990:
            p3802=self.conv1990(D180)
        elif  num280==991:
            p3802=self.conv1991(D180)
        elif  num280==992:
            p3802=self.conv1992(D180)
        elif  num280==993:
            p3802=self.conv1993(D180)
        elif  num280==994:
            p3802=self.conv1994(D180)
        elif  num280==995:
            p3802=self.conv1995(D180)
        elif  num280==996:
            p3802=self.conv1996(D180)
        elif  num280==997:
            p3802=self.conv1997(D180)
        elif  num280==998:
            p3802=self.conv1998(D180)
        elif  num280==999:
            p3802=self.conv1999(D180) 
        elif  num280==1000:
            p3802=self.conv11000(D180)
        elif  num280==1001:
            p3802=self.conv11001(D180)
        elif  num280==1002:
            p3802=self.conv11002(D180)
        elif  num280==1003:
            p3802=self.conv11003(D180)
        elif  num280==1004:
            p3802=self.conv11004(D180)
        elif  num280==1005:
            p3802=self.conv11005(D180)
        elif  num280==1006:
            p3802=self.conv11006(D180)
        elif  num280==1007:
            p3802=self.conv11007(D180)
        elif  num280==1008:
            p3802=self.conv11008(D180)
        elif  num280==1009:
            p3802=self.conv11009(D180) 
        elif  num280==1010:
            p3802=self.conv11010(D180)
        elif  num280==1011:
            p3802=self.conv11011(D180)
        elif  num280==1012:
            p3802=self.conv11012(D180)
        elif  num280==1013:
            p3802=self.conv11013(D180)
        elif  num280==1014:
            p3802=self.conv11014(D180)
        elif  num280==1015:
            p3802=self.conv11015(D180)
        elif  num280==1016:
            p3802=self.conv11016(D180)
        elif  num280==1017:
            p3802=self.conv11017(D180)
        elif  num280==1018:
            p3802=self.conv11018(D180)
        elif  num280==1019:
            p3802=self.conv11019(D180)
        elif  num280==1020:
            p3802=self.conv11020(D180)
        elif  num280==1021:
            p3802=self.conv11021(D180)
        elif  num280==1022:
            p3802=self.conv11022(D180)
        elif  num280==1023:
            p3802=self.conv11023(D180)
        elif  num280==1024:
            p3802=self.conv11024(D180) 
            
                
        if num08==1:
            p38=self.conv11(B18)
        elif num08==2:
            p38=self.conv12(B18)
        elif num08==3:
            p38=self.conv13(B18)
        elif num08==4:
            p38=self.conv14(B18)
        elif num08==5:
            p38=self.conv15(B18)
        elif num08==6:
            p38=self.conv16(B18)
        elif num08==7:
            p38=self.conv17(B18)
        elif num08==8:
            p38=self.conv18(B18)
        elif num08==9:
            p38=self.conv19(B18)
        elif num08==10:
            p38=self.conv110(B18)
        elif num08==11:
            p38=self.conv111(B18)
        elif num08==12:
            p38=self.conv112(B18)
        elif num08==13:
            p38=self.conv113(B18)
        elif num08==14:
            p38=self.conv114(B18)
        elif num08==15:
            p38=self.conv115(B18)
        elif num08==16:
            p38=self.conv116(B18)
        elif num08==17:
            p38=self.conv117(B18)
        elif num08==18:
            p38=self.conv118(B18)
        elif num08==19:
            p38=self.conv119(B18)
        elif num08==20:
            p38=self.conv120(B18)
        elif num08==21:
            p38=self.conv121(B18)
        elif num08==22:
            p38=self.conv122(B18)
        elif num08==23:
            p38=self.conv123(B18)
        elif num08==24:
            p38=self.conv124(B18)
        elif num08==25:
            p38=self.conv125(B18)
        elif num08==26:
            p38=self.conv126(B18)
        elif num08==27:
            p38=self.conv127(B18)
        elif num08==28:
            p38=self.conv128(B18)
        elif num08==29:
            p38=self.conv129(B18)
        elif num08==30:
            p38=self.conv130(B18)
        elif num08==31:
            p38=self.conv131(B18)
        elif num08==32:
            p38=self.conv132(B18)
        elif num08==33:
            p38=self.conv133(B18)
        elif num08==34:
            p38=self.conv134(B18)
        elif num08==35:
            p38=self.conv135(B18)
        elif num08==36:
            p38=self.conv136(B18)
        elif num08==37:
            p38=self.conv137(B18)
        elif num08==38:
            p38=self.conv138(B18)
        elif num08==39:
            p38=self.conv139(B18)
        elif num08==40:
            p38=self.conv140(B18)
        elif num08==41:
            p38=self.conv141(B18)
        elif num08==42:
            p38=self.conv142(B18)
        elif num08==43:
            p38=self.conv143(B18)
        elif num08==44:
            p38=self.conv144(B18)
        elif num08==45:
            p38=self.conv145(B18)
        elif num08==46:
            p38=self.conv146(B18)
        elif num08==47:
            p38=self.conv147(B18)
        elif num08==48:
            p38=self.conv148(B18)
        elif num08==49:
            p38=self.conv149(B18)
        elif num08==50:
            p38=self.conv150(B18)
        elif num08==51:
            p38=self.conv151(B18)
        elif num08==52:
            p38=self.conv152(B18)
        elif num08==53:
            p38=self.conv153(B18)
        elif num08==54:
            p38=self.conv154(B18)
        elif num08==55:
            p38=self.conv155(B18)
        elif num08==56:
            p38=self.conv156(B18)
        elif num08==57:
            p38=self.conv157(B18)
        elif num08==58:
            p38=self.conv158(B18)
        elif num08==59:
            p38=self.conv159(B18)
        elif num08==60:
            p38=self.conv160(B18)
        elif num08==61:
            p38=self.conv161(B18)
        elif num08==62:
            p38=self.conv162(B18)
        elif num08==63:
            p38=self.conv163(B18)
        elif num08==64:
            p38=self.conv164(B18)
        
        if  num18==1:
            p381=self.conv11(C18)
        elif  num18==2:
            p381=self.conv12(C18)
        elif  num18==3:
            p381=self.conv13(C18)
        elif  num18==4:
            p381=self.conv14(C18)
        elif  num18==5:
            p381=self.conv15(C18)
        elif  num18==6:
            p381=self.conv16(C18)
        elif  num18==7:
            p381=self.conv17(C18)
        elif  num18==8:
            p381=self.conv18(C18)
        elif  num18==9:
            p381=self.conv19(C18)
        elif  num18==10:
            p381=self.conv110(C18)
        elif  num18==11:
            p381=self.conv111(C18)
        elif  num18==12:
            p381=self.conv112(C18)
        elif  num18==13:
            p381=self.conv113(C18)
        elif  num18==14:
            p381=self.conv114(C18)
        elif  num18==15:
            p381=self.conv115(C18)
        elif  num18==16:
            p381=self.conv116(C18)
        elif  num18==17:
            p381=self.conv117(C18)
        elif  num18==18:
            p381=self.conv118(C18)
        elif  num18==19:
            p381=self.conv119(C18)
        elif  num18==20:
            p381=self.conv120(C18)
        elif  num18==21:
            p381=self.conv121(C18)
        elif  num18==22:
            p381=self.conv122(C18)
        elif  num18==23:
            p381=self.conv123(C18)
        elif  num18==24:
            p381=self.conv124(C18)
        elif  num18==25:
            p381=self.conv125(C18)
        elif  num18==26:
            p381=self.conv126(C18)
        elif  num18==27:
            p381=self.conv127(C18)
        elif  num18==28:
            p381=self.conv128(C18)
        elif  num18==29:
            p381=self.conv129(C18)
        elif  num18==30:
            p381=self.conv130(C18)
        elif  num18==31:
            p381=self.conv131(C18)
        elif  num18==32:
            p381=self.conv132(C18)
        elif  num18==33:
            p381=self.conv133(C18)
        elif  num18==34:
            p381=self.conv134(C18)
        elif  num18==35:
            p381=self.conv135(C18)
        elif  num18==36:
            p381=self.conv136(C18)
        elif  num18==37:
            p381=self.conv137(C18)
        elif  num18==38:
            p381=self.conv138(C18)
        elif  num18==39:
            p381=self.conv139(C18)
        elif  num18==40:
            p381=self.conv140(C18)
        elif  num18==41:
            p381=self.conv141(C18)
        elif  num18==42:
            p381=self.conv142(C18)
        elif  num18==43:
            p381=self.conv143(C18)
        elif  num18==44:
            p381=self.conv144(C18)
        elif  num18==45:
            p381=self.conv145(C18)
        elif  num18==46:
            p381=self.conv146(C18)
        elif  num18==47:
            p381=self.conv147(C18)
        elif  num18==48:
            p381=self.conv148(C18)
        elif  num18==49:
            p381=self.conv149(C18)
        elif  num18==50:
            p381=self.conv150(C18)
        elif  num18==51:
            p381=self.conv151(C18)
        elif  num18==52:
            p381=self.conv152(C18)
        elif  num18==53:
            p381=self.conv153(C18)
        elif  num18==54:
            p381=self.conv154(C18)
        elif  num18==55:
            p381=self.conv155(C18)
        elif  num18==56:
            p381=self.conv156(C18)
        elif  num18==57:
            p381=self.conv157(C18)
        elif  num18==58:
            p381=self.conv158(C18)
        elif  num18==59:
            p381=self.conv159(C18)
        elif  num18==60:
            p381=self.conv160(C18)
        elif  num18==61:
            p381=self.conv161(C18)
        elif  num18==62:
            p381=self.conv162(C18)
        elif  num18==63:
            p381=self.conv163(C18)
        elif  num18==64:
            p381=self.conv164(C18)
        elif  num18==65:
            p381=self.conv165(C18)
        elif  num18==66:
            p381=self.conv166(C18)
        elif  num18==67:
            p381=self.conv167(C18)
        elif  num18==68:
            p381=self.conv168(C18)
        elif  num18==69:
            p381=self.conv169(C18)
        elif  num18==70:
            p381=self.conv170(C18)
        elif  num18==71:
            p381=self.conv171(C18)
        elif  num18==72:
            p381=self.conv172(C18)
        elif  num18==73:
            p381=self.conv173(C18)
        elif  num18==74:
            p381=self.conv174(C18)
        elif  num18==75:
            p381=self.conv175(C18)
        elif  num18==76:
            p381=self.conv176(C18)
        elif  num18==77:
            p381=self.conv177(C18)
        elif  num18==78:
            p381=self.conv178(C18)
        elif  num18==79:
            p381=self.conv179(C18)
        elif  num18==80:
            p381=self.conv180(C18)
        elif  num18==81:
            p381=self.conv181(C18)
        elif  num18==82:
            p381=self.conv182(C18)
        elif  num18==83:
            p381=self.conv183(C18)
        elif  num18==84:
            p381=self.conv184(C18)
        elif  num18==85:
            p381=self.conv185(C18)
        elif  num18==86:
            p381=self.conv186(C18)
        elif  num18==87:
            p381=self.conv187(C18)
        elif  num18==88:
            p381=self.conv188(C18)
        elif  num18==89:
            p381=self.conv189(C18)    
        elif  num18==90:
            p381=self.conv190(C18)
        elif  num18==91:
            p381=self.conv191(C18)
        elif  num18==92:
            p381=self.conv192(C18)
        elif  num18==93:
            p381=self.conv193(C18)
        elif  num18==94:
            p381=self.conv194(C18)
        elif  num18==95:
            p381=self.conv195(C18)
        elif  num18==96:
            p381=self.conv196(C18)
        elif  num18==97:
            p381=self.conv197(C18)
        elif  num18==98:
            p381=self.conv198(C18)
        elif  num18==99:
            p381=self.conv199(C18) 
        elif  num18==100:
            p381=self.conv1100(C18)
        elif  num18==101:
            p381=self.conv1101(C18)
        elif  num18==102:
            p381=self.conv1102(C18)
        elif  num18==103:
            p381=self.conv1103(C18)
        elif  num18==104:
            p381=self.conv1104(C18)
        elif  num18==105:
            p381=self.conv1105(C18)
        elif  num18==106:
            p381=self.conv1106(C18)
        elif  num18==107:
            p381=self.conv1107(C18)
        elif  num18==108:
            p381=self.conv1108(C18)
        elif  num18==109:
            p381=self.conv1109(C18)
        elif  num18==110:
            p381=self.conv1110(C18)
        elif  num18==111:
            p381=self.conv1111(C18)
        elif  num18==112:
            p381=self.conv1112(C18)
        elif  num18==113:
            p381=self.conv1113(C18)
        elif  num18==114:
            p381=self.conv1114(C18)
        elif  num18==115:
            p381=self.conv1115(C18)
        elif  num18==116:
            p381=self.conv1116(C18)
        elif  num18==117:
            p381=self.conv1117(C18)
        elif  num18==118:
            p381=self.conv1118(C18)
        elif  num18==119:
            p381=self.conv1119(C18) 
        elif  num18==120:
            p381=self.conv1120(C18)
        elif  num18==121:
            p381=self.conv1121(C18)
        elif  num18==122:
            p381=self.conv1122(C18)
        elif  num18==123:
            p381=self.conv1123(C18)
        elif  num18==124:
            p381=self.conv1124(C18)
        elif  num18==125:
            p381=self.conv1125(C18)
        elif  num18==126:
            p381=self.conv1126(C18)
        elif  num18==127:
            p381=self.conv1127(C18)
        elif  num18==128:
            p381=self.conv1128(C18)
        elif  num18==129:
            p381=self.conv1129(C18) 
        elif  num18==130:
            p381=self.conv1130(C18)
        elif  num18==131:
            p381=self.conv1131(C18)
        elif  num18==132:
            p381=self.conv1132(C18)
        elif  num18==133:
            p381=self.conv1133(C18)
        elif  num18==134:
            p381=self.conv1134(C18)
        elif  num18==135:
            p381=self.conv1135(C18)
        elif  num18==136:
            p381=self.conv1136(C18)
        elif  num18==137:
            p381=self.conv1137(C18)
        elif  num18==138:
            p381=self.conv1138(C18)
        elif  num18==139:
            p381=self.conv1139(C18)
        elif  num18==140:
            p381=self.conv1140(C18)
        elif  num18==141:
            p381=self.conv1141(C18)
        elif  num18==142:
            p381=self.conv1142(C18)
        elif  num18==143:
            p381=self.conv1143(C18)
        elif  num18==144:
            p381=self.conv1144(C18)
        elif  num18==145:
            p381=self.conv1145(C18)
        elif  num18==146:
            p381=self.conv1146(C18)
        elif  num18==147:
            p381=self.conv1147(C18)
        elif  num18==148:
            p381=self.conv1148(C18)
        elif  num18==149:
            p381=self.conv1149(C18) 
        elif  num18==150:
            p381=self.conv1150(C18)
        elif  num18==151:
            p381=self.conv1151(C18)
        elif  num18==152:
            p381=self.conv1152(C18)
        elif  num18==153:
            p381=self.conv1153(C18)
        elif  num18==154:
            p381=self.conv1154(C18)
        elif  num18==155:
            p381=self.conv1155(C18)
        elif  num18==156:
            p381=self.conv1156(C18)
        elif  num18==157:
            p381=self.conv1157(C18)
        elif  num18==158:
            p381=self.conv1158(C18)
        elif  num18==159:
            p381=self.conv1159(C18) 
        elif  num18==160:
            p381=self.conv1160(C18)
        elif  num18==161:
            p381=self.conv1161(C18)
        elif  num18==162:
            p381=self.conv1162(C18)
        elif  num18==163:
            p381=self.conv1163(C18)
        elif  num18==164:
            p381=self.conv1164(C18)
        elif  num18==165:
            p381=self.conv1165(C18)
        elif  num18==166:
            p381=self.conv1166(C18)
        elif  num18==167:
            p381=self.conv1167(C18)
        elif  num18==168:
            p381=self.conv1168(C18)
        elif  num18==169:
            p381=self.conv1169(C18) 
        elif  num18==170:
            p381=self.conv1170(C18)
        elif  num18==171:
            p381=self.conv1171(C18)
        elif  num18==172:
            p381=self.conv1172(C18)
        elif  num18==173:
            p381=self.conv1173(C18)
        elif  num18==174:
            p381=self.conv1174(C18)
        elif  num18==175:
            p381=self.conv1175(C18)
        elif  num18==176:
            p381=self.conv1176(C18)
        elif  num18==177:
            p381=self.conv1177(C18)
        elif  num18==178:
            p381=self.conv1178(C18)
        elif  num18==179:
            p381=self.conv1179(C18)                                                                                              
        elif  num18==180:
            p381=self.conv1180(C18)
        elif  num18==181:
            p381=self.conv1181(C18)
        elif  num18==182:
            p381=self.conv1182(C18)
        elif  num18==183:
            p381=self.conv1183(C18)
        elif  num18==184:
            p381=self.conv1184(C18)
        elif  num18==185:
            p381=self.conv1185(C18)
        elif  num18==186:
            p381=self.conv1186(C18)
        elif  num18==187:
            p381=self.conv1187(C18)
        elif  num18==188:
            p381=self.conv1188(C18)
        elif  num18==189:
            p381=self.conv1189(C18) 
        elif  num18==190:
            p381=self.conv1190(C18)
        elif  num18==191:
            p381=self.conv1191(C18)
        elif  num18==192:
            p381=self.conv1192(C18)
        elif  num18==193:
            p381=self.conv1193(C18)
        elif  num18==194:
            p381=self.conv1194(C18)
        elif  num18==195:
            p381=self.conv1195(C18)
        elif  num18==196:
            p381=self.conv1196(C18)
        elif  num18==197:
            p381=self.conv1197(C18)
        elif  num18==198:
            p381=self.conv1198(C18)
        elif  num18==199:
            p381=self.conv1199(C18)
        elif  num18==200:
            p381=self.conv1200(C18)
        elif  num18==201:
            p381=self.conv1201(C18)
        elif  num18==202:
            p381=self.conv1202(C18)
        elif  num18==203:
            p381=self.conv1203(C18)
        elif  num18==204:
            p381=self.conv1204(C18)
        elif  num18==205:
            p381=self.conv1205(C18)
        elif  num18==206:
            p381=self.conv1206(C18)
        elif  num18==207:
            p381=self.conv1207(C18)
        elif  num18==208:
            p381=self.conv1208(C18)
        elif  num18==209:
            p381=self.conv1209(C18)
        elif  num18==210:
            p381=self.conv1210(C18)
        elif  num18==211:
            p381=self.conv1211(C18)
        elif  num18==212:
            p381=self.conv1212(C18)
        elif  num18==213:
            p381=self.conv1213(C18)
        elif  num18==214:
            p381=self.conv1214(C18)
        elif  num18==215:
            p381=self.conv1215(C18)
        elif  num18==216:
            p381=self.conv1216(C18)
        elif  num18==217:
            p381=self.conv1217(C18)
        elif  num18==218:
            p381=self.conv1218(C18)
        elif  num18==219:
            p381=self.conv1219(C18)
        elif  num18==220:
            p381=self.conv1220(C18)
        elif  num18==221:
            p381=self.conv1221(C18)
        elif  num18==222:
            p381=self.conv1222(C18)
        elif  num18==223:
            p381=self.conv1223(C18)
        elif  num18==224:
            p381=self.conv1224(C18)
        elif  num18==225:
            p381=self.conv1225(C18)
        elif  num18==226:
            p381=self.conv1226(C18)
        elif  num18==227:
            p381=self.conv1227(C18)
        elif  num18==228:
            p381=self.conv1228(C18)
        elif  num18==229:
            p381=self.conv1229(C18)
        elif  num18==230:
            p381=self.conv1230(C18)
        elif  num18==231:
            p381=self.conv1231(C18)
        elif  num18==232:
            p381=self.conv1232(C18)
        elif  num18==233:
            p381=self.conv1233(C18)
        elif  num18==234:
            p381=self.conv1234(C18)
        elif  num18==235:
            p381=self.conv1235(C18)
        elif  num18==236:
            p381=self.conv1236(C18)
        elif  num18==237:
            p381=self.conv1237(C18)
        elif  num18==238:
            p381=self.conv1238(C18)
        elif  num18==239:
            p381=self.conv1239(C18) 
        elif  num18==240:
            p381=self.conv1240(C18)
        elif  num18==241:
            p381=self.conv1241(C18)
        elif  num18==242:
            p381=self.conv1242(C18)
        elif  num18==243:
            p381=self.conv1243(C18)
        elif  num18==244:
            p381=self.conv1244(C18)
        elif  num18==245:
            p381=self.conv1245(C18)
        elif  num18==246:
            p381=self.conv1246(C18)
        elif  num18==247:
            p381=self.conv1247(C18)
        elif  num18==248:
            p381=self.conv1248(C18)
        elif  num18==249:
            p381=self.conv1249(C18)
        elif  num18==250:
            p381=self.conv1250(C18)
        elif  num18==251:
            p381=self.conv1251(C18)
        elif  num18==252:
            p381=self.conv1252(C18)
        elif  num18==253:
            p381=self.conv1253(C18)
        elif  num18==254:
            p381=self.conv1254(C18)
        elif  num18==255:
            p381=self.conv1255(C18)
        elif  num18==256:
            p381=self.conv1256(C18)
            
        if  num28==1:
            p382=self.conv11(D18)
        elif  num28==2:
            p382=self.conv12(D18)
        elif  num28==3:
            p382=self.conv13(D18)
        elif  num28==4:
            p382=self.conv14(D18)
        elif  num28==5:
            p382=self.conv15(D18)
        elif  num28==6:
            p382=self.conv16(D18)
        elif  num28==7:
            p382=self.conv17(D18)
        elif  num28==8:
            p382=self.conv18(D18)
        elif  num28==9:
            p382=self.conv19(D18)
        elif  num28==10:
            p382=self.conv110(D18)
        elif  num28==11:
            p382=self.conv111(D18)
        elif  num28==12:
            p382=self.conv112(D18)
        elif  num28==13:
            p382=self.conv113(D18)
        elif  num28==14:
            p382=self.conv114(D18)
        elif  num28==15:
            p382=self.conv115(D18)
        elif  num28==16:
            p382=self.conv116(D18)
        elif  num28==17:
            p382=self.conv117(D18)
        elif  num28==18:
            p382=self.conv118(D18)
        elif  num28==19:
            p382=self.conv119(D18)
        elif  num28==20:
            p382=self.conv120(D18)
        elif  num28==21:
            p382=self.conv121(D18)
        elif  num28==22:
            p382=self.conv122(D18)
        elif  num28==23:
            p382=self.conv123(D18)
        elif  num28==24:
            p382=self.conv124(D18)
        elif  num28==25:
            p382=self.conv125(D18)
        elif  num28==26:
            p382=self.conv126(D18)
        elif  num28==27:
            p382=self.conv127(D18)
        elif  num28==28:
            p382=self.conv128(D18)
        elif  num28==29:
            p382=self.conv129(D18)
        elif  num28==30:
            p382=self.conv130(D18)
        elif  num28==31:
            p382=self.conv131(D18)
        elif  num28==32:
            p382=self.conv132(D18)
        elif  num28==33:
            p382=self.conv133(D18)
        elif  num28==34:
            p382=self.conv134(D18)
        elif  num28==35:
            p382=self.conv135(D18)
        elif  num28==36:
            p382=self.conv136(D18)
        elif  num28==37:
            p382=self.conv137(D18)
        elif  num28==38:
            p382=self.conv138(D18)
        elif  num28==39:
            p382=self.conv139(D18)
        elif  num28==40:
            p382=self.conv140(D18)
        elif  num28==41:
            p382=self.conv141(D18)
        elif  num28==42:
            p382=self.conv142(D18)
        elif  num28==43:
            p382=self.conv143(D18)
        elif  num28==44:
            p382=self.conv144(D18)
        elif  num28==45:
            p382=self.conv145(D18)
        elif  num28==46:
            p382=self.conv146(D18)
        elif  num28==47:
            p382=self.conv147(D18)
        elif  num28==48:
            p382=self.conv148(D18)
        elif  num28==49:
            p382=self.conv149(D18)
        elif  num28==50:
            p382=self.conv150(D18)
        elif  num28==51:
            p382=self.conv151(D18)
        elif  num28==52:
            p382=self.conv152(D18)
        elif  num28==53:
            p382=self.conv153(D18)
        elif  num28==54:
            p382=self.conv154(D18)
        elif  num28==55:
            p382=self.conv155(D18)
        elif  num28==56:
            p382=self.conv156(D18)
        elif  num28==57:
            p382=self.conv157(D18)
        elif  num28==58:
            p382=self.conv158(D18)
        elif  num28==59:
            p382=self.conv159(D18)
        elif  num28==60:
            p382=self.conv160(D18)
        elif  num28==61:
            p382=self.conv161(D18)
        elif  num28==62:
            p382=self.conv162(D18)
        elif  num28==63:
            p382=self.conv163(D18)
        elif  num28==64:
            p382=self.conv164(D18)
        elif  num28==65:
            p382=self.conv165(D18)
        elif  num28==66:
            p382=self.conv166(D18)
        elif  num28==67:
            p382=self.conv167(D18)
        elif  num28==68:
            p382=self.conv168(D18)
        elif  num28==69:
            p382=self.conv169(D18)
        elif  num28==70:
            p382=self.conv170(D18)
        elif  num28==71:
            p382=self.conv171(D18)
        elif  num28==72:
            p382=self.conv172(D18)
        elif  num28==73:
            p382=self.conv173(D18)
        elif  num28==74:
            p382=self.conv174(D18)
        elif  num28==75:
            p382=self.conv175(D18)
        elif  num28==76:
            p382=self.conv176(D18)
        elif  num28==77:
            p382=self.conv177(D18)
        elif  num28==78:
            p382=self.conv178(D18)
        elif  num28==79:
            p382=self.conv179(D18)
        elif  num28==80:
            p382=self.conv180(D18)
        elif  num28==81:
            p382=self.conv181(D18)
        elif  num28==82:
            p382=self.conv182(D18)
        elif  num28==83:
            p382=self.conv183(D18)
        elif  num28==84:
            p382=self.conv184(D18)
        elif  num28==85:
            p382=self.conv185(D18)
        elif  num28==86:
            p382=self.conv186(D18)
        elif  num28==87:
            p382=self.conv187(D18)
        elif  num28==88:
            p382=self.conv188(D18)
        elif  num28==89:
            p382=self.conv189(D18)    
        elif  num28==90:
            p382=self.conv190(D18)
        elif  num28==91:
            p382=self.conv191(D18)
        elif  num28==92:
            p382=self.conv192(D18)
        elif  num28==93:
            p382=self.conv193(D18)
        elif  num28==94:
            p382=self.conv194(D18)
        elif  num28==95:
            p382=self.conv195(D18)
        elif  num28==96:
            p382=self.conv196(D18)
        elif  num28==97:
            p382=self.conv197(D18)
        elif  num28==98:
            p382=self.conv198(D18)
        elif  num28==99:
            p382=self.conv199(D18) 
        elif  num28==100:
            p382=self.conv1100(D18)
        elif  num28==101:
            p382=self.conv1101(D18)
        elif  num28==102:
            p382=self.conv1102(D18)
        elif  num28==103:
            p382=self.conv1103(D18)
        elif  num28==104:
            p382=self.conv1104(D18)
        elif  num28==105:
            p382=self.conv1105(D18)
        elif  num28==106:
            p382=self.conv1106(D18)
        elif  num28==107:
            p382=self.conv1107(D18)
        elif  num28==108:
            p382=self.conv1108(D18)
        elif  num28==109:
            p382=self.conv1109(D18)
        elif  num28==110:
            p382=self.conv1110(D18)
        elif  num28==111:
            p382=self.conv1111(D18)
        elif  num28==112:
            p382=self.conv1112(D18)
        elif  num28==113:
            p382=self.conv1113(D18)
        elif  num28==114:
            p382=self.conv1114(D18)
        elif  num28==115:
            p382=self.conv1115(D18)
        elif  num28==116:
            p382=self.conv1116(D18)
        elif  num28==117:
            p382=self.conv1117(D18)
        elif  num28==118:
            p382=self.conv1118(D18)
        elif  num28==119:
            p382=self.conv1119(D18) 
        elif  num28==120:
            p382=self.conv1120(D18)
        elif  num28==121:
            p382=self.conv1121(D18)
        elif  num28==122:
            p382=self.conv1122(D18)
        elif  num28==123:
            p382=self.conv1123(D18)
        elif  num28==124:
            p382=self.conv1124(D18)
        elif  num28==125:
            p382=self.conv1125(D18)
        elif  num28==126:
            p382=self.conv1126(D18)
        elif  num28==127:
            p382=self.conv1127(D18)
        elif  num28==128:
            p382=self.conv1128(D18)
        elif  num28==129:
            p382=self.conv1129(D18) 
        elif  num28==130:
            p382=self.conv1130(D18)
        elif  num28==131:
            p382=self.conv1131(D18)
        elif  num28==132:
            p382=self.conv1132(D18)
        elif  num28==133:
            p382=self.conv1133(D18)
        elif  num28==134:
            p382=self.conv1134(D18)
        elif  num28==135:
            p382=self.conv1135(D18)
        elif  num28==136:
            p382=self.conv1136(D18)
        elif  num28==137:
            p382=self.conv1137(D18)
        elif  num28==138:
            p382=self.conv1138(D18)
        elif  num28==139:
            p382=self.conv1139(D18)
        elif  num28==140:
            p382=self.conv1140(D18)
        elif  num28==141:
            p382=self.conv1141(D18)
        elif  num28==142:
            p382=self.conv1142(D18)
        elif  num28==143:
            p382=self.conv1143(D18)
        elif  num28==144:
            p382=self.conv1144(D18)
        elif  num28==145:
            p382=self.conv1145(D18)
        elif  num28==146:
            p382=self.conv1146(D18)
        elif  num28==147:
            p382=self.conv1147(D18)
        elif  num28==148:
            p382=self.conv1148(D18)
        elif  num28==149:
            p382=self.conv1149(D18) 
        elif  num28==150:
            p382=self.conv1150(D18)
        elif  num28==151:
            p382=self.conv1151(D18)
        elif  num28==152:
            p382=self.conv1152(D18)
        elif  num28==153:
            p382=self.conv1153(D18)
        elif  num28==154:
            p382=self.conv1154(D18)
        elif  num28==155:
            p382=self.conv1155(D18)
        elif  num28==156:
            p382=self.conv1156(D18)
        elif  num28==157:
            p382=self.conv1157(D18)
        elif  num28==158:
            p382=self.conv1158(D18)
        elif  num28==159:
            p382=self.conv1159(D18) 
        elif  num28==160:
            p382=self.conv1160(D18)
        elif  num28==161:
            p382=self.conv1161(D18)
        elif  num28==162:
            p382=self.conv1162(D18)
        elif  num28==163:
            p382=self.conv1163(D18)
        elif  num28==164:
            p382=self.conv1164(D18)
        elif  num28==165:
            p382=self.conv1165(D18)
        elif  num28==166:
            p382=self.conv1166(D18)
        elif  num28==167:
            p382=self.conv1167(D18)
        elif  num28==168:
            p382=self.conv1168(D18)
        elif  num28==169:
            p382=self.conv1169(D18) 
        elif  num28==170:
            p382=self.conv1170(D18)
        elif  num28==171:
            p382=self.conv1171(D18)
        elif  num28==172:
            p382=self.conv1172(D18)
        elif  num28==173:
            p382=self.conv1173(D18)
        elif  num28==174:
            p382=self.conv1174(D18)
        elif  num28==175:
            p382=self.conv1175(D18)
        elif  num28==176:
            p382=self.conv1176(D18)
        elif  num28==177:
            p382=self.conv1177(D18)
        elif  num28==178:
            p382=self.conv1178(D18)
        elif  num28==179:
            p382=self.conv1179(D18)                                                                                              
        elif  num28==180:
            p382=self.conv1180(D18)
        elif  num28==181:
            p382=self.conv1181(D18)
        elif  num28==182:
            p382=self.conv1182(D18)
        elif  num28==183:
            p382=self.conv1183(D18)
        elif  num28==184:
            p382=self.conv1184(D18)
        elif  num28==185:
            p382=self.conv1185(D18)
        elif  num28==186:
            p382=self.conv1186(D18)
        elif  num28==187:
            p382=self.conv1187(D18)
        elif  num28==188:
            p382=self.conv1188(D18)
        elif  num28==189:
            p382=self.conv1189(D18) 
        elif  num28==190:
            p382=self.conv1190(D18)
        elif  num28==191:
            p382=self.conv1191(D18)
        elif  num28==192:
            p382=self.conv1192(D18)
        elif  num28==193:
            p382=self.conv1193(D18)
        elif  num28==194:
            p382=self.conv1194(D18)
        elif  num28==195:
            p382=self.conv1195(D18)
        elif  num28==196:
            p382=self.conv1196(D18)
        elif  num28==197:
            p382=self.conv1197(D18)
        elif  num28==198:
            p382=self.conv1198(D18)
        elif  num28==199:
            p382=self.conv1199(D18)
        elif  num28==200:
            p382=self.conv1200(D18)
        elif  num28==201:
            p382=self.conv1201(D18)
        elif  num28==202:
            p382=self.conv1202(D18)
        elif  num28==203:
            p382=self.conv1203(D18)
        elif  num28==204:
            p382=self.conv1204(D18)
        elif  num28==205:
            p382=self.conv1205(D18)
        elif  num28==206:
            p382=self.conv1206(D18)
        elif  num28==207:
            p382=self.conv1207(D18)
        elif  num28==208:
            p382=self.conv1208(D18)
        elif  num28==209:
            p382=self.conv1209(D18)
        elif  num28==210:
            p382=self.conv1210(D18)
        elif  num28==211:
            p382=self.conv1211(D18)
        elif  num28==212:
            p382=self.conv1212(D18)
        elif  num28==213:
            p382=self.conv1213(D18)
        elif  num28==214:
            p382=self.conv1214(D18)
        elif  num28==215:
            p382=self.conv1215(D18)
        elif  num28==216:
            p382=self.conv1216(D18)
        elif  num28==217:
            p382=self.conv1217(D18)
        elif  num28==218:
            p382=self.conv1218(D18)
        elif  num28==219:
            p382=self.conv1219(D18)
        elif  num28==220:
            p382=self.conv1220(D18)
        elif  num28==221:
            p382=self.conv1221(D18)
        elif  num28==222:
            p382=self.conv1222(D18)
        elif  num28==223:
            p382=self.conv1223(D18)
        elif  num28==224:
            p382=self.conv1224(D18)
        elif  num28==225:
            p382=self.conv1225(D18)
        elif  num28==226:
            p382=self.conv1226(D18)
        elif  num28==227:
            p382=self.conv1227(D18)
        elif  num28==228:
            p382=self.conv1228(D18)
        elif  num28==229:
            p382=self.conv1229(D18)
        elif  num28==230:
            p382=self.conv1230(D18)
        elif  num28==231:
            p382=self.conv1231(D18)
        elif  num28==232:
            p382=self.conv1232(D18)
        elif  num28==233:
            p382=self.conv1233(D18)
        elif  num28==234:
            p382=self.conv1234(D18)
        elif  num28==235:
            p382=self.conv1235(D18)
        elif  num28==236:
            p382=self.conv1236(D18)
        elif  num28==237:
            p382=self.conv1237(D18)
        elif  num28==238:
            p382=self.conv1238(D18)
        elif  num28==239:
            p382=self.conv1239(D18) 
        elif  num28==240:
            p382=self.conv1240(D18)
        elif  num28==241:
            p382=self.conv1241(D18)
        elif  num28==242:
            p382=self.conv1242(D18)
        elif  num28==243:
            p382=self.conv1243(D18)
        elif  num28==244:
            p382=self.conv1244(D18)
        elif  num28==245:
            p382=self.conv1245(D18)
        elif  num28==246:
            p382=self.conv1246(D18)
        elif  num28==247:
            p382=self.conv1247(D18)
        elif  num28==248:
            p382=self.conv1248(D18)
        elif  num28==249:
            p382=self.conv1249(D18)
        elif  num28==250:
            p382=self.conv1250(D18)
        elif  num28==251:
            p382=self.conv1251(D18)
        elif  num28==252:
            p382=self.conv1252(D18)
        elif  num28==253:
            p382=self.conv1253(D18)
        elif  num28==254:
            p382=self.conv1254(D18)
        elif  num28==255:
            p382=self.conv1255(D18)
        elif  num28==256:
            p382=self.conv1256(D18)
        elif  num28==257:
            p382=self.conv1257(D18)
        elif  num28==258:
            p382=self.conv1258(D18)
        elif  num28==259:
            p382=self.conv1259(D18)
        elif  num28==260:
            p382=self.conv1260(D18)
        elif  num28==261:
            p382=self.conv1261(D18)
        elif  num28==262:
            p382=self.conv1262(D18)
        elif  num28==263:
            p382=self.conv1263(D18)
        elif  num28==264:
            p382=self.conv1264(D18)
        elif  num28==265:
            p382=self.conv1265(D18)
        elif  num28==266:
            p382=self.conv1266(D18)
        elif  num28==267:
            p382=self.conv1267(D18)
        elif  num28==268:
            p382=self.conv1268(D18)
        elif  num28==269:
            p382=self.conv1269(D18) 
        elif  num28==270:
            p382=self.conv1270(D18)
        elif  num28==271:
            p382=self.conv1271(D18)
        elif  num28==272:
            p382=self.conv1272(D18)
        elif  num28==273:
            p382=self.conv1273(D18)
        elif  num28==274:
            p382=self.conv1274(D18)
        elif  num28==275:
            p382=self.conv1275(D18)
        elif  num28==276:
            p382=self.conv1276(D18)
        elif  num28==277:
            p382=self.conv1277(D18)
        elif  num28==278:
            p382=self.conv1278(D18)
        elif  num28==279:
            p382=self.conv1279(D18)
        elif  num28==280:
            p382=self.conv1280(D18)
        elif  num28==281:
            p382=self.conv1281(D18)
        elif  num28==282:
            p382=self.conv1282(D18)
        elif  num28==283:
            p382=self.conv1283(D18)
        elif  num28==284:
            p382=self.conv1284(D18)
        elif  num28==285:
            p382=self.conv1285(D18)
        elif  num28==286:
            p382=self.conv1286(D18)
        elif  num28==287:
            p382=self.conv1287(D18)
        elif  num28==288:
            p382=self.conv1288(D18)
        elif  num28==289:
            p382=self.conv1289(D18) 
        elif  num28==290:
            p382=self.conv1290(D18)
        elif  num28==291:
            p382=self.conv1291(D18)
        elif  num28==292:
            p382=self.conv1292(D18)
        elif  num28==293:
            p382=self.conv1293(D18)
        elif  num28==294:
            p382=self.conv1294(D18)
        elif  num28==295:
            p382=self.conv1295(D18)
        elif  num28==296:
            p382=self.conv1296(D18)
        elif  num28==297:
            p382=self.conv1297(D18)
        elif  num28==298:
            p382=self.conv1298(D18)
        elif  num28==299:
            p382=self.conv1299(D18)
        elif  num28==300:
            p382=self.conv1300(D18)
        elif  num28==301:
            p382=self.conv1301(D18)
        elif  num28==302:
            p382=self.conv1302(D18)
        elif  num28==303:
            p382=self.conv1303(D18)
        elif  num28==304:
            p382=self.conv1304(D18)
        elif  num28==305:
            p382=self.conv1305(D18)
        elif  num28==306:
            p382=self.conv1306(D18)
        elif  num28==307:
            p382=self.conv1307(D18)
        elif  num28==308:
            p382=self.conv1308(D18)
        elif  num28==309:
            p382=self.conv1309(D18) 
        elif  num28==310:
            p382=self.conv1310(D18)
        elif  num28==311:
            p382=self.conv1311(D18)
        elif  num28==312:
            p382=self.conv1312(D18)
        elif  num28==313:
            p382=self.conv1313(D18)
        elif  num28==314:
            p382=self.conv1314(D18)
        elif  num28==315:
            p382=self.conv1315(D18)
        elif  num28==316:
            p382=self.conv1316(D18)
        elif  num28==317:
            p382=self.conv1317(D18)
        elif  num28==318:
            p382=self.conv1318(D18)
        elif  num28==319:
            p382=self.conv1319(D18)
        elif  num28==320:
            p382=self.conv1320(D18)
        elif  num28==321:
            p382=self.conv1321(D18)
        elif  num28==322:
            p382=self.conv1322(D18)
        elif  num28==323:
            p382=self.conv1323(D18)
        elif  num28==324:
            p382=self.conv1324(D18)
        elif  num28==325:
            p382=self.conv1325(D18)
        elif  num28==326:
            p382=self.conv1326(D18)
        elif  num28==327:
            p382=self.conv1327(D18)
        elif  num28==328:
            p382=self.conv1328(D18)
        elif  num28==329:
            p382=self.conv1329(D18)
        elif  num28==330:
            p382=self.conv1330(D18)
        elif  num28==331:
            p382=self.conv1331(D18)
        elif  num28==332:
            p382=self.conv1332(D18)
        elif  num28==333:
            p382=self.conv1333(D18)
        elif  num28==334:
            p382=self.conv1334(D18)
        elif  num28==335:
            p382=self.conv1335(D18)
        elif  num28==336:
            p382=self.conv1336(D18)
        elif  num28==337:
            p382=self.conv1337(D18)
        elif  num28==338:
            p382=self.conv1338(D18)
        elif  num28==339:
            p382=self.conv1339(D18)
        elif  num28==340:
            p382=self.conv1340(D18)
        elif  num28==341:
            p382=self.conv1341(D18)
        elif  num28==342:
            p382=self.conv1342(D18)
        elif  num28==343:
            p382=self.conv1343(D18)
        elif  num28==344:
            p382=self.conv1344(D18)
        elif  num28==345:
            p382=self.conv1345(D18)
        elif  num28==346:
            p382=self.conv1346(D18)
        elif  num28==347:
            p382=self.conv1347(D18)
        elif  num28==348:
            p382=self.conv1348(D18)
        elif  num28==349:
            p382=self.conv1349(D18)
        elif  num28==350:
            p382=self.conv1350(D18)
        elif  num28==351:
            p382=self.conv1351(D18)
        elif  num28==352:
            p382=self.conv1352(D18)
        elif  num28==353:
            p382=self.conv1335(D18)
        elif  num28==354:
            p382=self.conv1354(D18)
        elif  num28==355:
            p382=self.conv1355(D18)
        elif  num28==356:
            p382=self.conv1356(D18)
        elif  num28==357:
            p382=self.conv1357(D18)
        elif  num28==358:
            p382=self.conv1358(D18)
        elif  num28==359:
            p382=self.conv1359(D18) 
        elif  num28==360:
            p382=self.conv1360(D18)
        elif  num28==361:
            p382=self.conv1361(D18)
        elif  num28==362:
            p382=self.conv1362(D18)
        elif  num28==363:
            p382=self.conv1363(D18)
        elif  num28==364:
            p382=self.conv1364(D18)
        elif  num28==365:
            p382=self.conv1365(D18)
        elif  num28==366:
            p382=self.conv1366(D18)
        elif  num28==367:
            p382=self.conv1367(D18)
        elif  num28==368:
            p382=self.conv1368(D18)
        elif  num28==369:
            p382=self.conv1369(D18) 
        elif  num28==370:
            p382=self.conv1370(D18)
        elif  num28==371:
            p382=self.conv1371(D18)
        elif  num28==372:
            p382=self.conv1372(D18)
        elif  num28==373:
            p382=self.conv1373(D18)
        elif  num28==374:
            p382=self.conv1374(D18)
        elif  num28==375:
            p382=self.conv1375(D18)
        elif  num28==376:
            p382=self.conv1376(D18)
        elif  num28==377:
            p382=self.conv1377(D18)
        elif  num28==378:
            p382=self.conv1378(D18)
        elif  num28==379:
            p382=self.conv1379(D18) 
        elif  num28==380:
            p382=self.conv1380(D18)
        elif  num28==381:
            p382=self.conv1381(D18)
        elif  num28==382:
            p382=self.conv1382(D18)
        elif  num28==383:
            p382=self.conv1383(D18)
        elif  num28==384:
            p382=self.conv1384(D18)
        elif  num28==385:
            p382=self.conv1385(D18)
        elif  num28==386:
            p382=self.conv1386(D18)
        elif  num28==387:
            p382=self.conv1387(D18)
        elif  num28==388:
            p382=self.conv1388(D18)
        elif  num28==389:
            p382=self.conv1389(D18) 
        elif  num28==390:
            p382=self.conv1390(D18)
        elif  num28==391:
            p382=self.conv1391(D18)
        elif  num28==392:
            p382=self.conv1392(D18)
        elif  num28==393:
            p382=self.conv1393(D18)
        elif  num28==394:
            p382=self.conv1394(D18)
        elif  num28==395:
            p382=self.conv1395(D18)
        elif  num28==396:
            p382=self.conv1396(D18)
        elif  num28==397:
            p382=self.conv1397(D18)
        elif  num28==398:
            p382=self.conv1398(D18)
        elif  num28==399:
            p382=self.conv1399(D18)
        elif  num28==400:
            p382=self.conv1400(D18)
        elif  num28==401:
            p382=self.conv1401(D18)
        elif  num28==402:
            p382=self.conv1402(D18)
        elif  num28==403:
            p382=self.conv1403(D18)
        elif  num28==404:
            p382=self.conv1404(D18)
        elif  num28==405:
            p382=self.conv1405(D18)
        elif  num28==406:
            p382=self.conv1406(D18)
        elif  num28==407:
            p382=self.conv1407(D18)
        elif  num28==408:
            p382=self.conv1408(D18)
        elif  num28==409:
            p382=self.conv1409(D18)
        elif  num28==410:
            p382=self.conv1410(D18)
        elif  num28==411:
            p382=self.conv1411(D18)
        elif  num28==412:
            p382=self.conv1412(D18)
        elif  num28==413:
            p382=self.conv1413(D18)
        elif  num28==414:
            p382=self.conv1414(D18)
        elif  num28==415:
            p382=self.conv145(D18)
        elif  num28==416:
            p382=self.conv1416(D18)
        elif  num28==417:
            p382=self.conv1417(D18)
        elif  num28==418:
            p382=self.conv1418(D18)
        elif  num28==419:
            p382=self.conv1419(D18) 
        elif  num28==420:
            p382=self.conv1420(D18)
        elif  num28==421:
            p382=self.conv1421(D18)
        elif  num28==422:
            p382=self.conv1422(D18)
        elif  num28==423:
            p382=self.conv1423(D18)
        elif  num28==424:
            p382=self.conv1424(D18)
        elif  num28==425:
            p382=self.conv1425(D18)
        elif  num28==426:
            p382=self.conv1426(D18)
        elif  num28==427:
            p382=self.conv1427(D18)
        elif  num28==428:
            p382=self.conv1428(D18)
        elif  num28==429:
            p382=self.conv1429(D18) 
        elif  num28==430:
            p382=self.conv1430(D18)
        elif  num28==431:
            p382=self.conv1431(D18)
        elif  num28==432:
            p382=self.conv1432(D18)
        elif  num28==433:
            p382=self.conv1433(D18)
        elif  num28==434:
            p382=self.conv1434(D18)
        elif  num28==435:
            p382=self.conv1435(D18)
        elif  num28==436:
            p382=self.conv1436(D18)
        elif  num28==437:
            p382=self.conv1437(D18)
        elif  num28==438:
            p382=self.conv1438(D18)
        elif  num28==439:
            p382=self.conv1439(D18)
        elif  num28==440:
            p382=self.conv1440(D18)
        elif  num28==441:
            p382=self.conv1441(D18)
        elif  num28==442:
            p382=self.conv1442(D18)
        elif  num28==443:
            p382=self.conv1443(D18)
        elif  num28==444:
            p382=self.conv1444(D18)
        elif  num28==445:
            p382=self.conv1445(D18)
        elif  num28==446:
            p382=self.conv1446(D18)
        elif  num28==447:
            p382=self.conv1447(D18)
        elif  num28==448:
            p382=self.conv1448(D18)
        elif  num28==449:
            p382=self.conv1449(D18)
        elif  num28==450:
            p382=self.conv1450(D18)
        elif  num28==451:
            p382=self.conv1451(D18)
        elif  num28==452:
            p382=self.conv1452(D18)
        elif  num28==453:
            p382=self.conv1453(D18)
        elif  num28==454:
            p382=self.conv1454(D18)
        elif  num28==455:
            p382=self.conv1455(D18)
        elif  num28==456:
            p382=self.conv1456(D18)
        elif  num28==457:
            p382=self.conv1457(D18)
        elif  num28==458:
            p382=self.conv1458(D18)
        elif  num28==459:
            p382=self.conv1459(D18)
        elif  num28==460:
            p382=self.conv1460(D18)
        elif  num28==461:
            p382=self.conv1461(D18)
        elif  num28==462:
            p382=self.conv1462(D18)
        elif  num28==463:
            p382=self.conv1463(D18)
        elif  num28==464:
            p382=self.conv1464(D18)
        elif  num28==465:
            p382=self.conv1465(D18)
        elif  num28==466:
            p382=self.conv1466(D18)
        elif  num28==467:
            p382=self.conv1467(D18)
        elif  num28==468:
            p382=self.conv1468(D18)
        elif  num28==469:
            p382=self.conv1469(D18) 
        elif  num28==470:
            p382=self.conv1470(D18)
        elif  num28==471:
            p382=self.conv1471(D18)
        elif  num28==472:
            p382=self.conv1472(D18)
        elif  num28==473:
            p382=self.conv1473(D18)
        elif  num28==474:
            p382=self.conv1474(D18)
        elif  num28==475:
            p382=self.conv1475(D18)
        elif  num28==476:
            p382=self.conv1476(D18)
        elif  num28==477:
            p382=self.conv1477(D18)
        elif  num28==478:
            p382=self.conv1478(D18)
        elif  num28==479:
            p382=self.conv1479(D18)
        elif  num28==480:
            p382=self.conv1480(D18)
        elif  num28==481:
            p382=self.conv1481(D18)
        elif  num28==482:
            p382=self.conv1482(D18)
        elif  num28==483:
            p382=self.conv1483(D18)
        elif  num28==484:
            p382=self.conv1484(D18)
        elif  num28==485:
            p382=self.conv1485(D18)
        elif  num28==486:
            p382=self.conv1486(D18)
        elif  num28==487:
            p382=self.conv1487(D18)
        elif  num28==488:
            p382=self.conv1488(D18)
        elif  num28==489:
            p382=self.conv1489(D18)
        elif  num28==490:
            p382=self.conv1490(D18)
        elif  num28==491:
            p382=self.conv1491(D18)
        elif  num28==492:
            p382=self.conv1492(D18)
        elif  num28==493:
            p382=self.conv1493(D18)
        elif  num28==494:
            p382=self.conv1494(D18)
        elif  num28==495:
            p382=self.conv1495(D18)
        elif  num28==496:
            p382=self.conv1496(D18)
        elif  num28==497:
            p382=self.conv1497(D18)
        elif  num28==498:
            p382=self.conv1498(D18)
        elif  num28==499:
            p382=self.conv1499(D18)
        elif  num28==500:
            p382=self.conv1500(D18)
        elif  num28==501:
            p382=self.conv1501(D18)
        elif  num28==502:
            p382=self.conv1502(D18)
        elif  num28==503:
            p382=self.conv1503(D18)
        elif  num28==504:
            p382=self.conv1504(D18)
        elif  num28==505:
            p382=self.conv1505(D18)
        elif  num28==506:
            p382=self.conv1506(D18)
        elif  num28==507:
            p382=self.conv1507(D18)
        elif  num28==508:
            p382=self.conv1508(D18)
        elif  num28==509:
            p382=self.conv1509(D18)
        elif  num28==510:
            p382=self.conv1510(D18)
        elif  num28==511:
            p382=self.conv1511(D18)
        elif  num28==512:
            p382=self.conv1512(D18)
        elif  num28==513:
            p382=self.conv1513(D18)
        elif  num28==514:
            p382=self.conv1514(D18)
        elif  num28==515:
            p382=self.conv1515(D18)
        elif  num28==516:
            p382=self.conv1516(D18)
        elif  num28==517:
            p382=self.conv1517(D18)
        elif  num28==518:
            p382=self.conv1518(D18)
        elif  num28==519:
            p382=self.conv1519(D18)
        elif  num28==520:
            p382=self.conv1520(D18)
        elif  num28==521:
            p382=self.conv1521(D18)
        elif  num28==522:
            p382=self.conv1522(D18)
        elif  num28==523:
            p382=self.conv1523(D18)
        elif  num28==524:
            p382=self.conv1524(D18)
        elif  num28==525:
            p382=self.conv1525(D18)
        elif  num28==526:
            p382=self.conv1526(D18)
        elif  num28==527:
            p382=self.conv1527(D18)
        elif  num28==528:
            p382=self.conv1528(D18)
        elif  num28==529:
            p382=self.conv1529(D18)
        elif  num28==530:
            p382=self.conv1530(D18)
        elif  num28==531:
            p382=self.conv1531(D18)
        elif  num28==532:
            p382=self.conv1532(D18)
        elif  num28==533:
            p382=self.conv1533(D18)
        elif  num28==534:
            p382=self.conv1534(D18)
        elif  num28==535:
            p382=self.conv1535(D18)
        elif  num28==536:
            p382=self.conv1536(D18)
        elif  num28==537:
            p382=self.conv1537(D18)
        elif  num28==538:
            p382=self.conv1538(D18)
        elif  num28==539:
            p382=self.conv1539(D18)
        elif  num28==540:
            p382=self.conv1540(D18)
        elif  num28==541:
            p382=self.conv1541(D18)
        elif  num28==542:
            p382=self.conv1542(D18)
        elif  num28==543:
            p382=self.conv1543(D18)
        elif  num28==544:
            p382=self.conv1544(D18)
        elif  num28==545:
            p382=self.conv1545(D18)
        elif  num28==546:
            p382=self.conv1546(D18)
        elif  num28==547:
            p382=self.conv1547(D18)
        elif  num28==548:
            p382=self.conv1548(D18)
        elif  num28==549:
            p382=self.conv1549(D18) 
        elif  num28==550:
            p382=self.conv1550(D18)
        elif  num28==551:
            p382=self.conv1551(D18)
        elif  num28==552:
            p382=self.conv1552(D18)
        elif  num28==553:
            p382=self.conv1553(D18)
        elif  num28==554:
            p382=self.conv1554(D18)
        elif  num28==555:
            p382=self.conv1555(D18)
        elif  num28==556:
            p382=self.conv1556(D18)
        elif  num28==557:
            p382=self.conv1557(D18)
        elif  num28==558:
            p382=self.conv1558(D18)
        elif  num28==559:
            p382=self.conv1559(D18)
        elif  num28==560:
            p382=self.conv1560(D18)
        elif  num28==561:
            p382=self.conv1561(D18)
        elif  num28==562:
            p382=self.conv1562(D18)
        elif  num28==563:
            p382=self.conv1563(D18)
        elif  num28==564:
            p382=self.conv1564(D18)
        elif  num28==565:
            p382=self.conv1565(D18)
        elif  num28==566:
            p382=self.conv1566(D18)
        elif  num28==567:
            p382=self.conv1567(D18)
        elif  num28==568:
            p382=self.conv1568(D18)
        elif  num28==569:
            p382=self.conv1569(D18) 
        elif  num28==570:
            p382=self.conv1570(D18)
        elif  num28==571:
            p382=self.conv1571(D18)
        elif  num28==572:
            p382=self.conv1572(D18)
        elif  num28==573:
            p382=self.conv1573(D18)
        elif  num28==574:
            p382=self.conv1574(D18)
        elif  num28==575:
            p382=self.conv1575(D18)
        elif  num28==576:
            p382=self.conv1576(D18)
        elif  num28==577:
            p382=self.conv1577(D18)
        elif  num28==578:
            p382=self.conv1578(D18)
        elif  num28==579:
            p382=self.conv1579(D18) 
        elif  num28==580:
            p382=self.conv1580(D18)
        elif  num28==581:
            p382=self.conv1581(D18)
        elif  num28==582:
            p382=self.conv1582(D18)
        elif  num28==583:
            p382=self.conv1583(D18)
        elif  num28==584:
            p382=self.conv1584(D18)
        elif  num28==585:
            p382=self.conv1585(D18)
        elif  num28==586:
            p382=self.conv1586(D18)
        elif  num28==587:
            p382=self.conv1587(D18)
        elif  num28==588:
            p382=self.conv1588(D18)
        elif  num28==589:
            p382=self.conv1589(D18)
        elif  num28==590:
            p382=self.conv1590(D18)
        elif  num28==591:
            p382=self.conv1591(D18)
        elif  num28==592:
            p382=self.conv1592(D18)
        elif  num28==593:
            p382=self.conv1593(D18)
        elif  num28==594:
            p382=self.conv1594(D18)
        elif  num28==595:
            p382=self.conv1595(D18)
        elif  num28==596:
            p382=self.conv1596(D18)
        elif  num28==597:
            p382=self.conv1597(D18)
        elif  num28==598:
            p382=self.conv1598(D18)
        elif  num28==599:
            p382=self.conv1599(D18)
        elif  num28==600:
            p382=self.conv1600(D18)
        elif  num28==601:
            p382=self.conv1601(D18)
        elif  num28==602:
            p382=self.conv1602(D18)
        elif  num28==603:
            p382=self.conv1603(D18)
        elif  num28==604:
            p382=self.conv1604(D18)
        elif  num28==605:
            p382=self.conv1605(D18)
        elif  num28==606:
            p382=self.conv1606(D18)
        elif  num28==607:
            p382=self.conv1607(D18)
        elif  num28==608:
            p382=self.conv1608(D18)
        elif  num28==609:
            p382=self.conv1609(D18)                                                                                                                         
        elif  num28==610:
            p382=self.conv1610(D18)
        elif  num28==611:
            p382=self.conv1611(D18)
        elif  num28==612:
            p382=self.conv1612(D18)
        elif  num28==613:
            p382=self.conv1613(D18)
        elif  num28==614:
            p382=self.conv1614(D18)
        elif  num28==615:
            p382=self.conv1615(D18)
        elif  num28==616:
            p382=self.conv1616(D18)
        elif  num28==617:
            p382=self.conv1617(D18)
        elif  num28==618:
            p382=self.conv1618(D18)
        elif  num28==619:
            p382=self.conv1619(D18)                                                                                                                          
        elif  num28==620:
            p382=self.conv1620(D18)
        elif  num28==621:
            p382=self.conv1621(D18)
        elif  num28==622:
            p382=self.conv1622(D18)
        elif  num28==623:
            p382=self.conv1623(D18)
        elif  num28==624:
            p382=self.conv1624(D18)
        elif  num28==625:
            p382=self.conv1625(D18)
        elif  num28==626:
            p382=self.conv1626(D18)
        elif  num28==627:
            p382=self.conv1627(D18)
        elif  num28==628:
            p382=self.conv1628(D18)
        elif  num28==629:
            p382=self.conv1629(D18)  
        elif  num28==630:
            p382=self.conv1630(D18)
        elif  num28==631:
            p382=self.conv1631(D18)
        elif  num28==632:
            p382=self.conv1632(D18)
        elif  num28==633:
            p382=self.conv1633(D18)
        elif  num28==634:
            p382=self.conv1634(D18)
        elif  num28==635:
            p382=self.conv1635(D18)
        elif  num28==636:
            p382=self.conv1636(D18)
        elif  num28==637:
            p382=self.conv1637(D18)
        elif  num28==638:
            p382=self.conv1638(D18)
        elif  num28==639:
            p382=self.conv1639(D18)                                                                                                                          
        elif  num28==640:
            p382=self.conv1640(D18)
        elif  num28==641:
            p382=self.conv1641(D18)
        elif  num28==642:
            p382=self.conv1642(D18)
        elif  num28==643:
            p382=self.conv1643(D18)
        elif  num28==644:
            p382=self.conv1644(D18)
        elif  num28==645:
            p382=self.conv1645(D18)
        elif  num28==646:
            p382=self.conv1646(D18)
        elif  num28==647:
            p382=self.conv1647(D18)
        elif  num28==648:
            p382=self.conv1648(D18)
        elif  num28==649:
            p382=self.conv1649(D18)                                                                                                                          
        elif  num28==650:
            p382=self.conv1650(D18)
        elif  num28==651:
            p382=self.conv1651(D18)
        elif  num28==652:
            p382=self.conv1652(D18)
        elif  num28==653:
            p382=self.conv1653(D18)
        elif  num28==654:
            p382=self.conv1654(D18)
        elif  num28==655:
            p382=self.conv1655(D18)
        elif  num28==656:
            p382=self.conv1656(D18)
        elif  num28==657:
            p382=self.conv1657(D18)
        elif  num28==658:
            p382=self.conv1658(D18)
        elif  num28==659:
            p382=self.conv1659(D18)
        elif  num28==660:
            p382=self.conv1660(D18)
        elif  num28==661:
            p382=self.conv1661(D18)
        elif  num28==662:
            p382=self.conv1662(D18)
        elif  num28==663:
            p382=self.conv1663(D18)
        elif  num28==664:
            p382=self.conv1664(D18)
        elif  num28==665:
            p382=self.conv1665(D18)
        elif  num28==666:
            p382=self.conv1666(D18)
        elif  num28==667:
            p382=self.conv1667(D18)
        elif  num28==668:
            p382=self.conv1668(D18)
        elif  num28==669:
            p382=self.conv1669(D18) 
        elif  num28==670:
            p382=self.conv1670(D18)
        elif  num28==671:
            p382=self.conv1671(D18)
        elif  num28==672:
            p382=self.conv1672(D18)
        elif  num28==673:
            p382=self.conv1673(D18)
        elif  num28==674:
            p382=self.conv1674(D18)
        elif  num28==675:
            p382=self.conv1675(D18)
        elif  num28==676:
            p382=self.conv1676(D18)
        elif  num28==677:
            p382=self.conv1677(D18)
        elif  num28==678:
            p382=self.conv1678(D18)
        elif  num28==679:
            p382=self.conv1679(D18)
        elif  num28==680:
            p382=self.conv1680(D18)
        elif  num28==681:
            p382=self.conv1681(D18)
        elif  num28==682:
            p382=self.conv1682(D18)
        elif  num28==683:
            p382=self.conv1683(D18)
        elif  num28==684:
            p382=self.conv1684(D18)
        elif  num28==685:
            p382=self.conv1685(D18)
        elif  num28==686:
            p382=self.conv1686(D18)
        elif  num28==687:
            p382=self.conv1687(D18)
        elif  num28==688:
            p382=self.conv1688(D18)
        elif  num28==689:
            p382=self.conv1689(D18)
        elif  num28==690:
            p382=self.conv1690(D18)
        elif  num28==691:
            p382=self.conv1691(D18)
        elif  num28==692:
            p382=self.conv1692(D18)
        elif  num28==693:
            p382=self.conv1693(D18)
        elif  num28==694:
            p382=self.conv1694(D18)
        elif  num28==695:
            p382=self.conv1695(D18)
        elif  num28==696:
            p382=self.conv1696(D18)
        elif  num28==697:
            p382=self.conv1697(D18)
        elif  num28==698:
            p382=self.conv1698(D18)
        elif  num28==699:
            p382=self.conv1699(D18)
        elif  num28==700:
            p382=self.conv1700(D18)
        elif  num28==701:
            p382=self.conv1701(D18)
        elif  num28==702:
            p382=self.conv1702(D18)
        elif  num28==703:
            p382=self.conv1703(D18)
        elif  num28==704:
            p382=self.conv1704(D18)
        elif  num28==705:
            p382=self.conv1705(D18)
        elif  num28==706:
            p382=self.conv1706(D18)
        elif  num28==707:
            p382=self.conv1707(D18)
        elif  num28==708:
            p382=self.conv1708(D18)
        elif  num28==709:
            p382=self.conv1709(D18)
        elif  num28==710:
            p382=self.conv1710(D18)
        elif  num28==711:
            p382=self.conv1711(D18)
        elif  num28==712:
            p382=self.conv1712(D18)
        elif  num28==713:
            p382=self.conv1713(D18)
        elif  num28==714:
            p382=self.conv1714(D18)
        elif  num28==715:
            p382=self.conv1715(D18)
        elif  num28==716:
            p382=self.conv1716(D18)
        elif  num28==717:
            p382=self.conv1717(D18)
        elif  num28==718:
            p382=self.conv1718(D18)
        elif  num28==719:
            p382=self.conv1719(D18)
        elif  num28==720:
            p382=self.conv1720(D18)
        elif  num28==721:
            p382=self.conv1721(D18)
        elif  num28==722:
            p382=self.conv1722(D18)
        elif  num28==723:
            p382=self.conv1723(D18)
        elif  num28==724:
            p382=self.conv1724(D18)
        elif  num28==725:
            p382=self.conv1725(D18)
        elif  num28==726:
            p382=self.conv1726(D18)
        elif  num28==727:
            p382=self.conv1727(D18)
        elif  num28==728:
            p382=self.conv1728(D18)
        elif  num28==729:
            p382=self.conv1729(D18)
        elif  num28==730:
            p382=self.conv1730(D18)
        elif  num28==731:
            p382=self.conv1731(D18)
        elif  num28==732:
            p382=self.conv1732(D18)
        elif  num28==733:
            p382=self.conv1733(D18)
        elif  num28==734:
            p382=self.conv1734(D18)
        elif  num28==735:
            p382=self.conv1735(D18)
        elif  num28==736:
            p382=self.conv1736(D18)
        elif  num28==737:
            p382=self.conv1737(D18)
        elif  num28==738:
            p382=self.conv1738(D18)
        elif  num28==739:
            p382=self.conv1739(D18)                                                                                                                          
        elif  num28==740:
            p382=self.conv1740(D18)
        elif  num28==741:
            p382=self.conv1741(D18)
        elif  num28==742:
            p382=self.conv1742(D18)
        elif  num28==743:
            p382=self.conv1743(D18)
        elif  num28==744:
            p382=self.conv1744(D18)
        elif  num28==745:
            p382=self.conv1745(D18)
        elif  num28==746:
            p382=self.conv1746(D18)
        elif  num28==747:
            p382=self.conv1747(D18)
        elif  num28==748:
            p382=self.conv1748(D18)
        elif  num28==749:
            p382=self.conv1749(D18)
        elif  num28==750:
            p382=self.conv1750(D18)
        elif  num28==751:
            p382=self.conv1751(D18)
        elif  num28==752:
            p382=self.conv1752(D18)
        elif  num28==753:
            p382=self.conv1753(D18)
        elif  num28==754:
            p382=self.conv1754(D18)
        elif  num28==755:
            p382=self.conv1755(D18)
        elif  num28==756:
            p382=self.conv1756(D18)
        elif  num28==757:
            p382=self.conv1757(D18)
        elif  num28==758:
            p382=self.conv1758(D18)
        elif  num28==759:
            p382=self.conv1759(D18)
        elif  num28==760:
            p382=self.conv1760(D18)
        elif  num28==761:
            p382=self.conv1761(D18)
        elif  num28==762:
            p382=self.conv1762(D18)
        elif  num28==763:
            p382=self.conv1763(D18)
        elif  num28==764:
            p382=self.conv1764(D18)
        elif  num28==765:
            p382=self.conv1765(D18)
        elif  num28==766:
            p382=self.conv1766(D18)
        elif  num28==767:
            p382=self.conv1767(D18)
        elif  num28==768:
            p382=self.conv1768(D18)
        elif  num28==769:
            p382=self.conv1769(D18) 
        elif  num28==770:
            p382=self.conv1770(D18)
        elif  num28==771:
            p382=self.conv1771(D18)
        elif  num28==772:
            p382=self.conv1772(D18)
        elif  num28==773:
            p382=self.conv1773(D18)
        elif  num28==774:
            p382=self.conv1774(D18)
        elif  num28==775:
            p382=self.conv1775(D18)
        elif  num28==776:
            p382=self.conv1776(D18)
        elif  num28==777:
            p382=self.conv1777(D18)
        elif  num28==778:
            p382=self.conv1778(D18)
        elif  num28==779:
            p382=self.conv1779(D18) 
        elif  num28==780:
            p382=self.conv1780(D18)
        elif  num28==781:
            p382=self.conv1781(D18)
        elif  num28==782:
            p382=self.conv1782(D18)
        elif  num28==783:
            p382=self.conv1783(D18)
        elif  num28==784:
            p382=self.conv1784(D18)
        elif  num28==785:
            p382=self.conv1785(D18)
        elif  num28==786:
            p382=self.conv1786(D18)
        elif  num28==787:
            p382=self.conv1787(D18)
        elif  num28==788:
            p382=self.conv1788(D18)
        elif  num28==789:
            p382=self.conv1789(D18) 
        elif  num28==790:
            p382=self.conv1790(D18)
        elif  num28==791:
            p382=self.conv1791(D18)
        elif  num28==792:
            p382=self.conv1792(D18)
        elif  num28==793:
            p382=self.conv1793(D18)
        elif  num28==794:
            p382=self.conv1794(D18)
        elif  num28==795:
            p382=self.conv1795(D18)
        elif  num28==796:
            p382=self.conv1796(D18)
        elif  num28==797:
            p382=self.conv1797(D18)
        elif  num28==798:
            p382=self.conv1798(D18)
        elif  num28==799:
            p382=self.conv1799(D18) 
        elif  num28==800:
            p382=self.conv1800(D18)
        elif  num28==801:
            p382=self.conv1801(D18)
        elif  num28==802:
            p382=self.conv1802(D18)
        elif  num28==803:
            p382=self.conv1803(D18)
        elif  num28==804:
            p382=self.conv1804(D18)
        elif  num28==805:
            p382=self.conv1805(D18)
        elif  num28==806:
            p382=self.conv1806(D18)
        elif  num28==807:
            p382=self.conv1807(D18)
        elif  num28==808:
            p382=self.conv1808(D18)
        elif  num28==809:
            p382=self.conv1809(D18)
        elif  num28==810:
            p382=self.conv1810(D18)
        elif  num28==811:
            p382=self.conv1811(D18)
        elif  num28==812:
            p382=self.conv1812(D18)
        elif  num28==813:
            p382=self.conv1813(D18)
        elif  num28==814:
            p382=self.conv1814(D18)
        elif  num28==815:
            p382=self.conv1815(D18)
        elif  num28==816:
            p382=self.conv1816(D18)
        elif  num28==817:
            p382=self.conv1817(D18)
        elif  num28==818:
            p382=self.conv1818(D18)
        elif  num28==819:
            p382=self.conv1819(D18)
        elif  num28==820:
            p382=self.conv1820(D18)
        elif  num28==821:
            p382=self.conv1821(D18)
        elif  num28==822:
            p382=self.conv1822(D18)
        elif  num28==823:
            p382=self.conv1823(D18)
        elif  num28==824:
            p382=self.conv1824(D18)
        elif  num28==825:
            p382=self.conv1825(D18)
        elif  num28==826:
            p382=self.conv1826(D18)
        elif  num28==827:
            p382=self.conv1827(D18)
        elif  num28==828:
            p382=self.conv1828(D18)
        elif  num28==829:
            p382=self.conv1829(D18)                                                                                                                          
        elif  num28==830:
            p382=self.conv1830(D18)
        elif  num28==831:
            p382=self.conv1831(D18)
        elif  num28==832:
            p382=self.conv1832(D18)
        elif  num28==833:
            p382=self.conv1833(D18)
        elif  num28==834:
            p382=self.conv1834(D18)
        elif  num28==835:
            p382=self.conv1835(D18)
        elif  num28==836:
            p382=self.conv1836(D18)
        elif  num28==837:
            p382=self.conv1837(D18)
        elif  num28==838:
            p382=self.conv1838(D18)
        elif  num28==839:
            p382=self.conv1839(D18)
        elif  num28==840:
            p382=self.conv1840(D18)
        elif  num28==841:
            p382=self.conv1841(D18)
        elif  num28==842:
            p382=self.conv1842(D18)
        elif  num28==843:
            p382=self.conv1843(D18)
        elif  num28==844:
            p382=self.conv1844(D18)
        elif  num28==845:
            p382=self.conv1845(D18)
        elif  num28==846:
            p382=self.conv1846(D18)
        elif  num28==847:
            p382=self.conv1847(D18)
        elif  num28==848:
            p382=self.conv1848(D18)
        elif  num28==849:
            p382=self.conv1849(D18)
        elif  num28==850:
            p382=self.conv1850(D18)
        elif  num28==851:
            p382=self.conv1851(D18)
        elif  num28==852:
            p382=self.conv1852(D18)
        elif  num28==853:
            p382=self.conv1853(D18)
        elif  num28==854:
            p382=self.conv1854(D18)
        elif  num28==855:
            p382=self.conv1855(D18)
        elif  num28==856:
            p382=self.conv1856(D18)
        elif  num28==857:
            p382=self.conv1857(D18)
        elif  num28==858:
            p382=self.conv1858(D18)
        elif  num28==859:
            p382=self.conv1859(D18)
        elif  num28==860:
            p382=self.conv1860(D18)
        elif  num28==861:
            p382=self.conv1861(D18)
        elif  num28==862:
            p382=self.conv1862(D18)
        elif  num28==863:
            p382=self.conv1863(D18)
        elif  num28==864:
            p382=self.conv1864(D18)
        elif  num28==865:
            p382=self.conv1865(D18)
        elif  num28==866:
            p382=self.conv1866(D18)
        elif  num28==867:
            p382=self.conv1867(D18)
        elif  num28==868:
            p382=self.conv1868(D18)
        elif  num28==869:
            p382=self.conv1869(D18) 
        elif  num28==870:
            p382=self.conv1870(D18)
        elif  num28==871:
            p382=self.conv1871(D18)
        elif  num28==872:
            p382=self.conv1872(D18)
        elif  num28==873:
            p382=self.conv1873(D18)
        elif  num28==874:
            p382=self.conv1874(D18)
        elif  num28==875:
            p382=self.conv1875(D18)
        elif  num28==876:
            p382=self.conv1876(D18)
        elif  num28==877:
            p382=self.conv1877(D18)
        elif  num28==878:
            p382=self.conv1878(D18)
        elif  num28==879:
            p382=self.conv1879(D18)
        elif  num28==880:
            p382=self.conv1880(D18)
        elif  num28==881:
            p382=self.conv1881(D18)
        elif  num28==882:
            p382=self.conv1882(D18)
        elif  num28==883:
            p382=self.conv1883(D18)
        elif  num28==884:
            p382=self.conv1884(D18)
        elif  num28==885:
            p382=self.conv1885(D18)
        elif  num28==886:
            p382=self.conv1886(D18)
        elif  num28==887:
            p382=self.conv1887(D18)
        elif  num28==888:
            p382=self.conv1888(D18)
        elif  num28==889:
            p382=self.conv1889(D18)  
        elif  num28==890:
            p382=self.conv1890(D18)
        elif  num28==891:
            p382=self.conv1891(D18)
        elif  num28==892:
            p382=self.conv1892(D18)
        elif  num28==893:
            p382=self.conv1893(D18)
        elif  num28==894:
            p382=self.conv1894(D18)
        elif  num28==895:
            p382=self.conv1895(D18)
        elif  num28==896:
            p382=self.conv1896(D18)
        elif  num28==897:
            p382=self.conv1897(D18)
        elif  num28==898:
            p382=self.conv1898(D18)
        elif  num28==899:
            p382=self.conv1899(D18)
        elif  num28==900:
            p382=self.conv1900(D18)
        elif  num28==901:
            p382=self.conv1901(D18)
        elif  num28==902:
            p382=self.conv1902(D18)
        elif  num28==903:
            p382=self.conv1903(D18)
        elif  num28==904:
            p382=self.conv1904(D18)
        elif  num28==905:
            p382=self.conv1905(D18)
        elif  num28==906:
            p382=self.conv1906(D18)
        elif  num28==907:
            p382=self.conv1907(D18)
        elif  num28==908:
            p382=self.conv1908(D18)
        elif  num28==909:
            p382=self.conv1909(D18)
        elif  num28==910:
            p382=self.conv1910(D18)
        elif  num28==911:
            p382=self.conv1911(D18)
        elif  num28==912:
            p382=self.conv1912(D18)
        elif  num28==913:
            p382=self.conv1913(D18)
        elif  num28==914:
            p382=self.conv1914(D18)
        elif  num28==915:
            p382=self.conv1915(D18)
        elif  num28==916:
            p382=self.conv1916(D18)
        elif  num28==917:
            p382=self.conv1917(D18)
        elif  num28==918:
            p382=self.conv1918(D18)
        elif  num28==919:
            p382=self.conv1919(D18)
        elif  num28==920:
            p382=self.conv1920(D18)
        elif  num28==921:
            p382=self.conv1921(D18)
        elif  num28==922:
            p382=self.conv1922(D18)
        elif  num28==923:
            p382=self.conv1923(D18)
        elif  num28==924:
            p382=self.conv1924(D18)
        elif  num28==925:
            p382=self.conv1925(D18)
        elif  num28==926:
            p382=self.conv1926(D18)
        elif  num28==927:
            p382=self.conv1927(D18)
        elif  num28==928:
            p382=self.conv1928(D18)
        elif  num28==929:
            p382=self.conv1929(D18)
        elif  num28==930:
            p382=self.conv1930(D18)
        elif  num28==931:
            p382=self.conv1931(D18)
        elif  num28==932:
            p382=self.conv1932(D18)
        elif  num28==933:
            p382=self.conv1933(D18)
        elif  num28==934:
            p382=self.conv1934(D18)
        elif  num28==935:
            p382=self.conv1935(D18)
        elif  num28==936:
            p382=self.conv1936(D18)
        elif  num28==937:
            p382=self.conv1937(D18)
        elif  num28==938:
            p382=self.conv1938(D18)
        elif  num28==939:
            p382=self.conv1939(D18) 
        elif  num28==940:
            p382=self.conv1940(D18)
        elif  num28==941:
            p382=self.conv1941(D18)
        elif  num28==942:
            p382=self.conv1942(D18)
        elif  num28==943:
            p382=self.conv1943(D18)
        elif  num28==944:
            p382=self.conv1944(D18)
        elif  num28==945:
            p382=self.conv1945(D18)
        elif  num28==946:
            p382=self.conv1946(D18)
        elif  num28==947:
            p382=self.conv1947(D18)
        elif  num28==948:
            p382=self.conv1948(D18)
        elif  num28==949:
            p382=self.conv1949(D18)                                                                                                                          
        elif  num28==950:
            p382=self.conv1950(D18)
        elif  num28==951:
            p382=self.conv1951(D18)
        elif  num28==952:
            p382=self.conv1952(D18)
        elif  num28==953:
            p382=self.conv1953(D18)
        elif  num28==954:
            p382=self.conv1954(D18)
        elif  num28==955:
            p382=self.conv1955(D18)
        elif  num28==956:
            p382=self.conv1956(D18)
        elif  num28==957:
            p382=self.conv1957(D18)
        elif  num28==958:
            p382=self.conv1958(D18)
        elif  num28==959:
            p382=self.conv1959(D18)
        elif  num28==960:
            p382=self.conv1960(D18)
        elif  num28==961:
            p382=self.conv1961(D18)
        elif  num28==962:
            p382=self.conv1962(D18)
        elif  num28==963:
            p382=self.conv1963(D18)
        elif  num28==964:
            p382=self.conv1964(D18)
        elif  num28==965:
            p382=self.conv1965(D18)
        elif  num28==966:
            p382=self.conv1966(D18)
        elif  num28==967:
            p382=self.conv1967(D18)
        elif  num28==968:
            p382=self.conv1968(D18)
        elif  num28==969:
            p382=self.conv1969(D18) 
        elif  num28==970:
            p382=self.conv1970(D18)
        elif  num28==971:
            p382=self.conv1971(D18)
        elif  num28==972:
            p382=self.conv1972(D18)
        elif  num28==973:
            p382=self.conv1973(D18)
        elif  num28==974:
            p382=self.conv1974(D18)
        elif  num28==975:
            p382=self.conv1975(D18)
        elif  num28==976:
            p382=self.conv1976(D18)
        elif  num28==977:
            p382=self.conv1977(D18)
        elif  num28==978:
            p382=self.conv1978(D18)
        elif  num28==979:
            p382=self.conv1979(D18) 
        elif  num28==980:
            p382=self.conv1980(D18)
        elif  num28==981:
            p382=self.conv1981(D18)
        elif  num28==982:
            p382=self.conv1982(D18)
        elif  num28==983:
            p382=self.conv1983(D18)
        elif  num28==984:
            p382=self.conv1984(D18)
        elif  num28==985:
            p382=self.conv1985(D18)
        elif  num28==986:
            p382=self.conv1986(D18)
        elif  num28==987:
            p382=self.conv1987(D18)
        elif  num28==988:
            p382=self.conv1988(D18)
        elif  num28==989:
            p382=self.conv1989(D18)
        elif  num28==990:
            p382=self.conv1990(D18)
        elif  num28==991:
            p382=self.conv1991(D18)
        elif  num28==992:
            p382=self.conv1992(D18)
        elif  num28==993:
            p382=self.conv1993(D18)
        elif  num28==994:
            p382=self.conv1994(D18)
        elif  num28==995:
            p382=self.conv1995(D18)
        elif  num28==996:
            p382=self.conv1996(D18)
        elif  num28==997:
            p382=self.conv1997(D18)
        elif  num28==998:
            p382=self.conv1998(D18)
        elif  num28==999:
            p382=self.conv1999(D18) 
        elif  num28==1000:
            p382=self.conv11000(D18)
        elif  num28==1001:
            p382=self.conv11001(D18)
        elif  num28==1002:
            p382=self.conv11002(D18)
        elif  num28==1003:
            p382=self.conv11003(D18)
        elif  num28==1004:
            p382=self.conv11004(D18)
        elif  num28==1005:
            p382=self.conv11005(D18)
        elif  num28==1006:
            p382=self.conv11006(D18)
        elif  num28==1007:
            p382=self.conv11007(D18)
        elif  num28==1008:
            p382=self.conv11008(D18)
        elif  num28==1009:
            p382=self.conv11009(D18) 
        elif  num28==1010:
            p382=self.conv11010(D18)
        elif  num28==1011:
            p382=self.conv11011(D18)
        elif  num28==1012:
            p382=self.conv11012(D18)
        elif  num28==1013:
            p382=self.conv11013(D18)
        elif  num28==1014:
            p382=self.conv11014(D18)
        elif  num28==1015:
            p382=self.conv11015(D18)
        elif  num28==1016:
            p382=self.conv11016(D18)
        elif  num28==1017:
            p382=self.conv11017(D18)
        elif  num28==1018:
            p382=self.conv11018(D18)
        elif  num28==1019:
            p382=self.conv11019(D18)
        elif  num28==1020:
            p382=self.conv11020(D18)
        elif  num28==1021:
            p382=self.conv11021(D18)
        elif  num28==1022:
            p382=self.conv11022(D18)
        elif  num28==1023:
            p382=self.conv11023(D18)
        elif  num28==1024:
            p382=self.conv11024(D18) 
            
        if num081==1:
            p3881=self.conv11(B181)
        elif num081==2:
            p3881=self.conv12(B181)
        elif num081==3:
            p3881=self.conv13(B181)
        elif num081==4:
            p3881=self.conv14(B181)
        elif num081==5:
            p3881=self.conv15(B181)
        elif num081==6:
            p3881=self.conv16(B181)
        elif num081==7:
            p3881=self.conv17(B181)
        elif num081==8:
            p3881=self.conv18(B181)
        elif num081==9:
            p3881=self.conv19(B181)
        elif num081==10:
            p3881=self.conv110(B181)
        elif num081==11:
            p3881=self.conv111(B181)
        elif num081==12:
            p3881=self.conv112(B181)
        elif num081==13:
            p3881=self.conv113(B181)
        elif num081==14:
            p3881=self.conv114(B181)
        elif num081==15:
            p3881=self.conv115(B181)
        elif num081==16:
            p3881=self.conv116(B181)
        elif num081==17:
            p3881=self.conv117(B181)
        elif num081==18:
            p3881=self.conv118(B181)
        elif num081==19:
            p3881=self.conv119(B181)
        elif num081==20:
            p3881=self.conv120(B181)
        elif num081==21:
            p3881=self.conv121(B181)
        elif num081==22:
            p3881=self.conv122(B181)
        elif num081==23:
            p3881=self.conv123(B181)
        elif num081==24:
            p3881=self.conv124(B181)
        elif num081==25:
            p3881=self.conv125(B181)
        elif num081==26:
            p3881=self.conv126(B181)
        elif num081==27:
            p3881=self.conv127(B181)
        elif num081==28:
            p3881=self.conv128(B181)
        elif num081==29:
            p3881=self.conv129(B181)
        elif num081==30:
            p3881=self.conv130(B181)
        elif num081==31:
            p3881=self.conv131(B181)
        elif num081==32:
            p3881=self.conv132(B181)
        elif num081==33:
            p3881=self.conv133(B181)
        elif num081==34:
            p3881=self.conv134(B181)
        elif num081==35:
            p3881=self.conv135(B181)
        elif num081==36:
            p3881=self.conv136(B181)
        elif num081==37:
            p3881=self.conv137(B181)
        elif num081==38:
            p3881=self.conv138(B181)
        elif num081==39:
            p3881=self.conv139(B181)
        elif num081==40:
            p3881=self.conv140(B181)
        elif num081==41:
            p3881=self.conv141(B181)
        elif num081==42:
            p3881=self.conv142(B181)
        elif num081==43:
            p3881=self.conv143(B181)
        elif num081==44:
            p3881=self.conv144(B181)
        elif num081==45:
            p3881=self.conv145(B181)
        elif num081==46:
            p3881=self.conv146(B181)
        elif num081==47:
            p3881=self.conv147(B181)
        elif num081==48:
            p3881=self.conv148(B181)
        elif num081==49:
            p3881=self.conv149(B181)
        elif num081==50:
            p3881=self.conv150(B181)
        elif num081==51:
            p3881=self.conv151(B181)
        elif num081==52:
            p3881=self.conv152(B181)
        elif num081==53:
            p3881=self.conv153(B181)
        elif num081==54:
            p3881=self.conv154(B181)
        elif num081==55:
            p3881=self.conv155(B181)
        elif num081==56:
            p3881=self.conv156(B181)
        elif num081==57:
            p3881=self.conv157(B181)
        elif num081==58:
            p3881=self.conv158(B181)
        elif num081==59:
            p3881=self.conv159(B181)
        elif num081==60:
            p3881=self.conv160(B181)
        elif num081==61:
            p3881=self.conv161(B181)
        elif num081==62:
            p3881=self.conv162(B181)
        elif num081==63:
            p3881=self.conv163(B181)
        elif num081==64:
            p3881=self.conv164(B181)
        
        if  num181==1:
            p3811=self.conv11(C181)
        elif  num181==2:
            p3811=self.conv12(C181)
        elif  num181==3:
            p3811=self.conv13(C181)
        elif  num181==4:
            p3811=self.conv14(C181)
        elif  num181==5:
            p3811=self.conv15(C181)
        elif  num181==6:
            p3811=self.conv16(C181)
        elif  num181==7:
            p3811=self.conv17(C181)
        elif  num181==8:
            p3811=self.conv18(C181)
        elif  num181==9:
            p3811=self.conv19(C181)
        elif  num181==10:
            p3811=self.conv110(C181)
        elif  num181==11:
            p3811=self.conv111(C181)
        elif  num181==12:
            p3811=self.conv112(C181)
        elif  num181==13:
            p3811=self.conv113(C181)
        elif  num181==14:
            p3811=self.conv114(C181)
        elif  num181==15:
            p3811=self.conv115(C181)
        elif  num181==16:
            p3811=self.conv116(C181)
        elif  num181==17:
            p3811=self.conv117(C181)
        elif  num181==18:
            p3811=self.conv118(C181)
        elif  num181==19:
            p3811=self.conv119(C181)
        elif  num181==20:
            p3811=self.conv120(C181)
        elif  num181==21:
            p3811=self.conv121(C181)
        elif  num181==22:
            p3811=self.conv122(C181)
        elif  num181==23:
            p3811=self.conv123(C181)
        elif  num181==24:
            p3811=self.conv124(C181)
        elif  num181==25:
            p3811=self.conv125(C181)
        elif  num181==26:
            p3811=self.conv126(C181)
        elif  num181==27:
            p3811=self.conv127(C181)
        elif  num181==28:
            p3811=self.conv128(C181)
        elif  num181==29:
            p3811=self.conv129(C181)
        elif  num181==30:
            p3811=self.conv130(C181)
        elif  num181==31:
            p3811=self.conv131(C181)
        elif  num181==32:
            p3811=self.conv132(C181)
        elif  num181==33:
            p3811=self.conv133(C181)
        elif  num181==34:
            p3811=self.conv134(C181)
        elif  num181==35:
            p3811=self.conv135(C181)
        elif  num181==36:
            p3811=self.conv136(C181)
        elif  num181==37:
            p3811=self.conv137(C181)
        elif  num181==38:
            p3811=self.conv138(C181)
        elif  num181==39:
            p3811=self.conv139(C181)
        elif  num181==40:
            p3811=self.conv140(C181)
        elif  num181==41:
            p3811=self.conv141(C181)
        elif  num181==42:
            p3811=self.conv142(C181)
        elif  num181==43:
            p3811=self.conv143(C181)
        elif  num181==44:
            p3811=self.conv144(C181)
        elif  num181==45:
            p3811=self.conv145(C181)
        elif  num181==46:
            p3811=self.conv146(C181)
        elif  num181==47:
            p3811=self.conv147(C181)
        elif  num181==48:
            p3811=self.conv148(C181)
        elif  num181==49:
            p3811=self.conv149(C181)
        elif  num181==50:
            p3811=self.conv150(C181)
        elif  num181==51:
            p3811=self.conv151(C181)
        elif  num181==52:
            p3811=self.conv152(C181)
        elif  num181==53:
            p3811=self.conv153(C181)
        elif  num181==54:
            p3811=self.conv154(C181)
        elif  num181==55:
            p3811=self.conv155(C181)
        elif  num181==56:
            p3811=self.conv156(C181)
        elif  num181==57:
            p3811=self.conv157(C181)
        elif  num181==58:
            p3811=self.conv158(C181)
        elif  num181==59:
            p3811=self.conv159(C181)
        elif  num181==60:
            p3811=self.conv160(C181)
        elif  num181==61:
            p3811=self.conv161(C181)
        elif  num181==62:
            p3811=self.conv162(C181)
        elif  num181==63:
            p3811=self.conv163(C181)
        elif  num181==64:
            p3811=self.conv164(C181)
        elif  num181==65:
            p3811=self.conv165(C181)
        elif  num181==66:
            p3811=self.conv166(C181)
        elif  num181==67:
            p3811=self.conv167(C181)
        elif  num181==68:
            p3811=self.conv168(C181)
        elif  num181==69:
            p3811=self.conv169(C181)
        elif  num181==70:
            p3811=self.conv170(C181)
        elif  num181==71:
            p3811=self.conv171(C181)
        elif  num181==72:
            p3811=self.conv172(C181)
        elif  num181==73:
            p3811=self.conv173(C181)
        elif  num181==74:
            p3811=self.conv174(C181)
        elif  num181==75:
            p3811=self.conv175(C181)
        elif  num181==76:
            p3811=self.conv176(C181)
        elif  num181==77:
            p3811=self.conv177(C181)
        elif  num181==78:
            p3811=self.conv178(C181)
        elif  num181==79:
            p3811=self.conv179(C181)
        elif  num181==80:
            p3811=self.conv180(C181)
        elif  num181==81:
            p3811=self.conv181(C181)
        elif  num181==82:
            p3811=self.conv182(C181)
        elif  num181==83:
            p3811=self.conv183(C181)
        elif  num181==84:
            p3811=self.conv184(C181)
        elif  num181==85:
            p3811=self.conv185(C181)
        elif  num181==86:
            p3811=self.conv186(C181)
        elif  num181==87:
            p3811=self.conv187(C181)
        elif  num181==88:
            p3811=self.conv188(C181)
        elif  num181==89:
            p3811=self.conv189(C181)    
        elif  num181==90:
            p3811=self.conv190(C181)
        elif  num181==91:
            p3811=self.conv191(C181)
        elif  num181==92:
            p3811=self.conv192(C181)
        elif  num181==93:
            p3811=self.conv193(C181)
        elif  num181==94:
            p3811=self.conv194(C181)
        elif  num181==95:
            p3811=self.conv195(C181)
        elif  num181==96:
            p3811=self.conv196(C181)
        elif  num181==97:
            p3811=self.conv197(C181)
        elif  num181==98:
            p3811=self.conv198(C181)
        elif  num181==99:
            p3811=self.conv199(C181) 
        elif  num181==100:
            p3811=self.conv1100(C181)
        elif  num181==101:
            p3811=self.conv1101(C181)
        elif  num181==102:
            p3811=self.conv1102(C181)
        elif  num181==103:
            p3811=self.conv1103(C181)
        elif  num181==104:
            p3811=self.conv1104(C181)
        elif  num181==105:
            p3811=self.conv1105(C181)
        elif  num181==106:
            p3811=self.conv1106(C181)
        elif  num181==107:
            p3811=self.conv1107(C181)
        elif  num181==108:
            p3811=self.conv1108(C181)
        elif  num181==109:
            p3811=self.conv1109(C181)
        elif  num181==110:
            p3811=self.conv1110(C181)
        elif  num181==111:
            p3811=self.conv1111(C181)
        elif  num181==112:
            p3811=self.conv1112(C181)
        elif  num181==113:
            p3811=self.conv1113(C181)
        elif  num181==114:
            p3811=self.conv1114(C181)
        elif  num181==115:
            p3811=self.conv1115(C181)
        elif  num181==116:
            p3811=self.conv1116(C181)
        elif  num181==117:
            p3811=self.conv1117(C181)
        elif  num181==118:
            p3811=self.conv1118(C181)
        elif  num181==119:
            p3811=self.conv1119(C181) 
        elif  num181==120:
            p3811=self.conv1120(C181)
        elif  num181==121:
            p3811=self.conv1121(C181)
        elif  num181==122:
            p3811=self.conv1122(C181)
        elif  num181==123:
            p3811=self.conv1123(C181)
        elif  num181==124:
            p3811=self.conv1124(C181)
        elif  num181==125:
            p3811=self.conv1125(C181)
        elif  num181==126:
            p3811=self.conv1126(C181)
        elif  num181==127:
            p3811=self.conv1127(C181)
        elif  num181==128:
            p3811=self.conv1128(C181)
        elif  num181==129:
            p3811=self.conv1129(C181) 
        elif  num181==130:
            p3811=self.conv1130(C181)
        elif  num181==131:
            p3811=self.conv1131(C181)
        elif  num181==132:
            p3811=self.conv1132(C181)
        elif  num181==133:
            p3811=self.conv1133(C181)
        elif  num181==134:
            p3811=self.conv1134(C181)
        elif  num181==135:
            p3811=self.conv1135(C181)
        elif  num181==136:
            p3811=self.conv1136(C181)
        elif  num181==137:
            p3811=self.conv1137(C181)
        elif  num181==138:
            p3811=self.conv1138(C181)
        elif  num181==139:
            p3811=self.conv1139(C181)
        elif  num181==140:
            p3811=self.conv1140(C181)
        elif  num181==141:
            p3811=self.conv1141(C181)
        elif  num181==142:
            p3811=self.conv1142(C181)
        elif  num181==143:
            p3811=self.conv1143(C181)
        elif  num181==144:
            p3811=self.conv1144(C181)
        elif  num181==145:
            p3811=self.conv1145(C181)
        elif  num181==146:
            p3811=self.conv1146(C181)
        elif  num181==147:
            p3811=self.conv1147(C181)
        elif  num181==148:
            p3811=self.conv1148(C181)
        elif  num181==149:
            p3811=self.conv1149(C181) 
        elif  num181==150:
            p3811=self.conv1150(C181)
        elif  num181==151:
            p3811=self.conv1151(C181)
        elif  num181==152:
            p3811=self.conv1152(C181)
        elif  num181==153:
            p3811=self.conv1153(C181)
        elif  num181==154:
            p3811=self.conv1154(C181)
        elif  num181==155:
            p3811=self.conv1155(C181)
        elif  num181==156:
            p3811=self.conv1156(C181)
        elif  num181==157:
            p3811=self.conv1157(C181)
        elif  num181==158:
            p3811=self.conv1158(C181)
        elif  num181==159:
            p3811=self.conv1159(C181) 
        elif  num181==160:
            p3811=self.conv1160(C181)
        elif  num181==161:
            p3811=self.conv1161(C181)
        elif  num181==162:
            p3811=self.conv1162(C181)
        elif  num181==163:
            p3811=self.conv1163(C181)
        elif  num181==164:
            p3811=self.conv1164(C181)
        elif  num181==165:
            p3811=self.conv1165(C181)
        elif  num181==166:
            p3811=self.conv1166(C181)
        elif  num181==167:
            p3811=self.conv1167(C181)
        elif  num181==168:
            p3811=self.conv1168(C181)
        elif  num181==169:
            p3811=self.conv1169(C181) 
        elif  num181==170:
            p3811=self.conv1170(C181)
        elif  num181==171:
            p3811=self.conv1171(C181)
        elif  num181==172:
            p3811=self.conv1172(C181)
        elif  num181==173:
            p3811=self.conv1173(C181)
        elif  num181==174:
            p3811=self.conv1174(C181)
        elif  num181==175:
            p3811=self.conv1175(C181)
        elif  num181==176:
            p3811=self.conv1176(C181)
        elif  num181==177:
            p3811=self.conv1177(C181)
        elif  num181==178:
            p3811=self.conv1178(C181)
        elif  num181==179:
            p3811=self.conv1179(C181)                                                                                              
        elif  num181==180:
            p3811=self.conv1180(C181)
        elif  num181==181:
            p3811=self.conv1181(C181)
        elif  num181==182:
            p3811=self.conv1182(C181)
        elif  num181==183:
            p3811=self.conv1183(C181)
        elif  num181==184:
            p3811=self.conv1184(C181)
        elif  num181==185:
            p3811=self.conv1185(C181)
        elif  num181==186:
            p3811=self.conv1186(C181)
        elif  num181==187:
            p3811=self.conv1187(C181)
        elif  num181==188:
            p3811=self.conv1188(C181)
        elif  num181==189:
            p3811=self.conv1189(C181) 
        elif  num181==190:
            p3811=self.conv1190(C181)
        elif  num181==191:
            p3811=self.conv1191(C181)
        elif  num181==192:
            p3811=self.conv1192(C181)
        elif  num181==193:
            p3811=self.conv1193(C181)
        elif  num181==194:
            p3811=self.conv1194(C181)
        elif  num181==195:
            p3811=self.conv1195(C181)
        elif  num181==196:
            p3811=self.conv1196(C181)
        elif  num181==197:
            p3811=self.conv1197(C181)
        elif  num181==198:
            p3811=self.conv1198(C181)
        elif  num181==199:
            p3811=self.conv1199(C181)
        elif  num181==200:
            p3811=self.conv1200(C181)
        elif  num181==201:
            p3811=self.conv1201(C181)
        elif  num181==202:
            p3811=self.conv1202(C181)
        elif  num181==203:
            p3811=self.conv1203(C181)
        elif  num181==204:
            p3811=self.conv1204(C181)
        elif  num181==205:
            p3811=self.conv1205(C181)
        elif  num181==206:
            p3811=self.conv1206(C181)
        elif  num181==207:
            p3811=self.conv1207(C181)
        elif  num181==208:
            p3811=self.conv1208(C181)
        elif  num181==209:
            p3811=self.conv1209(C181)
        elif  num181==210:
            p3811=self.conv1210(C181)
        elif  num181==211:
            p3811=self.conv1211(C181)
        elif  num181==212:
            p3811=self.conv1212(C181)
        elif  num181==213:
            p3811=self.conv1213(C181)
        elif  num181==214:
            p3811=self.conv1214(C181)
        elif  num181==215:
            p3811=self.conv1215(C181)
        elif  num181==216:
            p3811=self.conv1216(C181)
        elif  num181==217:
            p3811=self.conv1217(C181)
        elif  num181==218:
            p3811=self.conv1218(C181)
        elif  num181==219:
            p3811=self.conv1219(C181)
        elif  num181==220:
            p3811=self.conv1220(C181)
        elif  num181==221:
            p3811=self.conv1221(C181)
        elif  num181==222:
            p3811=self.conv1222(C181)
        elif  num181==223:
            p3811=self.conv1223(C181)
        elif  num181==224:
            p3811=self.conv1224(C181)
        elif  num181==225:
            p3811=self.conv1225(C181)
        elif  num181==226:
            p3811=self.conv1226(C181)
        elif  num181==227:
            p3811=self.conv1227(C181)
        elif  num181==228:
            p3811=self.conv1228(C181)
        elif  num181==229:
            p3811=self.conv1229(C181)
        elif  num181==230:
            p3811=self.conv1230(C181)
        elif  num181==231:
            p3811=self.conv1231(C181)
        elif  num181==232:
            p3811=self.conv1232(C181)
        elif  num181==233:
            p3811=self.conv1233(C181)
        elif  num181==234:
            p3811=self.conv1234(C181)
        elif  num181==235:
            p3811=self.conv1235(C181)
        elif  num181==236:
            p3811=self.conv1236(C181)
        elif  num181==237:
            p3811=self.conv1237(C181)
        elif  num181==238:
            p3811=self.conv1238(C181)
        elif  num181==239:
            p3811=self.conv1239(C181) 
        elif  num181==240:
            p3811=self.conv1240(C181)
        elif  num181==241:
            p3811=self.conv1241(C181)
        elif  num181==242:
            p3811=self.conv1242(C181)
        elif  num181==243:
            p3811=self.conv1243(C181)
        elif  num181==244:
            p3811=self.conv1244(C181)
        elif  num181==245:
            p3811=self.conv1245(C181)
        elif  num181==246:
            p3811=self.conv1246(C181)
        elif  num181==247:
            p3811=self.conv1247(C181)
        elif  num181==248:
            p3811=self.conv1248(C181)
        elif  num181==249:
            p3811=self.conv1249(C181)
        elif  num181==250:
            p3811=self.conv1250(C181)
        elif  num181==251:
            p3811=self.conv1251(C181)
        elif  num181==252:
            p3811=self.conv1252(C181)
        elif  num181==253:
            p3811=self.conv1253(C181)
        elif  num181==254:
            p3811=self.conv1254(C181)
        elif  num181==255:
            p3811=self.conv1255(C181)
        elif  num181==256:
            p3811=self.conv1256(C181)
            
        if  num281==1:
            p3812=self.conv11(D181)
        elif  num281==2:
            p3812=self.conv12(D181)
        elif  num281==3:
            p3812=self.conv13(D181)
        elif  num281==4:
            p3812=self.conv14(D181)
        elif  num281==5:
            p3812=self.conv15(D181)
        elif  num281==6:
            p3812=self.conv16(D181)
        elif  num281==7:
            p3812=self.conv17(D181)
        elif  num281==8:
            p3812=self.conv18(D181)
        elif  num281==9:
            p3812=self.conv19(D181)
        elif  num281==10:
            p3812=self.conv110(D181)
        elif  num281==11:
            p3812=self.conv111(D181)
        elif  num281==12:
            p3812=self.conv112(D181)
        elif  num281==13:
            p3812=self.conv113(D181)
        elif  num281==14:
            p3812=self.conv114(D181)
        elif  num281==15:
            p3812=self.conv115(D181)
        elif  num281==16:
            p3812=self.conv116(D181)
        elif  num281==17:
            p3812=self.conv117(D181)
        elif  num281==18:
            p3812=self.conv118(D181)
        elif  num281==19:
            p3812=self.conv119(D181)
        elif  num281==20:
            p3812=self.conv120(D181)
        elif  num281==21:
            p3812=self.conv121(D181)
        elif  num281==22:
            p3812=self.conv122(D181)
        elif  num281==23:
            p3812=self.conv123(D181)
        elif  num281==24:
            p3812=self.conv124(D181)
        elif  num281==25:
            p3812=self.conv125(D181)
        elif  num281==26:
            p3812=self.conv126(D181)
        elif  num281==27:
            p3812=self.conv127(D181)
        elif  num281==28:
            p3812=self.conv128(D181)
        elif  num281==29:
            p3812=self.conv129(D181)
        elif  num281==30:
            p3812=self.conv130(D181)
        elif  num281==31:
            p3812=self.conv131(D181)
        elif  num281==32:
            p3812=self.conv132(D181)
        elif  num281==33:
            p3812=self.conv133(D181)
        elif  num281==34:
            p3812=self.conv134(D181)
        elif  num281==35:
            p3812=self.conv135(D181)
        elif  num281==36:
            p3812=self.conv136(D181)
        elif  num281==37:
            p3812=self.conv137(D181)
        elif  num281==38:
            p3812=self.conv138(D181)
        elif  num281==39:
            p3812=self.conv139(D181)
        elif  num281==40:
            p3812=self.conv140(D181)
        elif  num281==41:
            p3812=self.conv141(D181)
        elif  num281==42:
            p3812=self.conv142(D181)
        elif  num281==43:
            p3812=self.conv143(D181)
        elif  num281==44:
            p3812=self.conv144(D181)
        elif  num281==45:
            p3812=self.conv145(D181)
        elif  num281==46:
            p3812=self.conv146(D181)
        elif  num281==47:
            p3812=self.conv147(D181)
        elif  num281==48:
            p3812=self.conv148(D181)
        elif  num281==49:
            p3812=self.conv149(D181)
        elif  num281==50:
            p3812=self.conv150(D181)
        elif  num281==51:
            p3812=self.conv151(D181)
        elif  num281==52:
            p3812=self.conv152(D181)
        elif  num281==53:
            p3812=self.conv153(D181)
        elif  num281==54:
            p3812=self.conv154(D181)
        elif  num281==55:
            p3812=self.conv155(D181)
        elif  num281==56:
            p3812=self.conv156(D181)
        elif  num281==57:
            p3812=self.conv157(D181)
        elif  num281==58:
            p3812=self.conv158(D181)
        elif  num281==59:
            p3812=self.conv159(D181)
        elif  num281==60:
            p3812=self.conv160(D181)
        elif  num281==61:
            p3812=self.conv161(D181)
        elif  num281==62:
            p3812=self.conv162(D181)
        elif  num281==63:
            p3812=self.conv163(D181)
        elif  num281==64:
            p3812=self.conv164(D181)
        elif  num281==65:
            p3812=self.conv165(D181)
        elif  num281==66:
            p3812=self.conv166(D181)
        elif  num281==67:
            p3812=self.conv167(D181)
        elif  num281==68:
            p3812=self.conv168(D181)
        elif  num281==69:
            p3812=self.conv169(D181)
        elif  num281==70:
            p3812=self.conv170(D181)
        elif  num281==71:
            p3812=self.conv171(D181)
        elif  num281==72:
            p3812=self.conv172(D181)
        elif  num281==73:
            p3812=self.conv173(D181)
        elif  num281==74:
            p3812=self.conv174(D181)
        elif  num281==75:
            p3812=self.conv175(D181)
        elif  num281==76:
            p3812=self.conv176(D181)
        elif  num281==77:
            p3812=self.conv177(D181)
        elif  num281==78:
            p3812=self.conv178(D181)
        elif  num281==79:
            p3812=self.conv179(D181)
        elif  num281==80:
            p3812=self.conv180(D181)
        elif  num281==81:
            p3812=self.conv181(D181)
        elif  num281==82:
            p3812=self.conv182(D181)
        elif  num281==83:
            p3812=self.conv183(D181)
        elif  num281==84:
            p3812=self.conv184(D181)
        elif  num281==85:
            p3812=self.conv185(D181)
        elif  num281==86:
            p3812=self.conv186(D181)
        elif  num281==87:
            p3812=self.conv187(D181)
        elif  num281==88:
            p3812=self.conv188(D181)
        elif  num281==89:
            p3812=self.conv189(D181)    
        elif  num281==90:
            p3812=self.conv190(D181)
        elif  num281==91:
            p3812=self.conv191(D181)
        elif  num281==92:
            p3812=self.conv192(D181)
        elif  num281==93:
            p3812=self.conv193(D181)
        elif  num281==94:
            p3812=self.conv194(D181)
        elif  num281==95:
            p3812=self.conv195(D181)
        elif  num281==96:
            p3812=self.conv196(D181)
        elif  num281==97:
            p3812=self.conv197(D181)
        elif  num281==98:
            p3812=self.conv198(D181)
        elif  num281==99:
            p3812=self.conv199(D181) 
        elif  num281==100:
            p3812=self.conv1100(D181)
        elif  num281==101:
            p3812=self.conv1101(D181)
        elif  num281==102:
            p3812=self.conv1102(D181)
        elif  num281==103:
            p3812=self.conv1103(D181)
        elif  num281==104:
            p3812=self.conv1104(D181)
        elif  num281==105:
            p3812=self.conv1105(D181)
        elif  num281==106:
            p3812=self.conv1106(D181)
        elif  num281==107:
            p3812=self.conv1107(D181)
        elif  num281==108:
            p3812=self.conv1108(D181)
        elif  num281==109:
            p3812=self.conv1109(D181)
        elif  num281==110:
            p3812=self.conv1110(D181)
        elif  num281==111:
            p3812=self.conv1111(D181)
        elif  num281==112:
            p3812=self.conv1112(D181)
        elif  num281==113:
            p3812=self.conv1113(D181)
        elif  num281==114:
            p3812=self.conv1114(D181)
        elif  num281==115:
            p3812=self.conv1115(D181)
        elif  num281==116:
            p3812=self.conv1116(D181)
        elif  num281==117:
            p3812=self.conv1117(D181)
        elif  num281==118:
            p3812=self.conv1118(D181)
        elif  num281==119:
            p3812=self.conv1119(D181) 
        elif  num281==120:
            p3812=self.conv1120(D181)
        elif  num281==121:
            p3812=self.conv1121(D181)
        elif  num281==122:
            p3812=self.conv1122(D181)
        elif  num281==123:
            p3812=self.conv1123(D181)
        elif  num281==124:
            p3812=self.conv1124(D181)
        elif  num281==125:
            p3812=self.conv1125(D181)
        elif  num281==126:
            p3812=self.conv1126(D181)
        elif  num281==127:
            p3812=self.conv1127(D181)
        elif  num281==128:
            p3812=self.conv1128(D181)
        elif  num281==129:
            p3812=self.conv1129(D181) 
        elif  num281==130:
            p3812=self.conv1130(D181)
        elif  num281==131:
            p3812=self.conv1131(D181)
        elif  num281==132:
            p3812=self.conv1132(D181)
        elif  num281==133:
            p3812=self.conv1133(D181)
        elif  num281==134:
            p3812=self.conv1134(D181)
        elif  num281==135:
            p3812=self.conv1135(D181)
        elif  num281==136:
            p3812=self.conv1136(D181)
        elif  num281==137:
            p3812=self.conv1137(D181)
        elif  num281==138:
            p3812=self.conv1138(D181)
        elif  num281==139:
            p3812=self.conv1139(D181)
        elif  num281==140:
            p3812=self.conv1140(D181)
        elif  num281==141:
            p3812=self.conv1141(D181)
        elif  num281==142:
            p3812=self.conv1142(D181)
        elif  num281==143:
            p3812=self.conv1143(D181)
        elif  num281==144:
            p3812=self.conv1144(D181)
        elif  num281==145:
            p3812=self.conv1145(D181)
        elif  num281==146:
            p3812=self.conv1146(D181)
        elif  num281==147:
            p3812=self.conv1147(D181)
        elif  num281==148:
            p3812=self.conv1148(D181)
        elif  num281==149:
            p3812=self.conv1149(D181) 
        elif  num281==150:
            p3812=self.conv1150(D181)
        elif  num281==151:
            p3812=self.conv1151(D181)
        elif  num281==152:
            p3812=self.conv1152(D181)
        elif  num281==153:
            p3812=self.conv1153(D181)
        elif  num281==154:
            p3812=self.conv1154(D181)
        elif  num281==155:
            p3812=self.conv1155(D181)
        elif  num281==156:
            p3812=self.conv1156(D181)
        elif  num281==157:
            p3812=self.conv1157(D181)
        elif  num281==158:
            p3812=self.conv1158(D181)
        elif  num281==159:
            p3812=self.conv1159(D181) 
        elif  num281==160:
            p3812=self.conv1160(D181)
        elif  num281==161:
            p3812=self.conv1161(D181)
        elif  num281==162:
            p3812=self.conv1162(D181)
        elif  num281==163:
            p3812=self.conv1163(D181)
        elif  num281==164:
            p3812=self.conv1164(D181)
        elif  num281==165:
            p3812=self.conv1165(D181)
        elif  num281==166:
            p3812=self.conv1166(D181)
        elif  num281==167:
            p3812=self.conv1167(D181)
        elif  num281==168:
            p3812=self.conv1168(D181)
        elif  num281==169:
            p3812=self.conv1169(D181) 
        elif  num281==170:
            p3812=self.conv1170(D181)
        elif  num281==171:
            p3812=self.conv1171(D181)
        elif  num281==172:
            p3812=self.conv1172(D181)
        elif  num281==173:
            p3812=self.conv1173(D181)
        elif  num281==174:
            p3812=self.conv1174(D181)
        elif  num281==175:
            p3812=self.conv1175(D181)
        elif  num281==176:
            p3812=self.conv1176(D181)
        elif  num281==177:
            p3812=self.conv1177(D181)
        elif  num281==178:
            p3812=self.conv1178(D181)
        elif  num281==179:
            p3812=self.conv1179(D181)                                                                                              
        elif  num281==180:
            p3812=self.conv1180(D181)
        elif  num281==181:
            p3812=self.conv1181(D181)
        elif  num281==182:
            p3812=self.conv1182(D181)
        elif  num281==183:
            p3812=self.conv1183(D181)
        elif  num281==184:
            p3812=self.conv1184(D181)
        elif  num281==185:
            p3812=self.conv1185(D181)
        elif  num281==186:
            p3812=self.conv1186(D181)
        elif  num281==187:
            p3812=self.conv1187(D181)
        elif  num281==188:
            p3812=self.conv1188(D181)
        elif  num281==189:
            p3812=self.conv1189(D181) 
        elif  num281==190:
            p3812=self.conv1190(D181)
        elif  num281==191:
            p3812=self.conv1191(D181)
        elif  num281==192:
            p3812=self.conv1192(D181)
        elif  num281==193:
            p3812=self.conv1193(D181)
        elif  num281==194:
            p3812=self.conv1194(D181)
        elif  num281==195:
            p3812=self.conv1195(D181)
        elif  num281==196:
            p3812=self.conv1196(D181)
        elif  num281==197:
            p3812=self.conv1197(D181)
        elif  num281==198:
            p3812=self.conv1198(D181)
        elif  num281==199:
            p3812=self.conv1199(D181)
        elif  num281==200:
            p3812=self.conv1200(D181)
        elif  num281==201:
            p3812=self.conv1201(D181)
        elif  num281==202:
            p3812=self.conv1202(D181)
        elif  num281==203:
            p3812=self.conv1203(D181)
        elif  num281==204:
            p3812=self.conv1204(D181)
        elif  num281==205:
            p3812=self.conv1205(D181)
        elif  num281==206:
            p3812=self.conv1206(D181)
        elif  num281==207:
            p3812=self.conv1207(D181)
        elif  num281==208:
            p3812=self.conv1208(D181)
        elif  num281==209:
            p3812=self.conv1209(D181)
        elif  num281==210:
            p3812=self.conv1210(D181)
        elif  num281==211:
            p3812=self.conv1211(D181)
        elif  num281==212:
            p3812=self.conv1212(D181)
        elif  num281==213:
            p3812=self.conv1213(D181)
        elif  num281==214:
            p3812=self.conv1214(D181)
        elif  num281==215:
            p3812=self.conv1215(D181)
        elif  num281==216:
            p3812=self.conv1216(D181)
        elif  num281==217:
            p3812=self.conv1217(D181)
        elif  num281==218:
            p3812=self.conv1218(D181)
        elif  num281==219:
            p3812=self.conv1219(D181)
        elif  num281==220:
            p3812=self.conv1220(D181)
        elif  num281==221:
            p3812=self.conv1221(D181)
        elif  num281==222:
            p3812=self.conv1222(D181)
        elif  num281==223:
            p3812=self.conv1223(D181)
        elif  num281==224:
            p3812=self.conv1224(D181)
        elif  num281==225:
            p3812=self.conv1225(D181)
        elif  num281==226:
            p3812=self.conv1226(D181)
        elif  num281==227:
            p3812=self.conv1227(D181)
        elif  num281==228:
            p3812=self.conv1228(D181)
        elif  num281==229:
            p3812=self.conv1229(D181)
        elif  num281==230:
            p3812=self.conv1230(D181)
        elif  num281==231:
            p3812=self.conv1231(D181)
        elif  num281==232:
            p3812=self.conv1232(D181)
        elif  num281==233:
            p3812=self.conv1233(D181)
        elif  num281==234:
            p3812=self.conv1234(D181)
        elif  num281==235:
            p3812=self.conv1235(D181)
        elif  num281==236:
            p3812=self.conv1236(D181)
        elif  num281==237:
            p3812=self.conv1237(D181)
        elif  num281==238:
            p3812=self.conv1238(D181)
        elif  num281==239:
            p3812=self.conv1239(D181) 
        elif  num281==240:
            p3812=self.conv1240(D181)
        elif  num281==241:
            p3812=self.conv1241(D181)
        elif  num281==242:
            p3812=self.conv1242(D181)
        elif  num281==243:
            p3812=self.conv1243(D181)
        elif  num281==244:
            p3812=self.conv1244(D181)
        elif  num281==245:
            p3812=self.conv1245(D181)
        elif  num281==246:
            p3812=self.conv1246(D181)
        elif  num281==247:
            p3812=self.conv1247(D181)
        elif  num281==248:
            p3812=self.conv1248(D181)
        elif  num281==249:
            p3812=self.conv1249(D181)
        elif  num281==250:
            p3812=self.conv1250(D181)
        elif  num281==251:
            p3812=self.conv1251(D181)
        elif  num281==252:
            p3812=self.conv1252(D181)
        elif  num281==253:
            p3812=self.conv1253(D181)
        elif  num281==254:
            p3812=self.conv1254(D181)
        elif  num281==255:
            p3812=self.conv1255(D181)
        elif  num281==256:
            p3812=self.conv1256(D181)
        elif  num281==257:
            p3812=self.conv1257(D181)
        elif  num281==258:
            p3812=self.conv1258(D181)
        elif  num281==259:
            p3812=self.conv1259(D181)
        elif  num281==260:
            p3812=self.conv1260(D181)
        elif  num281==261:
            p3812=self.conv1261(D181)
        elif  num281==262:
            p3812=self.conv1262(D181)
        elif  num281==263:
            p3812=self.conv1263(D181)
        elif  num281==264:
            p3812=self.conv1264(D181)
        elif  num281==265:
            p3812=self.conv1265(D181)
        elif  num281==266:
            p3812=self.conv1266(D181)
        elif  num281==267:
            p3812=self.conv1267(D181)
        elif  num281==268:
            p3812=self.conv1268(D181)
        elif  num281==269:
            p3812=self.conv1269(D181) 
        elif  num281==270:
            p3812=self.conv1270(D181)
        elif  num281==271:
            p3812=self.conv1271(D181)
        elif  num281==272:
            p3812=self.conv1272(D181)
        elif  num281==273:
            p3812=self.conv1273(D181)
        elif  num281==274:
            p3812=self.conv1274(D181)
        elif  num281==275:
            p3812=self.conv1275(D181)
        elif  num281==276:
            p3812=self.conv1276(D181)
        elif  num281==277:
            p3812=self.conv1277(D181)
        elif  num281==278:
            p3812=self.conv1278(D181)
        elif  num281==279:
            p3812=self.conv1279(D181)
        elif  num281==280:
            p3812=self.conv1280(D181)
        elif  num281==281:
            p3812=self.conv1281(D181)
        elif  num281==282:
            p3812=self.conv1282(D181)
        elif  num281==283:
            p3812=self.conv1283(D181)
        elif  num281==284:
            p3812=self.conv1284(D181)
        elif  num281==285:
            p3812=self.conv1285(D181)
        elif  num281==286:
            p3812=self.conv1286(D181)
        elif  num281==287:
            p3812=self.conv1287(D181)
        elif  num281==288:
            p3812=self.conv1288(D181)
        elif  num281==289:
            p3812=self.conv1289(D181) 
        elif  num281==290:
            p3812=self.conv1290(D181)
        elif  num281==291:
            p3812=self.conv1291(D181)
        elif  num281==292:
            p3812=self.conv1292(D181)
        elif  num281==293:
            p3812=self.conv1293(D181)
        elif  num281==294:
            p3812=self.conv1294(D181)
        elif  num281==295:
            p3812=self.conv1295(D181)
        elif  num281==296:
            p3812=self.conv1296(D181)
        elif  num281==297:
            p3812=self.conv1297(D181)
        elif  num281==298:
            p3812=self.conv1298(D181)
        elif  num281==299:
            p3812=self.conv1299(D181)
        elif  num281==300:
            p3812=self.conv1300(D181)
        elif  num281==301:
            p3812=self.conv1301(D181)
        elif  num281==302:
            p3812=self.conv1302(D181)
        elif  num281==303:
            p3812=self.conv1303(D181)
        elif  num281==304:
            p3812=self.conv1304(D181)
        elif  num281==305:
            p3812=self.conv1305(D181)
        elif  num281==306:
            p3812=self.conv1306(D181)
        elif  num281==307:
            p3812=self.conv1307(D181)
        elif  num281==308:
            p3812=self.conv1308(D181)
        elif  num281==309:
            p3812=self.conv1309(D181) 
        elif  num281==310:
            p3812=self.conv1310(D181)
        elif  num281==311:
            p3812=self.conv1311(D181)
        elif  num281==312:
            p3812=self.conv1312(D181)
        elif  num281==313:
            p3812=self.conv1313(D181)
        elif  num281==314:
            p3812=self.conv1314(D181)
        elif  num281==315:
            p3812=self.conv1315(D181)
        elif  num281==316:
            p3812=self.conv1316(D181)
        elif  num281==317:
            p3812=self.conv1317(D181)
        elif  num281==318:
            p3812=self.conv1318(D181)
        elif  num281==319:
            p3812=self.conv1319(D181)
        elif  num281==320:
            p3812=self.conv1320(D181)
        elif  num281==321:
            p3812=self.conv1321(D181)
        elif  num281==322:
            p3812=self.conv1322(D181)
        elif  num281==323:
            p3812=self.conv1323(D181)
        elif  num281==324:
            p3812=self.conv1324(D181)
        elif  num281==325:
            p3812=self.conv1325(D181)
        elif  num281==326:
            p3812=self.conv1326(D181)
        elif  num281==327:
            p3812=self.conv1327(D181)
        elif  num281==328:
            p3812=self.conv1328(D181)
        elif  num281==329:
            p3812=self.conv1329(D181)
        elif  num281==330:
            p3812=self.conv1330(D181)
        elif  num281==331:
            p3812=self.conv1331(D181)
        elif  num281==332:
            p3812=self.conv1332(D181)
        elif  num281==333:
            p3812=self.conv1333(D181)
        elif  num281==334:
            p3812=self.conv1334(D181)
        elif  num281==335:
            p3812=self.conv1335(D181)
        elif  num281==336:
            p3812=self.conv1336(D181)
        elif  num281==337:
            p3812=self.conv1337(D181)
        elif  num281==338:
            p3812=self.conv1338(D181)
        elif  num281==339:
            p3812=self.conv1339(D181)
        elif  num281==340:
            p3812=self.conv1340(D181)
        elif  num281==341:
            p3812=self.conv1341(D181)
        elif  num281==342:
            p3812=self.conv1342(D181)
        elif  num281==343:
            p3812=self.conv1343(D181)
        elif  num281==344:
            p3812=self.conv1344(D181)
        elif  num281==345:
            p3812=self.conv1345(D181)
        elif  num281==346:
            p3812=self.conv1346(D181)
        elif  num281==347:
            p3812=self.conv1347(D181)
        elif  num281==348:
            p3812=self.conv1348(D181)
        elif  num281==349:
            p3812=self.conv1349(D181)
        elif  num281==350:
            p3812=self.conv1350(D181)
        elif  num281==351:
            p3812=self.conv1351(D181)
        elif  num281==352:
            p3812=self.conv1352(D181)
        elif  num281==353:
            p3812=self.conv1335(D181)
        elif  num281==354:
            p3812=self.conv1354(D181)
        elif  num281==355:
            p3812=self.conv1355(D181)
        elif  num281==356:
            p3812=self.conv1356(D181)
        elif  num281==357:
            p3812=self.conv1357(D181)
        elif  num281==358:
            p3812=self.conv1358(D181)
        elif  num281==359:
            p3812=self.conv1359(D181) 
        elif  num281==360:
            p3812=self.conv1360(D181)
        elif  num281==361:
            p3812=self.conv1361(D181)
        elif  num281==362:
            p3812=self.conv1362(D181)
        elif  num281==363:
            p3812=self.conv1363(D181)
        elif  num281==364:
            p3812=self.conv1364(D181)
        elif  num281==365:
            p3812=self.conv1365(D181)
        elif  num281==366:
            p3812=self.conv1366(D181)
        elif  num281==367:
            p3812=self.conv1367(D181)
        elif  num281==368:
            p3812=self.conv1368(D181)
        elif  num281==369:
            p3812=self.conv1369(D181) 
        elif  num281==370:
            p3812=self.conv1370(D181)
        elif  num281==371:
            p3812=self.conv1371(D181)
        elif  num281==372:
            p3812=self.conv1372(D181)
        elif  num281==373:
            p3812=self.conv1373(D181)
        elif  num281==374:
            p3812=self.conv1374(D181)
        elif  num281==375:
            p3812=self.conv1375(D181)
        elif  num281==376:
            p3812=self.conv1376(D181)
        elif  num281==377:
            p3812=self.conv1377(D181)
        elif  num281==378:
            p3812=self.conv1378(D181)
        elif  num281==379:
            p3812=self.conv1379(D181) 
        elif  num281==380:
            p3812=self.conv1380(D181)
        elif  num281==381:
            p3812=self.conv1381(D181)
        elif  num281==382:
            p3812=self.conv1382(D181)
        elif  num281==383:
            p3812=self.conv1383(D181)
        elif  num281==384:
            p3812=self.conv1384(D181)
        elif  num281==385:
            p3812=self.conv1385(D181)
        elif  num281==386:
            p3812=self.conv1386(D181)
        elif  num281==387:
            p3812=self.conv1387(D181)
        elif  num281==388:
            p3812=self.conv1388(D181)
        elif  num281==389:
            p3812=self.conv1389(D181) 
        elif  num281==390:
            p3812=self.conv1390(D181)
        elif  num281==391:
            p3812=self.conv1391(D181)
        elif  num281==392:
            p3812=self.conv1392(D181)
        elif  num281==393:
            p3812=self.conv1393(D181)
        elif  num281==394:
            p3812=self.conv1394(D181)
        elif  num281==395:
            p3812=self.conv1395(D181)
        elif  num281==396:
            p3812=self.conv1396(D181)
        elif  num281==397:
            p3812=self.conv1397(D181)
        elif  num281==398:
            p3812=self.conv1398(D181)
        elif  num281==399:
            p3812=self.conv1399(D181)
        elif  num281==400:
            p3812=self.conv1400(D181)
        elif  num281==401:
            p3812=self.conv1401(D181)
        elif  num281==402:
            p3812=self.conv1402(D181)
        elif  num281==403:
            p3812=self.conv1403(D181)
        elif  num281==404:
            p3812=self.conv1404(D181)
        elif  num281==405:
            p3812=self.conv1405(D181)
        elif  num281==406:
            p3812=self.conv1406(D181)
        elif  num281==407:
            p3812=self.conv1407(D181)
        elif  num281==408:
            p3812=self.conv1408(D181)
        elif  num281==409:
            p3812=self.conv1409(D181)
        elif  num281==410:
            p3812=self.conv1410(D181)
        elif  num281==411:
            p3812=self.conv1411(D181)
        elif  num281==412:
            p3812=self.conv1412(D181)
        elif  num281==413:
            p3812=self.conv1413(D181)
        elif  num281==414:
            p3812=self.conv1414(D181)
        elif  num281==415:
            p3812=self.conv145(D181)
        elif  num281==416:
            p3812=self.conv1416(D181)
        elif  num281==417:
            p3812=self.conv1417(D181)
        elif  num281==418:
            p3812=self.conv1418(D181)
        elif  num281==419:
            p3812=self.conv1419(D181) 
        elif  num281==420:
            p3812=self.conv1420(D181)
        elif  num281==421:
            p3812=self.conv1421(D181)
        elif  num281==422:
            p3812=self.conv1422(D181)
        elif  num281==423:
            p3812=self.conv1423(D181)
        elif  num281==424:
            p3812=self.conv1424(D181)
        elif  num281==425:
            p3812=self.conv1425(D181)
        elif  num281==426:
            p3812=self.conv1426(D181)
        elif  num281==427:
            p3812=self.conv1427(D181)
        elif  num281==428:
            p3812=self.conv1428(D181)
        elif  num281==429:
            p3812=self.conv1429(D181) 
        elif  num281==430:
            p3812=self.conv1430(D181)
        elif  num281==431:
            p3812=self.conv1431(D181)
        elif  num281==432:
            p3812=self.conv1432(D181)
        elif  num281==433:
            p3812=self.conv1433(D181)
        elif  num281==434:
            p3812=self.conv1434(D181)
        elif  num281==435:
            p3812=self.conv1435(D181)
        elif  num281==436:
            p3812=self.conv1436(D181)
        elif  num281==437:
            p3812=self.conv1437(D181)
        elif  num281==438:
            p3812=self.conv1438(D181)
        elif  num281==439:
            p3812=self.conv1439(D181)
        elif  num281==440:
            p3812=self.conv1440(D181)
        elif  num281==441:
            p3812=self.conv1441(D181)
        elif  num281==442:
            p3812=self.conv1442(D181)
        elif  num281==443:
            p3812=self.conv1443(D181)
        elif  num281==444:
            p3812=self.conv1444(D181)
        elif  num281==445:
            p3812=self.conv1445(D181)
        elif  num281==446:
            p3812=self.conv1446(D181)
        elif  num281==447:
            p3812=self.conv1447(D181)
        elif  num281==448:
            p3812=self.conv1448(D181)
        elif  num281==449:
            p3812=self.conv1449(D181)
        elif  num281==450:
            p3812=self.conv1450(D181)
        elif  num281==451:
            p3812=self.conv1451(D181)
        elif  num281==452:
            p3812=self.conv1452(D181)
        elif  num281==453:
            p3812=self.conv1453(D181)
        elif  num281==454:
            p3812=self.conv1454(D181)
        elif  num281==455:
            p3812=self.conv1455(D181)
        elif  num281==456:
            p3812=self.conv1456(D181)
        elif  num281==457:
            p3812=self.conv1457(D181)
        elif  num281==458:
            p3812=self.conv1458(D181)
        elif  num281==459:
            p3812=self.conv1459(D181)
        elif  num281==460:
            p3812=self.conv1460(D181)
        elif  num281==461:
            p3812=self.conv1461(D181)
        elif  num281==462:
            p3812=self.conv1462(D181)
        elif  num281==463:
            p3812=self.conv1463(D181)
        elif  num281==464:
            p3812=self.conv1464(D181)
        elif  num281==465:
            p3812=self.conv1465(D181)
        elif  num281==466:
            p3812=self.conv1466(D181)
        elif  num281==467:
            p3812=self.conv1467(D181)
        elif  num281==468:
            p3812=self.conv1468(D181)
        elif  num281==469:
            p3812=self.conv1469(D181) 
        elif  num281==470:
            p3812=self.conv1470(D181)
        elif  num281==471:
            p3812=self.conv1471(D181)
        elif  num281==472:
            p3812=self.conv1472(D181)
        elif  num281==473:
            p3812=self.conv1473(D181)
        elif  num281==474:
            p3812=self.conv1474(D181)
        elif  num281==475:
            p3812=self.conv1475(D181)
        elif  num281==476:
            p3812=self.conv1476(D181)
        elif  num281==477:
            p3812=self.conv1477(D181)
        elif  num281==478:
            p3812=self.conv1478(D181)
        elif  num281==479:
            p3812=self.conv1479(D181)
        elif  num281==480:
            p3812=self.conv1480(D181)
        elif  num281==481:
            p3812=self.conv1481(D181)
        elif  num281==482:
            p3812=self.conv1482(D181)
        elif  num281==483:
            p3812=self.conv1483(D181)
        elif  num281==484:
            p3812=self.conv1484(D181)
        elif  num281==485:
            p3812=self.conv1485(D181)
        elif  num281==486:
            p3812=self.conv1486(D181)
        elif  num281==487:
            p3812=self.conv1487(D181)
        elif  num281==488:
            p3812=self.conv1488(D181)
        elif  num281==489:
            p3812=self.conv1489(D181)
        elif  num281==490:
            p3812=self.conv1490(D181)
        elif  num281==491:
            p3812=self.conv1491(D181)
        elif  num281==492:
            p3812=self.conv1492(D181)
        elif  num281==493:
            p3812=self.conv1493(D181)
        elif  num281==494:
            p3812=self.conv1494(D181)
        elif  num281==495:
            p3812=self.conv1495(D181)
        elif  num281==496:
            p3812=self.conv1496(D181)
        elif  num281==497:
            p3812=self.conv1497(D181)
        elif  num281==498:
            p3812=self.conv1498(D181)
        elif  num281==499:
            p3812=self.conv1499(D181)
        elif  num281==500:
            p3812=self.conv1500(D181)
        elif  num281==501:
            p3812=self.conv1501(D181)
        elif  num281==502:
            p3812=self.conv1502(D181)
        elif  num281==503:
            p3812=self.conv1503(D181)
        elif  num281==504:
            p3812=self.conv1504(D181)
        elif  num281==505:
            p3812=self.conv1505(D181)
        elif  num281==506:
            p3812=self.conv1506(D181)
        elif  num281==507:
            p3812=self.conv1507(D181)
        elif  num281==508:
            p3812=self.conv1508(D181)
        elif  num281==509:
            p3812=self.conv1509(D181)
        elif  num281==510:
            p3812=self.conv1510(D181)
        elif  num281==511:
            p3812=self.conv1511(D181)
        elif  num281==512:
            p3812=self.conv1512(D181)
        elif  num281==513:
            p3812=self.conv1513(D181)
        elif  num281==514:
            p3812=self.conv1514(D181)
        elif  num281==515:
            p3812=self.conv1515(D181)
        elif  num281==516:
            p3812=self.conv1516(D181)
        elif  num281==517:
            p3812=self.conv1517(D181)
        elif  num281==518:
            p3812=self.conv1518(D181)
        elif  num281==519:
            p3812=self.conv1519(D181)
        elif  num281==520:
            p3812=self.conv1520(D181)
        elif  num281==521:
            p3812=self.conv1521(D181)
        elif  num281==522:
            p3812=self.conv1522(D181)
        elif  num281==523:
            p3812=self.conv1523(D181)
        elif  num281==524:
            p3812=self.conv1524(D181)
        elif  num281==525:
            p3812=self.conv1525(D181)
        elif  num281==526:
            p3812=self.conv1526(D181)
        elif  num281==527:
            p3812=self.conv1527(D181)
        elif  num281==528:
            p3812=self.conv1528(D181)
        elif  num281==529:
            p3812=self.conv1529(D181)
        elif  num281==530:
            p3812=self.conv1530(D181)
        elif  num281==531:
            p3812=self.conv1531(D181)
        elif  num281==532:
            p3812=self.conv1532(D181)
        elif  num281==533:
            p3812=self.conv1533(D181)
        elif  num281==534:
            p3812=self.conv1534(D181)
        elif  num281==535:
            p3812=self.conv1535(D181)
        elif  num281==536:
            p3812=self.conv1536(D181)
        elif  num281==537:
            p3812=self.conv1537(D181)
        elif  num281==538:
            p3812=self.conv1538(D181)
        elif  num281==539:
            p3812=self.conv1539(D181)
        elif  num281==540:
            p3812=self.conv1540(D181)
        elif  num281==541:
            p3812=self.conv1541(D181)
        elif  num281==542:
            p3812=self.conv1542(D181)
        elif  num281==543:
            p3812=self.conv1543(D181)
        elif  num281==544:
            p3812=self.conv1544(D181)
        elif  num281==545:
            p3812=self.conv1545(D181)
        elif  num281==546:
            p3812=self.conv1546(D181)
        elif  num281==547:
            p3812=self.conv1547(D181)
        elif  num281==548:
            p3812=self.conv1548(D181)
        elif  num281==549:
            p3812=self.conv1549(D181) 
        elif  num281==550:
            p3812=self.conv1550(D181)
        elif  num281==551:
            p3812=self.conv1551(D181)
        elif  num281==552:
            p3812=self.conv1552(D181)
        elif  num281==553:
            p3812=self.conv1553(D181)
        elif  num281==554:
            p3812=self.conv1554(D181)
        elif  num281==555:
            p3812=self.conv1555(D181)
        elif  num281==556:
            p3812=self.conv1556(D181)
        elif  num281==557:
            p3812=self.conv1557(D181)
        elif  num281==558:
            p3812=self.conv1558(D181)
        elif  num281==559:
            p3812=self.conv1559(D181)
        elif  num281==560:
            p3812=self.conv1560(D181)
        elif  num281==561:
            p3812=self.conv1561(D181)
        elif  num281==562:
            p3812=self.conv1562(D181)
        elif  num281==563:
            p3812=self.conv1563(D181)
        elif  num281==564:
            p3812=self.conv1564(D181)
        elif  num281==565:
            p3812=self.conv1565(D181)
        elif  num281==566:
            p3812=self.conv1566(D181)
        elif  num281==567:
            p3812=self.conv1567(D181)
        elif  num281==568:
            p3812=self.conv1568(D181)
        elif  num281==569:
            p3812=self.conv1569(D181) 
        elif  num281==570:
            p3812=self.conv1570(D181)
        elif  num281==571:
            p3812=self.conv1571(D181)
        elif  num281==572:
            p3812=self.conv1572(D181)
        elif  num281==573:
            p3812=self.conv1573(D181)
        elif  num281==574:
            p3812=self.conv1574(D181)
        elif  num281==575:
            p3812=self.conv1575(D181)
        elif  num281==576:
            p3812=self.conv1576(D181)
        elif  num281==577:
            p3812=self.conv1577(D181)
        elif  num281==578:
            p3812=self.conv1578(D181)
        elif  num281==579:
            p3812=self.conv1579(D181) 
        elif  num281==580:
            p3812=self.conv1580(D181)
        elif  num281==581:
            p3812=self.conv1581(D181)
        elif  num281==582:
            p3812=self.conv1582(D181)
        elif  num281==583:
            p3812=self.conv1583(D181)
        elif  num281==584:
            p3812=self.conv1584(D181)
        elif  num281==585:
            p3812=self.conv1585(D181)
        elif  num281==586:
            p3812=self.conv1586(D181)
        elif  num281==587:
            p3812=self.conv1587(D181)
        elif  num281==588:
            p3812=self.conv1588(D181)
        elif  num281==589:
            p3812=self.conv1589(D181)
        elif  num281==590:
            p3812=self.conv1590(D181)
        elif  num281==591:
            p3812=self.conv1591(D181)
        elif  num281==592:
            p3812=self.conv1592(D181)
        elif  num281==593:
            p3812=self.conv1593(D181)
        elif  num281==594:
            p3812=self.conv1594(D181)
        elif  num281==595:
            p3812=self.conv1595(D181)
        elif  num281==596:
            p3812=self.conv1596(D181)
        elif  num281==597:
            p3812=self.conv1597(D181)
        elif  num281==598:
            p3812=self.conv1598(D181)
        elif  num281==599:
            p3812=self.conv1599(D181)
        elif  num281==600:
            p3812=self.conv1600(D181)
        elif  num281==601:
            p3812=self.conv1601(D181)
        elif  num281==602:
            p3812=self.conv1602(D181)
        elif  num281==603:
            p3812=self.conv1603(D181)
        elif  num281==604:
            p3812=self.conv1604(D181)
        elif  num281==605:
            p3812=self.conv1605(D181)
        elif  num281==606:
            p3812=self.conv1606(D181)
        elif  num281==607:
            p3812=self.conv1607(D181)
        elif  num281==608:
            p3812=self.conv1608(D181)
        elif  num281==609:
            p3812=self.conv1609(D181)                                                                                                                         
        elif  num281==610:
            p3812=self.conv1610(D181)
        elif  num281==611:
            p3812=self.conv1611(D181)
        elif  num281==612:
            p3812=self.conv1612(D181)
        elif  num281==613:
            p3812=self.conv1613(D181)
        elif  num281==614:
            p3812=self.conv1614(D181)
        elif  num281==615:
            p3812=self.conv1615(D181)
        elif  num281==616:
            p3812=self.conv1616(D181)
        elif  num281==617:
            p3812=self.conv1617(D181)
        elif  num281==618:
            p3812=self.conv1618(D181)
        elif  num281==619:
            p3812=self.conv1619(D181)                                                                                                                          
        elif  num281==620:
            p3812=self.conv1620(D181)
        elif  num281==621:
            p3812=self.conv1621(D181)
        elif  num281==622:
            p3812=self.conv1622(D181)
        elif  num281==623:
            p3812=self.conv1623(D181)
        elif  num281==624:
            p3812=self.conv1624(D181)
        elif  num281==625:
            p3812=self.conv1625(D181)
        elif  num281==626:
            p3812=self.conv1626(D181)
        elif  num281==627:
            p3812=self.conv1627(D181)
        elif  num281==628:
            p3812=self.conv1628(D181)
        elif  num281==629:
            p3812=self.conv1629(D181)  
        elif  num281==630:
            p3812=self.conv1630(D181)
        elif  num281==631:
            p3812=self.conv1631(D181)
        elif  num281==632:
            p3812=self.conv1632(D181)
        elif  num281==633:
            p3812=self.conv1633(D181)
        elif  num281==634:
            p3812=self.conv1634(D181)
        elif  num281==635:
            p3812=self.conv1635(D181)
        elif  num281==636:
            p3812=self.conv1636(D181)
        elif  num281==637:
            p3812=self.conv1637(D181)
        elif  num281==638:
            p3812=self.conv1638(D181)
        elif  num281==639:
            p3812=self.conv1639(D181)                                                                                                                          
        elif  num281==640:
            p3812=self.conv1640(D181)
        elif  num281==641:
            p3812=self.conv1641(D181)
        elif  num281==642:
            p3812=self.conv1642(D181)
        elif  num281==643:
            p3812=self.conv1643(D181)
        elif  num281==644:
            p3812=self.conv1644(D181)
        elif  num281==645:
            p3812=self.conv1645(D181)
        elif  num281==646:
            p3812=self.conv1646(D181)
        elif  num281==647:
            p3812=self.conv1647(D181)
        elif  num281==648:
            p3812=self.conv1648(D181)
        elif  num281==649:
            p3812=self.conv1649(D181)                                                                                                                          
        elif  num281==650:
            p3812=self.conv1650(D181)
        elif  num281==651:
            p3812=self.conv1651(D181)
        elif  num281==652:
            p3812=self.conv1652(D181)
        elif  num281==653:
            p3812=self.conv1653(D181)
        elif  num281==654:
            p3812=self.conv1654(D181)
        elif  num281==655:
            p3812=self.conv1655(D181)
        elif  num281==656:
            p3812=self.conv1656(D181)
        elif  num281==657:
            p3812=self.conv1657(D181)
        elif  num281==658:
            p3812=self.conv1658(D181)
        elif  num281==659:
            p3812=self.conv1659(D181)
        elif  num281==660:
            p3812=self.conv1660(D181)
        elif  num281==661:
            p3812=self.conv1661(D181)
        elif  num281==662:
            p3812=self.conv1662(D181)
        elif  num281==663:
            p3812=self.conv1663(D181)
        elif  num281==664:
            p3812=self.conv1664(D181)
        elif  num281==665:
            p3812=self.conv1665(D181)
        elif  num281==666:
            p3812=self.conv1666(D181)
        elif  num281==667:
            p3812=self.conv1667(D181)
        elif  num281==668:
            p3812=self.conv1668(D181)
        elif  num281==669:
            p3812=self.conv1669(D181) 
        elif  num281==670:
            p3812=self.conv1670(D181)
        elif  num281==671:
            p3812=self.conv1671(D181)
        elif  num281==672:
            p3812=self.conv1672(D181)
        elif  num281==673:
            p3812=self.conv1673(D181)
        elif  num281==674:
            p3812=self.conv1674(D181)
        elif  num281==675:
            p3812=self.conv1675(D181)
        elif  num281==676:
            p3812=self.conv1676(D181)
        elif  num281==677:
            p3812=self.conv1677(D181)
        elif  num281==678:
            p3812=self.conv1678(D181)
        elif  num281==679:
            p3812=self.conv1679(D181)
        elif  num281==680:
            p3812=self.conv1680(D181)
        elif  num281==681:
            p3812=self.conv1681(D181)
        elif  num281==682:
            p3812=self.conv1682(D181)
        elif  num281==683:
            p3812=self.conv1683(D181)
        elif  num281==684:
            p3812=self.conv1684(D181)
        elif  num281==685:
            p3812=self.conv1685(D181)
        elif  num281==686:
            p3812=self.conv1686(D181)
        elif  num281==687:
            p3812=self.conv1687(D181)
        elif  num281==688:
            p3812=self.conv1688(D181)
        elif  num281==689:
            p3812=self.conv1689(D181)
        elif  num281==690:
            p3812=self.conv1690(D181)
        elif  num281==691:
            p3812=self.conv1691(D181)
        elif  num281==692:
            p3812=self.conv1692(D181)
        elif  num281==693:
            p3812=self.conv1693(D181)
        elif  num281==694:
            p3812=self.conv1694(D181)
        elif  num281==695:
            p3812=self.conv1695(D181)
        elif  num281==696:
            p3812=self.conv1696(D181)
        elif  num281==697:
            p3812=self.conv1697(D181)
        elif  num281==698:
            p3812=self.conv1698(D181)
        elif  num281==699:
            p3812=self.conv1699(D181)
        elif  num281==700:
            p3812=self.conv1700(D181)
        elif  num281==701:
            p3812=self.conv1701(D181)
        elif  num281==702:
            p3812=self.conv1702(D181)
        elif  num281==703:
            p3812=self.conv1703(D181)
        elif  num281==704:
            p3812=self.conv1704(D181)
        elif  num281==705:
            p3812=self.conv1705(D181)
        elif  num281==706:
            p3812=self.conv1706(D181)
        elif  num281==707:
            p3812=self.conv1707(D181)
        elif  num281==708:
            p3812=self.conv1708(D181)
        elif  num281==709:
            p3812=self.conv1709(D181)
        elif  num281==710:
            p3812=self.conv1710(D181)
        elif  num281==711:
            p3812=self.conv1711(D181)
        elif  num281==712:
            p3812=self.conv1712(D181)
        elif  num281==713:
            p3812=self.conv1713(D181)
        elif  num281==714:
            p3812=self.conv1714(D181)
        elif  num281==715:
            p3812=self.conv1715(D181)
        elif  num281==716:
            p3812=self.conv1716(D181)
        elif  num281==717:
            p3812=self.conv1717(D181)
        elif  num281==718:
            p3812=self.conv1718(D181)
        elif  num281==719:
            p3812=self.conv1719(D181)
        elif  num281==720:
            p3812=self.conv1720(D181)
        elif  num281==721:
            p3812=self.conv1721(D181)
        elif  num281==722:
            p3812=self.conv1722(D181)
        elif  num281==723:
            p3812=self.conv1723(D181)
        elif  num281==724:
            p3812=self.conv1724(D181)
        elif  num281==725:
            p3812=self.conv1725(D181)
        elif  num281==726:
            p3812=self.conv1726(D181)
        elif  num281==727:
            p3812=self.conv1727(D181)
        elif  num281==728:
            p3812=self.conv1728(D181)
        elif  num281==729:
            p3812=self.conv1729(D181)
        elif  num281==730:
            p3812=self.conv1730(D181)
        elif  num281==731:
            p3812=self.conv1731(D181)
        elif  num281==732:
            p3812=self.conv1732(D181)
        elif  num281==733:
            p3812=self.conv1733(D181)
        elif  num281==734:
            p3812=self.conv1734(D181)
        elif  num281==735:
            p3812=self.conv1735(D181)
        elif  num281==736:
            p3812=self.conv1736(D181)
        elif  num281==737:
            p3812=self.conv1737(D181)
        elif  num281==738:
            p3812=self.conv1738(D181)
        elif  num281==739:
            p3812=self.conv1739(D181)                                                                                                                          
        elif  num281==740:
            p3812=self.conv1740(D181)
        elif  num281==741:
            p3812=self.conv1741(D181)
        elif  num281==742:
            p3812=self.conv1742(D181)
        elif  num281==743:
            p3812=self.conv1743(D181)
        elif  num281==744:
            p3812=self.conv1744(D181)
        elif  num281==745:
            p3812=self.conv1745(D181)
        elif  num281==746:
            p3812=self.conv1746(D181)
        elif  num281==747:
            p3812=self.conv1747(D181)
        elif  num281==748:
            p3812=self.conv1748(D181)
        elif  num281==749:
            p3812=self.conv1749(D181)
        elif  num281==750:
            p3812=self.conv1750(D181)
        elif  num281==751:
            p3812=self.conv1751(D181)
        elif  num281==752:
            p3812=self.conv1752(D181)
        elif  num281==753:
            p3812=self.conv1753(D181)
        elif  num281==754:
            p3812=self.conv1754(D181)
        elif  num281==755:
            p3812=self.conv1755(D181)
        elif  num281==756:
            p3812=self.conv1756(D181)
        elif  num281==757:
            p3812=self.conv1757(D181)
        elif  num281==758:
            p3812=self.conv1758(D181)
        elif  num281==759:
            p3812=self.conv1759(D181)
        elif  num281==760:
            p3812=self.conv1760(D181)
        elif  num281==761:
            p3812=self.conv1761(D181)
        elif  num281==762:
            p3812=self.conv1762(D181)
        elif  num281==763:
            p3812=self.conv1763(D181)
        elif  num281==764:
            p3812=self.conv1764(D181)
        elif  num281==765:
            p3812=self.conv1765(D181)
        elif  num281==766:
            p3812=self.conv1766(D181)
        elif  num281==767:
            p3812=self.conv1767(D181)
        elif  num281==768:
            p3812=self.conv1768(D181)
        elif  num281==769:
            p3812=self.conv1769(D181) 
        elif  num281==770:
            p3812=self.conv1770(D181)
        elif  num281==771:
            p3812=self.conv1771(D181)
        elif  num281==772:
            p3812=self.conv1772(D181)
        elif  num281==773:
            p3812=self.conv1773(D181)
        elif  num281==774:
            p3812=self.conv1774(D181)
        elif  num281==775:
            p3812=self.conv1775(D181)
        elif  num281==776:
            p3812=self.conv1776(D181)
        elif  num281==777:
            p3812=self.conv1777(D181)
        elif  num281==778:
            p3812=self.conv1778(D181)
        elif  num281==779:
            p3812=self.conv1779(D181) 
        elif  num281==780:
            p3812=self.conv1780(D181)
        elif  num281==781:
            p3812=self.conv1781(D181)
        elif  num281==782:
            p3812=self.conv1782(D181)
        elif  num281==783:
            p3812=self.conv1783(D181)
        elif  num281==784:
            p3812=self.conv1784(D181)
        elif  num281==785:
            p3812=self.conv1785(D181)
        elif  num281==786:
            p3812=self.conv1786(D181)
        elif  num281==787:
            p3812=self.conv1787(D181)
        elif  num281==788:
            p3812=self.conv1788(D181)
        elif  num281==789:
            p3812=self.conv1789(D181) 
        elif  num281==790:
            p3812=self.conv1790(D181)
        elif  num281==791:
            p3812=self.conv1791(D181)
        elif  num281==792:
            p3812=self.conv1792(D181)
        elif  num281==793:
            p3812=self.conv1793(D181)
        elif  num281==794:
            p3812=self.conv1794(D181)
        elif  num281==795:
            p3812=self.conv1795(D181)
        elif  num281==796:
            p3812=self.conv1796(D181)
        elif  num281==797:
            p3812=self.conv1797(D181)
        elif  num281==798:
            p3812=self.conv1798(D181)
        elif  num281==799:
            p3812=self.conv1799(D181) 
        elif  num281==800:
            p3812=self.conv1800(D181)
        elif  num281==801:
            p3812=self.conv1801(D181)
        elif  num281==802:
            p3812=self.conv1802(D181)
        elif  num281==803:
            p3812=self.conv1803(D181)
        elif  num281==804:
            p3812=self.conv1804(D181)
        elif  num281==805:
            p3812=self.conv1805(D181)
        elif  num281==806:
            p3812=self.conv1806(D181)
        elif  num281==807:
            p3812=self.conv1807(D181)
        elif  num281==808:
            p3812=self.conv1808(D181)
        elif  num281==809:
            p3812=self.conv1809(D181)
        elif  num281==810:
            p3812=self.conv1810(D181)
        elif  num281==811:
            p3812=self.conv1811(D181)
        elif  num281==812:
            p3812=self.conv1812(D181)
        elif  num281==813:
            p3812=self.conv1813(D181)
        elif  num281==814:
            p3812=self.conv1814(D181)
        elif  num281==815:
            p3812=self.conv1815(D181)
        elif  num281==816:
            p3812=self.conv1816(D181)
        elif  num281==817:
            p3812=self.conv1817(D181)
        elif  num281==818:
            p3812=self.conv1818(D181)
        elif  num281==819:
            p3812=self.conv1819(D181)
        elif  num281==820:
            p3812=self.conv1820(D181)
        elif  num281==821:
            p3812=self.conv1821(D181)
        elif  num281==822:
            p3812=self.conv1822(D181)
        elif  num281==823:
            p3812=self.conv1823(D181)
        elif  num281==824:
            p3812=self.conv1824(D181)
        elif  num281==825:
            p3812=self.conv1825(D181)
        elif  num281==826:
            p3812=self.conv1826(D181)
        elif  num281==827:
            p3812=self.conv1827(D181)
        elif  num281==828:
            p3812=self.conv1828(D181)
        elif  num281==829:
            p3812=self.conv1829(D181)                                                                                                                          
        elif  num281==830:
            p3812=self.conv1830(D181)
        elif  num281==831:
            p3812=self.conv1831(D181)
        elif  num281==832:
            p3812=self.conv1832(D181)
        elif  num281==833:
            p3812=self.conv1833(D181)
        elif  num281==834:
            p3812=self.conv1834(D181)
        elif  num281==835:
            p3812=self.conv1835(D181)
        elif  num281==836:
            p3812=self.conv1836(D181)
        elif  num281==837:
            p3812=self.conv1837(D181)
        elif  num281==838:
            p3812=self.conv1838(D181)
        elif  num281==839:
            p3812=self.conv1839(D181)
        elif  num281==840:
            p3812=self.conv1840(D181)
        elif  num281==841:
            p3812=self.conv1841(D181)
        elif  num281==842:
            p3812=self.conv1842(D181)
        elif  num281==843:
            p3812=self.conv1843(D181)
        elif  num281==844:
            p3812=self.conv1844(D181)
        elif  num281==845:
            p3812=self.conv1845(D181)
        elif  num281==846:
            p3812=self.conv1846(D181)
        elif  num281==847:
            p3812=self.conv1847(D181)
        elif  num281==848:
            p3812=self.conv1848(D181)
        elif  num281==849:
            p3812=self.conv1849(D181)
        elif  num281==850:
            p3812=self.conv1850(D181)
        elif  num281==851:
            p3812=self.conv1851(D181)
        elif  num281==852:
            p3812=self.conv1852(D181)
        elif  num281==853:
            p3812=self.conv1853(D181)
        elif  num281==854:
            p3812=self.conv1854(D181)
        elif  num281==855:
            p3812=self.conv1855(D181)
        elif  num281==856:
            p3812=self.conv1856(D181)
        elif  num281==857:
            p3812=self.conv1857(D181)
        elif  num281==858:
            p3812=self.conv1858(D181)
        elif  num281==859:
            p3812=self.conv1859(D181)
        elif  num281==860:
            p3812=self.conv1860(D181)
        elif  num281==861:
            p3812=self.conv1861(D181)
        elif  num281==862:
            p3812=self.conv1862(D181)
        elif  num281==863:
            p3812=self.conv1863(D181)
        elif  num281==864:
            p3812=self.conv1864(D181)
        elif  num281==865:
            p3812=self.conv1865(D181)
        elif  num281==866:
            p3812=self.conv1866(D181)
        elif  num281==867:
            p3812=self.conv1867(D181)
        elif  num281==868:
            p3812=self.conv1868(D181)
        elif  num281==869:
            p3812=self.conv1869(D181) 
        elif  num281==870:
            p3812=self.conv1870(D181)
        elif  num281==871:
            p3812=self.conv1871(D181)
        elif  num281==872:
            p3812=self.conv1872(D181)
        elif  num281==873:
            p3812=self.conv1873(D181)
        elif  num281==874:
            p3812=self.conv1874(D181)
        elif  num281==875:
            p3812=self.conv1875(D181)
        elif  num281==876:
            p3812=self.conv1876(D181)
        elif  num281==877:
            p3812=self.conv1877(D181)
        elif  num281==878:
            p3812=self.conv1878(D181)
        elif  num281==879:
            p3812=self.conv1879(D181)
        elif  num281==880:
            p3812=self.conv1880(D181)
        elif  num281==881:
            p3812=self.conv1881(D181)
        elif  num281==882:
            p3812=self.conv1882(D181)
        elif  num281==883:
            p3812=self.conv1883(D181)
        elif  num281==884:
            p3812=self.conv1884(D181)
        elif  num281==885:
            p3812=self.conv1885(D181)
        elif  num281==886:
            p3812=self.conv1886(D181)
        elif  num281==887:
            p3812=self.conv1887(D181)
        elif  num281==888:
            p3812=self.conv1888(D181)
        elif  num281==889:
            p3812=self.conv1889(D181)  
        elif  num281==890:
            p3812=self.conv1890(D181)
        elif  num281==891:
            p3812=self.conv1891(D181)
        elif  num281==892:
            p3812=self.conv1892(D181)
        elif  num281==893:
            p3812=self.conv1893(D181)
        elif  num281==894:
            p3812=self.conv1894(D181)
        elif  num281==895:
            p3812=self.conv1895(D181)
        elif  num281==896:
            p3812=self.conv1896(D181)
        elif  num281==897:
            p3812=self.conv1897(D181)
        elif  num281==898:
            p3812=self.conv1898(D181)
        elif  num281==899:
            p3812=self.conv1899(D181)
        elif  num281==900:
            p3812=self.conv1900(D181)
        elif  num281==901:
            p3812=self.conv1901(D181)
        elif  num281==902:
            p3812=self.conv1902(D181)
        elif  num281==903:
            p3812=self.conv1903(D181)
        elif  num281==904:
            p3812=self.conv1904(D181)
        elif  num281==905:
            p3812=self.conv1905(D181)
        elif  num281==906:
            p3812=self.conv1906(D181)
        elif  num281==907:
            p3812=self.conv1907(D181)
        elif  num281==908:
            p3812=self.conv1908(D181)
        elif  num281==909:
            p3812=self.conv1909(D181)
        elif  num281==910:
            p3812=self.conv1910(D181)
        elif  num281==911:
            p3812=self.conv1911(D181)
        elif  num281==912:
            p3812=self.conv1912(D181)
        elif  num281==913:
            p3812=self.conv1913(D181)
        elif  num281==914:
            p3812=self.conv1914(D181)
        elif  num281==915:
            p3812=self.conv1915(D181)
        elif  num281==916:
            p3812=self.conv1916(D181)
        elif  num281==917:
            p3812=self.conv1917(D181)
        elif  num281==918:
            p3812=self.conv1918(D181)
        elif  num281==919:
            p3812=self.conv1919(D181)
        elif  num281==920:
            p3812=self.conv1920(D181)
        elif  num281==921:
            p3812=self.conv1921(D181)
        elif  num281==922:
            p3812=self.conv1922(D181)
        elif  num281==923:
            p3812=self.conv1923(D181)
        elif  num281==924:
            p3812=self.conv1924(D181)
        elif  num281==925:
            p3812=self.conv1925(D181)
        elif  num281==926:
            p3812=self.conv1926(D181)
        elif  num281==927:
            p3812=self.conv1927(D181)
        elif  num281==928:
            p3812=self.conv1928(D181)
        elif  num281==929:
            p3812=self.conv1929(D181)
        elif  num281==930:
            p3812=self.conv1930(D181)
        elif  num281==931:
            p3812=self.conv1931(D181)
        elif  num281==932:
            p3812=self.conv1932(D181)
        elif  num281==933:
            p3812=self.conv1933(D181)
        elif  num281==934:
            p3812=self.conv1934(D181)
        elif  num281==935:
            p3812=self.conv1935(D181)
        elif  num281==936:
            p3812=self.conv1936(D181)
        elif  num281==937:
            p3812=self.conv1937(D181)
        elif  num281==938:
            p3812=self.conv1938(D181)
        elif  num281==939:
            p3812=self.conv1939(D181) 
        elif  num281==940:
            p3812=self.conv1940(D181)
        elif  num281==941:
            p3812=self.conv1941(D181)
        elif  num281==942:
            p3812=self.conv1942(D181)
        elif  num281==943:
            p3812=self.conv1943(D181)
        elif  num281==944:
            p3812=self.conv1944(D181)
        elif  num281==945:
            p3812=self.conv1945(D181)
        elif  num281==946:
            p3812=self.conv1946(D181)
        elif  num281==947:
            p3812=self.conv1947(D181)
        elif  num281==948:
            p3812=self.conv1948(D181)
        elif  num281==949:
            p3812=self.conv1949(D181)                                                                                                                          
        elif  num281==950:
            p3812=self.conv1950(D181)
        elif  num281==951:
            p3812=self.conv1951(D181)
        elif  num281==952:
            p3812=self.conv1952(D181)
        elif  num281==953:
            p3812=self.conv1953(D181)
        elif  num281==954:
            p3812=self.conv1954(D181)
        elif  num281==955:
            p3812=self.conv1955(D181)
        elif  num281==956:
            p3812=self.conv1956(D181)
        elif  num281==957:
            p3812=self.conv1957(D181)
        elif  num281==958:
            p3812=self.conv1958(D181)
        elif  num281==959:
            p3812=self.conv1959(D181)
        elif  num281==960:
            p3812=self.conv1960(D181)
        elif  num281==961:
            p3812=self.conv1961(D181)
        elif  num281==962:
            p3812=self.conv1962(D181)
        elif  num281==963:
            p3812=self.conv1963(D181)
        elif  num281==964:
            p3812=self.conv1964(D181)
        elif  num281==965:
            p3812=self.conv1965(D181)
        elif  num281==966:
            p3812=self.conv1966(D181)
        elif  num281==967:
            p3812=self.conv1967(D181)
        elif  num281==968:
            p3812=self.conv1968(D181)
        elif  num281==969:
            p3812=self.conv1969(D181) 
        elif  num281==970:
            p3812=self.conv1970(D181)
        elif  num281==971:
            p3812=self.conv1971(D181)
        elif  num281==972:
            p3812=self.conv1972(D181)
        elif  num281==973:
            p3812=self.conv1973(D181)
        elif  num281==974:
            p3812=self.conv1974(D181)
        elif  num281==975:
            p3812=self.conv1975(D181)
        elif  num281==976:
            p3812=self.conv1976(D181)
        elif  num281==977:
            p3812=self.conv1977(D181)
        elif  num281==978:
            p3812=self.conv1978(D181)
        elif  num281==979:
            p3812=self.conv1979(D181) 
        elif  num281==980:
            p3812=self.conv1980(D181)
        elif  num281==981:
            p3812=self.conv1981(D181)
        elif  num281==982:
            p3812=self.conv1982(D181)
        elif  num281==983:
            p3812=self.conv1983(D181)
        elif  num281==984:
            p3812=self.conv1984(D181)
        elif  num281==985:
            p3812=self.conv1985(D181)
        elif  num281==986:
            p3812=self.conv1986(D181)
        elif  num281==987:
            p3812=self.conv1987(D181)
        elif  num281==988:
            p3812=self.conv1988(D181)
        elif  num281==989:
            p3812=self.conv1989(D181)
        elif  num281==990:
            p3812=self.conv1990(D181)
        elif  num281==991:
            p3812=self.conv1991(D181)
        elif  num281==992:
            p3812=self.conv1992(D181)
        elif  num281==993:
            p3812=self.conv1993(D181)
        elif  num281==994:
            p3812=self.conv1994(D181)
        elif  num281==995:
            p3812=self.conv1995(D181)
        elif  num281==996:
            p3812=self.conv1996(D181)
        elif  num281==997:
            p3812=self.conv1997(D181)
        elif  num281==998:
            p3812=self.conv1998(D181)
        elif  num281==999:
            p3812=self.conv1999(D181) 
        elif  num281==1000:
            p3812=self.conv11000(D181)
        elif  num281==1001:
            p3812=self.conv11001(D181)
        elif  num281==1002:
            p3812=self.conv11002(D181)
        elif  num281==1003:
            p3812=self.conv11003(D181)
        elif  num281==1004:
            p3812=self.conv11004(D181)
        elif  num281==1005:
            p3812=self.conv11005(D181)
        elif  num281==1006:
            p3812=self.conv11006(D181)
        elif  num281==1007:
            p3812=self.conv11007(D181)
        elif  num281==1008:
            p3812=self.conv11008(D181)
        elif  num281==1009:
            p3812=self.conv11009(D181) 
        elif  num281==1010:
            p3812=self.conv11010(D181)
        elif  num281==1011:
            p3812=self.conv11011(D181)
        elif  num281==1012:
            p3812=self.conv11012(D181)
        elif  num281==1013:
            p3812=self.conv11013(D181)
        elif  num281==1014:
            p3812=self.conv11014(D181)
        elif  num281==1015:
            p3812=self.conv11015(D181)
        elif  num281==1016:
            p3812=self.conv11016(D181)
        elif  num281==1017:
            p3812=self.conv11017(D181)
        elif  num281==1018:
            p3812=self.conv11018(D181)
        elif  num281==1019:
            p3812=self.conv11019(D181)
        elif  num281==1020:
            p3812=self.conv11020(D181)
        elif  num281==1021:
            p3812=self.conv11021(D181)
        elif  num281==1022:
            p3812=self.conv11022(D181)
        elif  num281==1023:
            p3812=self.conv11023(D181)
        elif  num281==1024:
            p3812=self.conv11024(D181) 
            
        if num012==1:
            p312=self.conv11(B112)
        elif num012==2:
            p312=self.conv12(B112)
        elif num012==3:
            p312=self.conv13(B112)
        elif num012==4:
            p312=self.conv14(B112)
        elif num012==5:
            p312=self.conv15(B112)
        elif num012==6:
            p312=self.conv16(B112)
        elif num012==7:
            p312=self.conv17(B112)
        elif num012==8:
            p312=self.conv18(B112)
        elif num012==9:
            p312=self.conv19(B112)
        elif num012==10:
            p312=self.conv110(B112)
        elif num012==11:
            p312=self.conv111(B112)
        elif num012==12:
            p312=self.conv112(B112)
        elif num012==13:
            p312=self.conv113(B112)
        elif num012==14:
            p312=self.conv114(B112)
        elif num012==15:
            p312=self.conv115(B112)
        elif num012==16:
            p312=self.conv116(B112)
        elif num012==17:
            p312=self.conv117(B112)
        elif num012==18:
            p312=self.conv118(B112)
        elif num012==19:
            p312=self.conv119(B112)
        elif num012==20:
            p312=self.conv120(B112)
        elif num012==21:
            p312=self.conv121(B112)
        elif num012==22:
            p312=self.conv122(B112)
        elif num012==23:
            p312=self.conv123(B112)
        elif num012==24:
            p312=self.conv124(B112)
        elif num012==25:
            p312=self.conv125(B112)
        elif num012==26:
            p312=self.conv126(B112)
        elif num012==27:
            p312=self.conv127(B112)
        elif num012==28:
            p312=self.conv128(B112)
        elif num012==29:
            p312=self.conv129(B112)
        elif num012==30:
            p312=self.conv130(B112)
        elif num012==31:
            p312=self.conv131(B112)
        elif num012==32:
            p312=self.conv132(B112)
        elif num012==33:
            p312=self.conv133(B112)
        elif num012==34:
            p312=self.conv134(B112)
        elif num012==35:
            p312=self.conv135(B112)
        elif num012==36:
            p312=self.conv136(B112)
        elif num012==37:
            p312=self.conv137(B112)
        elif num012==38:
            p312=self.conv138(B112)
        elif num012==39:
            p312=self.conv139(B112)
        elif num012==40:
            p312=self.conv140(B112)
        elif num012==41:
            p312=self.conv141(B112)
        elif num012==42:
            p312=self.conv142(B112)
        elif num012==43:
            p312=self.conv143(B112)
        elif num012==44:
            p312=self.conv144(B112)
        elif num012==45:
            p312=self.conv145(B112)
        elif num012==46:
            p312=self.conv146(B112)
        elif num012==47:
            p312=self.conv147(B112)
        elif num012==48:
            p312=self.conv148(B112)
        elif num012==49:
            p312=self.conv149(B112)
        elif num012==50:
            p312=self.conv150(B112)
        elif num012==51:
            p312=self.conv151(B112)
        elif num012==52:
            p312=self.conv152(B112)
        elif num012==53:
            p312=self.conv153(B112)
        elif num012==54:
            p312=self.conv154(B112)
        elif num012==55:
            p312=self.conv155(B112)
        elif num012==56:
            p312=self.conv156(B112)
        elif num012==57:
            p312=self.conv157(B112)
        elif num012==58:
            p312=self.conv158(B112)
        elif num012==59:
            p312=self.conv159(B112)
        elif num012==60:
            p312=self.conv160(B112)
        elif num012==61:
            p312=self.conv161(B112)
        elif num012==62:
            p312=self.conv162(B112)
        elif num012==63:
            p312=self.conv163(B112)
        elif num012==64:
            p312=self.conv164(B112)
        
        if  num112==1:
            p3121=self.conv11(C112)
        elif  num112==2:
            p3121=self.conv12(C112)
        elif  num112==3:
            p3121=self.conv13(C112)
        elif  num112==4:
            p3121=self.conv14(C112)
        elif  num112==5:
            p3121=self.conv15(C112)
        elif  num112==6:
            p3121=self.conv16(C112)
        elif  num112==7:
            p3121=self.conv17(C112)
        elif  num112==8:
            p3121=self.conv18(C112)
        elif  num112==9:
            p3121=self.conv19(C112)
        elif  num112==10:
            p3121=self.conv110(C112)
        elif  num112==11:
            p3121=self.conv111(C112)
        elif  num112==12:
            p3121=self.conv112(C112)
        elif  num112==13:
            p3121=self.conv113(C112)
        elif  num112==14:
            p3121=self.conv114(C112)
        elif  num112==15:
            p3121=self.conv115(C112)
        elif  num112==16:
            p3121=self.conv116(C112)
        elif  num112==17:
            p3121=self.conv117(C112)
        elif  num112==18:
            p3121=self.conv118(C112)
        elif  num112==19:
            p3121=self.conv119(C112)
        elif  num112==20:
            p3121=self.conv120(C112)
        elif  num112==21:
            p3121=self.conv121(C112)
        elif  num112==22:
            p3121=self.conv122(C112)
        elif  num112==23:
            p3121=self.conv123(C112)
        elif  num112==24:
            p3121=self.conv124(C112)
        elif  num112==25:
            p3121=self.conv125(C112)
        elif  num112==26:
            p3121=self.conv126(C112)
        elif  num112==27:
            p3121=self.conv127(C112)
        elif  num112==28:
            p3121=self.conv128(C112)
        elif  num112==29:
            p3121=self.conv129(C112)
        elif  num112==30:
            p3121=self.conv130(C112)
        elif  num112==31:
            p3121=self.conv131(C112)
        elif  num112==32:
            p3121=self.conv132(C112)
        elif  num112==33:
            p3121=self.conv133(C112)
        elif  num112==34:
            p3121=self.conv134(C112)
        elif  num112==35:
            p3121=self.conv135(C112)
        elif  num112==36:
            p3121=self.conv136(C112)
        elif  num112==37:
            p3121=self.conv137(C112)
        elif  num112==38:
            p3121=self.conv138(C112)
        elif  num112==39:
            p3121=self.conv139(C112)
        elif  num112==40:
            p3121=self.conv140(C112)
        elif  num112==41:
            p3121=self.conv141(C112)
        elif  num112==42:
            p3121=self.conv142(C112)
        elif  num112==43:
            p3121=self.conv143(C112)
        elif  num112==44:
            p3121=self.conv144(C112)
        elif  num112==45:
            p3121=self.conv145(C112)
        elif  num112==46:
            p3121=self.conv146(C112)
        elif  num112==47:
            p3121=self.conv147(C112)
        elif  num112==48:
            p3121=self.conv148(C112)
        elif  num112==49:
            p3121=self.conv149(C112)
        elif  num112==50:
            p3121=self.conv150(C112)
        elif  num112==51:
            p3121=self.conv151(C112)
        elif  num112==52:
            p3121=self.conv152(C112)
        elif  num112==53:
            p3121=self.conv153(C112)
        elif  num112==54:
            p3121=self.conv154(C112)
        elif  num112==55:
            p3121=self.conv155(C112)
        elif  num112==56:
            p3121=self.conv156(C112)
        elif  num112==57:
            p3121=self.conv157(C112)
        elif  num112==58:
            p3121=self.conv158(C112)
        elif  num112==59:
            p3121=self.conv159(C112)
        elif  num112==60:
            p3121=self.conv160(C112)
        elif  num112==61:
            p3121=self.conv161(C112)
        elif  num112==62:
            p3121=self.conv162(C112)
        elif  num112==63:
            p3121=self.conv163(C112)
        elif  num112==64:
            p3121=self.conv164(C112)
        elif  num112==65:
            p3121=self.conv165(C112)
        elif  num112==66:
            p3121=self.conv166(C112)
        elif  num112==67:
            p3121=self.conv167(C112)
        elif  num112==68:
            p3121=self.conv168(C112)
        elif  num112==69:
            p3121=self.conv169(C112)
        elif  num112==70:
            p3121=self.conv170(C112)
        elif  num112==71:
            p3121=self.conv171(C112)
        elif  num112==72:
            p3121=self.conv172(C112)
        elif  num112==73:
            p3121=self.conv173(C112)
        elif  num112==74:
            p3121=self.conv174(C112)
        elif  num112==75:
            p3121=self.conv175(C112)
        elif  num112==76:
            p3121=self.conv176(C112)
        elif  num112==77:
            p3121=self.conv177(C112)
        elif  num112==78:
            p3121=self.conv178(C112)
        elif  num112==79:
            p3121=self.conv179(C112)
        elif  num112==80:
            p3121=self.conv180(C112)
        elif  num112==81:
            p3121=self.conv181(C112)
        elif  num112==82:
            p3121=self.conv182(C112)
        elif  num112==83:
            p3121=self.conv183(C112)
        elif  num112==84:
            p3121=self.conv184(C112)
        elif  num112==85:
            p3121=self.conv185(C112)
        elif  num112==86:
            p3121=self.conv186(C112)
        elif  num112==87:
            p3121=self.conv187(C112)
        elif  num112==88:
            p3121=self.conv188(C112)
        elif  num112==89:
            p3121=self.conv189(C112)    
        elif  num112==90:
            p3121=self.conv190(C112)
        elif  num112==91:
            p3121=self.conv191(C112)
        elif  num112==92:
            p3121=self.conv192(C112)
        elif  num112==93:
            p3121=self.conv193(C112)
        elif  num112==94:
            p3121=self.conv194(C112)
        elif  num112==95:
            p3121=self.conv195(C112)
        elif  num112==96:
            p3121=self.conv196(C112)
        elif  num112==97:
            p3121=self.conv197(C112)
        elif  num112==98:
            p3121=self.conv198(C112)
        elif  num112==99:
            p3121=self.conv199(C112) 
        elif  num112==100:
            p3121=self.conv1100(C112)
        elif  num112==101:
            p3121=self.conv1101(C112)
        elif  num112==102:
            p3121=self.conv1102(C112)
        elif  num112==103:
            p3121=self.conv1103(C112)
        elif  num112==104:
            p3121=self.conv1104(C112)
        elif  num112==105:
            p3121=self.conv1105(C112)
        elif  num112==106:
            p3121=self.conv1106(C112)
        elif  num112==107:
            p3121=self.conv1107(C112)
        elif  num112==108:
            p3121=self.conv1108(C112)
        elif  num112==109:
            p3121=self.conv1109(C112)
        elif  num112==110:
            p3121=self.conv1110(C112)
        elif  num112==111:
            p3121=self.conv1111(C112)
        elif  num112==112:
            p3121=self.conv1112(C112)
        elif  num112==113:
            p3121=self.conv1113(C112)
        elif  num112==114:
            p3121=self.conv1114(C112)
        elif  num112==115:
            p3121=self.conv1115(C112)
        elif  num112==116:
            p3121=self.conv1116(C112)
        elif  num112==117:
            p3121=self.conv1117(C112)
        elif  num112==118:
            p3121=self.conv1118(C112)
        elif  num112==119:
            p3121=self.conv1119(C112) 
        elif  num112==120:
            p3121=self.conv1120(C112)
        elif  num112==121:
            p3121=self.conv1121(C112)
        elif  num112==122:
            p3121=self.conv1122(C112)
        elif  num112==123:
            p3121=self.conv1123(C112)
        elif  num112==124:
            p3121=self.conv1124(C112)
        elif  num112==125:
            p3121=self.conv1125(C112)
        elif  num112==126:
            p3121=self.conv1126(C112)
        elif  num112==127:
            p3121=self.conv1127(C112)
        elif  num112==128:
            p3121=self.conv1128(C112)
        elif  num112==129:
            p3121=self.conv1129(C112) 
        elif  num112==130:
            p3121=self.conv1130(C112)
        elif  num112==131:
            p3121=self.conv1131(C112)
        elif  num112==132:
            p3121=self.conv1132(C112)
        elif  num112==133:
            p3121=self.conv1133(C112)
        elif  num112==134:
            p3121=self.conv1134(C112)
        elif  num112==135:
            p3121=self.conv1135(C112)
        elif  num112==136:
            p3121=self.conv1136(C112)
        elif  num112==137:
            p3121=self.conv1137(C112)
        elif  num112==138:
            p3121=self.conv1138(C112)
        elif  num112==139:
            p3121=self.conv1139(C112)
        elif  num112==140:
            p3121=self.conv1140(C112)
        elif  num112==141:
            p3121=self.conv1141(C112)
        elif  num112==142:
            p3121=self.conv1142(C112)
        elif  num112==143:
            p3121=self.conv1143(C112)
        elif  num112==144:
            p3121=self.conv1144(C112)
        elif  num112==145:
            p3121=self.conv1145(C112)
        elif  num112==146:
            p3121=self.conv1146(C112)
        elif  num112==147:
            p3121=self.conv1147(C112)
        elif  num112==148:
            p3121=self.conv1148(C112)
        elif  num112==149:
            p3121=self.conv1149(C112) 
        elif  num112==150:
            p3121=self.conv1150(C112)
        elif  num112==151:
            p3121=self.conv1151(C112)
        elif  num112==152:
            p3121=self.conv1152(C112)
        elif  num112==153:
            p3121=self.conv1153(C112)
        elif  num112==154:
            p3121=self.conv1154(C112)
        elif  num112==155:
            p3121=self.conv1155(C112)
        elif  num112==156:
            p3121=self.conv1156(C112)
        elif  num112==157:
            p3121=self.conv1157(C112)
        elif  num112==158:
            p3121=self.conv1158(C112)
        elif  num112==159:
            p3121=self.conv1159(C112) 
        elif  num112==160:
            p3121=self.conv1160(C112)
        elif  num112==161:
            p3121=self.conv1161(C112)
        elif  num112==162:
            p3121=self.conv1162(C112)
        elif  num112==163:
            p3121=self.conv1163(C112)
        elif  num112==164:
            p3121=self.conv1164(C112)
        elif  num112==165:
            p3121=self.conv1165(C112)
        elif  num112==166:
            p3121=self.conv1166(C112)
        elif  num112==167:
            p3121=self.conv1167(C112)
        elif  num112==168:
            p3121=self.conv1168(C112)
        elif  num112==169:
            p3121=self.conv1169(C112) 
        elif  num112==170:
            p3121=self.conv1170(C112)
        elif  num112==171:
            p3121=self.conv1171(C112)
        elif  num112==172:
            p3121=self.conv1172(C112)
        elif  num112==173:
            p3121=self.conv1173(C112)
        elif  num112==174:
            p3121=self.conv1174(C112)
        elif  num112==175:
            p3121=self.conv1175(C112)
        elif  num112==176:
            p3121=self.conv1176(C112)
        elif  num112==177:
            p3121=self.conv1177(C112)
        elif  num112==178:
            p3121=self.conv1178(C112)
        elif  num112==179:
            p3121=self.conv1179(C112)                                                                                              
        elif  num112==180:
            p3121=self.conv1180(C112)
        elif  num112==181:
            p3121=self.conv1181(C112)
        elif  num112==182:
            p3121=self.conv1182(C112)
        elif  num112==183:
            p3121=self.conv1183(C112)
        elif  num112==184:
            p3121=self.conv1184(C112)
        elif  num112==185:
            p3121=self.conv1185(C112)
        elif  num112==186:
            p3121=self.conv1186(C112)
        elif  num112==187:
            p3121=self.conv1187(C112)
        elif  num112==188:
            p3121=self.conv1188(C112)
        elif  num112==189:
            p3121=self.conv1189(C112) 
        elif  num112==190:
            p3121=self.conv1190(C112)
        elif  num112==191:
            p3121=self.conv1191(C112)
        elif  num112==192:
            p3121=self.conv1192(C112)
        elif  num112==193:
            p3121=self.conv1193(C112)
        elif  num112==194:
            p3121=self.conv1194(C112)
        elif  num112==195:
            p3121=self.conv1195(C112)
        elif  num112==196:
            p3121=self.conv1196(C112)
        elif  num112==197:
            p3121=self.conv1197(C112)
        elif  num112==198:
            p3121=self.conv1198(C112)
        elif  num112==199:
            p3121=self.conv1199(C112)
        elif  num112==200:
            p3121=self.conv1200(C112)
        elif  num112==201:
            p3121=self.conv1201(C112)
        elif  num112==202:
            p3121=self.conv1202(C112)
        elif  num112==203:
            p3121=self.conv1203(C112)
        elif  num112==204:
            p3121=self.conv1204(C112)
        elif  num112==205:
            p3121=self.conv1205(C112)
        elif  num112==206:
            p3121=self.conv1206(C112)
        elif  num112==207:
            p3121=self.conv1207(C112)
        elif  num112==208:
            p3121=self.conv1208(C112)
        elif  num112==209:
            p3121=self.conv1209(C112)
        elif  num112==210:
            p3121=self.conv1210(C112)
        elif  num112==211:
            p3121=self.conv1211(C112)
        elif  num112==212:
            p3121=self.conv1212(C112)
        elif  num112==213:
            p3121=self.conv1213(C112)
        elif  num112==214:
            p3121=self.conv1214(C112)
        elif  num112==215:
            p3121=self.conv1215(C112)
        elif  num112==216:
            p3121=self.conv1216(C112)
        elif  num112==217:
            p3121=self.conv1217(C112)
        elif  num112==218:
            p3121=self.conv1218(C112)
        elif  num112==219:
            p3121=self.conv1219(C112)
        elif  num112==220:
            p3121=self.conv1220(C112)
        elif  num112==221:
            p3121=self.conv1221(C112)
        elif  num112==222:
            p3121=self.conv1222(C112)
        elif  num112==223:
            p3121=self.conv1223(C112)
        elif  num112==224:
            p3121=self.conv1224(C112)
        elif  num112==225:
            p3121=self.conv1225(C112)
        elif  num112==226:
            p3121=self.conv1226(C112)
        elif  num112==227:
            p3121=self.conv1227(C112)
        elif  num112==228:
            p3121=self.conv1228(C112)
        elif  num112==229:
            p3121=self.conv1229(C112)
        elif  num112==230:
            p3121=self.conv1230(C112)
        elif  num112==231:
            p3121=self.conv1231(C112)
        elif  num112==232:
            p3121=self.conv1232(C112)
        elif  num112==233:
            p3121=self.conv1233(C112)
        elif  num112==234:
            p3121=self.conv1234(C112)
        elif  num112==235:
            p3121=self.conv1235(C112)
        elif  num112==236:
            p3121=self.conv1236(C112)
        elif  num112==237:
            p3121=self.conv1237(C112)
        elif  num112==238:
            p3121=self.conv1238(C112)
        elif  num112==239:
            p3121=self.conv1239(C112) 
        elif  num112==240:
            p3121=self.conv1240(C112)
        elif  num112==241:
            p3121=self.conv1241(C112)
        elif  num112==242:
            p3121=self.conv1242(C112)
        elif  num112==243:
            p3121=self.conv1243(C112)
        elif  num112==244:
            p3121=self.conv1244(C112)
        elif  num112==245:
            p3121=self.conv1245(C112)
        elif  num112==246:
            p3121=self.conv1246(C112)
        elif  num112==247:
            p3121=self.conv1247(C112)
        elif  num112==248:
            p3121=self.conv1248(C112)
        elif  num112==249:
            p3121=self.conv1249(C112)
        elif  num112==250:
            p3121=self.conv1250(C112)
        elif  num112==251:
            p3121=self.conv1251(C112)
        elif  num112==252:
            p3121=self.conv1252(C112)
        elif  num112==253:
            p3121=self.conv1253(C112)
        elif  num112==254:
            p3121=self.conv1254(C112)
        elif  num112==255:
            p3121=self.conv1255(C112)
        elif  num112==256:
            p3121=self.conv1256(C112)
            
        if  num212==1:
            p3122=self.conv11(D112)
        elif  num212==2:
            p3122=self.conv12(D112)
        elif  num212==3:
            p3122=self.conv13(D112)
        elif  num212==4:
            p3122=self.conv14(D112)
        elif  num212==5:
            p3122=self.conv15(D112)
        elif  num212==6:
            p3122=self.conv16(D112)
        elif  num212==7:
            p3122=self.conv17(D112)
        elif  num212==8:
            p3122=self.conv18(D112)
        elif  num212==9:
            p3122=self.conv19(D112)
        elif  num212==10:
            p3122=self.conv110(D112)
        elif  num212==11:
            p3122=self.conv111(D112)
        elif  num212==12:
            p3122=self.conv112(D112)
        elif  num212==13:
            p3122=self.conv113(D112)
        elif  num212==14:
            p3122=self.conv114(D112)
        elif  num212==15:
            p3122=self.conv115(D112)
        elif  num212==16:
            p3122=self.conv116(D112)
        elif  num212==17:
            p3122=self.conv117(D112)
        elif  num212==18:
            p3122=self.conv118(D112)
        elif  num212==19:
            p3122=self.conv119(D112)
        elif  num212==20:
            p3122=self.conv120(D112)
        elif  num212==21:
            p3122=self.conv121(D112)
        elif  num212==22:
            p3122=self.conv122(D112)
        elif  num212==23:
            p3122=self.conv123(D112)
        elif  num212==24:
            p3122=self.conv124(D112)
        elif  num212==25:
            p3122=self.conv125(D112)
        elif  num212==26:
            p3122=self.conv126(D112)
        elif  num212==27:
            p3122=self.conv127(D112)
        elif  num212==28:
            p3122=self.conv128(D112)
        elif  num212==29:
            p3122=self.conv129(D112)
        elif  num212==30:
            p3122=self.conv130(D112)
        elif  num212==31:
            p3122=self.conv131(D112)
        elif  num212==32:
            p3122=self.conv132(D112)
        elif  num212==33:
            p3122=self.conv133(D112)
        elif  num212==34:
            p3122=self.conv134(D112)
        elif  num212==35:
            p3122=self.conv135(D112)
        elif  num212==36:
            p3122=self.conv136(D112)
        elif  num212==37:
            p3122=self.conv137(D112)
        elif  num212==38:
            p3122=self.conv138(D112)
        elif  num212==39:
            p3122=self.conv139(D112)
        elif  num212==40:
            p3122=self.conv140(D112)
        elif  num212==41:
            p3122=self.conv141(D112)
        elif  num212==42:
            p3122=self.conv142(D112)
        elif  num212==43:
            p3122=self.conv143(D112)
        elif  num212==44:
            p3122=self.conv144(D112)
        elif  num212==45:
            p3122=self.conv145(D112)
        elif  num212==46:
            p3122=self.conv146(D112)
        elif  num212==47:
            p3122=self.conv147(D112)
        elif  num212==48:
            p3122=self.conv148(D112)
        elif  num212==49:
            p3122=self.conv149(D112)
        elif  num212==50:
            p3122=self.conv150(D112)
        elif  num212==51:
            p3122=self.conv151(D112)
        elif  num212==52:
            p3122=self.conv152(D112)
        elif  num212==53:
            p3122=self.conv153(D112)
        elif  num212==54:
            p3122=self.conv154(D112)
        elif  num212==55:
            p3122=self.conv155(D112)
        elif  num212==56:
            p3122=self.conv156(D112)
        elif  num212==57:
            p3122=self.conv157(D112)
        elif  num212==58:
            p3122=self.conv158(D112)
        elif  num212==59:
            p3122=self.conv159(D112)
        elif  num212==60:
            p3122=self.conv160(D112)
        elif  num212==61:
            p3122=self.conv161(D112)
        elif  num212==62:
            p3122=self.conv162(D112)
        elif  num212==63:
            p3122=self.conv163(D112)
        elif  num212==64:
            p3122=self.conv164(D112)
        elif  num212==65:
            p3122=self.conv165(D112)
        elif  num212==66:
            p3122=self.conv166(D112)
        elif  num212==67:
            p3122=self.conv167(D112)
        elif  num212==68:
            p3122=self.conv168(D112)
        elif  num212==69:
            p3122=self.conv169(D112)
        elif  num212==70:
            p3122=self.conv170(D112)
        elif  num212==71:
            p3122=self.conv171(D112)
        elif  num212==72:
            p3122=self.conv172(D112)
        elif  num212==73:
            p3122=self.conv173(D112)
        elif  num212==74:
            p3122=self.conv174(D112)
        elif  num212==75:
            p3122=self.conv175(D112)
        elif  num212==76:
            p3122=self.conv176(D112)
        elif  num212==77:
            p3122=self.conv177(D112)
        elif  num212==78:
            p3122=self.conv178(D112)
        elif  num212==79:
            p3122=self.conv179(D112)
        elif  num212==80:
            p3122=self.conv180(D112)
        elif  num212==81:
            p3122=self.conv181(D112)
        elif  num212==82:
            p3122=self.conv182(D112)
        elif  num212==83:
            p3122=self.conv183(D112)
        elif  num212==84:
            p3122=self.conv184(D112)
        elif  num212==85:
            p3122=self.conv185(D112)
        elif  num212==86:
            p3122=self.conv186(D112)
        elif  num212==87:
            p3122=self.conv187(D112)
        elif  num212==88:
            p3122=self.conv188(D112)
        elif  num212==89:
            p3122=self.conv189(D112)    
        elif  num212==90:
            p3122=self.conv190(D112)
        elif  num212==91:
            p3122=self.conv191(D112)
        elif  num212==92:
            p3122=self.conv192(D112)
        elif  num212==93:
            p3122=self.conv193(D112)
        elif  num212==94:
            p3122=self.conv194(D112)
        elif  num212==95:
            p3122=self.conv195(D112)
        elif  num212==96:
            p3122=self.conv196(D112)
        elif  num212==97:
            p3122=self.conv197(D112)
        elif  num212==98:
            p3122=self.conv198(D112)
        elif  num212==99:
            p3122=self.conv199(D112) 
        elif  num212==100:
            p3122=self.conv1100(D112)
        elif  num212==101:
            p3122=self.conv1101(D112)
        elif  num212==102:
            p3122=self.conv1102(D112)
        elif  num212==103:
            p3122=self.conv1103(D112)
        elif  num212==104:
            p3122=self.conv1104(D112)
        elif  num212==105:
            p3122=self.conv1105(D112)
        elif  num212==106:
            p3122=self.conv1106(D112)
        elif  num212==107:
            p3122=self.conv1107(D112)
        elif  num212==108:
            p3122=self.conv1108(D112)
        elif  num212==109:
            p3122=self.conv1109(D112)
        elif  num212==110:
            p3122=self.conv1110(D112)
        elif  num212==111:
            p3122=self.conv1111(D112)
        elif  num212==112:
            p3122=self.conv1112(D112)
        elif  num212==113:
            p3122=self.conv1113(D112)
        elif  num212==114:
            p3122=self.conv1114(D112)
        elif  num212==115:
            p3122=self.conv1115(D112)
        elif  num212==116:
            p3122=self.conv1116(D112)
        elif  num212==117:
            p3122=self.conv1117(D112)
        elif  num212==118:
            p3122=self.conv1118(D112)
        elif  num212==119:
            p3122=self.conv1119(D112) 
        elif  num212==120:
            p3122=self.conv1120(D112)
        elif  num212==121:
            p3122=self.conv1121(D112)
        elif  num212==122:
            p3122=self.conv1122(D112)
        elif  num212==123:
            p3122=self.conv1123(D112)
        elif  num212==124:
            p3122=self.conv1124(D112)
        elif  num212==125:
            p3122=self.conv1125(D112)
        elif  num212==126:
            p3122=self.conv1126(D112)
        elif  num212==127:
            p3122=self.conv1127(D112)
        elif  num212==128:
            p3122=self.conv1128(D112)
        elif  num212==129:
            p3122=self.conv1129(D112) 
        elif  num212==130:
            p3122=self.conv1130(D112)
        elif  num212==131:
            p3122=self.conv1131(D112)
        elif  num212==132:
            p3122=self.conv1132(D112)
        elif  num212==133:
            p3122=self.conv1133(D112)
        elif  num212==134:
            p3122=self.conv1134(D112)
        elif  num212==135:
            p3122=self.conv1135(D112)
        elif  num212==136:
            p3122=self.conv1136(D112)
        elif  num212==137:
            p3122=self.conv1137(D112)
        elif  num212==138:
            p3122=self.conv1138(D112)
        elif  num212==139:
            p3122=self.conv1139(D112)
        elif  num212==140:
            p3122=self.conv1140(D112)
        elif  num212==141:
            p3122=self.conv1141(D112)
        elif  num212==142:
            p3122=self.conv1142(D112)
        elif  num212==143:
            p3122=self.conv1143(D112)
        elif  num212==144:
            p3122=self.conv1144(D112)
        elif  num212==145:
            p3122=self.conv1145(D112)
        elif  num212==146:
            p3122=self.conv1146(D112)
        elif  num212==147:
            p3122=self.conv1147(D112)
        elif  num212==148:
            p3122=self.conv1148(D112)
        elif  num212==149:
            p3122=self.conv1149(D112) 
        elif  num212==150:
            p3122=self.conv1150(D112)
        elif  num212==151:
            p3122=self.conv1151(D112)
        elif  num212==152:
            p3122=self.conv1152(D112)
        elif  num212==153:
            p3122=self.conv1153(D112)
        elif  num212==154:
            p3122=self.conv1154(D112)
        elif  num212==155:
            p3122=self.conv1155(D112)
        elif  num212==156:
            p3122=self.conv1156(D112)
        elif  num212==157:
            p3122=self.conv1157(D112)
        elif  num212==158:
            p3122=self.conv1158(D112)
        elif  num212==159:
            p3122=self.conv1159(D112) 
        elif  num212==160:
            p3122=self.conv1160(D112)
        elif  num212==161:
            p3122=self.conv1161(D112)
        elif  num212==162:
            p3122=self.conv1162(D112)
        elif  num212==163:
            p3122=self.conv1163(D112)
        elif  num212==164:
            p3122=self.conv1164(D112)
        elif  num212==165:
            p3122=self.conv1165(D112)
        elif  num212==166:
            p3122=self.conv1166(D112)
        elif  num212==167:
            p3122=self.conv1167(D112)
        elif  num212==168:
            p3122=self.conv1168(D112)
        elif  num212==169:
            p3122=self.conv1169(D112) 
        elif  num212==170:
            p3122=self.conv1170(D112)
        elif  num212==171:
            p3122=self.conv1171(D112)
        elif  num212==172:
            p3122=self.conv1172(D112)
        elif  num212==173:
            p3122=self.conv1173(D112)
        elif  num212==174:
            p3122=self.conv1174(D112)
        elif  num212==175:
            p3122=self.conv1175(D112)
        elif  num212==176:
            p3122=self.conv1176(D112)
        elif  num212==177:
            p3122=self.conv1177(D112)
        elif  num212==178:
            p3122=self.conv1178(D112)
        elif  num212==179:
            p3122=self.conv1179(D112)                                                                                              
        elif  num212==180:
            p3122=self.conv1180(D112)
        elif  num212==181:
            p3122=self.conv1181(D112)
        elif  num212==182:
            p3122=self.conv1182(D112)
        elif  num212==183:
            p3122=self.conv1183(D112)
        elif  num212==184:
            p3122=self.conv1184(D112)
        elif  num212==185:
            p3122=self.conv1185(D112)
        elif  num212==186:
            p3122=self.conv1186(D112)
        elif  num212==187:
            p3122=self.conv1187(D112)
        elif  num212==188:
            p3122=self.conv1188(D112)
        elif  num212==189:
            p3122=self.conv1189(D112) 
        elif  num212==190:
            p3122=self.conv1190(D112)
        elif  num212==191:
            p3122=self.conv1191(D112)
        elif  num212==192:
            p3122=self.conv1192(D112)
        elif  num212==193:
            p3122=self.conv1193(D112)
        elif  num212==194:
            p3122=self.conv1194(D112)
        elif  num212==195:
            p3122=self.conv1195(D112)
        elif  num212==196:
            p3122=self.conv1196(D112)
        elif  num212==197:
            p3122=self.conv1197(D112)
        elif  num212==198:
            p3122=self.conv1198(D112)
        elif  num212==199:
            p3122=self.conv1199(D112)
        elif  num212==200:
            p3122=self.conv1200(D112)
        elif  num212==201:
            p3122=self.conv1201(D112)
        elif  num212==202:
            p3122=self.conv1202(D112)
        elif  num212==203:
            p3122=self.conv1203(D112)
        elif  num212==204:
            p3122=self.conv1204(D112)
        elif  num212==205:
            p3122=self.conv1205(D112)
        elif  num212==206:
            p3122=self.conv1206(D112)
        elif  num212==207:
            p3122=self.conv1207(D112)
        elif  num212==208:
            p3122=self.conv1208(D112)
        elif  num212==209:
            p3122=self.conv1209(D112)
        elif  num212==210:
            p3122=self.conv1210(D112)
        elif  num212==211:
            p3122=self.conv1211(D112)
        elif  num212==212:
            p3122=self.conv1212(D112)
        elif  num212==213:
            p3122=self.conv1213(D112)
        elif  num212==214:
            p3122=self.conv1214(D112)
        elif  num212==215:
            p3122=self.conv1215(D112)
        elif  num212==216:
            p3122=self.conv1216(D112)
        elif  num212==217:
            p3122=self.conv1217(D112)
        elif  num212==218:
            p3122=self.conv1218(D112)
        elif  num212==219:
            p3122=self.conv1219(D112)
        elif  num212==220:
            p3122=self.conv1220(D112)
        elif  num212==221:
            p3122=self.conv1221(D112)
        elif  num212==222:
            p3122=self.conv1222(D112)
        elif  num212==223:
            p3122=self.conv1223(D112)
        elif  num212==224:
            p3122=self.conv1224(D112)
        elif  num212==225:
            p3122=self.conv1225(D112)
        elif  num212==226:
            p3122=self.conv1226(D112)
        elif  num212==227:
            p3122=self.conv1227(D112)
        elif  num212==228:
            p3122=self.conv1228(D112)
        elif  num212==229:
            p3122=self.conv1229(D112)
        elif  num212==230:
            p3122=self.conv1230(D112)
        elif  num212==231:
            p3122=self.conv1231(D112)
        elif  num212==232:
            p3122=self.conv1232(D112)
        elif  num212==233:
            p3122=self.conv1233(D112)
        elif  num212==234:
            p3122=self.conv1234(D112)
        elif  num212==235:
            p3122=self.conv1235(D112)
        elif  num212==236:
            p3122=self.conv1236(D112)
        elif  num212==237:
            p3122=self.conv1237(D112)
        elif  num212==238:
            p3122=self.conv1238(D112)
        elif  num212==239:
            p3122=self.conv1239(D112) 
        elif  num212==240:
            p3122=self.conv1240(D112)
        elif  num212==241:
            p3122=self.conv1241(D112)
        elif  num212==242:
            p3122=self.conv1242(D112)
        elif  num212==243:
            p3122=self.conv1243(D112)
        elif  num212==244:
            p3122=self.conv1244(D112)
        elif  num212==245:
            p3122=self.conv1245(D112)
        elif  num212==246:
            p3122=self.conv1246(D112)
        elif  num212==247:
            p3122=self.conv1247(D112)
        elif  num212==248:
            p3122=self.conv1248(D112)
        elif  num212==249:
            p3122=self.conv1249(D112)
        elif  num212==250:
            p3122=self.conv1250(D112)
        elif  num212==251:
            p3122=self.conv1251(D112)
        elif  num212==252:
            p3122=self.conv1252(D112)
        elif  num212==253:
            p3122=self.conv1253(D112)
        elif  num212==254:
            p3122=self.conv1254(D112)
        elif  num212==255:
            p3122=self.conv1255(D112)
        elif  num212==256:
            p3122=self.conv1256(D112)
        elif  num212==257:
            p3122=self.conv1257(D112)
        elif  num212==258:
            p3122=self.conv1258(D112)
        elif  num212==259:
            p3122=self.conv1259(D112)
        elif  num212==260:
            p3122=self.conv1260(D112)
        elif  num212==261:
            p3122=self.conv1261(D112)
        elif  num212==262:
            p3122=self.conv1262(D112)
        elif  num212==263:
            p3122=self.conv1263(D112)
        elif  num212==264:
            p3122=self.conv1264(D112)
        elif  num212==265:
            p3122=self.conv1265(D112)
        elif  num212==266:
            p3122=self.conv1266(D112)
        elif  num212==267:
            p3122=self.conv1267(D112)
        elif  num212==268:
            p3122=self.conv1268(D112)
        elif  num212==269:
            p3122=self.conv1269(D112) 
        elif  num212==270:
            p3122=self.conv1270(D112)
        elif  num212==271:
            p3122=self.conv1271(D112)
        elif  num212==272:
            p3122=self.conv1272(D112)
        elif  num212==273:
            p3122=self.conv1273(D112)
        elif  num212==274:
            p3122=self.conv1274(D112)
        elif  num212==275:
            p3122=self.conv1275(D112)
        elif  num212==276:
            p3122=self.conv1276(D112)
        elif  num212==277:
            p3122=self.conv1277(D112)
        elif  num212==278:
            p3122=self.conv1278(D112)
        elif  num212==279:
            p3122=self.conv1279(D112)
        elif  num212==280:
            p3122=self.conv1280(D112)
        elif  num212==281:
            p3122=self.conv1281(D112)
        elif  num212==282:
            p3122=self.conv1282(D112)
        elif  num212==283:
            p3122=self.conv1283(D112)
        elif  num212==284:
            p3122=self.conv1284(D112)
        elif  num212==285:
            p3122=self.conv1285(D112)
        elif  num212==286:
            p3122=self.conv1286(D112)
        elif  num212==287:
            p3122=self.conv1287(D112)
        elif  num212==288:
            p3122=self.conv1288(D112)
        elif  num212==289:
            p3122=self.conv1289(D112) 
        elif  num212==290:
            p3122=self.conv1290(D112)
        elif  num212==291:
            p3122=self.conv1291(D112)
        elif  num212==292:
            p3122=self.conv1292(D112)
        elif  num212==293:
            p3122=self.conv1293(D112)
        elif  num212==294:
            p3122=self.conv1294(D112)
        elif  num212==295:
            p3122=self.conv1295(D112)
        elif  num212==296:
            p3122=self.conv1296(D112)
        elif  num212==297:
            p3122=self.conv1297(D112)
        elif  num212==298:
            p3122=self.conv1298(D112)
        elif  num212==299:
            p3122=self.conv1299(D112)
        elif  num212==300:
            p3122=self.conv1300(D112)
        elif  num212==301:
            p3122=self.conv1301(D112)
        elif  num212==302:
            p3122=self.conv1302(D112)
        elif  num212==303:
            p3122=self.conv1303(D112)
        elif  num212==304:
            p3122=self.conv1304(D112)
        elif  num212==305:
            p3122=self.conv1305(D112)
        elif  num212==306:
            p3122=self.conv1306(D112)
        elif  num212==307:
            p3122=self.conv1307(D112)
        elif  num212==308:
            p3122=self.conv1308(D112)
        elif  num212==309:
            p3122=self.conv1309(D112) 
        elif  num212==310:
            p3122=self.conv1310(D112)
        elif  num212==311:
            p3122=self.conv1311(D112)
        elif  num212==312:
            p3122=self.conv1312(D112)
        elif  num212==313:
            p3122=self.conv1313(D112)
        elif  num212==314:
            p3122=self.conv1314(D112)
        elif  num212==315:
            p3122=self.conv1315(D112)
        elif  num212==316:
            p3122=self.conv1316(D112)
        elif  num212==317:
            p3122=self.conv1317(D112)
        elif  num212==318:
            p3122=self.conv1318(D112)
        elif  num212==319:
            p3122=self.conv1319(D112)
        elif  num212==320:
            p3122=self.conv1320(D112)
        elif  num212==321:
            p3122=self.conv1321(D112)
        elif  num212==322:
            p3122=self.conv1322(D112)
        elif  num212==323:
            p3122=self.conv1323(D112)
        elif  num212==324:
            p3122=self.conv1324(D112)
        elif  num212==325:
            p3122=self.conv1325(D112)
        elif  num212==326:
            p3122=self.conv1326(D112)
        elif  num212==327:
            p3122=self.conv1327(D112)
        elif  num212==328:
            p3122=self.conv1328(D112)
        elif  num212==329:
            p3122=self.conv1329(D112)
        elif  num212==330:
            p3122=self.conv1330(D112)
        elif  num212==331:
            p3122=self.conv1331(D112)
        elif  num212==332:
            p3122=self.conv1332(D112)
        elif  num212==333:
            p3122=self.conv1333(D112)
        elif  num212==334:
            p3122=self.conv1334(D112)
        elif  num212==335:
            p3122=self.conv1335(D112)
        elif  num212==336:
            p3122=self.conv1336(D112)
        elif  num212==337:
            p3122=self.conv1337(D112)
        elif  num212==338:
            p3122=self.conv1338(D112)
        elif  num212==339:
            p3122=self.conv1339(D112)
        elif  num212==340:
            p3122=self.conv1340(D112)
        elif  num212==341:
            p3122=self.conv1341(D112)
        elif  num212==342:
            p3122=self.conv1342(D112)
        elif  num212==343:
            p3122=self.conv1343(D112)
        elif  num212==344:
            p3122=self.conv1344(D112)
        elif  num212==345:
            p3122=self.conv1345(D112)
        elif  num212==346:
            p3122=self.conv1346(D112)
        elif  num212==347:
            p3122=self.conv1347(D112)
        elif  num212==348:
            p3122=self.conv1348(D112)
        elif  num212==349:
            p3122=self.conv1349(D112)
        elif  num212==350:
            p3122=self.conv1350(D112)
        elif  num212==351:
            p3122=self.conv1351(D112)
        elif  num212==352:
            p3122=self.conv1352(D112)
        elif  num212==353:
            p3122=self.conv1335(D112)
        elif  num212==354:
            p3122=self.conv1354(D112)
        elif  num212==355:
            p3122=self.conv1355(D112)
        elif  num212==356:
            p3122=self.conv1356(D112)
        elif  num212==357:
            p3122=self.conv1357(D112)
        elif  num212==358:
            p3122=self.conv1358(D112)
        elif  num212==359:
            p3122=self.conv1359(D112) 
        elif  num212==360:
            p3122=self.conv1360(D112)
        elif  num212==361:
            p3122=self.conv1361(D112)
        elif  num212==362:
            p3122=self.conv1362(D112)
        elif  num212==363:
            p3122=self.conv1363(D112)
        elif  num212==364:
            p3122=self.conv1364(D112)
        elif  num212==365:
            p3122=self.conv1365(D112)
        elif  num212==366:
            p3122=self.conv1366(D112)
        elif  num212==367:
            p3122=self.conv1367(D112)
        elif  num212==368:
            p3122=self.conv1368(D112)
        elif  num212==369:
            p3122=self.conv1369(D112) 
        elif  num212==370:
            p3122=self.conv1370(D112)
        elif  num212==371:
            p3122=self.conv1371(D112)
        elif  num212==372:
            p3122=self.conv1372(D112)
        elif  num212==373:
            p3122=self.conv1373(D112)
        elif  num212==374:
            p3122=self.conv1374(D112)
        elif  num212==375:
            p3122=self.conv1375(D112)
        elif  num212==376:
            p3122=self.conv1376(D112)
        elif  num212==377:
            p3122=self.conv1377(D112)
        elif  num212==378:
            p3122=self.conv1378(D112)
        elif  num212==379:
            p3122=self.conv1379(D112) 
        elif  num212==380:
            p3122=self.conv1380(D112)
        elif  num212==381:
            p3122=self.conv1381(D112)
        elif  num212==382:
            p3122=self.conv1382(D112)
        elif  num212==383:
            p3122=self.conv1383(D112)
        elif  num212==384:
            p3122=self.conv1384(D112)
        elif  num212==385:
            p3122=self.conv1385(D112)
        elif  num212==386:
            p3122=self.conv1386(D112)
        elif  num212==387:
            p3122=self.conv1387(D112)
        elif  num212==388:
            p3122=self.conv1388(D112)
        elif  num212==389:
            p3122=self.conv1389(D112) 
        elif  num212==390:
            p3122=self.conv1390(D112)
        elif  num212==391:
            p3122=self.conv1391(D112)
        elif  num212==392:
            p3122=self.conv1392(D112)
        elif  num212==393:
            p3122=self.conv1393(D112)
        elif  num212==394:
            p3122=self.conv1394(D112)
        elif  num212==395:
            p3122=self.conv1395(D112)
        elif  num212==396:
            p3122=self.conv1396(D112)
        elif  num212==397:
            p3122=self.conv1397(D112)
        elif  num212==398:
            p3122=self.conv1398(D112)
        elif  num212==399:
            p3122=self.conv1399(D112)
        elif  num212==400:
            p3122=self.conv1400(D112)
        elif  num212==401:
            p3122=self.conv1401(D112)
        elif  num212==402:
            p3122=self.conv1402(D112)
        elif  num212==403:
            p3122=self.conv1403(D112)
        elif  num212==404:
            p3122=self.conv1404(D112)
        elif  num212==405:
            p3122=self.conv1405(D112)
        elif  num212==406:
            p3122=self.conv1406(D112)
        elif  num212==407:
            p3122=self.conv1407(D112)
        elif  num212==408:
            p3122=self.conv1408(D112)
        elif  num212==409:
            p3122=self.conv1409(D112)
        elif  num212==410:
            p3122=self.conv1410(D112)
        elif  num212==411:
            p3122=self.conv1411(D112)
        elif  num212==412:
            p3122=self.conv1412(D112)
        elif  num212==413:
            p3122=self.conv1413(D112)
        elif  num212==414:
            p3122=self.conv1414(D112)
        elif  num212==415:
            p3122=self.conv145(D112)
        elif  num212==416:
            p3122=self.conv1416(D112)
        elif  num212==417:
            p3122=self.conv1417(D112)
        elif  num212==418:
            p3122=self.conv1418(D112)
        elif  num212==419:
            p3122=self.conv1419(D112) 
        elif  num212==420:
            p3122=self.conv1420(D112)
        elif  num212==421:
            p3122=self.conv1421(D112)
        elif  num212==422:
            p3122=self.conv1422(D112)
        elif  num212==423:
            p3122=self.conv1423(D112)
        elif  num212==424:
            p3122=self.conv1424(D112)
        elif  num212==425:
            p3122=self.conv1425(D112)
        elif  num212==426:
            p3122=self.conv1426(D112)
        elif  num212==427:
            p3122=self.conv1427(D112)
        elif  num212==428:
            p3122=self.conv1428(D112)
        elif  num212==429:
            p3122=self.conv1429(D112) 
        elif  num212==430:
            p3122=self.conv1430(D112)
        elif  num212==431:
            p3122=self.conv1431(D112)
        elif  num212==432:
            p3122=self.conv1432(D112)
        elif  num212==433:
            p3122=self.conv1433(D112)
        elif  num212==434:
            p3122=self.conv1434(D112)
        elif  num212==435:
            p3122=self.conv1435(D112)
        elif  num212==436:
            p3122=self.conv1436(D112)
        elif  num212==437:
            p3122=self.conv1437(D112)
        elif  num212==438:
            p3122=self.conv1438(D112)
        elif  num212==439:
            p3122=self.conv1439(D112)
        elif  num212==440:
            p3122=self.conv1440(D112)
        elif  num212==441:
            p3122=self.conv1441(D112)
        elif  num212==442:
            p3122=self.conv1442(D112)
        elif  num212==443:
            p3122=self.conv1443(D112)
        elif  num212==444:
            p3122=self.conv1444(D112)
        elif  num212==445:
            p3122=self.conv1445(D112)
        elif  num212==446:
            p3122=self.conv1446(D112)
        elif  num212==447:
            p3122=self.conv1447(D112)
        elif  num212==448:
            p3122=self.conv1448(D112)
        elif  num212==449:
            p3122=self.conv1449(D112)
        elif  num212==450:
            p3122=self.conv1450(D112)
        elif  num212==451:
            p3122=self.conv1451(D112)
        elif  num212==452:
            p3122=self.conv1452(D112)
        elif  num212==453:
            p3122=self.conv1453(D112)
        elif  num212==454:
            p3122=self.conv1454(D112)
        elif  num212==455:
            p3122=self.conv1455(D112)
        elif  num212==456:
            p3122=self.conv1456(D112)
        elif  num212==457:
            p3122=self.conv1457(D112)
        elif  num212==458:
            p3122=self.conv1458(D112)
        elif  num212==459:
            p3122=self.conv1459(D112)
        elif  num212==460:
            p3122=self.conv1460(D112)
        elif  num212==461:
            p3122=self.conv1461(D112)
        elif  num212==462:
            p3122=self.conv1462(D112)
        elif  num212==463:
            p3122=self.conv1463(D112)
        elif  num212==464:
            p3122=self.conv1464(D112)
        elif  num212==465:
            p3122=self.conv1465(D112)
        elif  num212==466:
            p3122=self.conv1466(D112)
        elif  num212==467:
            p3122=self.conv1467(D112)
        elif  num212==468:
            p3122=self.conv1468(D112)
        elif  num212==469:
            p3122=self.conv1469(D112) 
        elif  num212==470:
            p3122=self.conv1470(D112)
        elif  num212==471:
            p3122=self.conv1471(D112)
        elif  num212==472:
            p3122=self.conv1472(D112)
        elif  num212==473:
            p3122=self.conv1473(D112)
        elif  num212==474:
            p3122=self.conv1474(D112)
        elif  num212==475:
            p3122=self.conv1475(D112)
        elif  num212==476:
            p3122=self.conv1476(D112)
        elif  num212==477:
            p3122=self.conv1477(D112)
        elif  num212==478:
            p3122=self.conv1478(D112)
        elif  num212==479:
            p3122=self.conv1479(D112)
        elif  num212==480:
            p3122=self.conv1480(D112)
        elif  num212==481:
            p3122=self.conv1481(D112)
        elif  num212==482:
            p3122=self.conv1482(D112)
        elif  num212==483:
            p3122=self.conv1483(D112)
        elif  num212==484:
            p3122=self.conv1484(D112)
        elif  num212==485:
            p3122=self.conv1485(D112)
        elif  num212==486:
            p3122=self.conv1486(D112)
        elif  num212==487:
            p3122=self.conv1487(D112)
        elif  num212==488:
            p3122=self.conv1488(D112)
        elif  num212==489:
            p3122=self.conv1489(D112)
        elif  num212==490:
            p3122=self.conv1490(D112)
        elif  num212==491:
            p3122=self.conv1491(D112)
        elif  num212==492:
            p3122=self.conv1492(D112)
        elif  num212==493:
            p3122=self.conv1493(D112)
        elif  num212==494:
            p3122=self.conv1494(D112)
        elif  num212==495:
            p3122=self.conv1495(D112)
        elif  num212==496:
            p3122=self.conv1496(D112)
        elif  num212==497:
            p3122=self.conv1497(D112)
        elif  num212==498:
            p3122=self.conv1498(D112)
        elif  num212==499:
            p3122=self.conv1499(D112)
        elif  num212==500:
            p3122=self.conv1500(D112)
        elif  num212==501:
            p3122=self.conv1501(D112)
        elif  num212==502:
            p3122=self.conv1502(D112)
        elif  num212==503:
            p3122=self.conv1503(D112)
        elif  num212==504:
            p3122=self.conv1504(D112)
        elif  num212==505:
            p3122=self.conv1505(D112)
        elif  num212==506:
            p3122=self.conv1506(D112)
        elif  num212==507:
            p3122=self.conv1507(D112)
        elif  num212==508:
            p3122=self.conv1508(D112)
        elif  num212==509:
            p3122=self.conv1509(D112)
        elif  num212==510:
            p3122=self.conv1510(D112)
        elif  num212==511:
            p3122=self.conv1511(D112)
        elif  num212==512:
            p3122=self.conv1512(D112)
        elif  num212==513:
            p3122=self.conv1513(D112)
        elif  num212==514:
            p3122=self.conv1514(D112)
        elif  num212==515:
            p3122=self.conv1515(D112)
        elif  num212==516:
            p3122=self.conv1516(D112)
        elif  num212==517:
            p3122=self.conv1517(D112)
        elif  num212==518:
            p3122=self.conv1518(D112)
        elif  num212==519:
            p3122=self.conv1519(D112)
        elif  num212==520:
            p3122=self.conv1520(D112)
        elif  num212==521:
            p3122=self.conv1521(D112)
        elif  num212==522:
            p3122=self.conv1522(D112)
        elif  num212==523:
            p3122=self.conv1523(D112)
        elif  num212==524:
            p3122=self.conv1524(D112)
        elif  num212==525:
            p3122=self.conv1525(D112)
        elif  num212==526:
            p3122=self.conv1526(D112)
        elif  num212==527:
            p3122=self.conv1527(D112)
        elif  num212==528:
            p3122=self.conv1528(D112)
        elif  num212==529:
            p3122=self.conv1529(D112)
        elif  num212==530:
            p3122=self.conv1530(D112)
        elif  num212==531:
            p3122=self.conv1531(D112)
        elif  num212==532:
            p3122=self.conv1532(D112)
        elif  num212==533:
            p3122=self.conv1533(D112)
        elif  num212==534:
            p3122=self.conv1534(D112)
        elif  num212==535:
            p3122=self.conv1535(D112)
        elif  num212==536:
            p3122=self.conv1536(D112)
        elif  num212==537:
            p3122=self.conv1537(D112)
        elif  num212==538:
            p3122=self.conv1538(D112)
        elif  num212==539:
            p3122=self.conv1539(D112)
        elif  num212==540:
            p3122=self.conv1540(D112)
        elif  num212==541:
            p3122=self.conv1541(D112)
        elif  num212==542:
            p3122=self.conv1542(D112)
        elif  num212==543:
            p3122=self.conv1543(D112)
        elif  num212==544:
            p3122=self.conv1544(D112)
        elif  num212==545:
            p3122=self.conv1545(D112)
        elif  num212==546:
            p3122=self.conv1546(D112)
        elif  num212==547:
            p3122=self.conv1547(D112)
        elif  num212==548:
            p3122=self.conv1548(D112)
        elif  num212==549:
            p3122=self.conv1549(D112) 
        elif  num212==550:
            p3122=self.conv1550(D112)
        elif  num212==551:
            p3122=self.conv1551(D112)
        elif  num212==552:
            p3122=self.conv1552(D112)
        elif  num212==553:
            p3122=self.conv1553(D112)
        elif  num212==554:
            p3122=self.conv1554(D112)
        elif  num212==555:
            p3122=self.conv1555(D112)
        elif  num212==556:
            p3122=self.conv1556(D112)
        elif  num212==557:
            p3122=self.conv1557(D112)
        elif  num212==558:
            p3122=self.conv1558(D112)
        elif  num212==559:
            p3122=self.conv1559(D112)
        elif  num212==560:
            p3122=self.conv1560(D112)
        elif  num212==561:
            p3122=self.conv1561(D112)
        elif  num212==562:
            p3122=self.conv1562(D112)
        elif  num212==563:
            p3122=self.conv1563(D112)
        elif  num212==564:
            p3122=self.conv1564(D112)
        elif  num212==565:
            p3122=self.conv1565(D112)
        elif  num212==566:
            p3122=self.conv1566(D112)
        elif  num212==567:
            p3122=self.conv1567(D112)
        elif  num212==568:
            p3122=self.conv1568(D112)
        elif  num212==569:
            p3122=self.conv1569(D112) 
        elif  num212==570:
            p3122=self.conv1570(D112)
        elif  num212==571:
            p3122=self.conv1571(D112)
        elif  num212==572:
            p3122=self.conv1572(D112)
        elif  num212==573:
            p3122=self.conv1573(D112)
        elif  num212==574:
            p3122=self.conv1574(D112)
        elif  num212==575:
            p3122=self.conv1575(D112)
        elif  num212==576:
            p3122=self.conv1576(D112)
        elif  num212==577:
            p3122=self.conv1577(D112)
        elif  num212==578:
            p3122=self.conv1578(D112)
        elif  num212==579:
            p3122=self.conv1579(D112) 
        elif  num212==580:
            p3122=self.conv1580(D112)
        elif  num212==581:
            p3122=self.conv1581(D112)
        elif  num212==582:
            p3122=self.conv1582(D112)
        elif  num212==583:
            p3122=self.conv1583(D112)
        elif  num212==584:
            p3122=self.conv1584(D112)
        elif  num212==585:
            p3122=self.conv1585(D112)
        elif  num212==586:
            p3122=self.conv1586(D112)
        elif  num212==587:
            p3122=self.conv1587(D112)
        elif  num212==588:
            p3122=self.conv1588(D112)
        elif  num212==589:
            p3122=self.conv1589(D112)
        elif  num212==590:
            p3122=self.conv1590(D112)
        elif  num212==591:
            p3122=self.conv1591(D112)
        elif  num212==592:
            p3122=self.conv1592(D112)
        elif  num212==593:
            p3122=self.conv1593(D112)
        elif  num212==594:
            p3122=self.conv1594(D112)
        elif  num212==595:
            p3122=self.conv1595(D112)
        elif  num212==596:
            p3122=self.conv1596(D112)
        elif  num212==597:
            p3122=self.conv1597(D112)
        elif  num212==598:
            p3122=self.conv1598(D112)
        elif  num212==599:
            p3122=self.conv1599(D112)
        elif  num212==600:
            p3122=self.conv1600(D112)
        elif  num212==601:
            p3122=self.conv1601(D112)
        elif  num212==602:
            p3122=self.conv1602(D112)
        elif  num212==603:
            p3122=self.conv1603(D112)
        elif  num212==604:
            p3122=self.conv1604(D112)
        elif  num212==605:
            p3122=self.conv1605(D112)
        elif  num212==606:
            p3122=self.conv1606(D112)
        elif  num212==607:
            p3122=self.conv1607(D112)
        elif  num212==608:
            p3122=self.conv1608(D112)
        elif  num212==609:
            p3122=self.conv1609(D112)                                                                                                                         
        elif  num212==610:
            p3122=self.conv1610(D112)
        elif  num212==611:
            p3122=self.conv1611(D112)
        elif  num212==612:
            p3122=self.conv1612(D112)
        elif  num212==613:
            p3122=self.conv1613(D112)
        elif  num212==614:
            p3122=self.conv1614(D112)
        elif  num212==615:
            p3122=self.conv1615(D112)
        elif  num212==616:
            p3122=self.conv1616(D112)
        elif  num212==617:
            p3122=self.conv1617(D112)
        elif  num212==618:
            p3122=self.conv1618(D112)
        elif  num212==619:
            p3122=self.conv1619(D112)                                                                                                                          
        elif  num212==620:
            p3122=self.conv1620(D112)
        elif  num212==621:
            p3122=self.conv1621(D112)
        elif  num212==622:
            p3122=self.conv1622(D112)
        elif  num212==623:
            p3122=self.conv1623(D112)
        elif  num212==624:
            p3122=self.conv1624(D112)
        elif  num212==625:
            p3122=self.conv1625(D112)
        elif  num212==626:
            p3122=self.conv1626(D112)
        elif  num212==627:
            p3122=self.conv1627(D112)
        elif  num212==628:
            p3122=self.conv1628(D112)
        elif  num212==629:
            p3122=self.conv1629(D112)  
        elif  num212==630:
            p3122=self.conv1630(D112)
        elif  num212==631:
            p3122=self.conv1631(D112)
        elif  num212==632:
            p3122=self.conv1632(D112)
        elif  num212==633:
            p3122=self.conv1633(D112)
        elif  num212==634:
            p3122=self.conv1634(D112)
        elif  num212==635:
            p3122=self.conv1635(D112)
        elif  num212==636:
            p3122=self.conv1636(D112)
        elif  num212==637:
            p3122=self.conv1637(D112)
        elif  num212==638:
            p3122=self.conv1638(D112)
        elif  num212==639:
            p3122=self.conv1639(D112)                                                                                                                          
        elif  num212==640:
            p3122=self.conv1640(D112)
        elif  num212==641:
            p3122=self.conv1641(D112)
        elif  num212==642:
            p3122=self.conv1642(D112)
        elif  num212==643:
            p3122=self.conv1643(D112)
        elif  num212==644:
            p3122=self.conv1644(D112)
        elif  num212==645:
            p3122=self.conv1645(D112)
        elif  num212==646:
            p3122=self.conv1646(D112)
        elif  num212==647:
            p3122=self.conv1647(D112)
        elif  num212==648:
            p3122=self.conv1648(D112)
        elif  num212==649:
            p3122=self.conv1649(D112)                                                                                                                          
        elif  num212==650:
            p3122=self.conv1650(D112)
        elif  num212==651:
            p3122=self.conv1651(D112)
        elif  num212==652:
            p3122=self.conv1652(D112)
        elif  num212==653:
            p3122=self.conv1653(D112)
        elif  num212==654:
            p3122=self.conv1654(D112)
        elif  num212==655:
            p3122=self.conv1655(D112)
        elif  num212==656:
            p3122=self.conv1656(D112)
        elif  num212==657:
            p3122=self.conv1657(D112)
        elif  num212==658:
            p3122=self.conv1658(D112)
        elif  num212==659:
            p3122=self.conv1659(D112)
        elif  num212==660:
            p3122=self.conv1660(D112)
        elif  num212==661:
            p3122=self.conv1661(D112)
        elif  num212==662:
            p3122=self.conv1662(D112)
        elif  num212==663:
            p3122=self.conv1663(D112)
        elif  num212==664:
            p3122=self.conv1664(D112)
        elif  num212==665:
            p3122=self.conv1665(D112)
        elif  num212==666:
            p3122=self.conv1666(D112)
        elif  num212==667:
            p3122=self.conv1667(D112)
        elif  num212==668:
            p3122=self.conv1668(D112)
        elif  num212==669:
            p3122=self.conv1669(D112) 
        elif  num212==670:
            p3122=self.conv1670(D112)
        elif  num212==671:
            p3122=self.conv1671(D112)
        elif  num212==672:
            p3122=self.conv1672(D112)
        elif  num212==673:
            p3122=self.conv1673(D112)
        elif  num212==674:
            p3122=self.conv1674(D112)
        elif  num212==675:
            p3122=self.conv1675(D112)
        elif  num212==676:
            p3122=self.conv1676(D112)
        elif  num212==677:
            p3122=self.conv1677(D112)
        elif  num212==678:
            p3122=self.conv1678(D112)
        elif  num212==679:
            p3122=self.conv1679(D112)
        elif  num212==680:
            p3122=self.conv1680(D112)
        elif  num212==681:
            p3122=self.conv1681(D112)
        elif  num212==682:
            p3122=self.conv1682(D112)
        elif  num212==683:
            p3122=self.conv1683(D112)
        elif  num212==684:
            p3122=self.conv1684(D112)
        elif  num212==685:
            p3122=self.conv1685(D112)
        elif  num212==686:
            p3122=self.conv1686(D112)
        elif  num212==687:
            p3122=self.conv1687(D112)
        elif  num212==688:
            p3122=self.conv1688(D112)
        elif  num212==689:
            p3122=self.conv1689(D112)
        elif  num212==690:
            p3122=self.conv1690(D112)
        elif  num212==691:
            p3122=self.conv1691(D112)
        elif  num212==692:
            p3122=self.conv1692(D112)
        elif  num212==693:
            p3122=self.conv1693(D112)
        elif  num212==694:
            p3122=self.conv1694(D112)
        elif  num212==695:
            p3122=self.conv1695(D112)
        elif  num212==696:
            p3122=self.conv1696(D112)
        elif  num212==697:
            p3122=self.conv1697(D112)
        elif  num212==698:
            p3122=self.conv1698(D112)
        elif  num212==699:
            p3122=self.conv1699(D112)
        elif  num212==700:
            p3122=self.conv1700(D112)
        elif  num212==701:
            p3122=self.conv1701(D112)
        elif  num212==702:
            p3122=self.conv1702(D112)
        elif  num212==703:
            p3122=self.conv1703(D112)
        elif  num212==704:
            p3122=self.conv1704(D112)
        elif  num212==705:
            p3122=self.conv1705(D112)
        elif  num212==706:
            p3122=self.conv1706(D112)
        elif  num212==707:
            p3122=self.conv1707(D112)
        elif  num212==708:
            p3122=self.conv1708(D112)
        elif  num212==709:
            p3122=self.conv1709(D112)
        elif  num212==710:
            p3122=self.conv1710(D112)
        elif  num212==711:
            p3122=self.conv1711(D112)
        elif  num212==712:
            p3122=self.conv1712(D112)
        elif  num212==713:
            p3122=self.conv1713(D112)
        elif  num212==714:
            p3122=self.conv1714(D112)
        elif  num212==715:
            p3122=self.conv1715(D112)
        elif  num212==716:
            p3122=self.conv1716(D112)
        elif  num212==717:
            p3122=self.conv1717(D112)
        elif  num212==718:
            p3122=self.conv1718(D112)
        elif  num212==719:
            p3122=self.conv1719(D112)
        elif  num212==720:
            p3122=self.conv1720(D112)
        elif  num212==721:
            p3122=self.conv1721(D112)
        elif  num212==722:
            p3122=self.conv1722(D112)
        elif  num212==723:
            p3122=self.conv1723(D112)
        elif  num212==724:
            p3122=self.conv1724(D112)
        elif  num212==725:
            p3122=self.conv1725(D112)
        elif  num212==726:
            p3122=self.conv1726(D112)
        elif  num212==727:
            p3122=self.conv1727(D112)
        elif  num212==728:
            p3122=self.conv1728(D112)
        elif  num212==729:
            p3122=self.conv1729(D112)
        elif  num212==730:
            p3122=self.conv1730(D112)
        elif  num212==731:
            p3122=self.conv1731(D112)
        elif  num212==732:
            p3122=self.conv1732(D112)
        elif  num212==733:
            p3122=self.conv1733(D112)
        elif  num212==734:
            p3122=self.conv1734(D112)
        elif  num212==735:
            p3122=self.conv1735(D112)
        elif  num212==736:
            p3122=self.conv1736(D112)
        elif  num212==737:
            p3122=self.conv1737(D112)
        elif  num212==738:
            p3122=self.conv1738(D112)
        elif  num212==739:
            p3122=self.conv1739(D112)                                                                                                                          
        elif  num212==740:
            p3122=self.conv1740(D112)
        elif  num212==741:
            p3122=self.conv1741(D112)
        elif  num212==742:
            p3122=self.conv1742(D112)
        elif  num212==743:
            p3122=self.conv1743(D112)
        elif  num212==744:
            p3122=self.conv1744(D112)
        elif  num212==745:
            p3122=self.conv1745(D112)
        elif  num212==746:
            p3122=self.conv1746(D112)
        elif  num212==747:
            p3122=self.conv1747(D112)
        elif  num212==748:
            p3122=self.conv1748(D112)
        elif  num212==749:
            p3122=self.conv1749(D112)
        elif  num212==750:
            p3122=self.conv1750(D112)
        elif  num212==751:
            p3122=self.conv1751(D112)
        elif  num212==752:
            p3122=self.conv1752(D112)
        elif  num212==753:
            p3122=self.conv1753(D112)
        elif  num212==754:
            p3122=self.conv1754(D112)
        elif  num212==755:
            p3122=self.conv1755(D112)
        elif  num212==756:
            p3122=self.conv1756(D112)
        elif  num212==757:
            p3122=self.conv1757(D112)
        elif  num212==758:
            p3122=self.conv1758(D112)
        elif  num212==759:
            p3122=self.conv1759(D112)
        elif  num212==760:
            p3122=self.conv1760(D112)
        elif  num212==761:
            p3122=self.conv1761(D112)
        elif  num212==762:
            p3122=self.conv1762(D112)
        elif  num212==763:
            p3122=self.conv1763(D112)
        elif  num212==764:
            p3122=self.conv1764(D112)
        elif  num212==765:
            p3122=self.conv1765(D112)
        elif  num212==766:
            p3122=self.conv1766(D112)
        elif  num212==767:
            p3122=self.conv1767(D112)
        elif  num212==768:
            p3122=self.conv1768(D112)
        elif  num212==769:
            p3122=self.conv1769(D112) 
        elif  num212==770:
            p3122=self.conv1770(D112)
        elif  num212==771:
            p3122=self.conv1771(D112)
        elif  num212==772:
            p3122=self.conv1772(D112)
        elif  num212==773:
            p3122=self.conv1773(D112)
        elif  num212==774:
            p3122=self.conv1774(D112)
        elif  num212==775:
            p3122=self.conv1775(D112)
        elif  num212==776:
            p3122=self.conv1776(D112)
        elif  num212==777:
            p3122=self.conv1777(D112)
        elif  num212==778:
            p3122=self.conv1778(D112)
        elif  num212==779:
            p3122=self.conv1779(D112) 
        elif  num212==780:
            p3122=self.conv1780(D112)
        elif  num212==781:
            p3122=self.conv1781(D112)
        elif  num212==782:
            p3122=self.conv1782(D112)
        elif  num212==783:
            p3122=self.conv1783(D112)
        elif  num212==784:
            p3122=self.conv1784(D112)
        elif  num212==785:
            p3122=self.conv1785(D112)
        elif  num212==786:
            p3122=self.conv1786(D112)
        elif  num212==787:
            p3122=self.conv1787(D112)
        elif  num212==788:
            p3122=self.conv1788(D112)
        elif  num212==789:
            p3122=self.conv1789(D112) 
        elif  num212==790:
            p3122=self.conv1790(D112)
        elif  num212==791:
            p3122=self.conv1791(D112)
        elif  num212==792:
            p3122=self.conv1792(D112)
        elif  num212==793:
            p3122=self.conv1793(D112)
        elif  num212==794:
            p3122=self.conv1794(D112)
        elif  num212==795:
            p3122=self.conv1795(D112)
        elif  num212==796:
            p3122=self.conv1796(D112)
        elif  num212==797:
            p3122=self.conv1797(D112)
        elif  num212==798:
            p3122=self.conv1798(D112)
        elif  num212==799:
            p3122=self.conv1799(D112) 
        elif  num212==800:
            p3122=self.conv1800(D112)
        elif  num212==801:
            p3122=self.conv1801(D112)
        elif  num212==802:
            p3122=self.conv1802(D112)
        elif  num212==803:
            p3122=self.conv1803(D112)
        elif  num212==804:
            p3122=self.conv1804(D112)
        elif  num212==805:
            p3122=self.conv1805(D112)
        elif  num212==806:
            p3122=self.conv1806(D112)
        elif  num212==807:
            p3122=self.conv1807(D112)
        elif  num212==808:
            p3122=self.conv1808(D112)
        elif  num212==809:
            p3122=self.conv1809(D112)
        elif  num212==810:
            p3122=self.conv1810(D112)
        elif  num212==811:
            p3122=self.conv1811(D112)
        elif  num212==812:
            p3122=self.conv1812(D112)
        elif  num212==813:
            p3122=self.conv1813(D112)
        elif  num212==814:
            p3122=self.conv1814(D112)
        elif  num212==815:
            p3122=self.conv1815(D112)
        elif  num212==816:
            p3122=self.conv1816(D112)
        elif  num212==817:
            p3122=self.conv1817(D112)
        elif  num212==818:
            p3122=self.conv1818(D112)
        elif  num212==819:
            p3122=self.conv1819(D112)
        elif  num212==820:
            p3122=self.conv1820(D112)
        elif  num212==821:
            p3122=self.conv1821(D112)
        elif  num212==822:
            p3122=self.conv1822(D112)
        elif  num212==823:
            p3122=self.conv1823(D112)
        elif  num212==824:
            p3122=self.conv1824(D112)
        elif  num212==825:
            p3122=self.conv1825(D112)
        elif  num212==826:
            p3122=self.conv1826(D112)
        elif  num212==827:
            p3122=self.conv1827(D112)
        elif  num212==828:
            p3122=self.conv1828(D112)
        elif  num212==829:
            p3122=self.conv1829(D112)                                                                                                                          
        elif  num212==830:
            p3122=self.conv1830(D112)
        elif  num212==831:
            p3122=self.conv1831(D112)
        elif  num212==832:
            p3122=self.conv1832(D112)
        elif  num212==833:
            p3122=self.conv1833(D112)
        elif  num212==834:
            p3122=self.conv1834(D112)
        elif  num212==835:
            p3122=self.conv1835(D112)
        elif  num212==836:
            p3122=self.conv1836(D112)
        elif  num212==837:
            p3122=self.conv1837(D112)
        elif  num212==838:
            p3122=self.conv1838(D112)
        elif  num212==839:
            p3122=self.conv1839(D112)
        elif  num212==840:
            p3122=self.conv1840(D112)
        elif  num212==841:
            p3122=self.conv1841(D112)
        elif  num212==842:
            p3122=self.conv1842(D112)
        elif  num212==843:
            p3122=self.conv1843(D112)
        elif  num212==844:
            p3122=self.conv1844(D112)
        elif  num212==845:
            p3122=self.conv1845(D112)
        elif  num212==846:
            p3122=self.conv1846(D112)
        elif  num212==847:
            p3122=self.conv1847(D112)
        elif  num212==848:
            p3122=self.conv1848(D112)
        elif  num212==849:
            p3122=self.conv1849(D112)
        elif  num212==850:
            p3122=self.conv1850(D112)
        elif  num212==851:
            p3122=self.conv1851(D112)
        elif  num212==852:
            p3122=self.conv1852(D112)
        elif  num212==853:
            p3122=self.conv1853(D112)
        elif  num212==854:
            p3122=self.conv1854(D112)
        elif  num212==855:
            p3122=self.conv1855(D112)
        elif  num212==856:
            p3122=self.conv1856(D112)
        elif  num212==857:
            p3122=self.conv1857(D112)
        elif  num212==858:
            p3122=self.conv1858(D112)
        elif  num212==859:
            p3122=self.conv1859(D112)
        elif  num212==860:
            p3122=self.conv1860(D112)
        elif  num212==861:
            p3122=self.conv1861(D112)
        elif  num212==862:
            p3122=self.conv1862(D112)
        elif  num212==863:
            p3122=self.conv1863(D112)
        elif  num212==864:
            p3122=self.conv1864(D112)
        elif  num212==865:
            p3122=self.conv1865(D112)
        elif  num212==866:
            p3122=self.conv1866(D112)
        elif  num212==867:
            p3122=self.conv1867(D112)
        elif  num212==868:
            p3122=self.conv1868(D112)
        elif  num212==869:
            p3122=self.conv1869(D112) 
        elif  num212==870:
            p3122=self.conv1870(D112)
        elif  num212==871:
            p3122=self.conv1871(D112)
        elif  num212==872:
            p3122=self.conv1872(D112)
        elif  num212==873:
            p3122=self.conv1873(D112)
        elif  num212==874:
            p3122=self.conv1874(D112)
        elif  num212==875:
            p3122=self.conv1875(D112)
        elif  num212==876:
            p3122=self.conv1876(D112)
        elif  num212==877:
            p3122=self.conv1877(D112)
        elif  num212==878:
            p3122=self.conv1878(D112)
        elif  num212==879:
            p3122=self.conv1879(D112)
        elif  num212==880:
            p3122=self.conv1880(D112)
        elif  num212==881:
            p3122=self.conv1881(D112)
        elif  num212==882:
            p3122=self.conv1882(D112)
        elif  num212==883:
            p3122=self.conv1883(D112)
        elif  num212==884:
            p3122=self.conv1884(D112)
        elif  num212==885:
            p3122=self.conv1885(D112)
        elif  num212==886:
            p3122=self.conv1886(D112)
        elif  num212==887:
            p3122=self.conv1887(D112)
        elif  num212==888:
            p3122=self.conv1888(D112)
        elif  num212==889:
            p3122=self.conv1889(D112)  
        elif  num212==890:
            p3122=self.conv1890(D112)
        elif  num212==891:
            p3122=self.conv1891(D112)
        elif  num212==892:
            p3122=self.conv1892(D112)
        elif  num212==893:
            p3122=self.conv1893(D112)
        elif  num212==894:
            p3122=self.conv1894(D112)
        elif  num212==895:
            p3122=self.conv1895(D112)
        elif  num212==896:
            p3122=self.conv1896(D112)
        elif  num212==897:
            p3122=self.conv1897(D112)
        elif  num212==898:
            p3122=self.conv1898(D112)
        elif  num212==899:
            p3122=self.conv1899(D112)
        elif  num212==900:
            p3122=self.conv1900(D112)
        elif  num212==901:
            p3122=self.conv1901(D112)
        elif  num212==902:
            p3122=self.conv1902(D112)
        elif  num212==903:
            p3122=self.conv1903(D112)
        elif  num212==904:
            p3122=self.conv1904(D112)
        elif  num212==905:
            p3122=self.conv1905(D112)
        elif  num212==906:
            p3122=self.conv1906(D112)
        elif  num212==907:
            p3122=self.conv1907(D112)
        elif  num212==908:
            p3122=self.conv1908(D112)
        elif  num212==909:
            p3122=self.conv1909(D112)
        elif  num212==910:
            p3122=self.conv1910(D112)
        elif  num212==911:
            p3122=self.conv1911(D112)
        elif  num212==912:
            p3122=self.conv1912(D112)
        elif  num212==913:
            p3122=self.conv1913(D112)
        elif  num212==914:
            p3122=self.conv1914(D112)
        elif  num212==915:
            p3122=self.conv1915(D112)
        elif  num212==916:
            p3122=self.conv1916(D112)
        elif  num212==917:
            p3122=self.conv1917(D112)
        elif  num212==918:
            p3122=self.conv1918(D112)
        elif  num212==919:
            p3122=self.conv1919(D112)
        elif  num212==920:
            p3122=self.conv1920(D112)
        elif  num212==921:
            p3122=self.conv1921(D112)
        elif  num212==922:
            p3122=self.conv1922(D112)
        elif  num212==923:
            p3122=self.conv1923(D112)
        elif  num212==924:
            p3122=self.conv1924(D112)
        elif  num212==925:
            p3122=self.conv1925(D112)
        elif  num212==926:
            p3122=self.conv1926(D112)
        elif  num212==927:
            p3122=self.conv1927(D112)
        elif  num212==928:
            p3122=self.conv1928(D112)
        elif  num212==929:
            p3122=self.conv1929(D112)
        elif  num212==930:
            p3122=self.conv1930(D112)
        elif  num212==931:
            p3122=self.conv1931(D112)
        elif  num212==932:
            p3122=self.conv1932(D112)
        elif  num212==933:
            p3122=self.conv1933(D112)
        elif  num212==934:
            p3122=self.conv1934(D112)
        elif  num212==935:
            p3122=self.conv1935(D112)
        elif  num212==936:
            p3122=self.conv1936(D112)
        elif  num212==937:
            p3122=self.conv1937(D112)
        elif  num212==938:
            p3122=self.conv1938(D112)
        elif  num212==939:
            p3122=self.conv1939(D112) 
        elif  num212==940:
            p3122=self.conv1940(D112)
        elif  num212==941:
            p3122=self.conv1941(D112)
        elif  num212==942:
            p3122=self.conv1942(D112)
        elif  num212==943:
            p3122=self.conv1943(D112)
        elif  num212==944:
            p3122=self.conv1944(D112)
        elif  num212==945:
            p3122=self.conv1945(D112)
        elif  num212==946:
            p3122=self.conv1946(D112)
        elif  num212==947:
            p3122=self.conv1947(D112)
        elif  num212==948:
            p3122=self.conv1948(D112)
        elif  num212==949:
            p3122=self.conv1949(D112)                                                                                                                          
        elif  num212==950:
            p3122=self.conv1950(D112)
        elif  num212==951:
            p3122=self.conv1951(D112)
        elif  num212==952:
            p3122=self.conv1952(D112)
        elif  num212==953:
            p3122=self.conv1953(D112)
        elif  num212==954:
            p3122=self.conv1954(D112)
        elif  num212==955:
            p3122=self.conv1955(D112)
        elif  num212==956:
            p3122=self.conv1956(D112)
        elif  num212==957:
            p3122=self.conv1957(D112)
        elif  num212==958:
            p3122=self.conv1958(D112)
        elif  num212==959:
            p3122=self.conv1959(D112)
        elif  num212==960:
            p3122=self.conv1960(D112)
        elif  num212==961:
            p3122=self.conv1961(D112)
        elif  num212==962:
            p3122=self.conv1962(D112)
        elif  num212==963:
            p3122=self.conv1963(D112)
        elif  num212==964:
            p3122=self.conv1964(D112)
        elif  num212==965:
            p3122=self.conv1965(D112)
        elif  num212==966:
            p3122=self.conv1966(D112)
        elif  num212==967:
            p3122=self.conv1967(D112)
        elif  num212==968:
            p3122=self.conv1968(D112)
        elif  num212==969:
            p3122=self.conv1969(D112) 
        elif  num212==970:
            p3122=self.conv1970(D112)
        elif  num212==971:
            p3122=self.conv1971(D112)
        elif  num212==972:
            p3122=self.conv1972(D112)
        elif  num212==973:
            p3122=self.conv1973(D112)
        elif  num212==974:
            p3122=self.conv1974(D112)
        elif  num212==975:
            p3122=self.conv1975(D112)
        elif  num212==976:
            p3122=self.conv1976(D112)
        elif  num212==977:
            p3122=self.conv1977(D112)
        elif  num212==978:
            p3122=self.conv1978(D112)
        elif  num212==979:
            p3122=self.conv1979(D112) 
        elif  num212==980:
            p3122=self.conv1980(D112)
        elif  num212==981:
            p3122=self.conv1981(D112)
        elif  num212==982:
            p3122=self.conv1982(D112)
        elif  num212==983:
            p3122=self.conv1983(D112)
        elif  num212==984:
            p3122=self.conv1984(D112)
        elif  num212==985:
            p3122=self.conv1985(D112)
        elif  num212==986:
            p3122=self.conv1986(D112)
        elif  num212==987:
            p3122=self.conv1987(D112)
        elif  num212==988:
            p3122=self.conv1988(D112)
        elif  num212==989:
            p3122=self.conv1989(D112)
        elif  num212==990:
            p3122=self.conv1990(D112)
        elif  num212==991:
            p3122=self.conv1991(D112)
        elif  num212==992:
            p3122=self.conv1992(D112)
        elif  num212==993:
            p3122=self.conv1993(D112)
        elif  num212==994:
            p3122=self.conv1994(D112)
        elif  num212==995:
            p3122=self.conv1995(D112)
        elif  num212==996:
            p3122=self.conv1996(D112)
        elif  num212==997:
            p3122=self.conv1997(D112)
        elif  num212==998:
            p3122=self.conv1998(D112)
        elif  num212==999:
            p3122=self.conv1999(D112) 
        elif  num212==1000:
            p3122=self.conv11000(D112)
        elif  num212==1001:
            p3122=self.conv11001(D112)
        elif  num212==1002:
            p3122=self.conv11002(D112)
        elif  num212==1003:
            p3122=self.conv11003(D112)
        elif  num212==1004:
            p3122=self.conv11004(D112)
        elif  num212==1005:
            p3122=self.conv11005(D112)
        elif  num212==1006:
            p3122=self.conv11006(D112)
        elif  num212==1007:
            p3122=self.conv11007(D112)
        elif  num212==1008:
            p3122=self.conv11008(D112)
        elif  num212==1009:
            p3122=self.conv11009(D112) 
        elif  num212==1010:
            p3122=self.conv11010(D112)
        elif  num212==1011:
            p3122=self.conv11011(D112)
        elif  num212==1012:
            p3122=self.conv11012(D112)
        elif  num212==1013:
            p3122=self.conv11013(D112)
        elif  num212==1014:
            p3122=self.conv11014(D112)
        elif  num212==1015:
            p3122=self.conv11015(D112)
        elif  num212==1016:
            p3122=self.conv11016(D112)
        elif  num212==1017:
            p3122=self.conv11017(D112)
        elif  num212==1018:
            p3122=self.conv11018(D112)
        elif  num212==1019:
            p3122=self.conv11019(D112)
        elif  num212==1020:
            p3122=self.conv11020(D112)
        elif  num212==1021:
            p3122=self.conv11021(D112)
        elif  num212==1022:
            p3122=self.conv11022(D112)
        elif  num212==1023:
            p3122=self.conv11023(D112)
        elif  num212==1024:
            p3122=self.conv11024(D112) 
            
        if num0120==1:
            p3120=self.conv11(B1120)
        elif num0120==2:
            p3120=self.conv12(B1120)
        elif num0120==3:
            p3120=self.conv13(B1120)
        elif num0120==4:
            p3120=self.conv14(B1120)
        elif num0120==5:
            p3120=self.conv15(B1120)
        elif num0120==6:
            p3120=self.conv16(B1120)
        elif num0120==7:
            p3120=self.conv17(B1120)
        elif num0120==8:
            p3120=self.conv18(B1120)
        elif num0120==9:
            p3120=self.conv19(B1120)
        elif num0120==10:
            p3120=self.conv110(B1120)
        elif num0120==11:
            p3120=self.conv111(B1120)
        elif num0120==12:
            p3120=self.conv112(B1120)
        elif num0120==13:
            p3120=self.conv113(B1120)
        elif num0120==14:
            p3120=self.conv114(B1120)
        elif num0120==15:
            p3120=self.conv115(B1120)
        elif num0120==16:
            p3120=self.conv116(B1120)
        elif num0120==17:
            p3120=self.conv117(B1120)
        elif num0120==18:
            p3120=self.conv118(B1120)
        elif num0120==19:
            p3120=self.conv119(B1120)
        elif num0120==20:
            p3120=self.conv120(B1120)
        elif num0120==21:
            p3120=self.conv121(B1120)
        elif num0120==22:
            p3120=self.conv122(B1120)
        elif num0120==23:
            p3120=self.conv123(B1120)
        elif num0120==24:
            p3120=self.conv124(B1120)
        elif num0120==25:
            p3120=self.conv125(B1120)
        elif num0120==26:
            p3120=self.conv126(B1120)
        elif num0120==27:
            p3120=self.conv127(B1120)
        elif num0120==28:
            p3120=self.conv128(B1120)
        elif num0120==29:
            p3120=self.conv129(B1120)
        elif num0120==30:
            p3120=self.conv130(B1120)
        elif num0120==31:
            p3120=self.conv131(B1120)
        elif num0120==32:
            p3120=self.conv132(B1120)
        elif num0120==33:
            p3120=self.conv133(B1120)
        elif num0120==34:
            p3120=self.conv134(B1120)
        elif num0120==35:
            p3120=self.conv135(B1120)
        elif num0120==36:
            p3120=self.conv136(B1120)
        elif num0120==37:
            p3120=self.conv137(B1120)
        elif num0120==38:
            p3120=self.conv138(B1120)
        elif num0120==39:
            p3120=self.conv139(B1120)
        elif num0120==40:
            p3120=self.conv140(B1120)
        elif num0120==41:
            p3120=self.conv141(B1120)
        elif num0120==42:
            p3120=self.conv142(B1120)
        elif num0120==43:
            p3120=self.conv143(B1120)
        elif num0120==44:
            p3120=self.conv144(B1120)
        elif num0120==45:
            p3120=self.conv145(B1120)
        elif num0120==46:
            p3120=self.conv146(B1120)
        elif num0120==47:
            p3120=self.conv147(B1120)
        elif num0120==48:
            p3120=self.conv148(B1120)
        elif num0120==49:
            p3120=self.conv149(B1120)
        elif num0120==50:
            p3120=self.conv150(B1120)
        elif num0120==51:
            p3120=self.conv151(B1120)
        elif num0120==52:
            p3120=self.conv152(B1120)
        elif num0120==53:
            p3120=self.conv153(B1120)
        elif num0120==54:
            p3120=self.conv154(B1120)
        elif num0120==55:
            p3120=self.conv155(B1120)
        elif num0120==56:
            p3120=self.conv156(B1120)
        elif num0120==57:
            p3120=self.conv157(B1120)
        elif num0120==58:
            p3120=self.conv158(B1120)
        elif num0120==59:
            p3120=self.conv159(B1120)
        elif num0120==60:
            p3120=self.conv160(B1120)
        elif num0120==61:
            p3120=self.conv161(B1120)
        elif num0120==62:
            p3120=self.conv162(B1120)
        elif num0120==63:
            p3120=self.conv163(B1120)
        elif num0120==64:
            p3120=self.conv164(B1120)
        
        if  num1120==1:
            p31201=self.conv11(C1120)
        elif  num1120==2:
            p31201=self.conv12(C1120)
        elif  num1120==3:
            p31201=self.conv13(C1120)
        elif  num1120==4:
            p31201=self.conv14(C1120)
        elif  num1120==5:
            p31201=self.conv15(C1120)
        elif  num1120==6:
            p31201=self.conv16(C1120)
        elif  num1120==7:
            p31201=self.conv17(C1120)
        elif  num1120==8:
            p31201=self.conv18(C1120)
        elif  num1120==9:
            p31201=self.conv19(C1120)
        elif  num1120==10:
            p31201=self.conv110(C1120)
        elif  num1120==11:
            p31201=self.conv111(C1120)
        elif  num1120==12:
            p31201=self.conv112(C1120)
        elif  num1120==13:
            p31201=self.conv113(C1120)
        elif  num1120==14:
            p31201=self.conv114(C1120)
        elif  num1120==15:
            p31201=self.conv115(C1120)
        elif  num1120==16:
            p31201=self.conv116(C1120)
        elif  num1120==17:
            p31201=self.conv117(C1120)
        elif  num1120==18:
            p31201=self.conv118(C1120)
        elif  num1120==19:
            p31201=self.conv119(C1120)
        elif  num1120==20:
            p31201=self.conv120(C1120)
        elif  num1120==21:
            p31201=self.conv121(C1120)
        elif  num1120==22:
            p31201=self.conv122(C1120)
        elif  num1120==23:
            p31201=self.conv123(C1120)
        elif  num1120==24:
            p31201=self.conv124(C1120)
        elif  num1120==25:
            p31201=self.conv125(C1120)
        elif  num1120==26:
            p31201=self.conv126(C1120)
        elif  num1120==27:
            p31201=self.conv127(C1120)
        elif  num1120==28:
            p31201=self.conv128(C1120)
        elif  num1120==29:
            p31201=self.conv129(C1120)
        elif  num1120==30:
            p31201=self.conv130(C1120)
        elif  num1120==31:
            p31201=self.conv131(C1120)
        elif  num1120==32:
            p31201=self.conv132(C1120)
        elif  num1120==33:
            p31201=self.conv133(C1120)
        elif  num1120==34:
            p31201=self.conv134(C1120)
        elif  num1120==35:
            p31201=self.conv135(C1120)
        elif  num1120==36:
            p31201=self.conv136(C1120)
        elif  num1120==37:
            p31201=self.conv137(C1120)
        elif  num1120==38:
            p31201=self.conv138(C1120)
        elif  num1120==39:
            p31201=self.conv139(C1120)
        elif  num1120==40:
            p31201=self.conv140(C1120)
        elif  num1120==41:
            p31201=self.conv141(C1120)
        elif  num1120==42:
            p31201=self.conv142(C1120)
        elif  num1120==43:
            p31201=self.conv143(C1120)
        elif  num1120==44:
            p31201=self.conv144(C1120)
        elif  num1120==45:
            p31201=self.conv145(C1120)
        elif  num1120==46:
            p31201=self.conv146(C1120)
        elif  num1120==47:
            p31201=self.conv147(C1120)
        elif  num1120==48:
            p31201=self.conv148(C1120)
        elif  num1120==49:
            p31201=self.conv149(C1120)
        elif  num1120==50:
            p31201=self.conv150(C1120)
        elif  num1120==51:
            p31201=self.conv151(C1120)
        elif  num1120==52:
            p31201=self.conv152(C1120)
        elif  num1120==53:
            p31201=self.conv153(C1120)
        elif  num1120==54:
            p31201=self.conv154(C1120)
        elif  num1120==55:
            p31201=self.conv155(C1120)
        elif  num1120==56:
            p31201=self.conv156(C1120)
        elif  num1120==57:
            p31201=self.conv157(C1120)
        elif  num1120==58:
            p31201=self.conv158(C1120)
        elif  num1120==59:
            p31201=self.conv159(C1120)
        elif  num1120==60:
            p31201=self.conv160(C1120)
        elif  num1120==61:
            p31201=self.conv161(C1120)
        elif  num1120==62:
            p31201=self.conv162(C1120)
        elif  num1120==63:
            p31201=self.conv163(C1120)
        elif  num1120==64:
            p31201=self.conv164(C1120)
        elif  num1120==65:
            p31201=self.conv165(C1120)
        elif  num1120==66:
            p31201=self.conv166(C1120)
        elif  num1120==67:
            p31201=self.conv167(C1120)
        elif  num1120==68:
            p31201=self.conv168(C1120)
        elif  num1120==69:
            p31201=self.conv169(C1120)
        elif  num1120==70:
            p31201=self.conv170(C1120)
        elif  num1120==71:
            p31201=self.conv171(C1120)
        elif  num1120==72:
            p31201=self.conv172(C1120)
        elif  num1120==73:
            p31201=self.conv173(C1120)
        elif  num1120==74:
            p31201=self.conv174(C1120)
        elif  num1120==75:
            p31201=self.conv175(C1120)
        elif  num1120==76:
            p31201=self.conv176(C1120)
        elif  num1120==77:
            p31201=self.conv177(C1120)
        elif  num1120==78:
            p31201=self.conv178(C1120)
        elif  num1120==79:
            p31201=self.conv179(C1120)
        elif  num1120==80:
            p31201=self.conv180(C1120)
        elif  num1120==81:
            p31201=self.conv181(C1120)
        elif  num1120==82:
            p31201=self.conv182(C1120)
        elif  num1120==83:
            p31201=self.conv183(C1120)
        elif  num1120==84:
            p31201=self.conv184(C1120)
        elif  num1120==85:
            p31201=self.conv185(C1120)
        elif  num1120==86:
            p31201=self.conv186(C1120)
        elif  num1120==87:
            p31201=self.conv187(C1120)
        elif  num1120==88:
            p31201=self.conv188(C1120)
        elif  num1120==89:
            p31201=self.conv189(C1120)    
        elif  num1120==90:
            p31201=self.conv190(C1120)
        elif  num1120==91:
            p31201=self.conv191(C1120)
        elif  num1120==92:
            p31201=self.conv192(C1120)
        elif  num1120==93:
            p31201=self.conv193(C1120)
        elif  num1120==94:
            p31201=self.conv194(C1120)
        elif  num1120==95:
            p31201=self.conv195(C1120)
        elif  num1120==96:
            p31201=self.conv196(C1120)
        elif  num1120==97:
            p31201=self.conv197(C1120)
        elif  num1120==98:
            p31201=self.conv198(C1120)
        elif  num1120==99:
            p31201=self.conv199(C1120) 
        elif  num1120==100:
            p31201=self.conv1100(C1120)
        elif  num1120==101:
            p31201=self.conv1101(C1120)
        elif  num1120==102:
            p31201=self.conv1102(C1120)
        elif  num1120==103:
            p31201=self.conv1103(C1120)
        elif  num1120==104:
            p31201=self.conv1104(C1120)
        elif  num1120==105:
            p31201=self.conv1105(C1120)
        elif  num1120==106:
            p31201=self.conv1106(C1120)
        elif  num1120==107:
            p31201=self.conv1107(C1120)
        elif  num1120==108:
            p31201=self.conv1108(C1120)
        elif  num1120==109:
            p31201=self.conv1109(C1120)
        elif  num1120==110:
            p31201=self.conv1110(C1120)
        elif  num1120==111:
            p31201=self.conv1111(C1120)
        elif  num1120==112:
            p31201=self.conv1112(C1120)
        elif  num1120==113:
            p31201=self.conv1113(C1120)
        elif  num1120==114:
            p31201=self.conv1114(C1120)
        elif  num1120==115:
            p31201=self.conv1115(C1120)
        elif  num1120==116:
            p31201=self.conv1116(C1120)
        elif  num1120==117:
            p31201=self.conv1117(C1120)
        elif  num1120==118:
            p31201=self.conv1118(C1120)
        elif  num1120==119:
            p31201=self.conv1119(C1120) 
        elif  num1120==120:
            p31201=self.conv1120(C1120)
        elif  num1120==121:
            p31201=self.conv1121(C1120)
        elif  num1120==122:
            p31201=self.conv1122(C1120)
        elif  num1120==123:
            p31201=self.conv1123(C1120)
        elif  num1120==124:
            p31201=self.conv1124(C1120)
        elif  num1120==125:
            p31201=self.conv1125(C1120)
        elif  num1120==126:
            p31201=self.conv1126(C1120)
        elif  num1120==127:
            p31201=self.conv1127(C1120)
        elif  num1120==128:
            p31201=self.conv1128(C1120)
        elif  num1120==129:
            p31201=self.conv1129(C1120) 
        elif  num1120==130:
            p31201=self.conv1130(C1120)
        elif  num1120==131:
            p31201=self.conv1131(C1120)
        elif  num1120==132:
            p31201=self.conv1132(C1120)
        elif  num1120==133:
            p31201=self.conv1133(C1120)
        elif  num1120==134:
            p31201=self.conv1134(C1120)
        elif  num1120==135:
            p31201=self.conv1135(C1120)
        elif  num1120==136:
            p31201=self.conv1136(C1120)
        elif  num1120==137:
            p31201=self.conv1137(C1120)
        elif  num1120==138:
            p31201=self.conv1138(C1120)
        elif  num1120==139:
            p31201=self.conv1139(C1120)
        elif  num1120==140:
            p31201=self.conv1140(C1120)
        elif  num1120==141:
            p31201=self.conv1141(C1120)
        elif  num1120==142:
            p31201=self.conv1142(C1120)
        elif  num1120==143:
            p31201=self.conv1143(C1120)
        elif  num1120==144:
            p31201=self.conv1144(C1120)
        elif  num1120==145:
            p31201=self.conv1145(C1120)
        elif  num1120==146:
            p31201=self.conv1146(C1120)
        elif  num1120==147:
            p31201=self.conv1147(C1120)
        elif  num1120==148:
            p31201=self.conv1148(C1120)
        elif  num1120==149:
            p31201=self.conv1149(C1120) 
        elif  num1120==150:
            p31201=self.conv1150(C1120)
        elif  num1120==151:
            p31201=self.conv1151(C1120)
        elif  num1120==152:
            p31201=self.conv1152(C1120)
        elif  num1120==153:
            p31201=self.conv1153(C1120)
        elif  num1120==154:
            p31201=self.conv1154(C1120)
        elif  num1120==155:
            p31201=self.conv1155(C1120)
        elif  num1120==156:
            p31201=self.conv1156(C1120)
        elif  num1120==157:
            p31201=self.conv1157(C1120)
        elif  num1120==158:
            p31201=self.conv1158(C1120)
        elif  num1120==159:
            p31201=self.conv1159(C1120) 
        elif  num1120==160:
            p31201=self.conv1160(C1120)
        elif  num1120==161:
            p31201=self.conv1161(C1120)
        elif  num1120==162:
            p31201=self.conv1162(C1120)
        elif  num1120==163:
            p31201=self.conv1163(C1120)
        elif  num1120==164:
            p31201=self.conv1164(C1120)
        elif  num1120==165:
            p31201=self.conv1165(C1120)
        elif  num1120==166:
            p31201=self.conv1166(C1120)
        elif  num1120==167:
            p31201=self.conv1167(C1120)
        elif  num1120==168:
            p31201=self.conv1168(C1120)
        elif  num1120==169:
            p31201=self.conv1169(C1120) 
        elif  num1120==170:
            p31201=self.conv1170(C1120)
        elif  num1120==171:
            p31201=self.conv1171(C1120)
        elif  num1120==172:
            p31201=self.conv1172(C1120)
        elif  num1120==173:
            p31201=self.conv1173(C1120)
        elif  num1120==174:
            p31201=self.conv1174(C1120)
        elif  num1120==175:
            p31201=self.conv1175(C1120)
        elif  num1120==176:
            p31201=self.conv1176(C1120)
        elif  num1120==177:
            p31201=self.conv1177(C1120)
        elif  num1120==178:
            p31201=self.conv1178(C1120)
        elif  num1120==179:
            p31201=self.conv1179(C1120)                                                                                              
        elif  num1120==180:
            p31201=self.conv1180(C1120)
        elif  num1120==181:
            p31201=self.conv1181(C1120)
        elif  num1120==182:
            p31201=self.conv1182(C1120)
        elif  num1120==183:
            p31201=self.conv1183(C1120)
        elif  num1120==184:
            p31201=self.conv1184(C1120)
        elif  num1120==185:
            p31201=self.conv1185(C1120)
        elif  num1120==186:
            p31201=self.conv1186(C1120)
        elif  num1120==187:
            p31201=self.conv1187(C1120)
        elif  num1120==188:
            p31201=self.conv1188(C1120)
        elif  num1120==189:
            p31201=self.conv1189(C1120) 
        elif  num1120==190:
            p31201=self.conv1190(C1120)
        elif  num1120==191:
            p31201=self.conv1191(C1120)
        elif  num1120==192:
            p31201=self.conv1192(C1120)
        elif  num1120==193:
            p31201=self.conv1193(C1120)
        elif  num1120==194:
            p31201=self.conv1194(C1120)
        elif  num1120==195:
            p31201=self.conv1195(C1120)
        elif  num1120==196:
            p31201=self.conv1196(C1120)
        elif  num1120==197:
            p31201=self.conv1197(C1120)
        elif  num1120==198:
            p31201=self.conv1198(C1120)
        elif  num1120==199:
            p31201=self.conv1199(C1120)
        elif  num1120==200:
            p31201=self.conv1200(C1120)
        elif  num1120==201:
            p31201=self.conv1201(C1120)
        elif  num1120==202:
            p31201=self.conv1202(C1120)
        elif  num1120==203:
            p31201=self.conv1203(C1120)
        elif  num1120==204:
            p31201=self.conv1204(C1120)
        elif  num1120==205:
            p31201=self.conv1205(C1120)
        elif  num1120==206:
            p31201=self.conv1206(C1120)
        elif  num1120==207:
            p31201=self.conv1207(C1120)
        elif  num1120==208:
            p31201=self.conv1208(C1120)
        elif  num1120==209:
            p31201=self.conv1209(C1120)
        elif  num1120==210:
            p31201=self.conv1210(C1120)
        elif  num1120==211:
            p31201=self.conv1211(C1120)
        elif  num1120==212:
            p31201=self.conv1212(C1120)
        elif  num1120==213:
            p31201=self.conv1213(C1120)
        elif  num1120==214:
            p31201=self.conv1214(C1120)
        elif  num1120==215:
            p31201=self.conv1215(C1120)
        elif  num1120==216:
            p31201=self.conv1216(C1120)
        elif  num1120==217:
            p31201=self.conv1217(C1120)
        elif  num1120==218:
            p31201=self.conv1218(C1120)
        elif  num1120==219:
            p31201=self.conv1219(C1120)
        elif  num1120==220:
            p31201=self.conv1220(C1120)
        elif  num1120==221:
            p31201=self.conv1221(C1120)
        elif  num1120==222:
            p31201=self.conv1222(C1120)
        elif  num1120==223:
            p31201=self.conv1223(C1120)
        elif  num1120==224:
            p31201=self.conv1224(C1120)
        elif  num1120==225:
            p31201=self.conv1225(C1120)
        elif  num1120==226:
            p31201=self.conv1226(C1120)
        elif  num1120==227:
            p31201=self.conv1227(C1120)
        elif  num1120==228:
            p31201=self.conv1228(C1120)
        elif  num1120==229:
            p31201=self.conv1229(C1120)
        elif  num1120==230:
            p31201=self.conv1230(C1120)
        elif  num1120==231:
            p31201=self.conv1231(C1120)
        elif  num1120==232:
            p31201=self.conv1232(C1120)
        elif  num1120==233:
            p31201=self.conv1233(C1120)
        elif  num1120==234:
            p31201=self.conv1234(C1120)
        elif  num1120==235:
            p31201=self.conv1235(C1120)
        elif  num1120==236:
            p31201=self.conv1236(C1120)
        elif  num1120==237:
            p31201=self.conv1237(C1120)
        elif  num1120==238:
            p31201=self.conv1238(C1120)
        elif  num1120==239:
            p31201=self.conv1239(C1120) 
        elif  num1120==240:
            p31201=self.conv1240(C1120)
        elif  num1120==241:
            p31201=self.conv1241(C1120)
        elif  num1120==242:
            p31201=self.conv1242(C1120)
        elif  num1120==243:
            p31201=self.conv1243(C1120)
        elif  num1120==244:
            p31201=self.conv1244(C1120)
        elif  num1120==245:
            p31201=self.conv1245(C1120)
        elif  num1120==246:
            p31201=self.conv1246(C1120)
        elif  num1120==247:
            p31201=self.conv1247(C1120)
        elif  num1120==248:
            p31201=self.conv1248(C1120)
        elif  num1120==249:
            p31201=self.conv1249(C1120)
        elif  num1120==250:
            p31201=self.conv1250(C1120)
        elif  num1120==251:
            p31201=self.conv1251(C1120)
        elif  num1120==252:
            p31201=self.conv1252(C1120)
        elif  num1120==253:
            p31201=self.conv1253(C1120)
        elif  num1120==254:
            p31201=self.conv1254(C1120)
        elif  num1120==255:
            p31201=self.conv1255(C1120)
        elif  num1120==256:
            p31201=self.conv1256(C1120)
            
        if  num2120==1:
            p31202=self.conv11(D1120)
        elif  num2120==2:
            p31202=self.conv12(D1120)
        elif  num2120==3:
            p31202=self.conv13(D1120)
        elif  num2120==4:
            p31202=self.conv14(D1120)
        elif  num2120==5:
            p31202=self.conv15(D1120)
        elif  num2120==6:
            p31202=self.conv16(D1120)
        elif  num2120==7:
            p31202=self.conv17(D1120)
        elif  num2120==8:
            p31202=self.conv18(D1120)
        elif  num2120==9:
            p31202=self.conv19(D1120)
        elif  num2120==10:
            p31202=self.conv110(D1120)
        elif  num2120==11:
            p31202=self.conv111(D1120)
        elif  num2120==12:
            p31202=self.conv112(D1120)
        elif  num2120==13:
            p31202=self.conv113(D1120)
        elif  num2120==14:
            p31202=self.conv114(D1120)
        elif  num2120==15:
            p31202=self.conv115(D1120)
        elif  num2120==16:
            p31202=self.conv116(D1120)
        elif  num2120==17:
            p31202=self.conv117(D1120)
        elif  num2120==18:
            p31202=self.conv118(D1120)
        elif  num2120==19:
            p31202=self.conv119(D1120)
        elif  num2120==20:
            p31202=self.conv120(D1120)
        elif  num2120==21:
            p31202=self.conv121(D1120)
        elif  num2120==22:
            p31202=self.conv122(D1120)
        elif  num2120==23:
            p31202=self.conv123(D1120)
        elif  num2120==24:
            p31202=self.conv124(D1120)
        elif  num2120==25:
            p31202=self.conv125(D1120)
        elif  num2120==26:
            p31202=self.conv126(D1120)
        elif  num2120==27:
            p31202=self.conv127(D1120)
        elif  num2120==28:
            p31202=self.conv128(D1120)
        elif  num2120==29:
            p31202=self.conv129(D1120)
        elif  num2120==30:
            p31202=self.conv130(D1120)
        elif  num2120==31:
            p31202=self.conv131(D1120)
        elif  num2120==32:
            p31202=self.conv132(D1120)
        elif  num2120==33:
            p31202=self.conv133(D1120)
        elif  num2120==34:
            p31202=self.conv134(D1120)
        elif  num2120==35:
            p31202=self.conv135(D1120)
        elif  num2120==36:
            p31202=self.conv136(D1120)
        elif  num2120==37:
            p31202=self.conv137(D1120)
        elif  num2120==38:
            p31202=self.conv138(D1120)
        elif  num2120==39:
            p31202=self.conv139(D1120)
        elif  num2120==40:
            p31202=self.conv140(D1120)
        elif  num2120==41:
            p31202=self.conv141(D1120)
        elif  num2120==42:
            p31202=self.conv142(D1120)
        elif  num2120==43:
            p31202=self.conv143(D1120)
        elif  num2120==44:
            p31202=self.conv144(D1120)
        elif  num2120==45:
            p31202=self.conv145(D1120)
        elif  num2120==46:
            p31202=self.conv146(D1120)
        elif  num2120==47:
            p31202=self.conv147(D1120)
        elif  num2120==48:
            p31202=self.conv148(D1120)
        elif  num2120==49:
            p31202=self.conv149(D1120)
        elif  num2120==50:
            p31202=self.conv150(D1120)
        elif  num2120==51:
            p31202=self.conv151(D1120)
        elif  num2120==52:
            p31202=self.conv152(D1120)
        elif  num2120==53:
            p31202=self.conv153(D1120)
        elif  num2120==54:
            p31202=self.conv154(D1120)
        elif  num2120==55:
            p31202=self.conv155(D1120)
        elif  num2120==56:
            p31202=self.conv156(D1120)
        elif  num2120==57:
            p31202=self.conv157(D1120)
        elif  num2120==58:
            p31202=self.conv158(D1120)
        elif  num2120==59:
            p31202=self.conv159(D1120)
        elif  num2120==60:
            p31202=self.conv160(D1120)
        elif  num2120==61:
            p31202=self.conv161(D1120)
        elif  num2120==62:
            p31202=self.conv162(D1120)
        elif  num2120==63:
            p31202=self.conv163(D1120)
        elif  num2120==64:
            p31202=self.conv164(D1120)
        elif  num2120==65:
            p31202=self.conv165(D1120)
        elif  num2120==66:
            p31202=self.conv166(D1120)
        elif  num2120==67:
            p31202=self.conv167(D1120)
        elif  num2120==68:
            p31202=self.conv168(D1120)
        elif  num2120==69:
            p31202=self.conv169(D1120)
        elif  num2120==70:
            p31202=self.conv170(D1120)
        elif  num2120==71:
            p31202=self.conv171(D1120)
        elif  num2120==72:
            p31202=self.conv172(D1120)
        elif  num2120==73:
            p31202=self.conv173(D1120)
        elif  num2120==74:
            p31202=self.conv174(D1120)
        elif  num2120==75:
            p31202=self.conv175(D1120)
        elif  num2120==76:
            p31202=self.conv176(D1120)
        elif  num2120==77:
            p31202=self.conv177(D1120)
        elif  num2120==78:
            p31202=self.conv178(D1120)
        elif  num2120==79:
            p31202=self.conv179(D1120)
        elif  num2120==80:
            p31202=self.conv180(D1120)
        elif  num2120==81:
            p31202=self.conv181(D1120)
        elif  num2120==82:
            p31202=self.conv182(D1120)
        elif  num2120==83:
            p31202=self.conv183(D1120)
        elif  num2120==84:
            p31202=self.conv184(D1120)
        elif  num2120==85:
            p31202=self.conv185(D1120)
        elif  num2120==86:
            p31202=self.conv186(D1120)
        elif  num2120==87:
            p31202=self.conv187(D1120)
        elif  num2120==88:
            p31202=self.conv188(D1120)
        elif  num2120==89:
            p31202=self.conv189(D1120)    
        elif  num2120==90:
            p31202=self.conv190(D1120)
        elif  num2120==91:
            p31202=self.conv191(D1120)
        elif  num2120==92:
            p31202=self.conv192(D1120)
        elif  num2120==93:
            p31202=self.conv193(D1120)
        elif  num2120==94:
            p31202=self.conv194(D1120)
        elif  num2120==95:
            p31202=self.conv195(D1120)
        elif  num2120==96:
            p31202=self.conv196(D1120)
        elif  num2120==97:
            p31202=self.conv197(D1120)
        elif  num2120==98:
            p31202=self.conv198(D1120)
        elif  num2120==99:
            p31202=self.conv199(D1120) 
        elif  num2120==100:
            p31202=self.conv1100(D1120)
        elif  num2120==101:
            p31202=self.conv1101(D1120)
        elif  num2120==102:
            p31202=self.conv1102(D1120)
        elif  num2120==103:
            p31202=self.conv1103(D1120)
        elif  num2120==104:
            p31202=self.conv1104(D1120)
        elif  num2120==105:
            p31202=self.conv1105(D1120)
        elif  num2120==106:
            p31202=self.conv1106(D1120)
        elif  num2120==107:
            p31202=self.conv1107(D1120)
        elif  num2120==108:
            p31202=self.conv1108(D1120)
        elif  num2120==109:
            p31202=self.conv1109(D1120)
        elif  num2120==110:
            p31202=self.conv1110(D1120)
        elif  num2120==111:
            p31202=self.conv1111(D1120)
        elif  num2120==112:
            p31202=self.conv1112(D1120)
        elif  num2120==113:
            p31202=self.conv1113(D1120)
        elif  num2120==114:
            p31202=self.conv1114(D1120)
        elif  num2120==115:
            p31202=self.conv1115(D1120)
        elif  num2120==116:
            p31202=self.conv1116(D1120)
        elif  num2120==117:
            p31202=self.conv1117(D1120)
        elif  num2120==118:
            p31202=self.conv1118(D1120)
        elif  num2120==119:
            p31202=self.conv1119(D1120) 
        elif  num2120==120:
            p31202=self.conv1120(D1120)
        elif  num2120==121:
            p31202=self.conv1121(D1120)
        elif  num2120==122:
            p31202=self.conv1122(D1120)
        elif  num2120==123:
            p31202=self.conv1123(D1120)
        elif  num2120==124:
            p31202=self.conv1124(D1120)
        elif  num2120==125:
            p31202=self.conv1125(D1120)
        elif  num2120==126:
            p31202=self.conv1126(D1120)
        elif  num2120==127:
            p31202=self.conv1127(D1120)
        elif  num2120==128:
            p31202=self.conv1128(D1120)
        elif  num2120==129:
            p31202=self.conv1129(D1120) 
        elif  num2120==130:
            p31202=self.conv1130(D1120)
        elif  num2120==131:
            p31202=self.conv1131(D1120)
        elif  num2120==132:
            p31202=self.conv1132(D1120)
        elif  num2120==133:
            p31202=self.conv1133(D1120)
        elif  num2120==134:
            p31202=self.conv1134(D1120)
        elif  num2120==135:
            p31202=self.conv1135(D1120)
        elif  num2120==136:
            p31202=self.conv1136(D1120)
        elif  num2120==137:
            p31202=self.conv1137(D1120)
        elif  num2120==138:
            p31202=self.conv1138(D1120)
        elif  num2120==139:
            p31202=self.conv1139(D1120)
        elif  num2120==140:
            p31202=self.conv1140(D1120)
        elif  num2120==141:
            p31202=self.conv1141(D1120)
        elif  num2120==142:
            p31202=self.conv1142(D1120)
        elif  num2120==143:
            p31202=self.conv1143(D1120)
        elif  num2120==144:
            p31202=self.conv1144(D1120)
        elif  num2120==145:
            p31202=self.conv1145(D1120)
        elif  num2120==146:
            p31202=self.conv1146(D1120)
        elif  num2120==147:
            p31202=self.conv1147(D1120)
        elif  num2120==148:
            p31202=self.conv1148(D1120)
        elif  num2120==149:
            p31202=self.conv1149(D1120) 
        elif  num2120==150:
            p31202=self.conv1150(D1120)
        elif  num2120==151:
            p31202=self.conv1151(D1120)
        elif  num2120==152:
            p31202=self.conv1152(D1120)
        elif  num2120==153:
            p31202=self.conv1153(D1120)
        elif  num2120==154:
            p31202=self.conv1154(D1120)
        elif  num2120==155:
            p31202=self.conv1155(D1120)
        elif  num2120==156:
            p31202=self.conv1156(D1120)
        elif  num2120==157:
            p31202=self.conv1157(D1120)
        elif  num2120==158:
            p31202=self.conv1158(D1120)
        elif  num2120==159:
            p31202=self.conv1159(D1120) 
        elif  num2120==160:
            p31202=self.conv1160(D1120)
        elif  num2120==161:
            p31202=self.conv1161(D1120)
        elif  num2120==162:
            p31202=self.conv1162(D1120)
        elif  num2120==163:
            p31202=self.conv1163(D1120)
        elif  num2120==164:
            p31202=self.conv1164(D1120)
        elif  num2120==165:
            p31202=self.conv1165(D1120)
        elif  num2120==166:
            p31202=self.conv1166(D1120)
        elif  num2120==167:
            p31202=self.conv1167(D1120)
        elif  num2120==168:
            p31202=self.conv1168(D1120)
        elif  num2120==169:
            p31202=self.conv1169(D1120) 
        elif  num2120==170:
            p31202=self.conv1170(D1120)
        elif  num2120==171:
            p31202=self.conv1171(D1120)
        elif  num2120==172:
            p31202=self.conv1172(D1120)
        elif  num2120==173:
            p31202=self.conv1173(D1120)
        elif  num2120==174:
            p31202=self.conv1174(D1120)
        elif  num2120==175:
            p31202=self.conv1175(D1120)
        elif  num2120==176:
            p31202=self.conv1176(D1120)
        elif  num2120==177:
            p31202=self.conv1177(D1120)
        elif  num2120==178:
            p31202=self.conv1178(D1120)
        elif  num2120==179:
            p31202=self.conv1179(D1120)                                                                                              
        elif  num2120==180:
            p31202=self.conv1180(D1120)
        elif  num2120==181:
            p31202=self.conv1181(D1120)
        elif  num2120==182:
            p31202=self.conv1182(D1120)
        elif  num2120==183:
            p31202=self.conv1183(D1120)
        elif  num2120==184:
            p31202=self.conv1184(D1120)
        elif  num2120==185:
            p31202=self.conv1185(D1120)
        elif  num2120==186:
            p31202=self.conv1186(D1120)
        elif  num2120==187:
            p31202=self.conv1187(D1120)
        elif  num2120==188:
            p31202=self.conv1188(D1120)
        elif  num2120==189:
            p31202=self.conv1189(D1120) 
        elif  num2120==190:
            p31202=self.conv1190(D1120)
        elif  num2120==191:
            p31202=self.conv1191(D1120)
        elif  num2120==192:
            p31202=self.conv1192(D1120)
        elif  num2120==193:
            p31202=self.conv1193(D1120)
        elif  num2120==194:
            p31202=self.conv1194(D1120)
        elif  num2120==195:
            p31202=self.conv1195(D1120)
        elif  num2120==196:
            p31202=self.conv1196(D1120)
        elif  num2120==197:
            p31202=self.conv1197(D1120)
        elif  num2120==198:
            p31202=self.conv1198(D1120)
        elif  num2120==199:
            p31202=self.conv1199(D1120)
        elif  num2120==200:
            p31202=self.conv1200(D1120)
        elif  num2120==201:
            p31202=self.conv1201(D1120)
        elif  num2120==202:
            p31202=self.conv1202(D1120)
        elif  num2120==203:
            p31202=self.conv1203(D1120)
        elif  num2120==204:
            p31202=self.conv1204(D1120)
        elif  num2120==205:
            p31202=self.conv1205(D1120)
        elif  num2120==206:
            p31202=self.conv1206(D1120)
        elif  num2120==207:
            p31202=self.conv1207(D1120)
        elif  num2120==208:
            p31202=self.conv1208(D1120)
        elif  num2120==209:
            p31202=self.conv1209(D1120)
        elif  num2120==210:
            p31202=self.conv1210(D1120)
        elif  num2120==211:
            p31202=self.conv1211(D1120)
        elif  num2120==212:
            p31202=self.conv1212(D1120)
        elif  num2120==213:
            p31202=self.conv1213(D1120)
        elif  num2120==214:
            p31202=self.conv1214(D1120)
        elif  num2120==215:
            p31202=self.conv1215(D1120)
        elif  num2120==216:
            p31202=self.conv1216(D1120)
        elif  num2120==217:
            p31202=self.conv1217(D1120)
        elif  num2120==218:
            p31202=self.conv1218(D1120)
        elif  num2120==219:
            p31202=self.conv1219(D1120)
        elif  num2120==220:
            p31202=self.conv1220(D1120)
        elif  num2120==221:
            p31202=self.conv1221(D1120)
        elif  num2120==222:
            p31202=self.conv1222(D1120)
        elif  num2120==223:
            p31202=self.conv1223(D1120)
        elif  num2120==224:
            p31202=self.conv1224(D1120)
        elif  num2120==225:
            p31202=self.conv1225(D1120)
        elif  num2120==226:
            p31202=self.conv1226(D1120)
        elif  num2120==227:
            p31202=self.conv1227(D1120)
        elif  num2120==228:
            p31202=self.conv1228(D1120)
        elif  num2120==229:
            p31202=self.conv1229(D1120)
        elif  num2120==230:
            p31202=self.conv1230(D1120)
        elif  num2120==231:
            p31202=self.conv1231(D1120)
        elif  num2120==232:
            p31202=self.conv1232(D1120)
        elif  num2120==233:
            p31202=self.conv1233(D1120)
        elif  num2120==234:
            p31202=self.conv1234(D1120)
        elif  num2120==235:
            p31202=self.conv1235(D1120)
        elif  num2120==236:
            p31202=self.conv1236(D1120)
        elif  num2120==237:
            p31202=self.conv1237(D1120)
        elif  num2120==238:
            p31202=self.conv1238(D1120)
        elif  num2120==239:
            p31202=self.conv1239(D1120) 
        elif  num2120==240:
            p31202=self.conv1240(D1120)
        elif  num2120==241:
            p31202=self.conv1241(D1120)
        elif  num2120==242:
            p31202=self.conv1242(D1120)
        elif  num2120==243:
            p31202=self.conv1243(D1120)
        elif  num2120==244:
            p31202=self.conv1244(D1120)
        elif  num2120==245:
            p31202=self.conv1245(D1120)
        elif  num2120==246:
            p31202=self.conv1246(D1120)
        elif  num2120==247:
            p31202=self.conv1247(D1120)
        elif  num2120==248:
            p31202=self.conv1248(D1120)
        elif  num2120==249:
            p31202=self.conv1249(D1120)
        elif  num2120==250:
            p31202=self.conv1250(D1120)
        elif  num2120==251:
            p31202=self.conv1251(D1120)
        elif  num2120==252:
            p31202=self.conv1252(D1120)
        elif  num2120==253:
            p31202=self.conv1253(D1120)
        elif  num2120==254:
            p31202=self.conv1254(D1120)
        elif  num2120==255:
            p31202=self.conv1255(D1120)
        elif  num2120==256:
            p31202=self.conv1256(D1120)
        elif  num2120==257:
            p31202=self.conv1257(D1120)
        elif  num2120==258:
            p31202=self.conv1258(D1120)
        elif  num2120==259:
            p31202=self.conv1259(D1120)
        elif  num2120==260:
            p31202=self.conv1260(D1120)
        elif  num2120==261:
            p31202=self.conv1261(D1120)
        elif  num2120==262:
            p31202=self.conv1262(D1120)
        elif  num2120==263:
            p31202=self.conv1263(D1120)
        elif  num2120==264:
            p31202=self.conv1264(D1120)
        elif  num2120==265:
            p31202=self.conv1265(D1120)
        elif  num2120==266:
            p31202=self.conv1266(D1120)
        elif  num2120==267:
            p31202=self.conv1267(D1120)
        elif  num2120==268:
            p31202=self.conv1268(D1120)
        elif  num2120==269:
            p31202=self.conv1269(D1120) 
        elif  num2120==270:
            p31202=self.conv1270(D1120)
        elif  num2120==271:
            p31202=self.conv1271(D1120)
        elif  num2120==272:
            p31202=self.conv1272(D1120)
        elif  num2120==273:
            p31202=self.conv1273(D1120)
        elif  num2120==274:
            p31202=self.conv1274(D1120)
        elif  num2120==275:
            p31202=self.conv1275(D1120)
        elif  num2120==276:
            p31202=self.conv1276(D1120)
        elif  num2120==277:
            p31202=self.conv1277(D1120)
        elif  num2120==278:
            p31202=self.conv1278(D1120)
        elif  num2120==279:
            p31202=self.conv1279(D1120)
        elif  num2120==280:
            p31202=self.conv1280(D1120)
        elif  num2120==281:
            p31202=self.conv1281(D1120)
        elif  num2120==282:
            p31202=self.conv1282(D1120)
        elif  num2120==283:
            p31202=self.conv1283(D1120)
        elif  num2120==284:
            p31202=self.conv1284(D1120)
        elif  num2120==285:
            p31202=self.conv1285(D1120)
        elif  num2120==286:
            p31202=self.conv1286(D1120)
        elif  num2120==287:
            p31202=self.conv1287(D1120)
        elif  num2120==288:
            p31202=self.conv1288(D1120)
        elif  num2120==289:
            p31202=self.conv1289(D1120) 
        elif  num2120==290:
            p31202=self.conv1290(D1120)
        elif  num2120==291:
            p31202=self.conv1291(D1120)
        elif  num2120==292:
            p31202=self.conv1292(D1120)
        elif  num2120==293:
            p31202=self.conv1293(D1120)
        elif  num2120==294:
            p31202=self.conv1294(D1120)
        elif  num2120==295:
            p31202=self.conv1295(D1120)
        elif  num2120==296:
            p31202=self.conv1296(D1120)
        elif  num2120==297:
            p31202=self.conv1297(D1120)
        elif  num2120==298:
            p31202=self.conv1298(D1120)
        elif  num2120==299:
            p31202=self.conv1299(D1120)
        elif  num2120==300:
            p31202=self.conv1300(D1120)
        elif  num2120==301:
            p31202=self.conv1301(D1120)
        elif  num2120==302:
            p31202=self.conv1302(D1120)
        elif  num2120==303:
            p31202=self.conv1303(D1120)
        elif  num2120==304:
            p31202=self.conv1304(D1120)
        elif  num2120==305:
            p31202=self.conv1305(D1120)
        elif  num2120==306:
            p31202=self.conv1306(D1120)
        elif  num2120==307:
            p31202=self.conv1307(D1120)
        elif  num2120==308:
            p31202=self.conv1308(D1120)
        elif  num2120==309:
            p31202=self.conv1309(D1120) 
        elif  num2120==310:
            p31202=self.conv1310(D1120)
        elif  num2120==311:
            p31202=self.conv1311(D1120)
        elif  num2120==312:
            p31202=self.conv1312(D1120)
        elif  num2120==313:
            p31202=self.conv1313(D1120)
        elif  num2120==314:
            p31202=self.conv1314(D1120)
        elif  num2120==315:
            p31202=self.conv1315(D1120)
        elif  num2120==316:
            p31202=self.conv1316(D1120)
        elif  num2120==317:
            p31202=self.conv1317(D1120)
        elif  num2120==318:
            p31202=self.conv1318(D1120)
        elif  num2120==319:
            p31202=self.conv1319(D1120)
        elif  num2120==320:
            p31202=self.conv1320(D1120)
        elif  num2120==321:
            p31202=self.conv1321(D1120)
        elif  num2120==322:
            p31202=self.conv1322(D1120)
        elif  num2120==323:
            p31202=self.conv1323(D1120)
        elif  num2120==324:
            p31202=self.conv1324(D1120)
        elif  num2120==325:
            p31202=self.conv1325(D1120)
        elif  num2120==326:
            p31202=self.conv1326(D1120)
        elif  num2120==327:
            p31202=self.conv1327(D1120)
        elif  num2120==328:
            p31202=self.conv1328(D1120)
        elif  num2120==329:
            p31202=self.conv1329(D1120)
        elif  num2120==330:
            p31202=self.conv1330(D1120)
        elif  num2120==331:
            p31202=self.conv1331(D1120)
        elif  num2120==332:
            p31202=self.conv1332(D1120)
        elif  num2120==333:
            p31202=self.conv1333(D1120)
        elif  num2120==334:
            p31202=self.conv1334(D1120)
        elif  num2120==335:
            p31202=self.conv1335(D1120)
        elif  num2120==336:
            p31202=self.conv1336(D1120)
        elif  num2120==337:
            p31202=self.conv1337(D1120)
        elif  num2120==338:
            p31202=self.conv1338(D1120)
        elif  num2120==339:
            p31202=self.conv1339(D1120)
        elif  num2120==340:
            p31202=self.conv1340(D1120)
        elif  num2120==341:
            p31202=self.conv1341(D1120)
        elif  num2120==342:
            p31202=self.conv1342(D1120)
        elif  num2120==343:
            p31202=self.conv1343(D1120)
        elif  num2120==344:
            p31202=self.conv1344(D1120)
        elif  num2120==345:
            p31202=self.conv1345(D1120)
        elif  num2120==346:
            p31202=self.conv1346(D1120)
        elif  num2120==347:
            p31202=self.conv1347(D1120)
        elif  num2120==348:
            p31202=self.conv1348(D1120)
        elif  num2120==349:
            p31202=self.conv1349(D1120)
        elif  num2120==350:
            p31202=self.conv1350(D1120)
        elif  num2120==351:
            p31202=self.conv1351(D1120)
        elif  num2120==352:
            p31202=self.conv1352(D1120)
        elif  num2120==353:
            p31202=self.conv1335(D1120)
        elif  num2120==354:
            p31202=self.conv1354(D1120)
        elif  num2120==355:
            p31202=self.conv1355(D1120)
        elif  num2120==356:
            p31202=self.conv1356(D1120)
        elif  num2120==357:
            p31202=self.conv1357(D1120)
        elif  num2120==358:
            p31202=self.conv1358(D1120)
        elif  num2120==359:
            p31202=self.conv1359(D1120) 
        elif  num2120==360:
            p31202=self.conv1360(D1120)
        elif  num2120==361:
            p31202=self.conv1361(D1120)
        elif  num2120==362:
            p31202=self.conv1362(D1120)
        elif  num2120==363:
            p31202=self.conv1363(D1120)
        elif  num2120==364:
            p31202=self.conv1364(D1120)
        elif  num2120==365:
            p31202=self.conv1365(D1120)
        elif  num2120==366:
            p31202=self.conv1366(D1120)
        elif  num2120==367:
            p31202=self.conv1367(D1120)
        elif  num2120==368:
            p31202=self.conv1368(D1120)
        elif  num2120==369:
            p31202=self.conv1369(D1120) 
        elif  num2120==370:
            p31202=self.conv1370(D1120)
        elif  num2120==371:
            p31202=self.conv1371(D1120)
        elif  num2120==372:
            p31202=self.conv1372(D1120)
        elif  num2120==373:
            p31202=self.conv1373(D1120)
        elif  num2120==374:
            p31202=self.conv1374(D1120)
        elif  num2120==375:
            p31202=self.conv1375(D1120)
        elif  num2120==376:
            p31202=self.conv1376(D1120)
        elif  num2120==377:
            p31202=self.conv1377(D1120)
        elif  num2120==378:
            p31202=self.conv1378(D1120)
        elif  num2120==379:
            p31202=self.conv1379(D1120) 
        elif  num2120==380:
            p31202=self.conv1380(D1120)
        elif  num2120==381:
            p31202=self.conv1381(D1120)
        elif  num2120==382:
            p31202=self.conv1382(D1120)
        elif  num2120==383:
            p31202=self.conv1383(D1120)
        elif  num2120==384:
            p31202=self.conv1384(D1120)
        elif  num2120==385:
            p31202=self.conv1385(D1120)
        elif  num2120==386:
            p31202=self.conv1386(D1120)
        elif  num2120==387:
            p31202=self.conv1387(D1120)
        elif  num2120==388:
            p31202=self.conv1388(D1120)
        elif  num2120==389:
            p31202=self.conv1389(D1120) 
        elif  num2120==390:
            p31202=self.conv1390(D1120)
        elif  num2120==391:
            p31202=self.conv1391(D1120)
        elif  num2120==392:
            p31202=self.conv1392(D1120)
        elif  num2120==393:
            p31202=self.conv1393(D1120)
        elif  num2120==394:
            p31202=self.conv1394(D1120)
        elif  num2120==395:
            p31202=self.conv1395(D1120)
        elif  num2120==396:
            p31202=self.conv1396(D1120)
        elif  num2120==397:
            p31202=self.conv1397(D1120)
        elif  num2120==398:
            p31202=self.conv1398(D1120)
        elif  num2120==399:
            p31202=self.conv1399(D1120)
        elif  num2120==400:
            p31202=self.conv1400(D1120)
        elif  num2120==401:
            p31202=self.conv1401(D1120)
        elif  num2120==402:
            p31202=self.conv1402(D1120)
        elif  num2120==403:
            p31202=self.conv1403(D1120)
        elif  num2120==404:
            p31202=self.conv1404(D1120)
        elif  num2120==405:
            p31202=self.conv1405(D1120)
        elif  num2120==406:
            p31202=self.conv1406(D1120)
        elif  num2120==407:
            p31202=self.conv1407(D1120)
        elif  num2120==408:
            p31202=self.conv1408(D1120)
        elif  num2120==409:
            p31202=self.conv1409(D1120)
        elif  num2120==410:
            p31202=self.conv1410(D1120)
        elif  num2120==411:
            p31202=self.conv1411(D1120)
        elif  num2120==412:
            p31202=self.conv1412(D1120)
        elif  num2120==413:
            p31202=self.conv1413(D1120)
        elif  num2120==414:
            p31202=self.conv1414(D1120)
        elif  num2120==415:
            p31202=self.conv145(D1120)
        elif  num2120==416:
            p31202=self.conv1416(D1120)
        elif  num2120==417:
            p31202=self.conv1417(D1120)
        elif  num2120==418:
            p31202=self.conv1418(D1120)
        elif  num2120==419:
            p31202=self.conv1419(D1120) 
        elif  num2120==420:
            p31202=self.conv1420(D1120)
        elif  num2120==421:
            p31202=self.conv1421(D1120)
        elif  num2120==422:
            p31202=self.conv1422(D1120)
        elif  num2120==423:
            p31202=self.conv1423(D1120)
        elif  num2120==424:
            p31202=self.conv1424(D1120)
        elif  num2120==425:
            p31202=self.conv1425(D1120)
        elif  num2120==426:
            p31202=self.conv1426(D1120)
        elif  num2120==427:
            p31202=self.conv1427(D1120)
        elif  num2120==428:
            p31202=self.conv1428(D1120)
        elif  num2120==429:
            p31202=self.conv1429(D1120) 
        elif  num2120==430:
            p31202=self.conv1430(D1120)
        elif  num2120==431:
            p31202=self.conv1431(D1120)
        elif  num2120==432:
            p31202=self.conv1432(D1120)
        elif  num2120==433:
            p31202=self.conv1433(D1120)
        elif  num2120==434:
            p31202=self.conv1434(D1120)
        elif  num2120==435:
            p31202=self.conv1435(D1120)
        elif  num2120==436:
            p31202=self.conv1436(D1120)
        elif  num2120==437:
            p31202=self.conv1437(D1120)
        elif  num2120==438:
            p31202=self.conv1438(D1120)
        elif  num2120==439:
            p31202=self.conv1439(D1120)
        elif  num2120==440:
            p31202=self.conv1440(D1120)
        elif  num2120==441:
            p31202=self.conv1441(D1120)
        elif  num2120==442:
            p31202=self.conv1442(D1120)
        elif  num2120==443:
            p31202=self.conv1443(D1120)
        elif  num2120==444:
            p31202=self.conv1444(D1120)
        elif  num2120==445:
            p31202=self.conv1445(D1120)
        elif  num2120==446:
            p31202=self.conv1446(D1120)
        elif  num2120==447:
            p31202=self.conv1447(D1120)
        elif  num2120==448:
            p31202=self.conv1448(D1120)
        elif  num2120==449:
            p31202=self.conv1449(D1120)
        elif  num2120==450:
            p31202=self.conv1450(D1120)
        elif  num2120==451:
            p31202=self.conv1451(D1120)
        elif  num2120==452:
            p31202=self.conv1452(D1120)
        elif  num2120==453:
            p31202=self.conv1453(D1120)
        elif  num2120==454:
            p31202=self.conv1454(D1120)
        elif  num2120==455:
            p31202=self.conv1455(D1120)
        elif  num2120==456:
            p31202=self.conv1456(D1120)
        elif  num2120==457:
            p31202=self.conv1457(D1120)
        elif  num2120==458:
            p31202=self.conv1458(D1120)
        elif  num2120==459:
            p31202=self.conv1459(D1120)
        elif  num2120==460:
            p31202=self.conv1460(D1120)
        elif  num2120==461:
            p31202=self.conv1461(D1120)
        elif  num2120==462:
            p31202=self.conv1462(D1120)
        elif  num2120==463:
            p31202=self.conv1463(D1120)
        elif  num2120==464:
            p31202=self.conv1464(D1120)
        elif  num2120==465:
            p31202=self.conv1465(D1120)
        elif  num2120==466:
            p31202=self.conv1466(D1120)
        elif  num2120==467:
            p31202=self.conv1467(D1120)
        elif  num2120==468:
            p31202=self.conv1468(D1120)
        elif  num2120==469:
            p31202=self.conv1469(D1120) 
        elif  num2120==470:
            p31202=self.conv1470(D1120)
        elif  num2120==471:
            p31202=self.conv1471(D1120)
        elif  num2120==472:
            p31202=self.conv1472(D1120)
        elif  num2120==473:
            p31202=self.conv1473(D1120)
        elif  num2120==474:
            p31202=self.conv1474(D1120)
        elif  num2120==475:
            p31202=self.conv1475(D1120)
        elif  num2120==476:
            p31202=self.conv1476(D1120)
        elif  num2120==477:
            p31202=self.conv1477(D1120)
        elif  num2120==478:
            p31202=self.conv1478(D1120)
        elif  num2120==479:
            p31202=self.conv1479(D1120)
        elif  num2120==480:
            p31202=self.conv1480(D1120)
        elif  num2120==481:
            p31202=self.conv1481(D1120)
        elif  num2120==482:
            p31202=self.conv1482(D1120)
        elif  num2120==483:
            p31202=self.conv1483(D1120)
        elif  num2120==484:
            p31202=self.conv1484(D1120)
        elif  num2120==485:
            p31202=self.conv1485(D1120)
        elif  num2120==486:
            p31202=self.conv1486(D1120)
        elif  num2120==487:
            p31202=self.conv1487(D1120)
        elif  num2120==488:
            p31202=self.conv1488(D1120)
        elif  num2120==489:
            p31202=self.conv1489(D1120)
        elif  num2120==490:
            p31202=self.conv1490(D1120)
        elif  num2120==491:
            p31202=self.conv1491(D1120)
        elif  num2120==492:
            p31202=self.conv1492(D1120)
        elif  num2120==493:
            p31202=self.conv1493(D1120)
        elif  num2120==494:
            p31202=self.conv1494(D1120)
        elif  num2120==495:
            p31202=self.conv1495(D1120)
        elif  num2120==496:
            p31202=self.conv1496(D1120)
        elif  num2120==497:
            p31202=self.conv1497(D1120)
        elif  num2120==498:
            p31202=self.conv1498(D1120)
        elif  num2120==499:
            p31202=self.conv1499(D1120)
        elif  num2120==500:
            p31202=self.conv1500(D1120)
        elif  num2120==501:
            p31202=self.conv1501(D1120)
        elif  num2120==502:
            p31202=self.conv1502(D1120)
        elif  num2120==503:
            p31202=self.conv1503(D1120)
        elif  num2120==504:
            p31202=self.conv1504(D1120)
        elif  num2120==505:
            p31202=self.conv1505(D1120)
        elif  num2120==506:
            p31202=self.conv1506(D1120)
        elif  num2120==507:
            p31202=self.conv1507(D1120)
        elif  num2120==508:
            p31202=self.conv1508(D1120)
        elif  num2120==509:
            p31202=self.conv1509(D1120)
        elif  num2120==510:
            p31202=self.conv1510(D1120)
        elif  num2120==511:
            p31202=self.conv1511(D1120)
        elif  num2120==512:
            p31202=self.conv1512(D1120)
        elif  num2120==513:
            p31202=self.conv1513(D1120)
        elif  num2120==514:
            p31202=self.conv1514(D1120)
        elif  num2120==515:
            p31202=self.conv1515(D1120)
        elif  num2120==516:
            p31202=self.conv1516(D1120)
        elif  num2120==517:
            p31202=self.conv1517(D1120)
        elif  num2120==518:
            p31202=self.conv1518(D1120)
        elif  num2120==519:
            p31202=self.conv1519(D1120)
        elif  num2120==520:
            p31202=self.conv1520(D1120)
        elif  num2120==521:
            p31202=self.conv1521(D1120)
        elif  num2120==522:
            p31202=self.conv1522(D1120)
        elif  num2120==523:
            p31202=self.conv1523(D1120)
        elif  num2120==524:
            p31202=self.conv1524(D1120)
        elif  num2120==525:
            p31202=self.conv1525(D1120)
        elif  num2120==526:
            p31202=self.conv1526(D1120)
        elif  num2120==527:
            p31202=self.conv1527(D1120)
        elif  num2120==528:
            p31202=self.conv1528(D1120)
        elif  num2120==529:
            p31202=self.conv1529(D1120)
        elif  num2120==530:
            p31202=self.conv1530(D1120)
        elif  num2120==531:
            p31202=self.conv1531(D1120)
        elif  num2120==532:
            p31202=self.conv1532(D1120)
        elif  num2120==533:
            p31202=self.conv1533(D1120)
        elif  num2120==534:
            p31202=self.conv1534(D1120)
        elif  num2120==535:
            p31202=self.conv1535(D1120)
        elif  num2120==536:
            p31202=self.conv1536(D1120)
        elif  num2120==537:
            p31202=self.conv1537(D1120)
        elif  num2120==538:
            p31202=self.conv1538(D1120)
        elif  num2120==539:
            p31202=self.conv1539(D1120)
        elif  num2120==540:
            p31202=self.conv1540(D1120)
        elif  num2120==541:
            p31202=self.conv1541(D1120)
        elif  num2120==542:
            p31202=self.conv1542(D1120)
        elif  num2120==543:
            p31202=self.conv1543(D1120)
        elif  num2120==544:
            p31202=self.conv1544(D1120)
        elif  num2120==545:
            p31202=self.conv1545(D1120)
        elif  num2120==546:
            p31202=self.conv1546(D1120)
        elif  num2120==547:
            p31202=self.conv1547(D1120)
        elif  num2120==548:
            p31202=self.conv1548(D1120)
        elif  num2120==549:
            p31202=self.conv1549(D1120) 
        elif  num2120==550:
            p31202=self.conv1550(D1120)
        elif  num2120==551:
            p31202=self.conv1551(D1120)
        elif  num2120==552:
            p31202=self.conv1552(D1120)
        elif  num2120==553:
            p31202=self.conv1553(D1120)
        elif  num2120==554:
            p31202=self.conv1554(D1120)
        elif  num2120==555:
            p31202=self.conv1555(D1120)
        elif  num2120==556:
            p31202=self.conv1556(D1120)
        elif  num2120==557:
            p31202=self.conv1557(D1120)
        elif  num2120==558:
            p31202=self.conv1558(D1120)
        elif  num2120==559:
            p31202=self.conv1559(D1120)
        elif  num2120==560:
            p31202=self.conv1560(D1120)
        elif  num2120==561:
            p31202=self.conv1561(D1120)
        elif  num2120==562:
            p31202=self.conv1562(D1120)
        elif  num2120==563:
            p31202=self.conv1563(D1120)
        elif  num2120==564:
            p31202=self.conv1564(D1120)
        elif  num2120==565:
            p31202=self.conv1565(D1120)
        elif  num2120==566:
            p31202=self.conv1566(D1120)
        elif  num2120==567:
            p31202=self.conv1567(D1120)
        elif  num2120==568:
            p31202=self.conv1568(D1120)
        elif  num2120==569:
            p31202=self.conv1569(D1120) 
        elif  num2120==570:
            p31202=self.conv1570(D1120)
        elif  num2120==571:
            p31202=self.conv1571(D1120)
        elif  num2120==572:
            p31202=self.conv1572(D1120)
        elif  num2120==573:
            p31202=self.conv1573(D1120)
        elif  num2120==574:
            p31202=self.conv1574(D1120)
        elif  num2120==575:
            p31202=self.conv1575(D1120)
        elif  num2120==576:
            p31202=self.conv1576(D1120)
        elif  num2120==577:
            p31202=self.conv1577(D1120)
        elif  num2120==578:
            p31202=self.conv1578(D1120)
        elif  num2120==579:
            p31202=self.conv1579(D1120) 
        elif  num2120==580:
            p31202=self.conv1580(D1120)
        elif  num2120==581:
            p31202=self.conv1581(D1120)
        elif  num2120==582:
            p31202=self.conv1582(D1120)
        elif  num2120==583:
            p31202=self.conv1583(D1120)
        elif  num2120==584:
            p31202=self.conv1584(D1120)
        elif  num2120==585:
            p31202=self.conv1585(D1120)
        elif  num2120==586:
            p31202=self.conv1586(D1120)
        elif  num2120==587:
            p31202=self.conv1587(D1120)
        elif  num2120==588:
            p31202=self.conv1588(D1120)
        elif  num2120==589:
            p31202=self.conv1589(D1120)
        elif  num2120==590:
            p31202=self.conv1590(D1120)
        elif  num2120==591:
            p31202=self.conv1591(D1120)
        elif  num2120==592:
            p31202=self.conv1592(D1120)
        elif  num2120==593:
            p31202=self.conv1593(D1120)
        elif  num2120==594:
            p31202=self.conv1594(D1120)
        elif  num2120==595:
            p31202=self.conv1595(D1120)
        elif  num2120==596:
            p31202=self.conv1596(D1120)
        elif  num2120==597:
            p31202=self.conv1597(D1120)
        elif  num2120==598:
            p31202=self.conv1598(D1120)
        elif  num2120==599:
            p31202=self.conv1599(D1120)
        elif  num2120==600:
            p31202=self.conv1600(D1120)
        elif  num2120==601:
            p31202=self.conv1601(D1120)
        elif  num2120==602:
            p31202=self.conv1602(D1120)
        elif  num2120==603:
            p31202=self.conv1603(D1120)
        elif  num2120==604:
            p31202=self.conv1604(D1120)
        elif  num2120==605:
            p31202=self.conv1605(D1120)
        elif  num2120==606:
            p31202=self.conv1606(D1120)
        elif  num2120==607:
            p31202=self.conv1607(D1120)
        elif  num2120==608:
            p31202=self.conv1608(D1120)
        elif  num2120==609:
            p31202=self.conv1609(D1120)                                                                                                                         
        elif  num2120==610:
            p31202=self.conv1610(D1120)
        elif  num2120==611:
            p31202=self.conv1611(D1120)
        elif  num2120==612:
            p31202=self.conv1612(D1120)
        elif  num2120==613:
            p31202=self.conv1613(D1120)
        elif  num2120==614:
            p31202=self.conv1614(D1120)
        elif  num2120==615:
            p31202=self.conv1615(D1120)
        elif  num2120==616:
            p31202=self.conv1616(D1120)
        elif  num2120==617:
            p31202=self.conv1617(D1120)
        elif  num2120==618:
            p31202=self.conv1618(D1120)
        elif  num2120==619:
            p31202=self.conv1619(D1120)                                                                                                                        
             
        elif  num2120==620:
            p31202=self.conv1620(D1120)
        elif  num2120==621:
            p31202=self.conv1621(D1120)
        elif  num2120==622:
            p31202=self.conv1622(D1120)
        elif  num2120==623:
            p31202=self.conv1623(D1120)
        elif  num2120==624:
            p31202=self.conv1624(D1120)
        elif  num2120==625:
            p31202=self.conv1625(D1120)
        elif  num2120==626:
            p31202=self.conv1626(D1120)
        elif  num2120==627:
            p31202=self.conv1627(D1120)
        elif  num2120==628:
            p31202=self.conv1628(D1120)
        elif  num2120==629:
            p31202=self.conv1629(D1120)  
        elif  num2120==630:
            p31202=self.conv1630(D1120)
        elif  num2120==631:
            p31202=self.conv1631(D1120)
        elif  num2120==632:
            p31202=self.conv1632(D1120)
        elif  num2120==633:
            p31202=self.conv1633(D1120)
        elif  num2120==634:
            p31202=self.conv1634(D1120)
        elif  num2120==635:
            p31202=self.conv1635(D1120)
        elif  num2120==636:
            p31202=self.conv1636(D1120)
        elif  num2120==637:
            p31202=self.conv1637(D1120)
        elif  num2120==638:
            p31202=self.conv1638(D1120)
        elif  num2120==639:
            p31202=self.conv1639(D1120)                                                                                                                         
            
        elif  num2120==640:
            p31202=self.conv1640(D1120)
        elif  num2120==641:
            p31202=self.conv1641(D1120)
        elif  num2120==642:
            p31202=self.conv1642(D1120)
        elif  num2120==643:
            p31202=self.conv1643(D1120)
        elif  num2120==644:
            p31202=self.conv1644(D1120)
        elif  num2120==645:
            p31202=self.conv1645(D1120)
        elif  num2120==646:
            p31202=self.conv1646(D1120)
        elif  num2120==647:
            p31202=self.conv1647(D1120)
        elif  num2120==648:
            p31202=self.conv1648(D1120)
        elif  num2120==649:
            p31202=self.conv1649(D1120)                                                                                                                        
             
        elif  num2120==650:
            p31202=self.conv1650(D1120)
        elif  num2120==651:
            p31202=self.conv1651(D1120)
        elif  num2120==652:
            p31202=self.conv1652(D1120)
        elif  num2120==653:
            p31202=self.conv1653(D1120)
        elif  num2120==654:
            p31202=self.conv1654(D1120)
        elif  num2120==655:
            p31202=self.conv1655(D1120)
        elif  num2120==656:
            p31202=self.conv1656(D1120)
        elif  num2120==657:
            p31202=self.conv1657(D1120)
        elif  num2120==658:
            p31202=self.conv1658(D1120)
        elif  num2120==659:
            p31202=self.conv1659(D1120)
        elif  num2120==660:
            p31202=self.conv1660(D1120)
        elif  num2120==661:
            p31202=self.conv1661(D1120)
        elif  num2120==662:
            p31202=self.conv1662(D1120)
        elif  num2120==663:
            p31202=self.conv1663(D1120)
        elif  num2120==664:
            p31202=self.conv1664(D1120)
        elif  num2120==665:
            p31202=self.conv1665(D1120)
        elif  num2120==666:
            p31202=self.conv1666(D1120)
        elif  num2120==667:
            p31202=self.conv1667(D1120)
        elif  num2120==668:
            p31202=self.conv1668(D1120)
        elif  num2120==669:
            p31202=self.conv1669(D1120) 
        elif  num2120==670:
            p31202=self.conv1670(D1120)
        elif  num2120==671:
            p31202=self.conv1671(D1120)
        elif  num2120==672:
            p31202=self.conv1672(D1120)
        elif  num2120==673:
            p31202=self.conv1673(D1120)
        elif  num2120==674:
            p31202=self.conv1674(D1120)
        elif  num2120==675:
            p31202=self.conv1675(D1120)
        elif  num2120==676:
            p31202=self.conv1676(D1120)
        elif  num2120==677:
            p31202=self.conv1677(D1120)
        elif  num2120==678:
            p31202=self.conv1678(D1120)
        elif  num2120==679:
            p31202=self.conv1679(D1120)
        elif  num2120==680:
            p31202=self.conv1680(D1120)
        elif  num2120==681:
            p31202=self.conv1681(D1120)
        elif  num2120==682:
            p31202=self.conv1682(D1120)
        elif  num2120==683:
            p31202=self.conv1683(D1120)
        elif  num2120==684:
            p31202=self.conv1684(D1120)
        elif  num2120==685:
            p31202=self.conv1685(D1120)
        elif  num2120==686:
            p31202=self.conv1686(D1120)
        elif  num2120==687:
            p31202=self.conv1687(D1120)
        elif  num2120==688:
            p31202=self.conv1688(D1120)
        elif  num2120==689:
            p31202=self.conv1689(D1120)
        elif  num2120==690:
            p31202=self.conv1690(D1120)
        elif  num2120==691:
            p31202=self.conv1691(D1120)
        elif  num2120==692:
            p31202=self.conv1692(D1120)
        elif  num2120==693:
            p31202=self.conv1693(D1120)
        elif  num2120==694:
            p31202=self.conv1694(D1120)
        elif  num2120==695:
            p31202=self.conv1695(D1120)
        elif  num2120==696:
            p31202=self.conv1696(D1120)
        elif  num2120==697:
            p31202=self.conv1697(D1120)
        elif  num2120==698:
            p31202=self.conv1698(D1120)
        elif  num2120==699:
            p31202=self.conv1699(D1120)
        elif  num2120==700:
            p31202=self.conv1700(D1120)
        elif  num2120==701:
            p31202=self.conv1701(D1120)
        elif  num2120==702:
            p31202=self.conv1702(D1120)
        elif  num2120==703:
            p31202=self.conv1703(D1120)
        elif  num2120==704:
            p31202=self.conv1704(D1120)
        elif  num2120==705:
            p31202=self.conv1705(D1120)
        elif  num2120==706:
            p31202=self.conv1706(D1120)
        elif  num2120==707:
            p31202=self.conv1707(D1120)
        elif  num2120==708:
            p31202=self.conv1708(D1120)
        elif  num2120==709:
            p31202=self.conv1709(D1120)
        elif  num2120==710:
            p31202=self.conv1710(D1120)
        elif  num2120==711:
            p31202=self.conv1711(D1120)
        elif  num2120==712:
            p31202=self.conv1712(D1120)
        elif  num2120==713:
            p31202=self.conv1713(D1120)
        elif  num2120==714:
            p31202=self.conv1714(D1120)
        elif  num2120==715:
            p31202=self.conv1715(D1120)
        elif  num2120==716:
            p31202=self.conv1716(D1120)
        elif  num2120==717:
            p31202=self.conv1717(D1120)
        elif  num2120==718:
            p31202=self.conv1718(D1120)
        elif  num2120==719:
            p31202=self.conv1719(D1120)
        elif  num2120==720:
            p31202=self.conv1720(D1120)
        elif  num2120==721:
            p31202=self.conv1721(D1120)
        elif  num2120==722:
            p31202=self.conv1722(D1120)
        elif  num2120==723:
            p31202=self.conv1723(D1120)
        elif  num2120==724:
            p31202=self.conv1724(D1120)
        elif  num2120==725:
            p31202=self.conv1725(D1120)
        elif  num2120==726:
            p31202=self.conv1726(D1120)
        elif  num2120==727:
            p31202=self.conv1727(D1120)
        elif  num2120==728:
            p31202=self.conv1728(D1120)
        elif  num2120==729:
            p31202=self.conv1729(D1120)
        elif  num2120==730:
            p31202=self.conv1730(D1120)
        elif  num2120==731:
            p31202=self.conv1731(D1120)
        elif  num2120==732:
            p31202=self.conv1732(D1120)
        elif  num2120==733:
            p31202=self.conv1733(D1120)
        elif  num2120==734:
            p31202=self.conv1734(D1120)
        elif  num2120==735:
            p31202=self.conv1735(D1120)
        elif  num2120==736:
            p31202=self.conv1736(D1120)
        elif  num2120==737:
            p31202=self.conv1737(D1120)
        elif  num2120==738:
            p31202=self.conv1738(D1120)
        elif  num2120==739:
            p31202=self.conv1739(D1120)                                                                                                                         
            
        elif  num2120==740:
            p31202=self.conv1740(D1120)
        elif  num2120==741:
            p31202=self.conv1741(D1120)
        elif  num2120==742:
            p31202=self.conv1742(D1120)
        elif  num2120==743:
            p31202=self.conv1743(D1120)
        elif  num2120==744:
            p31202=self.conv1744(D1120)
        elif  num2120==745:
            p31202=self.conv1745(D1120)
        elif  num2120==746:
            p31202=self.conv1746(D1120)
        elif  num2120==747:
            p31202=self.conv1747(D1120)
        elif  num2120==748:
            p31202=self.conv1748(D1120)
        elif  num2120==749:
            p31202=self.conv1749(D1120)
        elif  num2120==750:
            p31202=self.conv1750(D1120)
        elif  num2120==751:
            p31202=self.conv1751(D1120)
        elif  num2120==752:
            p31202=self.conv1752(D1120)
        elif  num2120==753:
            p31202=self.conv1753(D1120)
        elif  num2120==754:
            p31202=self.conv1754(D1120)
        elif  num2120==755:
            p31202=self.conv1755(D1120)
        elif  num2120==756:
            p31202=self.conv1756(D1120)
        elif  num2120==757:
            p31202=self.conv1757(D1120)
        elif  num2120==758:
            p31202=self.conv1758(D1120)
        elif  num2120==759:
            p31202=self.conv1759(D1120)
        elif  num2120==760:
            p31202=self.conv1760(D1120)
        elif  num2120==761:
            p31202=self.conv1761(D1120)
        elif  num2120==762:
            p31202=self.conv1762(D1120)
        elif  num2120==763:
            p31202=self.conv1763(D1120)
        elif  num2120==764:
            p31202=self.conv1764(D1120)
        elif  num2120==765:
            p31202=self.conv1765(D1120)
        elif  num2120==766:
            p31202=self.conv1766(D1120)
        elif  num2120==767:
            p31202=self.conv1767(D1120)
        elif  num2120==768:
            p31202=self.conv1768(D1120)
        elif  num2120==769:
            p31202=self.conv1769(D1120) 
        elif  num2120==770:
            p31202=self.conv1770(D1120)
        elif  num2120==771:
            p31202=self.conv1771(D1120)
        elif  num2120==772:
            p31202=self.conv1772(D1120)
        elif  num2120==773:
            p31202=self.conv1773(D1120)
        elif  num2120==774:
            p31202=self.conv1774(D1120)
        elif  num2120==775:
            p31202=self.conv1775(D1120)
        elif  num2120==776:
            p31202=self.conv1776(D1120)
        elif  num2120==777:
            p31202=self.conv1777(D1120)
        elif  num2120==778:
            p31202=self.conv1778(D1120)
        elif  num2120==779:
            p31202=self.conv1779(D1120) 
        elif  num2120==780:
            p31202=self.conv1780(D1120)
        elif  num2120==781:
            p31202=self.conv1781(D1120)
        elif  num2120==782:
            p31202=self.conv1782(D1120)
        elif  num2120==783:
            p31202=self.conv1783(D1120)
        elif  num2120==784:
            p31202=self.conv1784(D1120)
        elif  num2120==785:
            p31202=self.conv1785(D1120)
        elif  num2120==786:
            p31202=self.conv1786(D1120)
        elif  num2120==787:
            p31202=self.conv1787(D1120)
        elif  num2120==788:
            p31202=self.conv1788(D1120)
        elif  num2120==789:
            p31202=self.conv1789(D1120) 
        elif  num2120==790:
            p31202=self.conv1790(D1120)
        elif  num2120==791:
            p31202=self.conv1791(D1120)
        elif  num2120==792:
            p31202=self.conv1792(D1120)
        elif  num2120==793:
            p31202=self.conv1793(D1120)
        elif  num2120==794:
            p31202=self.conv1794(D1120)
        elif  num2120==795:
            p31202=self.conv1795(D1120)
        elif  num2120==796:
            p31202=self.conv1796(D1120)
        elif  num2120==797:
            p31202=self.conv1797(D1120)
        elif  num2120==798:
            p31202=self.conv1798(D1120)
        elif  num2120==799:
            p31202=self.conv1799(D1120) 
        elif  num2120==800:
            p31202=self.conv1800(D1120)
        elif  num2120==801:
            p31202=self.conv1801(D1120)
        elif  num2120==802:
            p31202=self.conv1802(D1120)
        elif  num2120==803:
            p31202=self.conv1803(D1120)
        elif  num2120==804:
            p31202=self.conv1804(D1120)
        elif  num2120==805:
            p31202=self.conv1805(D1120)
        elif  num2120==806:
            p31202=self.conv1806(D1120)
        elif  num2120==807:
            p31202=self.conv1807(D1120)
        elif  num2120==808:
            p31202=self.conv1808(D1120)
        elif  num2120==809:
            p31202=self.conv1809(D1120)
        elif  num2120==810:
            p31202=self.conv1810(D1120)
        elif  num2120==811:
            p31202=self.conv1811(D1120)
        elif  num2120==812:
            p31202=self.conv1812(D1120)
        elif  num2120==813:
            p31202=self.conv1813(D1120)
        elif  num2120==814:
            p31202=self.conv1814(D1120)
        elif  num2120==815:
            p31202=self.conv1815(D1120)
        elif  num2120==816:
            p31202=self.conv1816(D1120)
        elif  num2120==817:
            p31202=self.conv1817(D1120)
        elif  num2120==818:
            p31202=self.conv1818(D1120)
        elif  num2120==819:
            p31202=self.conv1819(D1120)
        elif  num2120==820:
            p31202=self.conv1820(D1120)
        elif  num2120==821:
            p31202=self.conv1821(D1120)
        elif  num2120==822:
            p31202=self.conv1822(D1120)
        elif  num2120==823:
            p31202=self.conv1823(D1120)
        elif  num2120==824:
            p31202=self.conv1824(D1120)
        elif  num2120==825:
            p31202=self.conv1825(D1120)
        elif  num2120==826:
            p31202=self.conv1826(D1120)
        elif  num2120==827:
            p31202=self.conv1827(D1120)
        elif  num2120==828:
            p31202=self.conv1828(D1120)
        elif  num2120==829:
            p31202=self.conv1829(D1120)                                                                                                                        
             
        elif  num2120==830:
            p31202=self.conv1830(D1120)
        elif  num2120==831:
            p31202=self.conv1831(D1120)
        elif  num2120==832:
            p31202=self.conv1832(D1120)
        elif  num2120==833:
            p31202=self.conv1833(D1120)
        elif  num2120==834:
            p31202=self.conv1834(D1120)
        elif  num2120==835:
            p31202=self.conv1835(D1120)
        elif  num2120==836:
            p31202=self.conv1836(D1120)
        elif  num2120==837:
            p31202=self.conv1837(D1120)
        elif  num2120==838:
            p31202=self.conv1838(D1120)
        elif  num2120==839:
            p31202=self.conv1839(D1120)
        elif  num2120==840:
            p31202=self.conv1840(D1120)
        elif  num2120==841:
            p31202=self.conv1841(D1120)
        elif  num2120==842:
            p31202=self.conv1842(D1120)
        elif  num2120==843:
            p31202=self.conv1843(D1120)
        elif  num2120==844:
            p31202=self.conv1844(D1120)
        elif  num2120==845:
            p31202=self.conv1845(D1120)
        elif  num2120==846:
            p31202=self.conv1846(D1120)
        elif  num2120==847:
            p31202=self.conv1847(D1120)
        elif  num2120==848:
            p31202=self.conv1848(D1120)
        elif  num2120==849:
            p31202=self.conv1849(D1120)
        elif  num2120==850:
            p31202=self.conv1850(D1120)
        elif  num2120==851:
            p31202=self.conv1851(D1120)
        elif  num2120==852:
            p31202=self.conv1852(D1120)
        elif  num2120==853:
            p31202=self.conv1853(D1120)
        elif  num2120==854:
            p31202=self.conv1854(D1120)
        elif  num2120==855:
            p31202=self.conv1855(D1120)
        elif  num2120==856:
            p31202=self.conv1856(D1120)
        elif  num2120==857:
            p31202=self.conv1857(D1120)
        elif  num2120==858:
            p31202=self.conv1858(D1120)
        elif  num2120==859:
            p31202=self.conv1859(D1120)
        elif  num2120==860:
            p31202=self.conv1860(D1120)
        elif  num2120==861:
            p31202=self.conv1861(D1120)
        elif  num2120==862:
            p31202=self.conv1862(D1120)
        elif  num2120==863:
            p31202=self.conv1863(D1120)
        elif  num2120==864:
            p31202=self.conv1864(D1120)
        elif  num2120==865:
            p31202=self.conv1865(D1120)
        elif  num2120==866:
            p31202=self.conv1866(D1120)
        elif  num2120==867:
            p31202=self.conv1867(D1120)
        elif  num2120==868:
            p31202=self.conv1868(D1120)
        elif  num2120==869:
            p31202=self.conv1869(D1120) 
        elif  num2120==870:
            p31202=self.conv1870(D1120)
        elif  num2120==871:
            p31202=self.conv1871(D1120)
        elif  num2120==872:
            p31202=self.conv1872(D1120)
        elif  num2120==873:
            p31202=self.conv1873(D1120)
        elif  num2120==874:
            p31202=self.conv1874(D1120)
        elif  num2120==875:
            p31202=self.conv1875(D1120)
        elif  num2120==876:
            p31202=self.conv1876(D1120)
        elif  num2120==877:
            p31202=self.conv1877(D1120)
        elif  num2120==878:
            p31202=self.conv1878(D1120)
        elif  num2120==879:
            p31202=self.conv1879(D1120)
        elif  num2120==880:
            p31202=self.conv1880(D1120)
        elif  num2120==881:
            p31202=self.conv1881(D1120)
        elif  num2120==882:
            p31202=self.conv1882(D1120)
        elif  num2120==883:
            p31202=self.conv1883(D1120)
        elif  num2120==884:
            p31202=self.conv1884(D1120)
        elif  num2120==885:
            p31202=self.conv1885(D1120)
        elif  num2120==886:
            p31202=self.conv1886(D1120)
        elif  num2120==887:
            p31202=self.conv1887(D1120)
        elif  num2120==888:
            p31202=self.conv1888(D1120)
        elif  num2120==889:
            p31202=self.conv1889(D1120)  
        elif  num2120==890:
            p31202=self.conv1890(D1120)
        elif  num2120==891:
            p31202=self.conv1891(D1120)
        elif  num2120==892:
            p31202=self.conv1892(D1120)
        elif  num2120==893:
            p31202=self.conv1893(D1120)
        elif  num2120==894:
            p31202=self.conv1894(D1120)
        elif  num2120==895:
            p31202=self.conv1895(D1120)
        elif  num2120==896:
            p31202=self.conv1896(D1120)
        elif  num2120==897:
            p31202=self.conv1897(D1120)
        elif  num2120==898:
            p31202=self.conv1898(D1120)
        elif  num2120==899:
            p31202=self.conv1899(D1120)
        elif  num2120==900:
            p31202=self.conv1900(D1120)
        elif  num2120==901:
            p31202=self.conv1901(D1120)
        elif  num2120==902:
            p31202=self.conv1902(D1120)
        elif  num2120==903:
            p31202=self.conv1903(D1120)
        elif  num2120==904:
            p31202=self.conv1904(D1120)
        elif  num2120==905:
            p31202=self.conv1905(D1120)
        elif  num2120==906:
            p31202=self.conv1906(D1120)
        elif  num2120==907:
            p31202=self.conv1907(D1120)
        elif  num2120==908:
            p31202=self.conv1908(D1120)
        elif  num2120==909:
            p31202=self.conv1909(D1120)
        elif  num2120==910:
            p31202=self.conv1910(D1120)
        elif  num2120==911:
            p31202=self.conv1911(D1120)
        elif  num2120==912:
            p31202=self.conv1912(D1120)
        elif  num2120==913:
            p31202=self.conv1913(D1120)
        elif  num2120==914:
            p31202=self.conv1914(D1120)
        elif  num2120==915:
            p31202=self.conv1915(D1120)
        elif  num2120==916:
            p31202=self.conv1916(D1120)
        elif  num2120==917:
            p31202=self.conv1917(D1120)
        elif  num2120==918:
            p31202=self.conv1918(D1120)
        elif  num2120==919:
            p31202=self.conv1919(D1120)
        elif  num2120==920:
            p31202=self.conv1920(D1120)
        elif  num2120==921:
            p31202=self.conv1921(D1120)
        elif  num2120==922:
            p31202=self.conv1922(D1120)
        elif  num2120==923:
            p31202=self.conv1923(D1120)
        elif  num2120==924:
            p31202=self.conv1924(D1120)
        elif  num2120==925:
            p31202=self.conv1925(D1120)
        elif  num2120==926:
            p31202=self.conv1926(D1120)
        elif  num2120==927:
            p31202=self.conv1927(D1120)
        elif  num2120==928:
            p31202=self.conv1928(D1120)
        elif  num2120==929:
            p31202=self.conv1929(D1120)
        elif  num2120==930:
            p31202=self.conv1930(D1120)
        elif  num2120==931:
            p31202=self.conv1931(D1120)
        elif  num2120==932:
            p31202=self.conv1932(D1120)
        elif  num2120==933:
            p31202=self.conv1933(D1120)
        elif  num2120==934:
            p31202=self.conv1934(D1120)
        elif  num2120==935:
            p31202=self.conv1935(D1120)
        elif  num2120==936:
            p31202=self.conv1936(D1120)
        elif  num2120==937:
            p31202=self.conv1937(D1120)
        elif  num2120==938:
            p31202=self.conv1938(D1120)
        elif  num2120==939:
            p31202=self.conv1939(D1120) 
        elif  num2120==940:
            p31202=self.conv1940(D1120)
        elif  num2120==941:
            p31202=self.conv1941(D1120)
        elif  num2120==942:
            p31202=self.conv1942(D1120)
        elif  num2120==943:
            p31202=self.conv1943(D1120)
        elif  num2120==944:
            p31202=self.conv1944(D1120)
        elif  num2120==945:
            p31202=self.conv1945(D1120)
        elif  num2120==946:
            p31202=self.conv1946(D1120)
        elif  num2120==947:
            p31202=self.conv1947(D1120)
        elif  num2120==948:
            p31202=self.conv1948(D1120)
        elif  num2120==949:
            p31202=self.conv1949(D1120)                                                                                                                         
            
        elif  num2120==950:
            p31202=self.conv1950(D1120)
        elif  num2120==951:
            p31202=self.conv1951(D1120)
        elif  num2120==952:
            p31202=self.conv1952(D1120)
        elif  num2120==953:
            p31202=self.conv1953(D1120)
        elif  num2120==954:
            p31202=self.conv1954(D1120)
        elif  num2120==955:
            p31202=self.conv1955(D1120)
        elif  num2120==956:
            p31202=self.conv1956(D1120)
        elif  num2120==957:
            p31202=self.conv1957(D1120)
        elif  num2120==958:
            p31202=self.conv1958(D1120)
        elif  num2120==959:
            p31202=self.conv1959(D1120)
        elif  num2120==960:
            p31202=self.conv1960(D1120)
        elif  num2120==961:
            p31202=self.conv1961(D1120)
        elif  num2120==962:
            p31202=self.conv1962(D1120)
        elif  num2120==963:
            p31202=self.conv1963(D1120)
        elif  num2120==964:
            p31202=self.conv1964(D1120)
        elif  num2120==965:
            p31202=self.conv1965(D1120)
        elif  num2120==966:
            p31202=self.conv1966(D1120)
        elif  num2120==967:
            p31202=self.conv1967(D1120)
        elif  num2120==968:
            p31202=self.conv1968(D1120)
        elif  num2120==969:
            p31202=self.conv1969(D1120) 
        elif  num2120==970:
            p31202=self.conv1970(D1120)
        elif  num2120==971:
            p31202=self.conv1971(D1120)
        elif  num2120==972:
            p31202=self.conv1972(D1120)
        elif  num2120==973:
            p31202=self.conv1973(D1120)
        elif  num2120==974:
            p31202=self.conv1974(D1120)
        elif  num2120==975:
            p31202=self.conv1975(D1120)
        elif  num2120==976:
            p31202=self.conv1976(D1120)
        elif  num2120==977:
            p31202=self.conv1977(D1120)
        elif  num2120==978:
            p31202=self.conv1978(D1120)
        elif  num2120==979:
            p31202=self.conv1979(D1120) 
        elif  num2120==980:
            p31202=self.conv1980(D1120)
        elif  num2120==981:
            p31202=self.conv1981(D1120)
        elif  num2120==982:
            p31202=self.conv1982(D1120)
        elif  num2120==983:
            p31202=self.conv1983(D1120)
        elif  num2120==984:
            p31202=self.conv1984(D1120)
        elif  num2120==985:
            p31202=self.conv1985(D1120)
        elif  num2120==986:
            p31202=self.conv1986(D1120)
        elif  num2120==987:
            p31202=self.conv1987(D1120)
        elif  num2120==988:
            p31202=self.conv1988(D1120)
        elif  num2120==989:
            p31202=self.conv1989(D1120)
        elif  num2120==990:
            p31202=self.conv1990(D1120)
        elif  num2120==991:
            p31202=self.conv1991(D1120)
        elif  num2120==992:
            p31202=self.conv1992(D1120)
        elif  num2120==993:
            p31202=self.conv1993(D1120)
        elif  num2120==994:
            p31202=self.conv1994(D1120)
        elif  num2120==995:
            p31202=self.conv1995(D1120)
        elif  num2120==996:
            p31202=self.conv1996(D1120)
        elif  num2120==997:
            p31202=self.conv1997(D1120)
        elif  num2120==998:
            p31202=self.conv1998(D1120)
        elif  num2120==999:
            p31202=self.conv1999(D1120) 
        elif  num2120==1000:
            p31202=self.conv11000(D1120)
        elif  num2120==1001:
            p31202=self.conv11001(D1120)
        elif  num2120==1002:
            p31202=self.conv11002(D1120)
        elif  num2120==1003:
            p31202=self.conv11003(D1120)
        elif  num2120==1004:
            p31202=self.conv11004(D1120)
        elif  num2120==1005:
            p31202=self.conv11005(D1120)
        elif  num2120==1006:
            p31202=self.conv11006(D1120)
        elif  num2120==1007:
            p31202=self.conv11007(D1120)
        elif  num2120==1008:
            p31202=self.conv11008(D1120)
        elif  num2120==1009:
            p31202=self.conv11009(D1120) 
        elif  num2120==1010:
            p31202=self.conv11010(D1120)
        elif  num2120==1011:
            p31202=self.conv11011(D1120)
        elif  num2120==1012:
            p31202=self.conv11012(D1120)
        elif  num2120==1013:
            p31202=self.conv11013(D1120)
        elif  num2120==1014:
            p31202=self.conv11014(D1120)
        elif  num2120==1015:
            p31202=self.conv11015(D1120)
        elif  num2120==1016:
            p31202=self.conv11016(D1120)
        elif  num2120==1017:
            p31202=self.conv11017(D1120)
        elif  num2120==1018:
            p31202=self.conv11018(D1120)
        elif  num2120==1019:
            p31202=self.conv11019(D1120)
        elif  num2120==1020:
            p31202=self.conv11020(D1120)
        elif  num2120==1021:
            p31202=self.conv11021(D1120)
        elif  num2120==1022:
            p31202=self.conv11022(D1120)
        elif  num2120==1023:
            p31202=self.conv11023(D1120)
        elif  num2120==1024:
            p31202=self.conv11024(D1120) 
            
        if num0121==1:
            p312121=self.conv11(B1121)
        elif num0121==2:
            p312121=self.conv12(B1121)
        elif num0121==3:
            p312121=self.conv13(B1121)
        elif num0121==4:
            p312121=self.conv14(B1121)
        elif num0121==5:
            p312121=self.conv15(B1121)
        elif num0121==6:
            p312121=self.conv16(B1121)
        elif num0121==7:
            p312121=self.conv17(B1121)
        elif num0121==8:
            p312121=self.conv18(B1121)
        elif num0121==9:
            p312121=self.conv19(B1121)
        elif num0121==10:
            p312121=self.conv110(B1121)
        elif num0121==11:
            p312121=self.conv111(B1121)
        elif num0121==12:
            p312121=self.conv112(B1121)
        elif num0121==13:
            p312121=self.conv113(B1121)
        elif num0121==14:
            p312121=self.conv114(B1121)
        elif num0121==15:
            p312121=self.conv115(B1121)
        elif num0121==16:
            p312121=self.conv116(B1121)
        elif num0121==17:
            p312121=self.conv117(B1121)
        elif num0121==18:
            p312121=self.conv118(B1121)
        elif num0121==19:
            p312121=self.conv119(B1121)
        elif num0121==20:
            p312121=self.conv120(B1121)
        elif num0121==21:
            p312121=self.conv121(B1121)
        elif num0121==22:
            p312121=self.conv122(B1121)
        elif num0121==23:
            p312121=self.conv123(B1121)
        elif num0121==24:
            p312121=self.conv124(B1121)
        elif num0121==25:
            p312121=self.conv125(B1121)
        elif num0121==26:
            p312121=self.conv126(B1121)
        elif num0121==27:
            p312121=self.conv127(B1121)
        elif num0121==28:
            p312121=self.conv128(B1121)
        elif num0121==29:
            p312121=self.conv129(B1121)
        elif num0121==30:
            p312121=self.conv130(B1121)
        elif num0121==31:
            p312121=self.conv131(B1121)
        elif num0121==32:
            p312121=self.conv132(B1121)
        elif num0121==33:
            p312121=self.conv133(B1121)
        elif num0121==34:
            p312121=self.conv134(B1121)
        elif num0121==35:
            p312121=self.conv135(B1121)
        elif num0121==36:
            p312121=self.conv136(B1121)
        elif num0121==37:
            p312121=self.conv137(B1121)
        elif num0121==38:
            p312121=self.conv138(B1121)
        elif num0121==39:
            p312121=self.conv139(B1121)
        elif num0121==40:
            p312121=self.conv140(B1121)
        elif num0121==41:
            p312121=self.conv141(B1121)
        elif num0121==42:
            p312121=self.conv142(B1121)
        elif num0121==43:
            p312121=self.conv143(B1121)
        elif num0121==44:
            p312121=self.conv144(B1121)
        elif num0121==45:
            p312121=self.conv145(B1121)
        elif num0121==46:
            p312121=self.conv146(B1121)
        elif num0121==47:
            p312121=self.conv147(B1121)
        elif num0121==48:
            p312121=self.conv148(B1121)
        elif num0121==49:
            p312121=self.conv149(B1121)
        elif num0121==50:
            p312121=self.conv150(B1121)
        elif num0121==51:
            p312121=self.conv151(B1121)
        elif num0121==52:
            p312121=self.conv152(B1121)
        elif num0121==53:
            p312121=self.conv153(B1121)
        elif num0121==54:
            p312121=self.conv154(B1121)
        elif num0121==55:
            p312121=self.conv155(B1121)
        elif num0121==56:
            p312121=self.conv156(B1121)
        elif num0121==57:
            p312121=self.conv157(B1121)
        elif num0121==58:
            p312121=self.conv158(B1121)
        elif num0121==59:
            p312121=self.conv159(B1121)
        elif num0121==60:
            p312121=self.conv160(B1121)
        elif num0121==61:
            p312121=self.conv161(B1121)
        elif num0121==62:
            p312121=self.conv162(B1121)
        elif num0121==63:
            p312121=self.conv163(B1121)
        elif num0121==64:
            p312121=self.conv164(B1121)
        
        if  num1121==1:
            p31211=self.conv11(C1121)
        elif  num1121==2:
            p31211=self.conv12(C1121)
        elif  num1121==3:
            p31211=self.conv13(C1121)
        elif  num1121==4:
            p31211=self.conv14(C1121)
        elif  num1121==5:
            p31211=self.conv15(C1121)
        elif  num1121==6:
            p31211=self.conv16(C1121)
        elif  num1121==7:
            p31211=self.conv17(C1121)
        elif  num1121==8:
            p31211=self.conv18(C1121)
        elif  num1121==9:
            p31211=self.conv19(C1121)
        elif  num1121==10:
            p31211=self.conv110(C1121)
        elif  num1121==11:
            p31211=self.conv111(C1121)
        elif  num1121==12:
            p31211=self.conv112(C1121)
        elif  num1121==13:
            p31211=self.conv113(C1121)
        elif  num1121==14:
            p31211=self.conv114(C1121)
        elif  num1121==15:
            p31211=self.conv115(C1121)
        elif  num1121==16:
            p31211=self.conv116(C1121)
        elif  num1121==17:
            p31211=self.conv117(C1121)
        elif  num1121==18:
            p31211=self.conv118(C1121)
        elif  num1121==19:
            p31211=self.conv119(C1121)
        elif  num1121==20:
            p31211=self.conv120(C1121)
        elif  num1121==21:
            p31211=self.conv121(C1121)
        elif  num1121==22:
            p31211=self.conv122(C1121)
        elif  num1121==23:
            p31211=self.conv123(C1121)
        elif  num1121==24:
            p31211=self.conv124(C1121)
        elif  num1121==25:
            p31211=self.conv125(C1121)
        elif  num1121==26:
            p31211=self.conv126(C1121)
        elif  num1121==27:
            p31211=self.conv127(C1121)
        elif  num1121==28:
            p31211=self.conv128(C1121)
        elif  num1121==29:
            p31211=self.conv129(C1121)
        elif  num1121==30:
            p31211=self.conv130(C1121)
        elif  num1121==31:
            p31211=self.conv131(C1121)
        elif  num1121==32:
            p31211=self.conv132(C1121)
        elif  num1121==33:
            p31211=self.conv133(C1121)
        elif  num1121==34:
            p31211=self.conv134(C1121)
        elif  num1121==35:
            p31211=self.conv135(C1121)
        elif  num1121==36:
            p31211=self.conv136(C1121)
        elif  num1121==37:
            p31211=self.conv137(C1121)
        elif  num1121==38:
            p31211=self.conv138(C1121)
        elif  num1121==39:
            p31211=self.conv139(C1121)
        elif  num1121==40:
            p31211=self.conv140(C1121)
        elif  num1121==41:
            p31211=self.conv141(C1121)
        elif  num1121==42:
            p31211=self.conv142(C1121)
        elif  num1121==43:
            p31211=self.conv143(C1121)
        elif  num1121==44:
            p31211=self.conv144(C1121)
        elif  num1121==45:
            p31211=self.conv145(C1121)
        elif  num1121==46:
            p31211=self.conv146(C1121)
        elif  num1121==47:
            p31211=self.conv147(C1121)
        elif  num1121==48:
            p31211=self.conv148(C1121)
        elif  num1121==49:
            p31211=self.conv149(C1121)
        elif  num1121==50:
            p31211=self.conv150(C1121)
        elif  num1121==51:
            p31211=self.conv151(C1121)
        elif  num1121==52:
            p31211=self.conv152(C1121)
        elif  num1121==53:
            p31211=self.conv153(C1121)
        elif  num1121==54:
            p31211=self.conv154(C1121)
        elif  num1121==55:
            p31211=self.conv155(C1121)
        elif  num1121==56:
            p31211=self.conv156(C1121)
        elif  num1121==57:
            p31211=self.conv157(C1121)
        elif  num1121==58:
            p31211=self.conv158(C1121)
        elif  num1121==59:
            p31211=self.conv159(C1121)
        elif  num1121==60:
            p31211=self.conv160(C1121)
        elif  num1121==61:
            p31211=self.conv161(C1121)
        elif  num1121==62:
            p31211=self.conv162(C1121)
        elif  num1121==63:
            p31211=self.conv163(C1121)
        elif  num1121==64:
            p31211=self.conv164(C1121)
        elif  num1121==65:
            p31211=self.conv165(C1121)
        elif  num1121==66:
            p31211=self.conv166(C1121)
        elif  num1121==67:
            p31211=self.conv167(C1121)
        elif  num1121==68:
            p31211=self.conv168(C1121)
        elif  num1121==69:
            p31211=self.conv169(C1121)
        elif  num1121==70:
            p31211=self.conv170(C1121)
        elif  num1121==71:
            p31211=self.conv171(C1121)
        elif  num1121==72:
            p31211=self.conv172(C1121)
        elif  num1121==73:
            p31211=self.conv173(C1121)
        elif  num1121==74:
            p31211=self.conv174(C1121)
        elif  num1121==75:
            p31211=self.conv175(C1121)
        elif  num1121==76:
            p31211=self.conv176(C1121)
        elif  num1121==77:
            p31211=self.conv177(C1121)
        elif  num1121==78:
            p31211=self.conv178(C1121)
        elif  num1121==79:
            p31211=self.conv179(C1121)
        elif  num1121==80:
            p31211=self.conv180(C1121)
        elif  num1121==81:
            p31211=self.conv181(C1121)
        elif  num1121==82:
            p31211=self.conv182(C1121)
        elif  num1121==83:
            p31211=self.conv183(C1121)
        elif  num1121==84:
            p31211=self.conv184(C1121)
        elif  num1121==85:
            p31211=self.conv185(C1121)
        elif  num1121==86:
            p31211=self.conv186(C1121)
        elif  num1121==87:
            p31211=self.conv187(C1121)
        elif  num1121==88:
            p31211=self.conv188(C1121)
        elif  num1121==89:
            p31211=self.conv189(C1121)    
        elif  num1121==90:
            p31211=self.conv190(C1121)
        elif  num1121==91:
            p31211=self.conv191(C1121)
        elif  num1121==92:
            p31211=self.conv192(C1121)
        elif  num1121==93:
            p31211=self.conv193(C1121)
        elif  num1121==94:
            p31211=self.conv194(C1121)
        elif  num1121==95:
            p31211=self.conv195(C1121)
        elif  num1121==96:
            p31211=self.conv196(C1121)
        elif  num1121==97:
            p31211=self.conv197(C1121)
        elif  num1121==98:
            p31211=self.conv198(C1121)
        elif  num1121==99:
            p31211=self.conv199(C1121) 
        elif  num1121==100:
            p31211=self.conv1100(C1121)
        elif  num1121==101:
            p31211=self.conv1101(C1121)
        elif  num1121==102:
            p31211=self.conv1102(C1121)
        elif  num1121==103:
            p31211=self.conv1103(C1121)
        elif  num1121==104:
            p31211=self.conv1104(C1121)
        elif  num1121==105:
            p31211=self.conv1105(C1121)
        elif  num1121==106:
            p31211=self.conv1106(C1121)
        elif  num1121==107:
            p31211=self.conv1107(C1121)
        elif  num1121==108:
            p31211=self.conv1108(C1121)
        elif  num1121==109:
            p31211=self.conv1109(C1121)
        elif  num1121==110:
            p31211=self.conv1110(C1121)
        elif  num1121==111:
            p31211=self.conv1111(C1121)
        elif  num1121==112:
            p31211=self.conv1112(C1121)
        elif  num1121==113:
            p31211=self.conv1113(C1121)
        elif  num1121==114:
            p31211=self.conv1114(C1121)
        elif  num1121==115:
            p31211=self.conv1115(C1121)
        elif  num1121==116:
            p31211=self.conv1116(C1121)
        elif  num1121==117:
            p31211=self.conv1117(C1121)
        elif  num1121==118:
            p31211=self.conv1118(C1121)
        elif  num1121==119:
            p31211=self.conv1119(C1121) 
        elif  num1121==120:
            p31211=self.conv1120(C1121)
        elif  num1121==121:
            p31211=self.conv1121(C1121)
        elif  num1121==122:
            p31211=self.conv1122(C1121)
        elif  num1121==123:
            p31211=self.conv1123(C1121)
        elif  num1121==124:
            p31211=self.conv1124(C1121)
        elif  num1121==125:
            p31211=self.conv1125(C1121)
        elif  num1121==126:
            p31211=self.conv1126(C1121)
        elif  num1121==127:
            p31211=self.conv1127(C1121)
        elif  num1121==128:
            p31211=self.conv1128(C1121)
        elif  num1121==129:
            p31211=self.conv1129(C1121) 
        elif  num1121==130:
            p31211=self.conv1130(C1121)
        elif  num1121==131:
            p31211=self.conv1131(C1121)
        elif  num1121==132:
            p31211=self.conv1132(C1121)
        elif  num1121==133:
            p31211=self.conv1133(C1121)
        elif  num1121==134:
            p31211=self.conv1134(C1121)
        elif  num1121==135:
            p31211=self.conv1135(C1121)
        elif  num1121==136:
            p31211=self.conv1136(C1121)
        elif  num1121==137:
            p31211=self.conv1137(C1121)
        elif  num1121==138:
            p31211=self.conv1138(C1121)
        elif  num1121==139:
            p31211=self.conv1139(C1121)
        elif  num1121==140:
            p31211=self.conv1140(C1121)
        elif  num1121==141:
            p31211=self.conv1141(C1121)
        elif  num1121==142:
            p31211=self.conv1142(C1121)
        elif  num1121==143:
            p31211=self.conv1143(C1121)
        elif  num1121==144:
            p31211=self.conv1144(C1121)
        elif  num1121==145:
            p31211=self.conv1145(C1121)
        elif  num1121==146:
            p31211=self.conv1146(C1121)
        elif  num1121==147:
            p31211=self.conv1147(C1121)
        elif  num1121==148:
            p31211=self.conv1148(C1121)
        elif  num1121==149:
            p31211=self.conv1149(C1121) 
        elif  num1121==150:
            p31211=self.conv1150(C1121)
        elif  num1121==151:
            p31211=self.conv1151(C1121)
        elif  num1121==152:
            p31211=self.conv1152(C1121)
        elif  num1121==153:
            p31211=self.conv1153(C1121)
        elif  num1121==154:
            p31211=self.conv1154(C1121)
        elif  num1121==155:
            p31211=self.conv1155(C1121)
        elif  num1121==156:
            p31211=self.conv1156(C1121)
        elif  num1121==157:
            p31211=self.conv1157(C1121)
        elif  num1121==158:
            p31211=self.conv1158(C1121)
        elif  num1121==159:
            p31211=self.conv1159(C1121) 
        elif  num1121==160:
            p31211=self.conv1160(C1121)
        elif  num1121==161:
            p31211=self.conv1161(C1121)
        elif  num1121==162:
            p31211=self.conv1162(C1121)
        elif  num1121==163:
            p31211=self.conv1163(C1121)
        elif  num1121==164:
            p31211=self.conv1164(C1121)
        elif  num1121==165:
            p31211=self.conv1165(C1121)
        elif  num1121==166:
            p31211=self.conv1166(C1121)
        elif  num1121==167:
            p31211=self.conv1167(C1121)
        elif  num1121==168:
            p31211=self.conv1168(C1121)
        elif  num1121==169:
            p31211=self.conv1169(C1121) 
        elif  num1121==170:
            p31211=self.conv1170(C1121)
        elif  num1121==171:
            p31211=self.conv1171(C1121)
        elif  num1121==172:
            p31211=self.conv1172(C1121)
        elif  num1121==173:
            p31211=self.conv1173(C1121)
        elif  num1121==174:
            p31211=self.conv1174(C1121)
        elif  num1121==175:
            p31211=self.conv1175(C1121)
        elif  num1121==176:
            p31211=self.conv1176(C1121)
        elif  num1121==177:
            p31211=self.conv1177(C1121)
        elif  num1121==178:
            p31211=self.conv1178(C1121)
        elif  num1121==179:
            p31211=self.conv1179(C1121)                                                                                              
        elif  num1121==180:
            p31211=self.conv1180(C1121)
        elif  num1121==181:
            p31211=self.conv1181(C1121)
        elif  num1121==182:
            p31211=self.conv1182(C1121)
        elif  num1121==183:
            p31211=self.conv1183(C1121)
        elif  num1121==184:
            p31211=self.conv1184(C1121)
        elif  num1121==185:
            p31211=self.conv1185(C1121)
        elif  num1121==186:
            p31211=self.conv1186(C1121)
        elif  num1121==187:
            p31211=self.conv1187(C1121)
        elif  num1121==188:
            p31211=self.conv1188(C1121)
        elif  num1121==189:
            p31211=self.conv1189(C1121) 
        elif  num1121==190:
            p31211=self.conv1190(C1121)
        elif  num1121==191:
            p31211=self.conv1191(C1121)
        elif  num1121==192:
            p31211=self.conv1192(C1121)
        elif  num1121==193:
            p31211=self.conv1193(C1121)
        elif  num1121==194:
            p31211=self.conv1194(C1121)
        elif  num1121==195:
            p31211=self.conv1195(C1121)
        elif  num1121==196:
            p31211=self.conv1196(C1121)
        elif  num1121==197:
            p31211=self.conv1197(C1121)
        elif  num1121==198:
            p31211=self.conv1198(C1121)
        elif  num1121==199:
            p31211=self.conv1199(C1121)
        elif  num1121==200:
            p31211=self.conv1200(C1121)
        elif  num1121==201:
            p31211=self.conv1201(C1121)
        elif  num1121==202:
            p31211=self.conv1202(C1121)
        elif  num1121==203:
            p31211=self.conv1203(C1121)
        elif  num1121==204:
            p31211=self.conv1204(C1121)
        elif  num1121==205:
            p31211=self.conv1205(C1121)
        elif  num1121==206:
            p31211=self.conv1206(C1121)
        elif  num1121==207:
            p31211=self.conv1207(C1121)
        elif  num1121==208:
            p31211=self.conv1208(C1121)
        elif  num1121==209:
            p31211=self.conv1209(C1121)
        elif  num1121==210:
            p31211=self.conv1210(C1121)
        elif  num1121==211:
            p31211=self.conv1211(C1121)
        elif  num1121==212:
            p31211=self.conv1212(C1121)
        elif  num1121==213:
            p31211=self.conv1213(C1121)
        elif  num1121==214:
            p31211=self.conv1214(C1121)
        elif  num1121==215:
            p31211=self.conv1215(C1121)
        elif  num1121==216:
            p31211=self.conv1216(C1121)
        elif  num1121==217:
            p31211=self.conv1217(C1121)
        elif  num1121==218:
            p31211=self.conv1218(C1121)
        elif  num1121==219:
            p31211=self.conv1219(C1121)
        elif  num1121==220:
            p31211=self.conv1220(C1121)
        elif  num1121==221:
            p31211=self.conv1221(C1121)
        elif  num1121==222:
            p31211=self.conv1222(C1121)
        elif  num1121==223:
            p31211=self.conv1223(C1121)
        elif  num1121==224:
            p31211=self.conv1224(C1121)
        elif  num1121==225:
            p31211=self.conv1225(C1121)
        elif  num1121==226:
            p31211=self.conv1226(C1121)
        elif  num1121==227:
            p31211=self.conv1227(C1121)
        elif  num1121==228:
            p31211=self.conv1228(C1121)
        elif  num1121==229:
            p31211=self.conv1229(C1121)
        elif  num1121==230:
            p31211=self.conv1230(C1121)
        elif  num1121==231:
            p31211=self.conv1231(C1121)
        elif  num1121==232:
            p31211=self.conv1232(C1121)
        elif  num1121==233:
            p31211=self.conv1233(C1121)
        elif  num1121==234:
            p31211=self.conv1234(C1121)
        elif  num1121==235:
            p31211=self.conv1235(C1121)
        elif  num1121==236:
            p31211=self.conv1236(C1121)
        elif  num1121==237:
            p31211=self.conv1237(C1121)
        elif  num1121==238:
            p31211=self.conv1238(C1121)
        elif  num1121==239:
            p31211=self.conv1239(C1121) 
        elif  num1121==240:
            p31211=self.conv1240(C1121)
        elif  num1121==241:
            p31211=self.conv1241(C1121)
        elif  num1121==242:
            p31211=self.conv1242(C1121)
        elif  num1121==243:
            p31211=self.conv1243(C1121)
        elif  num1121==244:
            p31211=self.conv1244(C1121)
        elif  num1121==245:
            p31211=self.conv1245(C1121)
        elif  num1121==246:
            p31211=self.conv1246(C1121)
        elif  num1121==247:
            p31211=self.conv1247(C1121)
        elif  num1121==248:
            p31211=self.conv1248(C1121)
        elif  num1121==249:
            p31211=self.conv1249(C1121)
        elif  num1121==250:
            p31211=self.conv1250(C1121)
        elif  num1121==251:
            p31211=self.conv1251(C1121)
        elif  num1121==252:
            p31211=self.conv1252(C1121)
        elif  num1121==253:
            p31211=self.conv1253(C1121)
        elif  num1121==254:
            p31211=self.conv1254(C1121)
        elif  num1121==255:
            p31211=self.conv1255(C1121)
        elif  num1121==256:
            p31211=self.conv1256(C1121)
            
        if  num2121==1:
            p31212=self.conv11(D1121)
        elif  num2121==2:
            p31212=self.conv12(D1121)
        elif  num2121==3:
            p31212=self.conv13(D1121)
        elif  num2121==4:
            p31212=self.conv14(D1121)
        elif  num2121==5:
            p31212=self.conv15(D1121)
        elif  num2121==6:
            p31212=self.conv16(D1121)
        elif  num2121==7:
            p31212=self.conv17(D1121)
        elif  num2121==8:
            p31212=self.conv18(D1121)
        elif  num2121==9:
            p31212=self.conv19(D1121)
        elif  num2121==10:
            p31212=self.conv110(D1121)
        elif  num2121==11:
            p31212=self.conv111(D1121)
        elif  num2121==12:
            p31212=self.conv112(D1121)
        elif  num2121==13:
            p31212=self.conv113(D1121)
        elif  num2121==14:
            p31212=self.conv114(D1121)
        elif  num2121==15:
            p31212=self.conv115(D1121)
        elif  num2121==16:
            p31212=self.conv116(D1121)
        elif  num2121==17:
            p31212=self.conv117(D1121)
        elif  num2121==18:
            p31212=self.conv118(D1121)
        elif  num2121==19:
            p31212=self.conv119(D1121)
        elif  num2121==20:
            p31212=self.conv120(D1121)
        elif  num2121==21:
            p31212=self.conv121(D1121)
        elif  num2121==22:
            p31212=self.conv122(D1121)
        elif  num2121==23:
            p31212=self.conv123(D1121)
        elif  num2121==24:
            p31212=self.conv124(D1121)
        elif  num2121==25:
            p31212=self.conv125(D1121)
        elif  num2121==26:
            p31212=self.conv126(D1121)
        elif  num2121==27:
            p31212=self.conv127(D1121)
        elif  num2121==28:
            p31212=self.conv128(D1121)
        elif  num2121==29:
            p31212=self.conv129(D1121)
        elif  num2121==30:
            p31212=self.conv130(D1121)
        elif  num2121==31:
            p31212=self.conv131(D1121)
        elif  num2121==32:
            p31212=self.conv132(D1121)
        elif  num2121==33:
            p31212=self.conv133(D1121)
        elif  num2121==34:
            p31212=self.conv134(D1121)
        elif  num2121==35:
            p31212=self.conv135(D1121)
        elif  num2121==36:
            p31212=self.conv136(D1121)
        elif  num2121==37:
            p31212=self.conv137(D1121)
        elif  num2121==38:
            p31212=self.conv138(D1121)
        elif  num2121==39:
            p31212=self.conv139(D1121)
        elif  num2121==40:
            p31212=self.conv140(D1121)
        elif  num2121==41:
            p31212=self.conv141(D1121)
        elif  num2121==42:
            p31212=self.conv142(D1121)
        elif  num2121==43:
            p31212=self.conv143(D1121)
        elif  num2121==44:
            p31212=self.conv144(D1121)
        elif  num2121==45:
            p31212=self.conv145(D1121)
        elif  num2121==46:
            p31212=self.conv146(D1121)
        elif  num2121==47:
            p31212=self.conv147(D1121)
        elif  num2121==48:
            p31212=self.conv148(D1121)
        elif  num2121==49:
            p31212=self.conv149(D1121)
        elif  num2121==50:
            p31212=self.conv150(D1121)
        elif  num2121==51:
            p31212=self.conv151(D1121)
        elif  num2121==52:
            p31212=self.conv152(D1121)
        elif  num2121==53:
            p31212=self.conv153(D1121)
        elif  num2121==54:
            p31212=self.conv154(D1121)
        elif  num2121==55:
            p31212=self.conv155(D1121)
        elif  num2121==56:
            p31212=self.conv156(D1121)
        elif  num2121==57:
            p31212=self.conv157(D1121)
        elif  num2121==58:
            p31212=self.conv158(D1121)
        elif  num2121==59:
            p31212=self.conv159(D1121)
        elif  num2121==60:
            p31212=self.conv160(D1121)
        elif  num2121==61:
            p31212=self.conv161(D1121)
        elif  num2121==62:
            p31212=self.conv162(D1121)
        elif  num2121==63:
            p31212=self.conv163(D1121)
        elif  num2121==64:
            p31212=self.conv164(D1121)
        elif  num2121==65:
            p31212=self.conv165(D1121)
        elif  num2121==66:
            p31212=self.conv166(D1121)
        elif  num2121==67:
            p31212=self.conv167(D1121)
        elif  num2121==68:
            p31212=self.conv168(D1121)
        elif  num2121==69:
            p31212=self.conv169(D1121)
        elif  num2121==70:
            p31212=self.conv170(D1121)
        elif  num2121==71:
            p31212=self.conv171(D1121)
        elif  num2121==72:
            p31212=self.conv172(D1121)
        elif  num2121==73:
            p31212=self.conv173(D1121)
        elif  num2121==74:
            p31212=self.conv174(D1121)
        elif  num2121==75:
            p31212=self.conv175(D1121)
        elif  num2121==76:
            p31212=self.conv176(D1121)
        elif  num2121==77:
            p31212=self.conv177(D1121)
        elif  num2121==78:
            p31212=self.conv178(D1121)
        elif  num2121==79:
            p31212=self.conv179(D1121)
        elif  num2121==80:
            p31212=self.conv180(D1121)
        elif  num2121==81:
            p31212=self.conv181(D1121)
        elif  num2121==82:
            p31212=self.conv182(D1121)
        elif  num2121==83:
            p31212=self.conv183(D1121)
        elif  num2121==84:
            p31212=self.conv184(D1121)
        elif  num2121==85:
            p31212=self.conv185(D1121)
        elif  num2121==86:
            p31212=self.conv186(D1121)
        elif  num2121==87:
            p31212=self.conv187(D1121)
        elif  num2121==88:
            p31212=self.conv188(D1121)
        elif  num2121==89:
            p31212=self.conv189(D1121)    
        elif  num2121==90:
            p31212=self.conv190(D1121)
        elif  num2121==91:
            p31212=self.conv191(D1121)
        elif  num2121==92:
            p31212=self.conv192(D1121)
        elif  num2121==93:
            p31212=self.conv193(D1121)
        elif  num2121==94:
            p31212=self.conv194(D1121)
        elif  num2121==95:
            p31212=self.conv195(D1121)
        elif  num2121==96:
            p31212=self.conv196(D1121)
        elif  num2121==97:
            p31212=self.conv197(D1121)
        elif  num2121==98:
            p31212=self.conv198(D1121)
        elif  num2121==99:
            p31212=self.conv199(D1121) 
        elif  num2121==100:
            p31212=self.conv1100(D1121)
        elif  num2121==101:
            p31212=self.conv1101(D1121)
        elif  num2121==102:
            p31212=self.conv1102(D1121)
        elif  num2121==103:
            p31212=self.conv1103(D1121)
        elif  num2121==104:
            p31212=self.conv1104(D1121)
        elif  num2121==105:
            p31212=self.conv1105(D1121)
        elif  num2121==106:
            p31212=self.conv1106(D1121)
        elif  num2121==107:
            p31212=self.conv1107(D1121)
        elif  num2121==108:
            p31212=self.conv1108(D1121)
        elif  num2121==109:
            p31212=self.conv1109(D1121)
        elif  num2121==110:
            p31212=self.conv1110(D1121)
        elif  num2121==111:
            p31212=self.conv1111(D1121)
        elif  num2121==112:
            p31212=self.conv1112(D1121)
        elif  num2121==113:
            p31212=self.conv1113(D1121)
        elif  num2121==114:
            p31212=self.conv1114(D1121)
        elif  num2121==115:
            p31212=self.conv1115(D1121)
        elif  num2121==116:
            p31212=self.conv1116(D1121)
        elif  num2121==117:
            p31212=self.conv1117(D1121)
        elif  num2121==118:
            p31212=self.conv1118(D1121)
        elif  num2121==119:
            p31212=self.conv1119(D1121) 
        elif  num2121==120:
            p31212=self.conv1120(D1121)
        elif  num2121==121:
            p31212=self.conv1121(D1121)
        elif  num2121==122:
            p31212=self.conv1122(D1121)
        elif  num2121==123:
            p31212=self.conv1123(D1121)
        elif  num2121==124:
            p31212=self.conv1124(D1121)
        elif  num2121==125:
            p31212=self.conv1125(D1121)
        elif  num2121==126:
            p31212=self.conv1126(D1121)
        elif  num2121==127:
            p31212=self.conv1127(D1121)
        elif  num2121==128:
            p31212=self.conv1128(D1121)
        elif  num2121==129:
            p31212=self.conv1129(D1121) 
        elif  num2121==130:
            p31212=self.conv1130(D1121)
        elif  num2121==131:
            p31212=self.conv1131(D1121)
        elif  num2121==132:
            p31212=self.conv1132(D1121)
        elif  num2121==133:
            p31212=self.conv1133(D1121)
        elif  num2121==134:
            p31212=self.conv1134(D1121)
        elif  num2121==135:
            p31212=self.conv1135(D1121)
        elif  num2121==136:
            p31212=self.conv1136(D1121)
        elif  num2121==137:
            p31212=self.conv1137(D1121)
        elif  num2121==138:
            p31212=self.conv1138(D1121)
        elif  num2121==139:
            p31212=self.conv1139(D1121)
        elif  num2121==140:
            p31212=self.conv1140(D1121)
        elif  num2121==141:
            p31212=self.conv1141(D1121)
        elif  num2121==142:
            p31212=self.conv1142(D1121)
        elif  num2121==143:
            p31212=self.conv1143(D1121)
        elif  num2121==144:
            p31212=self.conv1144(D1121)
        elif  num2121==145:
            p31212=self.conv1145(D1121)
        elif  num2121==146:
            p31212=self.conv1146(D1121)
        elif  num2121==147:
            p31212=self.conv1147(D1121)
        elif  num2121==148:
            p31212=self.conv1148(D1121)
        elif  num2121==149:
            p31212=self.conv1149(D1121) 
        elif  num2121==150:
            p31212=self.conv1150(D1121)
        elif  num2121==151:
            p31212=self.conv1151(D1121)
        elif  num2121==152:
            p31212=self.conv1152(D1121)
        elif  num2121==153:
            p31212=self.conv1153(D1121)
        elif  num2121==154:
            p31212=self.conv1154(D1121)
        elif  num2121==155:
            p31212=self.conv1155(D1121)
        elif  num2121==156:
            p31212=self.conv1156(D1121)
        elif  num2121==157:
            p31212=self.conv1157(D1121)
        elif  num2121==158:
            p31212=self.conv1158(D1121)
        elif  num2121==159:
            p31212=self.conv1159(D1121) 
        elif  num2121==160:
            p31212=self.conv1160(D1121)
        elif  num2121==161:
            p31212=self.conv1161(D1121)
        elif  num2121==162:
            p31212=self.conv1162(D1121)
        elif  num2121==163:
            p31212=self.conv1163(D1121)
        elif  num2121==164:
            p31212=self.conv1164(D1121)
        elif  num2121==165:
            p31212=self.conv1165(D1121)
        elif  num2121==166:
            p31212=self.conv1166(D1121)
        elif  num2121==167:
            p31212=self.conv1167(D1121)
        elif  num2121==168:
            p31212=self.conv1168(D1121)
        elif  num2121==169:
            p31212=self.conv1169(D1121) 
        elif  num2121==170:
            p31212=self.conv1170(D1121)
        elif  num2121==171:
            p31212=self.conv1171(D1121)
        elif  num2121==172:
            p31212=self.conv1172(D1121)
        elif  num2121==173:
            p31212=self.conv1173(D1121)
        elif  num2121==174:
            p31212=self.conv1174(D1121)
        elif  num2121==175:
            p31212=self.conv1175(D1121)
        elif  num2121==176:
            p31212=self.conv1176(D1121)
        elif  num2121==177:
            p31212=self.conv1177(D1121)
        elif  num2121==178:
            p31212=self.conv1178(D1121)
        elif  num2121==179:
            p31212=self.conv1179(D1121)                                                                                              
        elif  num2121==180:
            p31212=self.conv1180(D1121)
        elif  num2121==181:
            p31212=self.conv1181(D1121)
        elif  num2121==182:
            p31212=self.conv1182(D1121)
        elif  num2121==183:
            p31212=self.conv1183(D1121)
        elif  num2121==184:
            p31212=self.conv1184(D1121)
        elif  num2121==185:
            p31212=self.conv1185(D1121)
        elif  num2121==186:
            p31212=self.conv1186(D1121)
        elif  num2121==187:
            p31212=self.conv1187(D1121)
        elif  num2121==188:
            p31212=self.conv1188(D1121)
        elif  num2121==189:
            p31212=self.conv1189(D1121) 
        elif  num2121==190:
            p31212=self.conv1190(D1121)
        elif  num2121==191:
            p31212=self.conv1191(D1121)
        elif  num2121==192:
            p31212=self.conv1192(D1121)
        elif  num2121==193:
            p31212=self.conv1193(D1121)
        elif  num2121==194:
            p31212=self.conv1194(D1121)
        elif  num2121==195:
            p31212=self.conv1195(D1121)
        elif  num2121==196:
            p31212=self.conv1196(D1121)
        elif  num2121==197:
            p31212=self.conv1197(D1121)
        elif  num2121==198:
            p31212=self.conv1198(D1121)
        elif  num2121==199:
            p31212=self.conv1199(D1121)
        elif  num2121==200:
            p31212=self.conv1200(D1121)
        elif  num2121==201:
            p31212=self.conv1201(D1121)
        elif  num2121==202:
            p31212=self.conv1202(D1121)
        elif  num2121==203:
            p31212=self.conv1203(D1121)
        elif  num2121==204:
            p31212=self.conv1204(D1121)
        elif  num2121==205:
            p31212=self.conv1205(D1121)
        elif  num2121==206:
            p31212=self.conv1206(D1121)
        elif  num2121==207:
            p31212=self.conv1207(D1121)
        elif  num2121==208:
            p31212=self.conv1208(D1121)
        elif  num2121==209:
            p31212=self.conv1209(D1121)
        elif  num2121==210:
            p31212=self.conv1210(D1121)
        elif  num2121==211:
            p31212=self.conv1211(D1121)
        elif  num2121==212:
            p31212=self.conv1212(D1121)
        elif  num2121==213:
            p31212=self.conv1213(D1121)
        elif  num2121==214:
            p31212=self.conv1214(D1121)
        elif  num2121==215:
            p31212=self.conv1215(D1121)
        elif  num2121==216:
            p31212=self.conv1216(D1121)
        elif  num2121==217:
            p31212=self.conv1217(D1121)
        elif  num2121==218:
            p31212=self.conv1218(D1121)
        elif  num2121==219:
            p31212=self.conv1219(D1121)
        elif  num2121==220:
            p31212=self.conv1220(D1121)
        elif  num2121==221:
            p31212=self.conv1221(D1121)
        elif  num2121==222:
            p31212=self.conv1222(D1121)
        elif  num2121==223:
            p31212=self.conv1223(D1121)
        elif  num2121==224:
            p31212=self.conv1224(D1121)
        elif  num2121==225:
            p31212=self.conv1225(D1121)
        elif  num2121==226:
            p31212=self.conv1226(D1121)
        elif  num2121==227:
            p31212=self.conv1227(D1121)
        elif  num2121==228:
            p31212=self.conv1228(D1121)
        elif  num2121==229:
            p31212=self.conv1229(D1121)
        elif  num2121==230:
            p31212=self.conv1230(D1121)
        elif  num2121==231:
            p31212=self.conv1231(D1121)
        elif  num2121==232:
            p31212=self.conv1232(D1121)
        elif  num2121==233:
            p31212=self.conv1233(D1121)
        elif  num2121==234:
            p31212=self.conv1234(D1121)
        elif  num2121==235:
            p31212=self.conv1235(D1121)
        elif  num2121==236:
            p31212=self.conv1236(D1121)
        elif  num2121==237:
            p31212=self.conv1237(D1121)
        elif  num2121==238:
            p31212=self.conv1238(D1121)
        elif  num2121==239:
            p31212=self.conv1239(D1121) 
        elif  num2121==240:
            p31212=self.conv1240(D1121)
        elif  num2121==241:
            p31212=self.conv1241(D1121)
        elif  num2121==242:
            p31212=self.conv1242(D1121)
        elif  num2121==243:
            p31212=self.conv1243(D1121)
        elif  num2121==244:
            p31212=self.conv1244(D1121)
        elif  num2121==245:
            p31212=self.conv1245(D1121)
        elif  num2121==246:
            p31212=self.conv1246(D1121)
        elif  num2121==247:
            p31212=self.conv1247(D1121)
        elif  num2121==248:
            p31212=self.conv1248(D1121)
        elif  num2121==249:
            p31212=self.conv1249(D1121)
        elif  num2121==250:
            p31212=self.conv1250(D1121)
        elif  num2121==251:
            p31212=self.conv1251(D1121)
        elif  num2121==252:
            p31212=self.conv1252(D1121)
        elif  num2121==253:
            p31212=self.conv1253(D1121)
        elif  num2121==254:
            p31212=self.conv1254(D1121)
        elif  num2121==255:
            p31212=self.conv1255(D1121)
        elif  num2121==256:
            p31212=self.conv1256(D1121)
        elif  num2121==257:
            p31212=self.conv1257(D1121)
        elif  num2121==258:
            p31212=self.conv1258(D1121)
        elif  num2121==259:
            p31212=self.conv1259(D1121)
        elif  num2121==260:
            p31212=self.conv1260(D1121)
        elif  num2121==261:
            p31212=self.conv1261(D1121)
        elif  num2121==262:
            p31212=self.conv1262(D1121)
        elif  num2121==263:
            p31212=self.conv1263(D1121)
        elif  num2121==264:
            p31212=self.conv1264(D1121)
        elif  num2121==265:
            p31212=self.conv1265(D1121)
        elif  num2121==266:
            p31212=self.conv1266(D1121)
        elif  num2121==267:
            p31212=self.conv1267(D1121)
        elif  num2121==268:
            p31212=self.conv1268(D1121)
        elif  num2121==269:
            p31212=self.conv1269(D1121) 
        elif  num2121==270:
            p31212=self.conv1270(D1121)
        elif  num2121==271:
            p31212=self.conv1271(D1121)
        elif  num2121==272:
            p31212=self.conv1272(D1121)
        elif  num2121==273:
            p31212=self.conv1273(D1121)
        elif  num2121==274:
            p31212=self.conv1274(D1121)
        elif  num2121==275:
            p31212=self.conv1275(D1121)
        elif  num2121==276:
            p31212=self.conv1276(D1121)
        elif  num2121==277:
            p31212=self.conv1277(D1121)
        elif  num2121==278:
            p31212=self.conv1278(D1121)
        elif  num2121==279:
            p31212=self.conv1279(D1121)
        elif  num2121==280:
            p31212=self.conv1280(D1121)
        elif  num2121==281:
            p31212=self.conv1281(D1121)
        elif  num2121==282:
            p31212=self.conv1282(D1121)
        elif  num2121==283:
            p31212=self.conv1283(D1121)
        elif  num2121==284:
            p31212=self.conv1284(D1121)
        elif  num2121==285:
            p31212=self.conv1285(D1121)
        elif  num2121==286:
            p31212=self.conv1286(D1121)
        elif  num2121==287:
            p31212=self.conv1287(D1121)
        elif  num2121==288:
            p31212=self.conv1288(D1121)
        elif  num2121==289:
            p31212=self.conv1289(D1121) 
        elif  num2121==290:
            p31212=self.conv1290(D1121)
        elif  num2121==291:
            p31212=self.conv1291(D1121)
        elif  num2121==292:
            p31212=self.conv1292(D1121)
        elif  num2121==293:
            p31212=self.conv1293(D1121)
        elif  num2121==294:
            p31212=self.conv1294(D1121)
        elif  num2121==295:
            p31212=self.conv1295(D1121)
        elif  num2121==296:
            p31212=self.conv1296(D1121)
        elif  num2121==297:
            p31212=self.conv1297(D1121)
        elif  num2121==298:
            p31212=self.conv1298(D1121)
        elif  num2121==299:
            p31212=self.conv1299(D1121)
        elif  num2121==300:
            p31212=self.conv1300(D1121)
        elif  num2121==301:
            p31212=self.conv1301(D1121)
        elif  num2121==302:
            p31212=self.conv1302(D1121)
        elif  num2121==303:
            p31212=self.conv1303(D1121)
        elif  num2121==304:
            p31212=self.conv1304(D1121)
        elif  num2121==305:
            p31212=self.conv1305(D1121)
        elif  num2121==306:
            p31212=self.conv1306(D1121)
        elif  num2121==307:
            p31212=self.conv1307(D1121)
        elif  num2121==308:
            p31212=self.conv1308(D1121)
        elif  num2121==309:
            p31212=self.conv1309(D1121) 
        elif  num2121==310:
            p31212=self.conv1310(D1121)
        elif  num2121==311:
            p31212=self.conv1311(D1121)
        elif  num2121==312:
            p31212=self.conv1312(D1121)
        elif  num2121==313:
            p31212=self.conv1313(D1121)
        elif  num2121==314:
            p31212=self.conv1314(D1121)
        elif  num2121==315:
            p31212=self.conv1315(D1121)
        elif  num2121==316:
            p31212=self.conv1316(D1121)
        elif  num2121==317:
            p31212=self.conv1317(D1121)
        elif  num2121==318:
            p31212=self.conv1318(D1121)
        elif  num2121==319:
            p31212=self.conv1319(D1121)
        elif  num2121==320:
            p31212=self.conv1320(D1121)
        elif  num2121==321:
            p31212=self.conv1321(D1121)
        elif  num2121==322:
            p31212=self.conv1322(D1121)
        elif  num2121==323:
            p31212=self.conv1323(D1121)
        elif  num2121==324:
            p31212=self.conv1324(D1121)
        elif  num2121==325:
            p31212=self.conv1325(D1121)
        elif  num2121==326:
            p31212=self.conv1326(D1121)
        elif  num2121==327:
            p31212=self.conv1327(D1121)
        elif  num2121==328:
            p31212=self.conv1328(D1121)
        elif  num2121==329:
            p31212=self.conv1329(D1121)
        elif  num2121==330:
            p31212=self.conv1330(D1121)
        elif  num2121==331:
            p31212=self.conv1331(D1121)
        elif  num2121==332:
            p31212=self.conv1332(D1121)
        elif  num2121==333:
            p31212=self.conv1333(D1121)
        elif  num2121==334:
            p31212=self.conv1334(D1121)
        elif  num2121==335:
            p31212=self.conv1335(D1121)
        elif  num2121==336:
            p31212=self.conv1336(D1121)
        elif  num2121==337:
            p31212=self.conv1337(D1121)
        elif  num2121==338:
            p31212=self.conv1338(D1121)
        elif  num2121==339:
            p31212=self.conv1339(D1121)
        elif  num2121==340:
            p31212=self.conv1340(D1121)
        elif  num2121==341:
            p31212=self.conv1341(D1121)
        elif  num2121==342:
            p31212=self.conv1342(D1121)
        elif  num2121==343:
            p31212=self.conv1343(D1121)
        elif  num2121==344:
            p31212=self.conv1344(D1121)
        elif  num2121==345:
            p31212=self.conv1345(D1121)
        elif  num2121==346:
            p31212=self.conv1346(D1121)
        elif  num2121==347:
            p31212=self.conv1347(D1121)
        elif  num2121==348:
            p31212=self.conv1348(D1121)
        elif  num2121==349:
            p31212=self.conv1349(D1121)
        elif  num2121==350:
            p31212=self.conv1350(D1121)
        elif  num2121==351:
            p31212=self.conv1351(D1121)
        elif  num2121==352:
            p31212=self.conv1352(D1121)
        elif  num2121==353:
            p31212=self.conv1335(D1121)
        elif  num2121==354:
            p31212=self.conv1354(D1121)
        elif  num2121==355:
            p31212=self.conv1355(D1121)
        elif  num2121==356:
            p31212=self.conv1356(D1121)
        elif  num2121==357:
            p31212=self.conv1357(D1121)
        elif  num2121==358:
            p31212=self.conv1358(D1121)
        elif  num2121==359:
            p31212=self.conv1359(D1121) 
        elif  num2121==360:
            p31212=self.conv1360(D1121)
        elif  num2121==361:
            p31212=self.conv1361(D1121)
        elif  num2121==362:
            p31212=self.conv1362(D1121)
        elif  num2121==363:
            p31212=self.conv1363(D1121)
        elif  num2121==364:
            p31212=self.conv1364(D1121)
        elif  num2121==365:
            p31212=self.conv1365(D1121)
        elif  num2121==366:
            p31212=self.conv1366(D1121)
        elif  num2121==367:
            p31212=self.conv1367(D1121)
        elif  num2121==368:
            p31212=self.conv1368(D1121)
        elif  num2121==369:
            p31212=self.conv1369(D1121) 
        elif  num2121==370:
            p31212=self.conv1370(D1121)
        elif  num2121==371:
            p31212=self.conv1371(D1121)
        elif  num2121==372:
            p31212=self.conv1372(D1121)
        elif  num2121==373:
            p31212=self.conv1373(D1121)
        elif  num2121==374:
            p31212=self.conv1374(D1121)
        elif  num2121==375:
            p31212=self.conv1375(D1121)
        elif  num2121==376:
            p31212=self.conv1376(D1121)
        elif  num2121==377:
            p31212=self.conv1377(D1121)
        elif  num2121==378:
            p31212=self.conv1378(D1121)
        elif  num2121==379:
            p31212=self.conv1379(D1121) 
        elif  num2121==380:
            p31212=self.conv1380(D1121)
        elif  num2121==381:
            p31212=self.conv1381(D1121)
        elif  num2121==382:
            p31212=self.conv1382(D1121)
        elif  num2121==383:
            p31212=self.conv1383(D1121)
        elif  num2121==384:
            p31212=self.conv1384(D1121)
        elif  num2121==385:
            p31212=self.conv1385(D1121)
        elif  num2121==386:
            p31212=self.conv1386(D1121)
        elif  num2121==387:
            p31212=self.conv1387(D1121)
        elif  num2121==388:
            p31212=self.conv1388(D1121)
        elif  num2121==389:
            p31212=self.conv1389(D1121) 
        elif  num2121==390:
            p31212=self.conv1390(D1121)
        elif  num2121==391:
            p31212=self.conv1391(D1121)
        elif  num2121==392:
            p31212=self.conv1392(D1121)
        elif  num2121==393:
            p31212=self.conv1393(D1121)
        elif  num2121==394:
            p31212=self.conv1394(D1121)
        elif  num2121==395:
            p31212=self.conv1395(D1121)
        elif  num2121==396:
            p31212=self.conv1396(D1121)
        elif  num2121==397:
            p31212=self.conv1397(D1121)
        elif  num2121==398:
            p31212=self.conv1398(D1121)
        elif  num2121==399:
            p31212=self.conv1399(D1121)
        elif  num2121==400:
            p31212=self.conv1400(D1121)
        elif  num2121==401:
            p31212=self.conv1401(D1121)
        elif  num2121==402:
            p31212=self.conv1402(D1121)
        elif  num2121==403:
            p31212=self.conv1403(D1121)
        elif  num2121==404:
            p31212=self.conv1404(D1121)
        elif  num2121==405:
            p31212=self.conv1405(D1121)
        elif  num2121==406:
            p31212=self.conv1406(D1121)
        elif  num2121==407:
            p31212=self.conv1407(D1121)
        elif  num2121==408:
            p31212=self.conv1408(D1121)
        elif  num2121==409:
            p31212=self.conv1409(D1121)
        elif  num2121==410:
            p31212=self.conv1410(D1121)
        elif  num2121==411:
            p31212=self.conv1411(D1121)
        elif  num2121==412:
            p31212=self.conv1412(D1121)
        elif  num2121==413:
            p31212=self.conv1413(D1121)
        elif  num2121==414:
            p31212=self.conv1414(D1121)
        elif  num2121==415:
            p31212=self.conv145(D1121)
        elif  num2121==416:
            p31212=self.conv1416(D1121)
        elif  num2121==417:
            p31212=self.conv1417(D1121)
        elif  num2121==418:
            p31212=self.conv1418(D1121)
        elif  num2121==419:
            p31212=self.conv1419(D1121) 
        elif  num2121==420:
            p31212=self.conv1420(D1121)
        elif  num2121==421:
            p31212=self.conv1421(D1121)
        elif  num2121==422:
            p31212=self.conv1422(D1121)
        elif  num2121==423:
            p31212=self.conv1423(D1121)
        elif  num2121==424:
            p31212=self.conv1424(D1121)
        elif  num2121==425:
            p31212=self.conv1425(D1121)
        elif  num2121==426:
            p31212=self.conv1426(D1121)
        elif  num2121==427:
            p31212=self.conv1427(D1121)
        elif  num2121==428:
            p31212=self.conv1428(D1121)
        elif  num2121==429:
            p31212=self.conv1429(D1121) 
        elif  num2121==430:
            p31212=self.conv1430(D1121)
        elif  num2121==431:
            p31212=self.conv1431(D1121)
        elif  num2121==432:
            p31212=self.conv1432(D1121)
        elif  num2121==433:
            p31212=self.conv1433(D1121)
        elif  num2121==434:
            p31212=self.conv1434(D1121)
        elif  num2121==435:
            p31212=self.conv1435(D1121)
        elif  num2121==436:
            p31212=self.conv1436(D1121)
        elif  num2121==437:
            p31212=self.conv1437(D1121)
        elif  num2121==438:
            p31212=self.conv1438(D1121)
        elif  num2121==439:
            p31212=self.conv1439(D1121)
        elif  num2121==440:
            p31212=self.conv1440(D1121)
        elif  num2121==441:
            p31212=self.conv1441(D1121)
        elif  num2121==442:
            p31212=self.conv1442(D1121)
        elif  num2121==443:
            p31212=self.conv1443(D1121)
        elif  num2121==444:
            p31212=self.conv1444(D1121)
        elif  num2121==445:
            p31212=self.conv1445(D1121)
        elif  num2121==446:
            p31212=self.conv1446(D1121)
        elif  num2121==447:
            p31212=self.conv1447(D1121)
        elif  num2121==448:
            p31212=self.conv1448(D1121)
        elif  num2121==449:
            p31212=self.conv1449(D1121)
        elif  num2121==450:
            p31212=self.conv1450(D1121)
        elif  num2121==451:
            p31212=self.conv1451(D1121)
        elif  num2121==452:
            p31212=self.conv1452(D1121)
        elif  num2121==453:
            p31212=self.conv1453(D1121)
        elif  num2121==454:
            p31212=self.conv1454(D1121)
        elif  num2121==455:
            p31212=self.conv1455(D1121)
        elif  num2121==456:
            p31212=self.conv1456(D1121)
        elif  num2121==457:
            p31212=self.conv1457(D1121)
        elif  num2121==458:
            p31212=self.conv1458(D1121)
        elif  num2121==459:
            p31212=self.conv1459(D1121)
        elif  num2121==460:
            p31212=self.conv1460(D1121)
        elif  num2121==461:
            p31212=self.conv1461(D1121)
        elif  num2121==462:
            p31212=self.conv1462(D1121)
        elif  num2121==463:
            p31212=self.conv1463(D1121)
        elif  num2121==464:
            p31212=self.conv1464(D1121)
        elif  num2121==465:
            p31212=self.conv1465(D1121)
        elif  num2121==466:
            p31212=self.conv1466(D1121)
        elif  num2121==467:
            p31212=self.conv1467(D1121)
        elif  num2121==468:
            p31212=self.conv1468(D1121)
        elif  num2121==469:
            p31212=self.conv1469(D1121) 
        elif  num2121==470:
            p31212=self.conv1470(D1121)
        elif  num2121==471:
            p31212=self.conv1471(D1121)
        elif  num2121==472:
            p31212=self.conv1472(D1121)
        elif  num2121==473:
            p31212=self.conv1473(D1121)
        elif  num2121==474:
            p31212=self.conv1474(D1121)
        elif  num2121==475:
            p31212=self.conv1475(D1121)
        elif  num2121==476:
            p31212=self.conv1476(D1121)
        elif  num2121==477:
            p31212=self.conv1477(D1121)
        elif  num2121==478:
            p31212=self.conv1478(D1121)
        elif  num2121==479:
            p31212=self.conv1479(D1121)
        elif  num2121==480:
            p31212=self.conv1480(D1121)
        elif  num2121==481:
            p31212=self.conv1481(D1121)
        elif  num2121==482:
            p31212=self.conv1482(D1121)
        elif  num2121==483:
            p31212=self.conv1483(D1121)
        elif  num2121==484:
            p31212=self.conv1484(D1121)
        elif  num2121==485:
            p31212=self.conv1485(D1121)
        elif  num2121==486:
            p31212=self.conv1486(D1121)
        elif  num2121==487:
            p31212=self.conv1487(D1121)
        elif  num2121==488:
            p31212=self.conv1488(D1121)
        elif  num2121==489:
            p31212=self.conv1489(D1121)
        elif  num2121==490:
            p31212=self.conv1490(D1121)
        elif  num2121==491:
            p31212=self.conv1491(D1121)
        elif  num2121==492:
            p31212=self.conv1492(D1121)
        elif  num2121==493:
            p31212=self.conv1493(D1121)
        elif  num2121==494:
            p31212=self.conv1494(D1121)
        elif  num2121==495:
            p31212=self.conv1495(D1121)
        elif  num2121==496:
            p31212=self.conv1496(D1121)
        elif  num2121==497:
            p31212=self.conv1497(D1121)
        elif  num2121==498:
            p31212=self.conv1498(D1121)
        elif  num2121==499:
            p31212=self.conv1499(D1121)
        elif  num2121==500:
            p31212=self.conv1500(D1121)
        elif  num2121==501:
            p31212=self.conv1501(D1121)
        elif  num2121==502:
            p31212=self.conv1502(D1121)
        elif  num2121==503:
            p31212=self.conv1503(D1121)
        elif  num2121==504:
            p31212=self.conv1504(D1121)
        elif  num2121==505:
            p31212=self.conv1505(D1121)
        elif  num2121==506:
            p31212=self.conv1506(D1121)
        elif  num2121==507:
            p31212=self.conv1507(D1121)
        elif  num2121==508:
            p31212=self.conv1508(D1121)
        elif  num2121==509:
            p31212=self.conv1509(D1121)
        elif  num2121==510:
            p31212=self.conv1510(D1121)
        elif  num2121==511:
            p31212=self.conv1511(D1121)
        elif  num2121==512:
            p31212=self.conv1512(D1121)
        elif  num2121==513:
            p31212=self.conv1513(D1121)
        elif  num2121==514:
            p31212=self.conv1514(D1121)
        elif  num2121==515:
            p31212=self.conv1515(D1121)
        elif  num2121==516:
            p31212=self.conv1516(D1121)
        elif  num2121==517:
            p31212=self.conv1517(D1121)
        elif  num2121==518:
            p31212=self.conv1518(D1121)
        elif  num2121==519:
            p31212=self.conv1519(D1121)
        elif  num2121==520:
            p31212=self.conv1520(D1121)
        elif  num2121==521:
            p31212=self.conv1521(D1121)
        elif  num2121==522:
            p31212=self.conv1522(D1121)
        elif  num2121==523:
            p31212=self.conv1523(D1121)
        elif  num2121==524:
            p31212=self.conv1524(D1121)
        elif  num2121==525:
            p31212=self.conv1525(D1121)
        elif  num2121==526:
            p31212=self.conv1526(D1121)
        elif  num2121==527:
            p31212=self.conv1527(D1121)
        elif  num2121==528:
            p31212=self.conv1528(D1121)
        elif  num2121==529:
            p31212=self.conv1529(D1121)
        elif  num2121==530:
            p31212=self.conv1530(D1121)
        elif  num2121==531:
            p31212=self.conv1531(D1121)
        elif  num2121==532:
            p31212=self.conv1532(D1121)
        elif  num2121==533:
            p31212=self.conv1533(D1121)
        elif  num2121==534:
            p31212=self.conv1534(D1121)
        elif  num2121==535:
            p31212=self.conv1535(D1121)
        elif  num2121==536:
            p31212=self.conv1536(D1121)
        elif  num2121==537:
            p31212=self.conv1537(D1121)
        elif  num2121==538:
            p31212=self.conv1538(D1121)
        elif  num2121==539:
            p31212=self.conv1539(D1121)
        elif  num2121==540:
            p31212=self.conv1540(D1121)
        elif  num2121==541:
            p31212=self.conv1541(D1121)
        elif  num2121==542:
            p31212=self.conv1542(D1121)
        elif  num2121==543:
            p31212=self.conv1543(D1121)
        elif  num2121==544:
            p31212=self.conv1544(D1121)
        elif  num2121==545:
            p31212=self.conv1545(D1121)
        elif  num2121==546:
            p31212=self.conv1546(D1121)
        elif  num2121==547:
            p31212=self.conv1547(D1121)
        elif  num2121==548:
            p31212=self.conv1548(D1121)
        elif  num2121==549:
            p31212=self.conv1549(D1121) 
        elif  num2121==550:
            p31212=self.conv1550(D1121)
        elif  num2121==551:
            p31212=self.conv1551(D1121)
        elif  num2121==552:
            p31212=self.conv1552(D1121)
        elif  num2121==553:
            p31212=self.conv1553(D1121)
        elif  num2121==554:
            p31212=self.conv1554(D1121)
        elif  num2121==555:
            p31212=self.conv1555(D1121)
        elif  num2121==556:
            p31212=self.conv1556(D1121)
        elif  num2121==557:
            p31212=self.conv1557(D1121)
        elif  num2121==558:
            p31212=self.conv1558(D1121)
        elif  num2121==559:
            p31212=self.conv1559(D1121)
        elif  num2121==560:
            p31212=self.conv1560(D1121)
        elif  num2121==561:
            p31212=self.conv1561(D1121)
        elif  num2121==562:
            p31212=self.conv1562(D1121)
        elif  num2121==563:
            p31212=self.conv1563(D1121)
        elif  num2121==564:
            p31212=self.conv1564(D1121)
        elif  num2121==565:
            p31212=self.conv1565(D1121)
        elif  num2121==566:
            p31212=self.conv1566(D1121)
        elif  num2121==567:
            p31212=self.conv1567(D1121)
        elif  num2121==568:
            p31212=self.conv1568(D1121)
        elif  num2121==569:
            p31212=self.conv1569(D1121) 
        elif  num2121==570:
            p31212=self.conv1570(D1121)
        elif  num2121==571:
            p31212=self.conv1571(D1121)
        elif  num2121==572:
            p31212=self.conv1572(D1121)
        elif  num2121==573:
            p31212=self.conv1573(D1121)
        elif  num2121==574:
            p31212=self.conv1574(D1121)
        elif  num2121==575:
            p31212=self.conv1575(D1121)
        elif  num2121==576:
            p31212=self.conv1576(D1121)
        elif  num2121==577:
            p31212=self.conv1577(D1121)
        elif  num2121==578:
            p31212=self.conv1578(D1121)
        elif  num2121==579:
            p31212=self.conv1579(D1121) 
        elif  num2121==580:
            p31212=self.conv1580(D1121)
        elif  num2121==581:
            p31212=self.conv1581(D1121)
        elif  num2121==582:
            p31212=self.conv1582(D1121)
        elif  num2121==583:
            p31212=self.conv1583(D1121)
        elif  num2121==584:
            p31212=self.conv1584(D1121)
        elif  num2121==585:
            p31212=self.conv1585(D1121)
        elif  num2121==586:
            p31212=self.conv1586(D1121)
        elif  num2121==587:
            p31212=self.conv1587(D1121)
        elif  num2121==588:
            p31212=self.conv1588(D1121)
        elif  num2121==589:
            p31212=self.conv1589(D1121)
        elif  num2121==590:
            p31212=self.conv1590(D1121)
        elif  num2121==591:
            p31212=self.conv1591(D1121)
        elif  num2121==592:
            p31212=self.conv1592(D1121)
        elif  num2121==593:
            p31212=self.conv1593(D1121)
        elif  num2121==594:
            p31212=self.conv1594(D1121)
        elif  num2121==595:
            p31212=self.conv1595(D1121)
        elif  num2121==596:
            p31212=self.conv1596(D1121)
        elif  num2121==597:
            p31212=self.conv1597(D1121)
        elif  num2121==598:
            p31212=self.conv1598(D1121)
        elif  num2121==599:
            p31212=self.conv1599(D1121)
        elif  num2121==600:
            p31212=self.conv1600(D1121)
        elif  num2121==601:
            p31212=self.conv1601(D1121)
        elif  num2121==602:
            p31212=self.conv1602(D1121)
        elif  num2121==603:
            p31212=self.conv1603(D1121)
        elif  num2121==604:
            p31212=self.conv1604(D1121)
        elif  num2121==605:
            p31212=self.conv1605(D1121)
        elif  num2121==606:
            p31212=self.conv1606(D1121)
        elif  num2121==607:
            p31212=self.conv1607(D1121)
        elif  num2121==608:
            p31212=self.conv1608(D1121)
        elif  num2121==609:
            p31212=self.conv1609(D1121)                                                                                                                         
        elif  num2121==610:
            p31212=self.conv1610(D1121)
        elif  num2121==611:
            p31212=self.conv1611(D1121)
        elif  num2121==612:
            p31212=self.conv1612(D1121)
        elif  num2121==613:
            p31212=self.conv1613(D1121)
        elif  num2121==614:
            p31212=self.conv1614(D1121)
        elif  num2121==615:
            p31212=self.conv1615(D1121)
        elif  num2121==616:
            p31212=self.conv1616(D1121)
        elif  num2121==617:
            p31212=self.conv1617(D1121)
        elif  num2121==618:
            p31212=self.conv1618(D1121)
        elif  num2121==619:
            p31212=self.conv1619(D1121)                                                                                                                         
            
        elif  num2121==620:
            p31212=self.conv1620(D1121)
        elif  num2121==621:
            p31212=self.conv1621(D1121)
        elif  num2121==622:
            p31212=self.conv1622(D1121)
        elif  num2121==623:
            p31212=self.conv1623(D1121)
        elif  num2121==624:
            p31212=self.conv1624(D1121)
        elif  num2121==625:
            p31212=self.conv1625(D1121)
        elif  num2121==626:
            p31212=self.conv1626(D1121)
        elif  num2121==627:
            p31212=self.conv1627(D1121)
        elif  num2121==628:
            p31212=self.conv1628(D1121)
        elif  num2121==629:
            p31212=self.conv1629(D1121)  
        elif  num2121==630:
            p31212=self.conv1630(D1121)
        elif  num2121==631:
            p31212=self.conv1631(D1121)
        elif  num2121==632:
            p31212=self.conv1632(D1121)
        elif  num2121==633:
            p31212=self.conv1633(D1121)
        elif  num2121==634:
            p31212=self.conv1634(D1121)
        elif  num2121==635:
            p31212=self.conv1635(D1121)
        elif  num2121==636:
            p31212=self.conv1636(D1121)
        elif  num2121==637:
            p31212=self.conv1637(D1121)
        elif  num2121==638:
            p31212=self.conv1638(D1121)
        elif  num2121==639:
            p31212=self.conv1639(D1121)                                                                                                                         
            
        elif  num2121==640:
            p31212=self.conv1640(D1121)
        elif  num2121==641:
            p31212=self.conv1641(D1121)
        elif  num2121==642:
            p31212=self.conv1642(D1121)
        elif  num2121==643:
            p31212=self.conv1643(D1121)
        elif  num2121==644:
            p31212=self.conv1644(D1121)
        elif  num2121==645:
            p31212=self.conv1645(D1121)
        elif  num2121==646:
            p31212=self.conv1646(D1121)
        elif  num2121==647:
            p31212=self.conv1647(D1121)
        elif  num2121==648:
            p31212=self.conv1648(D1121)
        elif  num2121==649:
            p31212=self.conv1649(D1121)                                                                                                                         
            
        elif  num2121==650:
            p31212=self.conv1650(D1121)
        elif  num2121==651:
            p31212=self.conv1651(D1121)
        elif  num2121==652:
            p31212=self.conv1652(D1121)
        elif  num2121==653:
            p31212=self.conv1653(D1121)
        elif  num2121==654:
            p31212=self.conv1654(D1121)
        elif  num2121==655:
            p31212=self.conv1655(D1121)
        elif  num2121==656:
            p31212=self.conv1656(D1121)
        elif  num2121==657:
            p31212=self.conv1657(D1121)
        elif  num2121==658:
            p31212=self.conv1658(D1121)
        elif  num2121==659:
            p31212=self.conv1659(D1121)
        elif  num2121==660:
            p31212=self.conv1660(D1121)
        elif  num2121==661:
            p31212=self.conv1661(D1121)
        elif  num2121==662:
            p31212=self.conv1662(D1121)
        elif  num2121==663:
            p31212=self.conv1663(D1121)
        elif  num2121==664:
            p31212=self.conv1664(D1121)
        elif  num2121==665:
            p31212=self.conv1665(D1121)
        elif  num2121==666:
            p31212=self.conv1666(D1121)
        elif  num2121==667:
            p31212=self.conv1667(D1121)
        elif  num2121==668:
            p31212=self.conv1668(D1121)
        elif  num2121==669:
            p31212=self.conv1669(D1121) 
        elif  num2121==670:
            p31212=self.conv1670(D1121)
        elif  num2121==671:
            p31212=self.conv1671(D1121)
        elif  num2121==672:
            p31212=self.conv1672(D1121)
        elif  num2121==673:
            p31212=self.conv1673(D1121)
        elif  num2121==674:
            p31212=self.conv1674(D1121)
        elif  num2121==675:
            p31212=self.conv1675(D1121)
        elif  num2121==676:
            p31212=self.conv1676(D1121)
        elif  num2121==677:
            p31212=self.conv1677(D1121)
        elif  num2121==678:
            p31212=self.conv1678(D1121)
        elif  num2121==679:
            p31212=self.conv1679(D1121)
        elif  num2121==680:
            p31212=self.conv1680(D1121)
        elif  num2121==681:
            p31212=self.conv1681(D1121)
        elif  num2121==682:
            p31212=self.conv1682(D1121)
        elif  num2121==683:
            p31212=self.conv1683(D1121)
        elif  num2121==684:
            p31212=self.conv1684(D1121)
        elif  num2121==685:
            p31212=self.conv1685(D1121)
        elif  num2121==686:
            p31212=self.conv1686(D1121)
        elif  num2121==687:
            p31212=self.conv1687(D1121)
        elif  num2121==688:
            p31212=self.conv1688(D1121)
        elif  num2121==689:
            p31212=self.conv1689(D1121)
        elif  num2121==690:
            p31212=self.conv1690(D1121)
        elif  num2121==691:
            p31212=self.conv1691(D1121)
        elif  num2121==692:
            p31212=self.conv1692(D1121)
        elif  num2121==693:
            p31212=self.conv1693(D1121)
        elif  num2121==694:
            p31212=self.conv1694(D1121)
        elif  num2121==695:
            p31212=self.conv1695(D1121)
        elif  num2121==696:
            p31212=self.conv1696(D1121)
        elif  num2121==697:
            p31212=self.conv1697(D1121)
        elif  num2121==698:
            p31212=self.conv1698(D1121)
        elif  num2121==699:
            p31212=self.conv1699(D1121)
        elif  num2121==700:
            p31212=self.conv1700(D1121)
        elif  num2121==701:
            p31212=self.conv1701(D1121)
        elif  num2121==702:
            p31212=self.conv1702(D1121)
        elif  num2121==703:
            p31212=self.conv1703(D1121)
        elif  num2121==704:
            p31212=self.conv1704(D1121)
        elif  num2121==705:
            p31212=self.conv1705(D1121)
        elif  num2121==706:
            p31212=self.conv1706(D1121)
        elif  num2121==707:
            p31212=self.conv1707(D1121)
        elif  num2121==708:
            p31212=self.conv1708(D1121)
        elif  num2121==709:
            p31212=self.conv1709(D1121)
        elif  num2121==710:
            p31212=self.conv1710(D1121)
        elif  num2121==711:
            p31212=self.conv1711(D1121)
        elif  num2121==712:
            p31212=self.conv1712(D1121)
        elif  num2121==713:
            p31212=self.conv1713(D1121)
        elif  num2121==714:
            p31212=self.conv1714(D1121)
        elif  num2121==715:
            p31212=self.conv1715(D1121)
        elif  num2121==716:
            p31212=self.conv1716(D1121)
        elif  num2121==717:
            p31212=self.conv1717(D1121)
        elif  num2121==718:
            p31212=self.conv1718(D1121)
        elif  num2121==719:
            p31212=self.conv1719(D1121)
        elif  num2121==720:
            p31212=self.conv1720(D1121)
        elif  num2121==721:
            p31212=self.conv1721(D1121)
        elif  num2121==722:
            p31212=self.conv1722(D1121)
        elif  num2121==723:
            p31212=self.conv1723(D1121)
        elif  num2121==724:
            p31212=self.conv1724(D1121)
        elif  num2121==725:
            p31212=self.conv1725(D1121)
        elif  num2121==726:
            p31212=self.conv1726(D1121)
        elif  num2121==727:
            p31212=self.conv1727(D1121)
        elif  num2121==728:
            p31212=self.conv1728(D1121)
        elif  num2121==729:
            p31212=self.conv1729(D1121)
        elif  num2121==730:
            p31212=self.conv1730(D1121)
        elif  num2121==731:
            p31212=self.conv1731(D1121)
        elif  num2121==732:
            p31212=self.conv1732(D1121)
        elif  num2121==733:
            p31212=self.conv1733(D1121)
        elif  num2121==734:
            p31212=self.conv1734(D1121)
        elif  num2121==735:
            p31212=self.conv1735(D1121)
        elif  num2121==736:
            p31212=self.conv1736(D1121)
        elif  num2121==737:
            p31212=self.conv1737(D1121)
        elif  num2121==738:
            p31212=self.conv1738(D1121)
        elif  num2121==739:
            p31212=self.conv1739(D1121)                                                                                                                         
            
        elif  num2121==740:
            p31212=self.conv1740(D1121)
        elif  num2121==741:
            p31212=self.conv1741(D1121)
        elif  num2121==742:
            p31212=self.conv1742(D1121)
        elif  num2121==743:
            p31212=self.conv1743(D1121)
        elif  num2121==744:
            p31212=self.conv1744(D1121)
        elif  num2121==745:
            p31212=self.conv1745(D1121)
        elif  num2121==746:
            p31212=self.conv1746(D1121)
        elif  num2121==747:
            p31212=self.conv1747(D1121)
        elif  num2121==748:
            p31212=self.conv1748(D1121)
        elif  num2121==749:
            p31212=self.conv1749(D1121)
        elif  num2121==750:
            p31212=self.conv1750(D1121)
        elif  num2121==751:
            p31212=self.conv1751(D1121)
        elif  num2121==752:
            p31212=self.conv1752(D1121)
        elif  num2121==753:
            p31212=self.conv1753(D1121)
        elif  num2121==754:
            p31212=self.conv1754(D1121)
        elif  num2121==755:
            p31212=self.conv1755(D1121)
        elif  num2121==756:
            p31212=self.conv1756(D1121)
        elif  num2121==757:
            p31212=self.conv1757(D1121)
        elif  num2121==758:
            p31212=self.conv1758(D1121)
        elif  num2121==759:
            p31212=self.conv1759(D1121)
        elif  num2121==760:
            p31212=self.conv1760(D1121)
        elif  num2121==761:
            p31212=self.conv1761(D1121)
        elif  num2121==762:
            p31212=self.conv1762(D1121)
        elif  num2121==763:
            p31212=self.conv1763(D1121)
        elif  num2121==764:
            p31212=self.conv1764(D1121)
        elif  num2121==765:
            p31212=self.conv1765(D1121)
        elif  num2121==766:
            p31212=self.conv1766(D1121)
        elif  num2121==767:
            p31212=self.conv1767(D1121)
        elif  num2121==768:
            p31212=self.conv1768(D1121)
        elif  num2121==769:
            p31212=self.conv1769(D1121) 
        elif  num2121==770:
            p31212=self.conv1770(D1121)
        elif  num2121==771:
            p31212=self.conv1771(D1121)
        elif  num2121==772:
            p31212=self.conv1772(D1121)
        elif  num2121==773:
            p31212=self.conv1773(D1121)
        elif  num2121==774:
            p31212=self.conv1774(D1121)
        elif  num2121==775:
            p31212=self.conv1775(D1121)
        elif  num2121==776:
            p31212=self.conv1776(D1121)
        elif  num2121==777:
            p31212=self.conv1777(D1121)
        elif  num2121==778:
            p31212=self.conv1778(D1121)
        elif  num2121==779:
            p31212=self.conv1779(D1121) 
        elif  num2121==780:
            p31212=self.conv1780(D1121)
        elif  num2121==781:
            p31212=self.conv1781(D1121)
        elif  num2121==782:
            p31212=self.conv1782(D1121)
        elif  num2121==783:
            p31212=self.conv1783(D1121)
        elif  num2121==784:
            p31212=self.conv1784(D1121)
        elif  num2121==785:
            p31212=self.conv1785(D1121)
        elif  num2121==786:
            p31212=self.conv1786(D1121)
        elif  num2121==787:
            p31212=self.conv1787(D1121)
        elif  num2121==788:
            p31212=self.conv1788(D1121)
        elif  num2121==789:
            p31212=self.conv1789(D1121) 
        elif  num2121==790:
            p31212=self.conv1790(D1121)
        elif  num2121==791:
            p31212=self.conv1791(D1121)
        elif  num2121==792:
            p31212=self.conv1792(D1121)
        elif  num2121==793:
            p31212=self.conv1793(D1121)
        elif  num2121==794:
            p31212=self.conv1794(D1121)
        elif  num2121==795:
            p31212=self.conv1795(D1121)
        elif  num2121==796:
            p31212=self.conv1796(D1121)
        elif  num2121==797:
            p31212=self.conv1797(D1121)
        elif  num2121==798:
            p31212=self.conv1798(D1121)
        elif  num2121==799:
            p31212=self.conv1799(D1121) 
        elif  num2121==800:
            p31212=self.conv1800(D1121)
        elif  num2121==801:
            p31212=self.conv1801(D1121)
        elif  num2121==802:
            p31212=self.conv1802(D1121)
        elif  num2121==803:
            p31212=self.conv1803(D1121)
        elif  num2121==804:
            p31212=self.conv1804(D1121)
        elif  num2121==805:
            p31212=self.conv1805(D1121)
        elif  num2121==806:
            p31212=self.conv1806(D1121)
        elif  num2121==807:
            p31212=self.conv1807(D1121)
        elif  num2121==808:
            p31212=self.conv1808(D1121)
        elif  num2121==809:
            p31212=self.conv1809(D1121)
        elif  num2121==810:
            p31212=self.conv1810(D1121)
        elif  num2121==811:
            p31212=self.conv1811(D1121)
        elif  num2121==812:
            p31212=self.conv1812(D1121)
        elif  num2121==813:
            p31212=self.conv1813(D1121)
        elif  num2121==814:
            p31212=self.conv1814(D1121)
        elif  num2121==815:
            p31212=self.conv1815(D1121)
        elif  num2121==816:
            p31212=self.conv1816(D1121)
        elif  num2121==817:
            p31212=self.conv1817(D1121)
        elif  num2121==818:
            p31212=self.conv1818(D1121)
        elif  num2121==819:
            p31212=self.conv1819(D1121)
        elif  num2121==820:
            p31212=self.conv1820(D1121)
        elif  num2121==821:
            p31212=self.conv1821(D1121)
        elif  num2121==822:
            p31212=self.conv1822(D1121)
        elif  num2121==823:
            p31212=self.conv1823(D1121)
        elif  num2121==824:
            p31212=self.conv1824(D1121)
        elif  num2121==825:
            p31212=self.conv1825(D1121)
        elif  num2121==826:
            p31212=self.conv1826(D1121)
        elif  num2121==827:
            p31212=self.conv1827(D1121)
        elif  num2121==828:
            p31212=self.conv1828(D1121)
        elif  num2121==829:
            p31212=self.conv1829(D1121)                                                                                                                         
            
        elif  num2121==830:
            p31212=self.conv1830(D1121)
        elif  num2121==831:
            p31212=self.conv1831(D1121)
        elif  num2121==832:
            p31212=self.conv1832(D1121)
        elif  num2121==833:
            p31212=self.conv1833(D1121)
        elif  num2121==834:
            p31212=self.conv1834(D1121)
        elif  num2121==835:
            p31212=self.conv1835(D1121)
        elif  num2121==836:
            p31212=self.conv1836(D1121)
        elif  num2121==837:
            p31212=self.conv1837(D1121)
        elif  num2121==838:
            p31212=self.conv1838(D1121)
        elif  num2121==839:
            p31212=self.conv1839(D1121)
        elif  num2121==840:
            p31212=self.conv1840(D1121)
        elif  num2121==841:
            p31212=self.conv1841(D1121)
        elif  num2121==842:
            p31212=self.conv1842(D1121)
        elif  num2121==843:
            p31212=self.conv1843(D1121)
        elif  num2121==844:
            p31212=self.conv1844(D1121)
        elif  num2121==845:
            p31212=self.conv1845(D1121)
        elif  num2121==846:
            p31212=self.conv1846(D1121)
        elif  num2121==847:
            p31212=self.conv1847(D1121)
        elif  num2121==848:
            p31212=self.conv1848(D1121)
        elif  num2121==849:
            p31212=self.conv1849(D1121)
        elif  num2121==850:
            p31212=self.conv1850(D1121)
        elif  num2121==851:
            p31212=self.conv1851(D1121)
        elif  num2121==852:
            p31212=self.conv1852(D1121)
        elif  num2121==853:
            p31212=self.conv1853(D1121)
        elif  num2121==854:
            p31212=self.conv1854(D1121)
        elif  num2121==855:
            p31212=self.conv1855(D1121)
        elif  num2121==856:
            p31212=self.conv1856(D1121)
        elif  num2121==857:
            p31212=self.conv1857(D1121)
        elif  num2121==858:
            p31212=self.conv1858(D1121)
        elif  num2121==859:
            p31212=self.conv1859(D1121)
        elif  num2121==860:
            p31212=self.conv1860(D1121)
        elif  num2121==861:
            p31212=self.conv1861(D1121)
        elif  num2121==862:
            p31212=self.conv1862(D1121)
        elif  num2121==863:
            p31212=self.conv1863(D1121)
        elif  num2121==864:
            p31212=self.conv1864(D1121)
        elif  num2121==865:
            p31212=self.conv1865(D1121)
        elif  num2121==866:
            p31212=self.conv1866(D1121)
        elif  num2121==867:
            p31212=self.conv1867(D1121)
        elif  num2121==868:
            p31212=self.conv1868(D1121)
        elif  num2121==869:
            p31212=self.conv1869(D1121) 
        elif  num2121==870:
            p31212=self.conv1870(D1121)
        elif  num2121==871:
            p31212=self.conv1871(D1121)
        elif  num2121==872:
            p31212=self.conv1872(D1121)
        elif  num2121==873:
            p31212=self.conv1873(D1121)
        elif  num2121==874:
            p31212=self.conv1874(D1121)
        elif  num2121==875:
            p31212=self.conv1875(D1121)
        elif  num2121==876:
            p31212=self.conv1876(D1121)
        elif  num2121==877:
            p31212=self.conv1877(D1121)
        elif  num2121==878:
            p31212=self.conv1878(D1121)
        elif  num2121==879:
            p31212=self.conv1879(D1121)
        elif  num2121==880:
            p31212=self.conv1880(D1121)
        elif  num2121==881:
            p31212=self.conv1881(D1121)
        elif  num2121==882:
            p31212=self.conv1882(D1121)
        elif  num2121==883:
            p31212=self.conv1883(D1121)
        elif  num2121==884:
            p31212=self.conv1884(D1121)
        elif  num2121==885:
            p31212=self.conv1885(D1121)
        elif  num2121==886:
            p31212=self.conv1886(D1121)
        elif  num2121==887:
            p31212=self.conv1887(D1121)
        elif  num2121==888:
            p31212=self.conv1888(D1121)
        elif  num2121==889:
            p31212=self.conv1889(D1121)  
        elif  num2121==890:
            p31212=self.conv1890(D1121)
        elif  num2121==891:
            p31212=self.conv1891(D1121)
        elif  num2121==892:
            p31212=self.conv1892(D1121)
        elif  num2121==893:
            p31212=self.conv1893(D1121)
        elif  num2121==894:
            p31212=self.conv1894(D1121)
        elif  num2121==895:
            p31212=self.conv1895(D1121)
        elif  num2121==896:
            p31212=self.conv1896(D1121)
        elif  num2121==897:
            p31212=self.conv1897(D1121)
        elif  num2121==898:
            p31212=self.conv1898(D1121)
        elif  num2121==899:
            p31212=self.conv1899(D1121)
        elif  num2121==900:
            p31212=self.conv1900(D1121)
        elif  num2121==901:
            p31212=self.conv1901(D1121)
        elif  num2121==902:
            p31212=self.conv1902(D1121)
        elif  num2121==903:
            p31212=self.conv1903(D1121)
        elif  num2121==904:
            p31212=self.conv1904(D1121)
        elif  num2121==905:
            p31212=self.conv1905(D1121)
        elif  num2121==906:
            p31212=self.conv1906(D1121)
        elif  num2121==907:
            p31212=self.conv1907(D1121)
        elif  num2121==908:
            p31212=self.conv1908(D1121)
        elif  num2121==909:
            p31212=self.conv1909(D1121)
        elif  num2121==910:
            p31212=self.conv1910(D1121)
        elif  num2121==911:
            p31212=self.conv1911(D1121)
        elif  num2121==912:
            p31212=self.conv1912(D1121)
        elif  num2121==913:
            p31212=self.conv1913(D1121)
        elif  num2121==914:
            p31212=self.conv1914(D1121)
        elif  num2121==915:
            p31212=self.conv1915(D1121)
        elif  num2121==916:
            p31212=self.conv1916(D1121)
        elif  num2121==917:
            p31212=self.conv1917(D1121)
        elif  num2121==918:
            p31212=self.conv1918(D1121)
        elif  num2121==919:
            p31212=self.conv1919(D1121)
        elif  num2121==920:
            p31212=self.conv1920(D1121)
        elif  num2121==921:
            p31212=self.conv1921(D1121)
        elif  num2121==922:
            p31212=self.conv1922(D1121)
        elif  num2121==923:
            p31212=self.conv1923(D1121)
        elif  num2121==924:
            p31212=self.conv1924(D1121)
        elif  num2121==925:
            p31212=self.conv1925(D1121)
        elif  num2121==926:
            p31212=self.conv1926(D1121)
        elif  num2121==927:
            p31212=self.conv1927(D1121)
        elif  num2121==928:
            p31212=self.conv1928(D1121)
        elif  num2121==929:
            p31212=self.conv1929(D1121)
        elif  num2121==930:
            p31212=self.conv1930(D1121)
        elif  num2121==931:
            p31212=self.conv1931(D1121)
        elif  num2121==932:
            p31212=self.conv1932(D1121)
        elif  num2121==933:
            p31212=self.conv1933(D1121)
        elif  num2121==934:
            p31212=self.conv1934(D1121)
        elif  num2121==935:
            p31212=self.conv1935(D1121)
        elif  num2121==936:
            p31212=self.conv1936(D1121)
        elif  num2121==937:
            p31212=self.conv1937(D1121)
        elif  num2121==938:
            p31212=self.conv1938(D1121)
        elif  num2121==939:
            p31212=self.conv1939(D1121) 
        elif  num2121==940:
            p31212=self.conv1940(D1121)
        elif  num2121==941:
            p31212=self.conv1941(D1121)
        elif  num2121==942:
            p31212=self.conv1942(D1121)
        elif  num2121==943:
            p31212=self.conv1943(D1121)
        elif  num2121==944:
            p31212=self.conv1944(D1121)
        elif  num2121==945:
            p31212=self.conv1945(D1121)
        elif  num2121==946:
            p31212=self.conv1946(D1121)
        elif  num2121==947:
            p31212=self.conv1947(D1121)
        elif  num2121==948:
            p31212=self.conv1948(D1121)
        elif  num2121==949:
            p31212=self.conv1949(D1121)                                                                                                                         
            
        elif  num2121==950:
            p31212=self.conv1950(D1121)
        elif  num2121==951:
            p31212=self.conv1951(D1121)
        elif  num2121==952:
            p31212=self.conv1952(D1121)
        elif  num2121==953:
            p31212=self.conv1953(D1121)
        elif  num2121==954:
            p31212=self.conv1954(D1121)
        elif  num2121==955:
            p31212=self.conv1955(D1121)
        elif  num2121==956:
            p31212=self.conv1956(D1121)
        elif  num2121==957:
            p31212=self.conv1957(D1121)
        elif  num2121==958:
            p31212=self.conv1958(D1121)
        elif  num2121==959:
            p31212=self.conv1959(D1121)
        elif  num2121==960:
            p31212=self.conv1960(D1121)
        elif  num2121==961:
            p31212=self.conv1961(D1121)
        elif  num2121==962:
            p31212=self.conv1962(D1121)
        elif  num2121==963:
            p31212=self.conv1963(D1121)
        elif  num2121==964:
            p31212=self.conv1964(D1121)
        elif  num2121==965:
            p31212=self.conv1965(D1121)
        elif  num2121==966:
            p31212=self.conv1966(D1121)
        elif  num2121==967:
            p31212=self.conv1967(D1121)
        elif  num2121==968:
            p31212=self.conv1968(D1121)
        elif  num2121==969:
            p31212=self.conv1969(D1121) 
        elif  num2121==970:
            p31212=self.conv1970(D1121)
        elif  num2121==971:
            p31212=self.conv1971(D1121)
        elif  num2121==972:
            p31212=self.conv1972(D1121)
        elif  num2121==973:
            p31212=self.conv1973(D1121)
        elif  num2121==974:
            p31212=self.conv1974(D1121)
        elif  num2121==975:
            p31212=self.conv1975(D1121)
        elif  num2121==976:
            p31212=self.conv1976(D1121)
        elif  num2121==977:
            p31212=self.conv1977(D1121)
        elif  num2121==978:
            p31212=self.conv1978(D1121)
        elif  num2121==979:
            p31212=self.conv1979(D1121) 
        elif  num2121==980:
            p31212=self.conv1980(D1121)
        elif  num2121==981:
            p31212=self.conv1981(D1121)
        elif  num2121==982:
            p31212=self.conv1982(D1121)
        elif  num2121==983:
            p31212=self.conv1983(D1121)
        elif  num2121==984:
            p31212=self.conv1984(D1121)
        elif  num2121==985:
            p31212=self.conv1985(D1121)
        elif  num2121==986:
            p31212=self.conv1986(D1121)
        elif  num2121==987:
            p31212=self.conv1987(D1121)
        elif  num2121==988:
            p31212=self.conv1988(D1121)
        elif  num2121==989:
            p31212=self.conv1989(D1121)
        elif  num2121==990:
            p31212=self.conv1990(D1121)
        elif  num2121==991:
            p31212=self.conv1991(D1121)
        elif  num2121==992:
            p31212=self.conv1992(D1121)
        elif  num2121==993:
            p31212=self.conv1993(D1121)
        elif  num2121==994:
            p31212=self.conv1994(D1121)
        elif  num2121==995:
            p31212=self.conv1995(D1121)
        elif  num2121==996:
            p31212=self.conv1996(D1121)
        elif  num2121==997:
            p31212=self.conv1997(D1121)
        elif  num2121==998:
            p31212=self.conv1998(D1121)
        elif  num2121==999:
            p31212=self.conv1999(D1121) 
        elif  num2121==1000:
            p31212=self.conv11000(D1121)
        elif  num2121==1001:
            p31212=self.conv11001(D1121)
        elif  num2121==1002:
            p31212=self.conv11002(D1121)
        elif  num2121==1003:
            p31212=self.conv11003(D1121)
        elif  num2121==1004:
            p31212=self.conv11004(D1121)
        elif  num2121==1005:
            p31212=self.conv11005(D1121)
        elif  num2121==1006:
            p31212=self.conv11006(D1121)
        elif  num2121==1007:
            p31212=self.conv11007(D1121)
        elif  num2121==1008:
            p31212=self.conv11008(D1121)
        elif  num2121==1009:
            p31212=self.conv11009(D1121) 
        elif  num2121==1010:
            p31212=self.conv11010(D1121)
        elif  num2121==1011:
            p31212=self.conv11011(D1121)
        elif  num2121==1012:
            p31212=self.conv11012(D1121)
        elif  num2121==1013:
            p31212=self.conv11013(D1121)
        elif  num2121==1014:
            p31212=self.conv11014(D1121)
        elif  num2121==1015:
            p31212=self.conv11015(D1121)
        elif  num2121==1016:
            p31212=self.conv11016(D1121)
        elif  num2121==1017:
            p31212=self.conv11017(D1121)
        elif  num2121==1018:
            p31212=self.conv11018(D1121)
        elif  num2121==1019:
            p31212=self.conv11019(D1121)
        elif  num2121==1020:
            p31212=self.conv11020(D1121)
        elif  num2121==1021:
            p31212=self.conv11021(D1121)
        elif  num2121==1022:
            p31212=self.conv11022(D1121)
        elif  num2121==1023:
            p31212=self.conv11023(D1121)
        elif  num2121==1024:
            p31212=self.conv11024(D1121) 
#        print(num2)
#        print(p33.size())
#        print(p331.size())
#        print(p332.size())
#        print(p34.size())
#        print(p341.size())
#        print(p342.size())
#        print(p340.size())
#        print(p3401.size())
#        print(p3402.size())
#        print(p3441.size())
#        print(p3411.size())
#        print(p3412.size())
#        print(p380.size())
#        print(p3801.size())
#        print(p3802.size())
#        print(p38.size())
#        print(p381.size())
#        print(p382.size())
#        print(p3881.size())
#        print(p3811.size())
#        print(p3812.size())
#        print(p312.size())
#        print(p3121.size())
#        print(p3122.size())
#        print(p3120.size())
#        print(p31201.size())
#        print(p31202.size())
#        print(p312121.size())
#        print(p31211.size())
#        print(p31212.size())
        
       
        pB = torch.cat([p33, p3,p34,p340,p3441,p380,p38,p3881,p312,p3120,p312121], dim=1)
        #pBB=self.net3(pB)
        #pB = torch.cat([p33, p3,p34,p340,p3441,p380,p380,p38,p312,p3120,p3120], dim=1)
        
       # print(D1.size())
        c4 = self.conv4(pB)
        p4 = self.pool4(c4)
        pC = torch.cat([p331, p4,p341,p3401,p3411,p3801,p381,p3811,p3121,p31201,p31211], dim=1)
        #pCC=self.net2(pC)
        #pC = torch.cat([p331, p4,p341,p3401,p3401,p381,p3801,p3801,p3121,p31201,p31201], dim=1)
       # pC = torch.cat([p331, p4], dim=1)
        c5 = self.conv5(pC)
        p5 = self.pool5(c5)
        
        #pD = torch.cat([p332, p5], dim=1)
        pD = torch.cat([p332, p5,p342,p3402,p3412,p3802,p382,p3812,p3122,p31202,p31212], dim=1)
       # pDD=self.net1(pD)
        c51 = self.conv51(pD)
        p51 = self.pool5(c51)
        c52 = self.conv52(p51)
        p52 = self.pool5(c52)
        c53= self.conv53(p52)

        up_63= self.up63(c53)
        merge63 = torch.cat([up_63, c52], dim=1)
        c63= self.conv63(merge63)  
        up_62 = self.up62(c63)
        merge62 = torch.cat([up_62, c51], dim=1)
        c62= self.conv62(merge62)      
        up_61= self.up61(c62)
        merge61 = torch.cat([up_61, c5], dim=1)
        c61 = self.conv61(merge61)
        up_6 = self.up6(c61)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Tanh()(c10)
       # hro = GuidedFilter(r=3,eps=1e-8)(out, out)
        #hr_yo = self.sobel(hro)
       # pro = self.sobel(y)
        
        return out#, pr,hr,hr_y,pro,hro,hr_yo
        
def define_G1(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_id='cuda:1'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    net =ResnetGenerator(input_nc, output_nc)
   
    return init_net(net, init_type, init_gain, gpu_id)        


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(3, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.sa = SpatialAttention()
        self.down2 = Down(ngf * 2, ngf * 4, norm_layer, use_bias)
        self.se = SELayer(ngf * 4, 16)
        self.sobel = SobelOperator(1e-4)
        model = []
        for i in range(n_blocks):
            model += [ResBlock(ngf * 4, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)
           
        self.up1 = Up(ngf * 4, ngf * 2, norm_layer, use_bias)
        self.sa1 = SpatialAttention()
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf, output_nc)

    def forward(self, input):
        out = {}
   #     pr = self.sobel(input)
        out['in'] = self.inc(input)
        out['d1'] = self.down1(out['in'])
        #out['sa']= self.sa(out['d1']) * out['d1']
        out['d2'] = self.down2(out['d1'])
        #out['se']= self.se(out['d2'])
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        #out['sa1']= self.sa(out['u1']) * out['u1']
        out['u2'] = self.up2(out['u1'])

        return self.outc(out['u2'])


class Inconv(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Inconv, self).__init__()
        self.inconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0,
                      bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.inconv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=2, padding=1, bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.down(x)
        return x


# Define a Resnet block
class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return nn.ReLU(True)(out)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, norm_layer, use_bias):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.Conv2d(in_ch, out_ch,
            #           kernel_size=3, stride=1,
            #           padding=1, bias=use_bias),
            nn.ConvTranspose2d(in_ch, out_ch,
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1,
                               bias=use_bias),
            norm_layer(out_ch),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Outconv, self).__init__()
        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, out_ch, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.outconv(x)
        return x

#class Unet0(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(Unet0, self).__init__()
#
##        self.conv1 = DoubleConv(in_ch, 32)
##        self.pool1 = nn.MaxPool2d(2)
##        self.conv2 = DoubleConv(32, 64)
##        self.pool2 = nn.MaxPool2d(2)
##        self.conv3 = DoubleConv(64, 128)
##        self.pool3 = nn.MaxPool2d(2)
##        self.conv4 = DoubleConv(128, 256)
##        self.pool4 = nn.MaxPool2d(2)
##        self.conv5 = DoubleConv(256, 512)
#        
#        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
#        self.bn1 =nn.BatchNorm2d(32)
#        self.re=nn.ReLU(inplace=True)
#        self.conv11=nn.Conv2d(64, 32, 1, padding=0)
#        self.pool1 = nn.MaxPool2d(2)
#        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#        self.bn2 =nn.BatchNorm2d(64)
#        self.conv22=nn.Conv2d(128, 64, 1, padding=0)
#        self.conv222=nn.Conv2d(96, 64, 1, padding=0)
#        self.pool2 = nn.MaxPool2d(2)
#        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#        self.bn3 =nn.BatchNorm2d(128)
#        self.conv33=nn.Conv2d(256, 128, 1, padding=0)
#        self.conv333=nn.Conv2d(192, 128, 1, padding=0)
#        self.pool3 = nn.MaxPool2d(2)
#        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#        self.bn4 =nn.BatchNorm2d(256)
#        self.conv44=nn.Conv2d(512, 256, 1, padding=0)
#        self.conv444=nn.Conv2d(384, 256, 1, padding=0)
#        self.pool4 = nn.MaxPool2d(2)
#        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
#        self.bn5 =nn.BatchNorm2d(512)
#        self.conv55=nn.Conv2d(1024, 512, 1, padding=0)
#        self.conv555=nn.Conv2d(768, 512, 1, padding=0)
#        self.up6 = nn.ConvTranspose2d(512, 256, 2, stride=2)
#        self.conv6 = DoubleConv(512, 256)
#        self.up7 = nn.ConvTranspose2d(256, 128, 2, stride=2)
#        self.conv7 = DoubleConv(256, 128)
#        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
#        self.conv8 = DoubleConv(128, 64)
#        self.up9 = nn.ConvTranspose2d(64, 32, 2, stride=2)
#        self.conv9 = DoubleConv(64, 32)
#        self.conv10 = nn.Conv2d(32, out_ch, 1)
#
#    def forward(self, x1,y1):
#        c11 = self.conv1(y1)
#        c111 = self.conv1(x1)
#        c01 =torch.cat((c111, c11), 1)
#        c1 = self.conv11(c01)
#        c0112=self.bn1(c11)
#        c01112=self.bn1(c111)
#        c0113=self.re(c0112)
#        c01113=self.re(c01112)
#        p11 = self.pool1(c0113)
#        p111 = self.pool1(c01113)
#        p1 = self.pool1(c1)
#        
#        c22 = self.conv2(p11)
#        c222 = self.conv2(p111)
#        c02 =torch.cat((c222, c22), 1)
#        c002 =torch.cat((c222, p1), 1)#96
#        c0020 =torch.cat((p1, c22), 1)
#        c202 = self.conv222(c002)
#        c2002 = self.conv222(c0020)
#        c2 = self.conv22(c02)
#        c0222=self.bn2(c202)
#        c02222=self.bn2(c2002)
#        c0223=self.re(c0222)
#        c02223=self.re(c02222)
#        p22 = self.pool2(c0223)
#        p222 = self.pool2(c02223)
#        p2 = self.pool2(c2)
#        
#        c33 = self.conv3(p22)
#        c333 = self.conv3(p222)
#        c03 =torch.cat((c333, c33), 1)
#        c003 =torch.cat((c333, p2), 1)
#        c0030 =torch.cat((p2, c33), 1)
#        c303 = self.conv333(c003)
#        c3003 = self.conv333(c0030)
#        c3 = self.conv33(c03)
#        c0332=self.bn3(c303)
#        c03332=self.bn3(c3003)
#        c0333=self.re(c0332)
#        c03333=self.re(c03332)
#        p33 = self.pool3(c0333)
#        p333 = self.pool3(c03333)
#        p3 = self.pool3(c3)
#        
#        c44 = self.conv4(p33)
#        c444 = self.conv4(p333)
#        c04 =torch.cat((c444, c44), 1)
#        c004 =torch.cat((c444, p3), 1)
#        c0040 =torch.cat((p3, c44), 1)
#        c404 = self.conv444(c004)
#        c4004 = self.conv444(c0040)
#        c4 = self.conv44(c04)
#        c0442=self.bn4(c404)
#        c04442=self.bn4(c4004)
#        c0443=self.re(c0442)
#        c04443=self.re(c04442)
#        p44 = self.pool4(c0443)
#        p444 = self.pool4(c04443)
#        p4 = self.pool4(c4)
#        
#        c55 = self.conv5(p44)
#        c555 = self.conv5(p444)
#        c05 =torch.cat((c555, c55), 1)
##        c005 =torch.cat((c555, p4), 1)
##        c0050 =torch.cat((p4, c55), 1)
##        c505 = self.conv555(c005)
##        c5005 = self.conv555(c0050)
#        c5 = self.conv55(c05)
#        cc5=self.bn5(c5)
#        cc53=self.re(cc5)
##        c0552=self.bn5(c505)
##        c05552=self.bn5(c5005)
##        c0553=self.re(c0552)
##        c05553=self.re(c05552)
#        
#       
##        up_6 = self.up6(c05553)#x1
##        merge6 = torch.cat([up_6, c444], dim=1)
##        c6 = self.conv6(merge6)
##        up_7 = self.up7(c6)
##        merge7 = torch.cat([up_7, c333], dim=1)
##        c7 = self.conv7(merge7)
##        up_8 = self.up8(c7)
##        merge8 = torch.cat([up_8, c222], dim=1)
##        c8 = self.conv8(merge8)
##        up_9 = self.up9(c8)
##        merge9 = torch.cat([up_9, c111], dim=1)
##        c9 = self.conv9(merge9)
##        c10 = self.conv10(c9)
##        out = nn.Sigmoid()(c10)
##  
##        up_66 = self.up6(c0553)
##        merge66 = torch.cat([up_66, c44], dim=1)
##        c66 = self.conv6(merge66)
##        up_77 = self.up7(c66)
##        merge77 = torch.cat([up_77, c33], dim=1)
##        c77 = self.conv7(merge77)
##        up_88 = self.up8(c77)
##        merge88 = torch.cat([up_88, c22], dim=1)
##        c88 = self.conv8(merge88)
##        up_99 = self.up9(c88)
##        merge99 = torch.cat([up_99, c11], dim=1)
##        c99 = self.conv9(merge99)
##        c1010 = self.conv10(c99)
##        out1 = nn.Sigmoid()(c1010)
#        
#        up_6 = self.up6(cc53)#x1
#        merge6 = torch.cat([up_6, c444], dim=1)
#        c6 = self.conv6(merge6)
#        up_7 = self.up7(c6)
#        merge7 = torch.cat([up_7, c333], dim=1)
#        c7 = self.conv7(merge7)
#        up_8 = self.up8(c7)
#        merge8 = torch.cat([up_8, c222], dim=1)
#        c8 = self.conv8(merge8)
#        up_9 = self.up9(c8)
#        merge9 = torch.cat([up_9, c111], dim=1)
#        c9 = self.conv9(merge9)
#        c10 = self.conv10(c9)
#        out = nn.Sigmoid()(c10)
##  
##        up_66 = self.up6(c0553)
##        merge66 = torch.cat([up_66, c4], dim=1)
##        c66 = self.conv6(merge66)
##        up_77 = self.up7(c66)
##        merge77 = torch.cat([up_77, c3], dim=1)
##        c77 = self.conv7(merge77)
##        up_88 = self.up8(c77)
##        merge88 = torch.cat([up_88, c2], dim=1)
##        c88 = self.conv8(merge88)
##        up_99 = self.up9(c88)
##        merge99 = torch.cat([up_99, c1], dim=1)
##        c99 = self.conv9(merge99)
##        c1010 = self.conv10(c99)
##        out1 = nn.Sigmoid()(c1010)
#
#        return out


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain, gpu_id)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
            #self.loss = nn.MSELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
