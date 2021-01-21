import torch
from torch import nn
import math
import numpy as np
from torch.autograd import Variable

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class SGN(nn.Module):
    def __init__(self, num_classes, dataset, seg, args, bias = True):
        super(SGN, self).__init__()
        self.dim1 = 256
        self.dataset = dataset
        self.seg = seg
        num_joint = 25
        bs = args.batch_size
        if args.train:
            self.spa = self.one_hot(bs, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(bs, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()
        else:
            self.spa = self.one_hot(32 * 5, num_joint, self.seg)
            self.spa = self.spa.permute(0, 3, 2, 1).cuda()
            self.tem = self.one_hot(32 * 5, self.seg, num_joint)
            self.tem = self.tem.permute(0, 3, 1, 2).cuda()

        self.tem_embed = embed(self.seg, 64*4, norm=False, bias=bias)
        self.tem_embed2 = embed(self.seg, 64, norm=False, bias=bias)
        self.spa_embed = embed(num_joint, 64, norm=False, bias=bias)
        self.joint_embed = embed(3, 64, norm=True, bias=bias)
        self.dif_embed = embed(3, 64, norm=True, bias=bias)
        self.bone_embed = embed(3, 64,norm=True, bias = bias)
        self.bone_dif_embed = embed(3, 64,norm=True, bias = bias)
        self.motion_embed = embed(3, 64,norm=True, bias = bias)

        self.conv_cat = nn.Conv2d(64*6, 128, 1, bias=bias)
        self.conv_cat2 = nn.Conv2d(256*2, 256, 1, bias=bias)
        self.bn_cat = nn.BatchNorm2d(128)
        self.bn_cat2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.cnn = local(self.dim1, self.dim1 * 2, bias=bias)

        self.compute_g1 = compute_g_spa(self.dim1 // 2, self.dim1, bias=bias)
        self.gcn1 = unit_gcn(self.dim1 // 2, self.dim1 // 2)
        self.gcn2 = unit_gcn(self.dim1 // 2, self.dim1)
        self.gcn3 = unit_gcn(self.dim1, self.dim1)
        
        self.fc = nn.Linear(self.dim1 * 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        bs, step, dim = input.size()
        num_joints = dim //3
        input = input.view((bs, step, num_joints, 3))
        input = input.permute(0, 3, 2, 1).contiguous()

        paris = {'data':(
                (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
                (13, 1),
                (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
                (25, 12)
            )
        }
        bone = torch.zeros_like(input)
        for v1, v2 in paris['data']:
            v1 -= 1
            v2 -= 1
            bone[:, :, v1, :] = input[:, :, v1, :] - input[:, :, v2, :]
        
        bone_dif = bone[:, :, :, 1:] - bone[:, :, :, 0:-1]
        bone_dif = torch.cat([bone_dif.new(bs, bone_dif.size(1), num_joints, 1).zero_(), bone_dif], dim=-1)
   
        motion = torch.zeros_like(input)
        for t in range(step - 1):
            motion[:,:,:,t] = input[:, :, :, t+1] - input[:, :, :, t]

        bone = self.bone_embed(bone)
        bone_dif = self.bone_dif_embed(bone_dif)
        motion = self.motion_embed(motion)

        dif = input[:, :, :, 1:] - input[:, :, :, 0:-1]
        dif = torch.cat([dif.new(bs, dif.size(1), num_joints, 1).zero_(), dif], dim=-1)

        pos = self.joint_embed(input) 
        tem1 = self.tem_embed(self.tem)
        spa1 = self.spa_embed(self.spa)
        dif = self.dif_embed(dif)

        input = torch.cat([pos,dif,motion, bone,bone_dif, spa1],1)
        input = self.relu(self.bn_cat(self.conv_cat(input)))

        g = self.compute_g1(input)
        input = self.gcn1(input, g)
        input = self.gcn2(input, g)
        input = self.gcn3(input, g)
        input = torch.cat([input,tem1],1)
        input = self.relu(self.bn_cat2(self.conv_cat2(input)))
        input = self.cnn(input)
        output = self.maxpool(input)
        output = torch.flatten(output, 1)
        output = self.fc(output)

        return output

    def one_hot(self, bs, spa, tem):

        y = torch.arange(spa).unsqueeze(-1)
        y_onehot = torch.FloatTensor(spa, spa)

        y_onehot.zero_()
        y_onehot.scatter_(1, y, 1)

        y_onehot = y_onehot.unsqueeze(0).unsqueeze(0)
        y_onehot = y_onehot.repeat(bs, tem, 1, 1)

        return y_onehot

class norm_data(nn.Module):
    def __init__(self, dim= 64):
        super(norm_data, self).__init__()

        self.bn = nn.BatchNorm1d(dim* 25)

    def forward(self, x):
        bs, c, num_joints, step = x.size()
        x = x.view(bs, -1, step)
        x = self.bn(x)
        x = x.view(bs, -1, num_joints, step).contiguous()
        return x

class embed(nn.Module):
    def __init__(self, dim = 3, dim1 = 128, norm = True, bias = False):
        super(embed, self).__init__()

        if norm:
            self.cnn = nn.Sequential(
                norm_data(dim),
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                cnn1x1(dim, 64, bias=bias),
                nn.ReLU(),
                cnn1x1(64, dim1, bias=bias),
                nn.ReLU(),
            )

    def forward(self, x):
        x = self.cnn(x)
        return x

class cnn1x1(nn.Module):
    def __init__(self, dim1 = 3, dim2 =3, bias = True):
        super(cnn1x1, self).__init__()
        self.cnn = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.cnn(x)
        return x

class local(nn.Module):
    def __init__(self, dim1 = 3, dim2 = 3, bias = False):
        super(local, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 20))
        self.cnn1 = nn.Conv2d(dim1, dim1, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU()
        self.cnn2 = nn.Conv2d(dim1, dim2, kernel_size=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout = nn.Dropout2d(0.2)
        self.cnn3 = nn.Conv2d(dim2, dim2, kernel_size=(1,3), padding=(0, 1), bias=bias)
        self.bn3 = nn.BatchNorm2d(dim2)

    def forward(self, x1):
        x1 = self.maxpool(x1)
        x = self.cnn1(x1)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        return x

class compute_g_spa(nn.Module):
    def __init__(self, dim1 = 64 *3, dim2 = 64*3, bias = False):
        super(compute_g_spa, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.g1 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.g2 = cnn1x1(self.dim1, self.dim2, bias=bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1):
        N,C,V,T = x1.size()
        g1 = self.g1(x1).permute(0, 3, 2, 1).contiguous().view(N,V,C*2*T)
        g2 = self.g2(x1).permute(0, 3, 1, 2).contiguous().view(N,C*2*T,V)
        g3 = g1.matmul(g2)
        g = self.softmax(g3)
        return g

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, coff_embedding=4, num_subset=3):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.num_subset = num_subset

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def forward(self, x, A):
        # 消融实验 没有B
        PA = nn.Parameter(A)
        nn.init.constant_(PA, 1e-6)
        
        N, C, V, T = x.size()

        A = Variable(A, requires_grad=False)
        A = A + PA

        y = None
        for i in range(self.num_subset):
            A1 = A
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, V, T))
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)