from .submodule import *
import math


class Constancy_feature_extraction(nn.Module):
    def __init__(self):
        super(Constancy_feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),  # conv0_1/conv0_2/conv0_3
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))
        self.upforconstanncy1 = nn.Sequential(nn.ConvTranspose2d(32, 16, 4, 2, 1),
                                              nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 3, 1, 1),
                                              nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                              nn.Conv2d(16, 16, 1, 1, 0), nn.BatchNorm2d(16))
        self.upforconstanncy2 = nn.Sequential(nn.ConvTranspose2d(48, 32, 4, 2, 1),
                                              nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, 3, 1, 1),
                                              nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                              nn.Conv2d(32, 32, 1, 1, 0), nn.BatchNorm2d(32))
        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)  # conv1_x
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)  # conv2_x
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)  # conv3_x, 但论文中dila=2，这里是1
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)  # conv4_x，但论文中dila=4，这里是2

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))  # 4个分支，得到不同size的输出

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      # concate之后通道数是320，是layer2和layer4的输出以及各个branch连接后的feature map
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1,
                                                bias=False))  # 最后的feature维度是32

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        rem = output
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)
        temp = self.upforconstanncy1(output_feature)
        # print("temp.shape: ", temp.shape)
        temp = torch.cat((rem, temp), 1)
        constancy_feature = self.upforconstanncy2(temp)
        return output_feature, constancy_feature


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes * 2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))  # 3Dstack1_1

        self.conv2 = convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=2, pad=1),  # 3Dstack1_2
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes * 2, inplanes * 2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes * 2))  # 3Dstack1_3 ,上采样，还需要加上3Dstack1_1（conv1）的结果

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(inplanes * 2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(inplanes))  # 3Dstack1_4 ,上采样，并且恢复通道数为32，还需要加上3Dconv1的结果

    def forward(self, x, presqu, postsqu):

        out = self.conv1(x)  # in:1/4 out:1/8
        pre = self.conv2(out)  # in:1/8 out:1/8
        if postsqu is not None:
            pre = F.relu(pre + postsqu, inplace=True)
        else:
            pre = F.relu(pre, inplace=True)  # pre是3Dstack1_1的结果

        out = self.conv3(pre)  # in:1/8 out:1/16
        out = self.conv4(out)  # in:1/16 out:1/16

        if presqu is not None:
            post = F.relu(self.conv5(out) + presqu, inplace=True)  # in:1/16 out:1/8
        else:
            # print(" conv5_out: ", self.conv5(out).shape, " pre: ", pre.shape)
            post = F.relu(self.conv5(out) + pre, inplace=True)  # 否则加上3Dstack1_1的结果，post 是3Dstack1_3的结果

        out = self.conv6(post)  # in:1/8 out:1/4

        return out, pre, post


class Feature_Constancy(nn.Module):
    def __init__(self, maxdisp):
        super(Feature_Constancy, self).__init__()

        self.maxdisp = int(maxdisp)
        self.feature_extraction = Constancy_feature_extraction()

        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),  # 3Dconv0
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),  # 3Dconv1
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):
        refimg_fea, constancy_left = self.feature_extraction(left)
        targetimg_fea, constancy_right = self.feature_extraction(right)

        # matching 维度关系，batch/feature/disp/height/width,这里的视差范围是原来的1/4
        cost = torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, int(self.maxdisp / 4),
                                 refimg_fea.size()[2], refimg_fea.size()[3]).zero_().cuda()

        for i in range(int(self.maxdisp / 4)):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]  # 后一半是参考图像特征，并且，只取有效部分
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]  # 前一半是目标图像的特征，也只取对应有效部分
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea

        cost = cost.contiguous()

        cost = self.dres0(cost)

        cost = self.dres1(cost) + cost  # 论文中这里没有add

        out1, pre1, post1 = self.dres2(cost, None, None)
        out1 = out1 + cost

        out2, pre2, post2 = self.dres3(out1, pre1, post1)
        out2 = out2 + cost

        out3, pre3, post3 = self.dres4(out2, pre1, post2)
        out3 = out3 + cost

        cost1 = self.classif1(out1)  # output1
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)

        cost3 = F.upsample(cost3, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        # For your information: This formulation 'softmax(c)' learned "similarity"
        # while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
        # However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)

        if self.training:
            return pred1, pred2, pred3, constancy_left, constancy_right
        # 返回的disp  是batch*height*width
        else:
            return pred3
