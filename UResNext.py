from fastai.vision.all import *
import torchvision
class FPN(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch * 2),
                           nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i), mode='bilinear')
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class UnetBlock(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False,
                 self_attention: bool = False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, **kwargs)
        # self.shuf = nn.ConvTranspose2d(in_channels=up_in_c, out_channels=up_in_c//2, kernel_size=3, stride=2, padding=1, output_padding=1,
        #                            bias=True)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
                               xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
                     [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d, groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                                         nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False),
                                      nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c * (2 + len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

class UneXt50(nn.Module):
    def __init__(self, stride=1, input_layers=8,output_layers=4,**kwargs):
        super().__init__()
        # encoder
        m = torchvision.models.resnext50_32x4d(pretrained=True, progress=True, **kwargs)
        self.se = SELayer(input_layers, 1)
        self.conv1 = nn.Conv2d(input_layers, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.enc0 = nn.Sequential(self.se,self.conv1, self.bn1,nn.ReLU(inplace=True),self.conv2, m.bn1, nn.ReLU(inplace=True))
        # self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                                  m.layer1)  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        # aspp with customized dilatations
        self.aspp = ASPP(2048, 256, out_c=512, dilations=[stride * 1, stride * 2, stride * 3, stride * 4])
        self.drop_aspp = nn.Dropout2d(0.5)
        # decoder
        self.dec4 = UnetBlock(512, 1024, 256)
        self.dec3 = UnetBlock(256, 512, 128)
        self.dec2 = UnetBlock(128, 256, 64)
        self.dec1 = UnetBlock(64, 64, 32)
        self.fpn = FPN([512, 256, 128, 64], [16] * 4)
        self.drop = nn.Dropout2d(0.1)
        self.final_conv = nn.Conv2d(32 + 16 * 4, output_layers, kernel_size=1, stride=1, padding=0) #ConvLayer(32 + 16 * 4, 1, ks=1, norm_type=None, act_cls=None)
        # self.feature_conv1 = nn.Conv2d(32 + 16 * 4, 20, kernel_size=4, stride=4, padding=0)
        # self.feature_conv2 = nn.Conv2d(20, 2, kernel_size=4, stride=4, padding=0)
        # self.feature_conv1_1 = nn.Conv2d(32 + 16 * 4, 32, kernel_size=1, stride=1, padding=0)
        # self.feature_conv3 = nn.Linear(in_features=64,out_features=64)
        # self.feature_conv3_1 = nn.Linear(in_features=128, out_features=128)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)
        dec3 = self.dec4(self.drop_aspp(enc5), enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        feature = self.fpn([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(feature))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        # feature_out=self.feature_conv2(self.drop(self.feature_conv1(feature)))
        # feature_out=feature_out.flatten(2)
        # feature_out=self.feature_conv3(feature_out)
        # feature_out=feature[:,:,84:86,44:46]
        # feature_out=self.feature_conv1_1(feature_out)
        # feature_out = feature_out.flatten(1)
        # feature_out = self.feature_conv3_1(feature_out)
        # feature_out=feature_out.reshape(feature_out.shape[0], 2, feature_out.shape[1] // 2)
        return x , feature


class CatHead(nn.Module):
    def __init__(self, station_x=113.5, station_y=12.5,size=2):
        super().__init__()
        stride=(size-1)/2
        self.x1 = round(station_x - stride)
        self.x2 = round(station_x + stride)
        self.y1 = round(station_y - stride)
        self.y2 = round(station_y + stride)
        self.pooling=nn.AvgPool2d(size//2,size//2)
        self.feature_conv1_1 = nn.Conv2d(32 + 16 * 4, 32, kernel_size=1, stride=1, padding=0)
        self.feature_conv3_1 = nn.Linear(in_features=128, out_features=128)
    def forward(self, feature):
        feature_out = feature[:, :, self.x1:self.x2+1, self.y1:self.y2+1]
        feature_out = self.pooling(feature_out)#pooling
        feature_out = self.feature_conv1_1(feature_out)
        feature_out = feature_out.flatten(1)
        feature_out = self.feature_conv3_1(feature_out)
        feature_out = feature_out.reshape(feature_out.shape[0], 2, feature_out.shape[1] // 2)
        return feature_out



split_layers = lambda m: [list(m.enc0.parameters()) + list(m.enc1.parameters()) +
                          list(m.enc2.parameters()) + list(m.enc3.parameters()) +
                          list(m.enc4.parameters())+ list(m.aspp.parameters()) + list(m.dec4.parameters()) +
                          list(m.dec3.parameters()) + list(m.dec2.parameters()) +
                          list(m.dec1.parameters()) + list(m.fpn.parameters()) +
                          list(m.final_conv.parameters())]