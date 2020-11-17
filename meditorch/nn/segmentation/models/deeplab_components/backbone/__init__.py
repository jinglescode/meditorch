from . import resnet, xception, drn, mobilenet

def build_backbone(in_ch, backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(in_ch, output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(in_ch, output_stride, BatchNorm)
    # elif backbone == 'drn':
    #     return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(in_ch, output_stride, BatchNorm)
    else:
        raise NotImplementedError
