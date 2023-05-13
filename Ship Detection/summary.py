#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.DFRnet import DFR
from utils.utils import net_flops

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 1
    
    model, _ = DFR([input_shape[0], input_shape[1], 3], num_classes, backbone='HFF')
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)

