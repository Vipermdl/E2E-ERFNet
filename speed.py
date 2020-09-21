import torch, os, cv2, pdb
from models.model import E2ENet
from utils.common import merge_config
from utils.dist_utils import dist_print



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    net = E2ENet(Channels = 96, nums_lane=4, culomn_channels = cfg.griding_num, row_channels = cfg.row_num, initialed = True).cuda()

    x = torch.randn(2, 3, 256, 512).cuda()

    net(x)