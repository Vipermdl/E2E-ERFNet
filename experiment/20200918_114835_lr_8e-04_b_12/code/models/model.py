import torch, pdb
import torch.nn as nn
import torch.nn.functional as F

from models.ERFNet import ERFNet
from models.HRM import HRM_Block



class E2ENet(nn.Module):
    def __init__(self, Channels = 96, nums_lane=4, culomn_channels = 256, row_channels = 128, initialed = True):
        super(E2ENet, self).__init__()
        self.backbone = ERFNet(channels=Channels)
        self.share_hrm1 = HRM_Block(in_planes=Channels, stride=2, kernel_size=3)
        self.share_hrm2 = HRM_Block(in_planes=Channels, stride=2, kernel_size=3)
        self.share_hrm3 = HRM_Block(in_planes=Channels, stride=2, kernel_size=3)

        self.lane_maker_confidence = nn.Sequential(
            nn.Linear(Channels, Channels//2),
            nn.BatchNorm1d(Channels//2),
            nn.ReLU(inplace=True),
            nn.Linear(Channels//2, nums_lane)
        )

        self.lane_marker_block1 = nn.Sequential(
                        # maker_wise_hrms
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=1),
                        # vertex_wise_confidence_branch
                        nn.Conv2d(Channels, Channels // 2, kernel_size=1),
                        nn.BatchNorm2d(Channels // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(Channels // 2, 1, kernel_size=1),
                        # row_wise_vertex_location_branch
                        nn.Conv2d(Channels, row_channels, kernel_size=1),
                        nn.BatchNorm2d(row_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(row_channels, culomn_channels, kernel_size=1)
                        )
        self.lane_marker_block2 = nn.Sequential(
                        # maker_wise_hrms
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=1),
                        # vertex_wise_confidence_branch
                        nn.Conv2d(Channels, Channels // 2, kernel_size=1),
                        nn.BatchNorm2d(Channels // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(Channels // 2, 1, kernel_size=1),
                        # row_wise_vertex_location_branch
                        nn.Conv2d(Channels, row_channels, kernel_size=1),
                        nn.BatchNorm2d(row_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(row_channels, culomn_channels, kernel_size=1)
                        )
        self.lane_marker_block3 = nn.Sequential(
                        # maker_wise_hrms
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=1),
                        # vertex_wise_confidence_branch
                        nn.Conv2d(Channels, Channels // 2, kernel_size=1),
                        nn.BatchNorm2d(Channels // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(Channels // 2, 1, kernel_size=1),
                        # row_wise_vertex_location_branch
                        nn.Conv2d(Channels, row_channels, kernel_size=1),
                        nn.BatchNorm2d(row_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(row_channels, culomn_channels, kernel_size=1)
                        )
        self.lane_marker_block4 = nn.Sequential(
                        # maker_wise_hrms
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=3),
                        HRM_Block(in_planes=Channels, stride=2, kernel_size=1),
                        # vertex_wise_confidence_branch
                        nn.Conv2d(Channels, Channels // 2, kernel_size=1),
                        nn.BatchNorm2d(Channels // 2),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(Channels // 2, 1, kernel_size=1),
                        # row_wise_vertex_location_branch
                        nn.Conv2d(Channels, row_channels, kernel_size=1),
                        nn.BatchNorm2d(row_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(row_channels, culomn_channels, kernel_size=1)
                        )

        if initialed:
            self.initialize_weights()
            print("====Initialize Neural Network Successfully!====")



    def forward(self, x):
        x = self.backbone(x)
        x = self.share_hrm1(x)
        x = self.share_hrm2(x)
        x = self.share_hrm3(x)

        # Lane marker confidence branch
        lane_marker_confidence_x = nn.AdaptiveAvgPool2d((1, 1))(x)
        lane_marker_confidence_x = lane_marker_confidence_x.view(x.size(0), -1)
        lane_marker_confidence_x = self.lane_maker_confidence(lane_marker_confidence_x)

        # Lane marker branch
        branch_x = self.lane_marker_block1[:5](x)
        vertex_wise_confidence_x_1 = self.lane_marker_block1[5:9](branch_x).squeeze()
        row_wise_vertex_location_x_1 = self.lane_marker_block1[9:](branch_x).squeeze()
        
        branch_x = self.lane_marker_block2[:5](x)
        vertex_wise_confidence_x_2 = self.lane_marker_block2[5:9](branch_x).squeeze()
        row_wise_vertex_location_x_2 = self.lane_marker_block2[9:](branch_x).squeeze()
        
        branch_x = self.lane_marker_block3[:5](x)
        vertex_wise_confidence_x_3 = self.lane_marker_block3[5:9](branch_x).squeeze()
        row_wise_vertex_location_x_3 = self.lane_marker_block3[9:](branch_x).squeeze()
        
        branch_x = self.lane_marker_block4[:5](x)
        vertex_wise_confidence_x_4 = self.lane_marker_block4[5:9](branch_x).squeeze()
        row_wise_vertex_location_x_4 = self.lane_marker_block4[9:](branch_x).squeeze()
        
        
        
        return {"lane_exit_out": lane_marker_confidence_x, 
                "vertex_wise_confidence_out_1": vertex_wise_confidence_x_1, 
                "row_wise_vertex_location_out_1": row_wise_vertex_location_x_1, 
                "vertex_wise_confidence_out_2": vertex_wise_confidence_x_2, 
                "row_wise_vertex_location_out_2": row_wise_vertex_location_x_2,
                "vertex_wise_confidence_out_3": vertex_wise_confidence_x_3, 
                "row_wise_vertex_location_out_3": row_wise_vertex_location_x_3,
                "vertex_wise_confidence_out_4": vertex_wise_confidence_x_4, 
                "row_wise_vertex_location_out_4": row_wise_vertex_location_x_4}

    def initialize_weights(self):
        for model in self.modules():
            self.real_init_weights(model)

    def real_init_weights(self, m):
        if isinstance(m, list):
            for mini_m in m:
                self.real_init_weights(mini_m)
        else:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0.0, std=0.01)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Module):
                for mini_m in m.children():
                    self.real_init_weights(mini_m)
            else:
                print('unkonwn module', m)


if __name__ == '__main__':
    x = torch.randn(size=(2, 3, 256, 512))
    model = E2ENet(Channels=96)
    result = model(x)

    # pdb.set_trace()

