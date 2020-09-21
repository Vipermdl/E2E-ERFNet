import torch, pdb
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=256, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

class Multi_Loss(nn.Module):
    def __init__(self, lambda_1=10, lambda_2=1):
        super(Multi_Loss, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.vl_loss = SoftmaxFocalLoss(gamma=2, ignore_1b=256)
    
    def forward(self, preds, labels):
        
        lane_exit_label = labels["lane_exist_label"].float()
        lane_exit_out = preds["lane_exit_out"]
        
        vertex_wise_confidence_label_1 = labels["vertex_wise_confidence_label_1"].float()
        vertex_wise_confidence_label_2 = labels["vertex_wise_confidence_label_2"].float()
        vertex_wise_confidence_label_3 = labels["vertex_wise_confidence_label_3"].float()
        vertex_wise_confidence_label_4 = labels["vertex_wise_confidence_label_4"].float()
        
        vertex_wise_confidence_out_1 = preds["vertex_wise_confidence_out_1"]
        vertex_wise_confidence_out_2 = preds["vertex_wise_confidence_out_2"]
        vertex_wise_confidence_out_3 = preds["vertex_wise_confidence_out_3"]
        vertex_wise_confidence_out_4 = preds["vertex_wise_confidence_out_4"]
        
        row_wise_vertex_location_label_1 = labels["row_wise_vertex_location_label_1"].long()
        row_wise_vertex_location_label_2 = labels["row_wise_vertex_location_label_2"].long()
        row_wise_vertex_location_label_3 = labels["row_wise_vertex_location_label_3"].long()
        row_wise_vertex_location_label_4 = labels["row_wise_vertex_location_label_4"].long()
        
        row_wise_vertex_location_out_1 = preds["row_wise_vertex_location_out_1"]
        row_wise_vertex_location_out_2 = preds["row_wise_vertex_location_out_2"]
        row_wise_vertex_location_out_3 = preds["row_wise_vertex_location_out_3"]
        row_wise_vertex_location_out_4 = preds["row_wise_vertex_location_out_4"]
        
        loss_lc = self.bce_loss(lane_exit_out, lane_exit_label.cuda())
        
        loss_vc1 = self.bce_loss(vertex_wise_confidence_out_1, vertex_wise_confidence_label_1.cuda())
        loss_vc2 = self.bce_loss(vertex_wise_confidence_out_2, vertex_wise_confidence_label_2.cuda())
        loss_vc3 = self.bce_loss(vertex_wise_confidence_out_3, vertex_wise_confidence_label_3.cuda())
        loss_vc4 = self.bce_loss(vertex_wise_confidence_out_4, vertex_wise_confidence_label_4.cuda())
        
        loss_vl1 = self.vl_loss(row_wise_vertex_location_out_1, row_wise_vertex_location_label_1.cuda())
        loss_vl2 = self.vl_loss(row_wise_vertex_location_out_2, row_wise_vertex_location_label_2.cuda())
        loss_vl3 = self.vl_loss(row_wise_vertex_location_out_3, row_wise_vertex_location_label_3.cuda())
        loss_vl4 = self.vl_loss(row_wise_vertex_location_out_4, row_wise_vertex_location_label_4.cuda())
        
        loss_vl = (loss_vl1 + loss_vl2 + loss_vl3 + loss_vl4) / 4
        loss_vc = (loss_vc1 + loss_vc2 + loss_vc3 + loss_vc4) / 4
        
        return loss_vl + self.lambda_1 * loss_vc + self.lambda_2 * loss_lc
        
        
        
        
        