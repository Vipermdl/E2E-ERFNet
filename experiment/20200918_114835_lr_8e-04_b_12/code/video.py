import torch, os, cv2, pdb
from models.model import E2ENet
from utils.common import merge_config
from utils.dist_utils import dist_print
import scipy.special, tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')

    net = E2ENet(Channels = 96, nums_lane=4, culomn_channels = cfg.griding_num, row_channels = cfg.row_num, initialed = True).cuda()
    
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    row_anchor = np.linspace(90, 255, 128).tolist()
    col_sample = np.linspace(0, 1280 - 1, 256)
    col_sample_w = col_sample[1] - col_sample[0]
    
    filter_f = lambda x: int(np.round(x))
    
    main_file = cv2.VideoCapture("./test20200814_161254.mp4")
    main_ret, main_frame = main_file.read()
    
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join('./result.avi'), fourcc , 30.0, (1280,720))
    
    while main_ret:
        img = img_transforms(Image.fromarray(main_frame))
        imgs = img.unsqueeze(0).cuda()

        with torch.no_grad():
            out = net(imgs)
        
        lane_exit_out = out["lane_exit_out"].sigmoid()
        lane_exit_out = lane_exit_out > cfg.thresh_lc
        
        
        for lane_index in range(lane_exit_out.size(1)):
            if lane_exit_out[0][lane_index] == True:
                x_list = []
                y_list = []
                vertex_wise_confidence_out = out["vertex_wise_confidence_out_"+str(lane_index+1)].sigmoid()
                vertex_wise_confidence_out = vertex_wise_confidence_out > cfg.thresh_vc
                
                row_wise_vertex_location_out = F.log_softmax(out["row_wise_vertex_location_out_"+str(lane_index+1)], dim=0)
                row_wise_vertex_location_out = torch.argmax(row_wise_vertex_location_out, dim=0)
                row_wise_vertex_location_out[~vertex_wise_confidence_out]=256
                
                row_wise_vertex_location_out = row_wise_vertex_location_out.detach().cpu().numpy()
                
                estimator = RANSACRegressor(random_state=42, min_samples=2, residual_threshold=5.0)
                model = make_pipeline(PolynomialFeatures(2), estimator)
                
                for k in range(row_wise_vertex_location_out.shape[0]):
                    if row_wise_vertex_location_out[k] != 256:
                        x = int(row_wise_vertex_location_out[k] * col_sample_w)
                        y = int(row_anchor[k] / 256 * 720)
                        x_list.append(x)
                        y_list.append(y)
                        cv2.circle(main_frame, (x, y), 2, (255, 0, 0), -1)
                if len(x_list) <= 1:
                    continue
                X = np.array(x_list)
                y = np.array(y_list)
                X = X[:, np.newaxis]
                x_plot = np.linspace(X.min(), X.max())
                model.fit(X, y)
                y_plot = model.predict(x_plot[:, np.newaxis])
                
                for x, y in zip(x_plot, y_plot):
                    cv2.circle(main_frame, (filter_f(x), filter_f(y)), 2, (255, 0, 0), -1)
                    
        vout.write(main_frame)
        main_ret, main_frame = main_file.read()
            
    main_file.release()
    vout.release()
    exit()