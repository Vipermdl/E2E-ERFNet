from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from data.dataloader import get_test_loader
from evaluation.tusimple.lane import LaneEval
from utils.dist_utils import is_main_process, dist_print, get_rank, get_world_size, dist_tqdm, synchronize
import os, json, torch, scipy
import torch.nn.functional as F
import numpy as np
import platform


def run_test(net, data_root, exp_name, work_dir, distributed, cfg, batch_size=1):
    # torch.backends.cudnn.benchmark = True
    output_path = os.path.join(work_dir, exp_name)
    if not os.path.exists(output_path) and is_main_process():
        os.mkdir(output_path)
    synchronize()

    row_anchor = np.linspace(90, 255, 128).tolist()
    col_sample = np.linspace(0, 1640 - 1, 256)
    col_sample_w = col_sample[1] - col_sample[0]

    loader = get_test_loader(batch_size, data_root, 'CULane', distributed)

    filter_f = lambda x: int(np.round(x))

    # import pdb;pdb.set_trace()
    for i, data in enumerate(dist_tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            out = net(imgs)
            
        for j in range(len(names)):
            name = names[j]
            
            line_save_path = os.path.join(output_path, name[:-3] + 'lines.txt')
            save_dir, _ = os.path.split(line_save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(line_save_path, 'w') as writer:
                lane_exit_out = out["lane_exit_out"].sigmoid()
                lane_exit_out = lane_exit_out > cfg.thresh_lc
    
                for lane_index in range(lane_exit_out.size(1)):
                    if lane_exit_out[0][lane_index] == True:
                        x_list = []
                        y_list = []
                        vertex_wise_confidence_out = out["vertex_wise_confidence_out_" + str(lane_index + 1)].sigmoid()
                        vertex_wise_confidence_out = vertex_wise_confidence_out > cfg.thresh_vc
    
                        row_wise_vertex_location_out = F.log_softmax(
                            out["row_wise_vertex_location_out_" + str(lane_index + 1)], dim=0)
                        row_wise_vertex_location_out = torch.argmax(row_wise_vertex_location_out, dim=0)
                        row_wise_vertex_location_out[~vertex_wise_confidence_out] = 256
    
                        row_wise_vertex_location_out = row_wise_vertex_location_out.detach().cpu().numpy()
    
                        estimator = RANSACRegressor(random_state=42, min_samples=2, residual_threshold=10.0)
                        ##model = make_pipeline(PolynomialFeatures(2), estimator)
    
                        for k in range(row_wise_vertex_location_out.shape[0]):
                            if row_wise_vertex_location_out[k] != 256:
                                x = row_wise_vertex_location_out[k] * col_sample_w
                                y = row_anchor[k] / 256 * 590
                                x_list.append(x)
                                y_list.append(y)
                                #writer.write('%d %d ' % (filter_f(row_wise_vertex_location_out[k] * col_sample_w), filter_f(row_anchor[k] / 256 * 590)))
                        #writer.write('\n')
    
                        if len(x_list) <= 1:
                            continue
                        X = np.array(x_list)
                        y = np.array(y_list)
                        y = y[:, np.newaxis]
                        y_plot = np.linspace(y.min(), y.max())
                        estimator.fit(y, X)
                        x_plot = estimator.predict(y_plot[:, np.newaxis])
                                                
                        for x, y in zip(x_plot, y_plot):
                            writer.write('%d %d ' % (filter_f(x), filter_f(y)))
                        writer.write('\n')
                 
                  
def eval_lane(net, dataset, data_root, work_dir, distributed, cfg):
    net.eval()
    run_test(net, data_root, 'culane_eval_tmp', work_dir, distributed, cfg)
    synchronize()  # wait for all results
    if is_main_process():
        res = call_culane_eval(data_root, 'culane_eval_tmp', work_dir)
        TP, FP, FN = 0, 0, 0
        for k, v in res.items():
            val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
            val_tp, val_fp, val_fn = int(v['tp']), int(v['fp']), int(v['fn'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            dist_print(k, val)
        P = TP * 1.0 / (TP + FP)
        R = TP * 1.0 / (TP + FN)
        F = 2 * P * R / (P + R)
        dist_print(F)
    synchronize()


def read_helper(path):
    lines = open(path, 'r').readlines()[1:]
    lines = ' '.join(lines)
    values = lines.split(' ')[1::2]
    keys = lines.split(' ')[0::2]
    keys = [key[:-1] for key in keys]
    res = {k: v for k, v in zip(keys, values)}
    return res


def call_culane_eval(data_dir, exp_name, output_path):
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'
    detect_dir = os.path.join(output_path, exp_name) + '/'

    w_lane = 30
    iou = 0.5;  # Set iou to 0.3 or 0.5
    im_w = 1640
    im_h = 590
    frame = 1
    list0 = os.path.join(data_dir, 'list/test_split/test0_normal.txt')
    list1 = os.path.join(data_dir, 'list/test_split/test1_crowd.txt')
    list2 = os.path.join(data_dir, 'list/test_split/test2_hlight.txt')
    list3 = os.path.join(data_dir, 'list/test_split/test3_shadow.txt')
    list4 = os.path.join(data_dir, 'list/test_split/test4_noline.txt')
    list5 = os.path.join(data_dir, 'list/test_split/test5_arrow.txt')
    list6 = os.path.join(data_dir, 'list/test_split/test6_curve.txt')
    list7 = os.path.join(data_dir, 'list/test_split/test7_cross.txt')
    list8 = os.path.join(data_dir, 'list/test_split/test8_night.txt')
    if not os.path.exists(os.path.join(output_path, 'txt')):
        os.mkdir(os.path.join(output_path, 'txt'))
    out0 = os.path.join(output_path, 'txt', 'out0_normal.txt')
    out1 = os.path.join(output_path, 'txt', 'out1_crowd.txt')
    out2 = os.path.join(output_path, 'txt', 'out2_hlight.txt')
    out3 = os.path.join(output_path, 'txt', 'out3_shadow.txt')
    out4 = os.path.join(output_path, 'txt', 'out4_noline.txt')
    out5 = os.path.join(output_path, 'txt', 'out5_arrow.txt')
    out6 = os.path.join(output_path, 'txt', 'out6_curve.txt')
    out7 = os.path.join(output_path, 'txt', 'out7_cross.txt')
    out8 = os.path.join(output_path, 'txt', 'out8_night.txt')

    eval_cmd = './evaluation/culane/evaluate'
    if platform.system() == 'Windows':
        eval_cmd = eval_cmd.replace('/', os.sep)

    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list0,w_lane,iou,im_w,im_h,frame,out0))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list0, w_lane, iou, im_w, im_h, frame, out0))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list1,w_lane,iou,im_w,im_h,frame,out1))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list1, w_lane, iou, im_w, im_h, frame, out1))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list2,w_lane,iou,im_w,im_h,frame,out2))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list2, w_lane, iou, im_w, im_h, frame, out2))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list3,w_lane,iou,im_w,im_h,frame,out3))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list3, w_lane, iou, im_w, im_h, frame, out3))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list4,w_lane,iou,im_w,im_h,frame,out4))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list4, w_lane, iou, im_w, im_h, frame, out4))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list5,w_lane,iou,im_w,im_h,frame,out5))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list5, w_lane, iou, im_w, im_h, frame, out5))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list6,w_lane,iou,im_w,im_h,frame,out6))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list6, w_lane, iou, im_w, im_h, frame, out6))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list7,w_lane,iou,im_w,im_h,frame,out7))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list7, w_lane, iou, im_w, im_h, frame, out7))
    # print('./evaluate -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s'%(data_dir,detect_dir,data_dir,list8,w_lane,iou,im_w,im_h,frame,out8))
    os.system('%s -a %s -d %s -i %s -l %s -w %s -t %s -c %s -r %s -f %s -o %s' % (
    eval_cmd, data_dir, detect_dir, data_dir, list8, w_lane, iou, im_w, im_h, frame, out8))
    res_all = {}
    res_all['res_normal'] = read_helper(out0)
    res_all['res_crowd'] = read_helper(out1)
    res_all['res_night'] = read_helper(out8)
    res_all['res_noline'] = read_helper(out4)
    res_all['res_shadow'] = read_helper(out3)
    res_all['res_arrow'] = read_helper(out5)
    res_all['res_hlight'] = read_helper(out2)
    res_all['res_curve'] = read_helper(out6)
    res_all['res_cross'] = read_helper(out7)
    return res_all