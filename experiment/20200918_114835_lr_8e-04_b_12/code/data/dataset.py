import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos



def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneValDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneValDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, np.array(label), img_name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, simu_transform=None, griding_num=256, load_name=False, row_anchor=None, use_aux=False, segment_transform=None):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()

        img_name, label_name = l_info[0], l_info[1]
        lane_exist_label = np.array([int(x) for x in l_info[2:]])

        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)

        lane_pts = self._get_index(label)

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        
        cls_label["lane_exist_label"] = lane_exist_label

        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        # to_pts = np.zeros((n, num_lane))
        to_pts = {}
        for i in range(num_lane):
            pti = pts[i, :, 1]
            # to_pts[:, i] = np.asarray(
            #     [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
            vertex_wise_confidence = []
            row_wise_vertex_location = []
            for pt in pti:
                if pt != -1:
                    vertex_wise_confidence.append(1)
                    row_wise_vertex_location.append(int(pt // (col_sample[1] - col_sample[0])))
                else:
                    vertex_wise_confidence.append(0)
                    row_wise_vertex_location.append(num_cols)
            to_pts["vertex_wise_confidence_label_"+str(i+1)] = np.array(vertex_wise_confidence)
            to_pts["row_wise_vertex_location_label_"+str(i+1)] = np.array(row_wise_vertex_location)
        return to_pts

    def _get_index(self, label):
        w, h = label.size

        if h != 256:
            scale_f = lambda x: int((x * 1.0 / 256) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))

        all_idx = np.zeros((4, len(sample_tmp), 2))
        for i, r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, 5):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        all_idx_cp = all_idx.copy()
        for i in range(4):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue

            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue
            if len(valid_idx) < 6:
                continue

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp




if __name__ == '__main__':

    list_path = r"./dataset/list/val_gt.txt"
    path = r"./dataset/"

    row_anchor = np.linspace(90, 255, 128).tolist()

    # label_path = r"./dataset/laneseg_label_w16/driver_23_30frame/05171117_0771.MP4/00020.png"
    #
    # img = loader_func(label_path)
    #
    # lane_pts = _get_index(img, row_anchor)
    #
    # w, h = img.size
    #
    # cls_label = _grid_pts(lane_pts, num_cols=256, w=w)
    #
    #


    dataset = LaneClsDataset(path=path, list_path=list_path, row_anchor=row_anchor)


    img, _, label =  dataset.__getitem__(0)

    print(label)
    exit(0)

    #    label = label.tolist()
    col_sample = np.linspace(0, 1640 - 1, 256)
    col_sample_w = col_sample[1] - col_sample[0]

    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # img = cv2.resize(img, (512, 256))

    for i in range(label.shape[1]):
        for k in range(label.shape[0]):
            if label[k, i] != 256:
                # print(label[k, i])
                x = int(label[k, i] * col_sample_w)
                y = int(row_anchor[k] / 256 * 590)
                # x = label[k, i]
                # y = int(row_anchor[k])
                print(x, y)
                cv2.circle(img, (x, y), 2, (255, 0, 0), -1)
    cv2.imwrite("a.jpg", img)



    # pdb.set_trace()










