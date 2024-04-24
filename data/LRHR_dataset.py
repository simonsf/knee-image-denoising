from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import numpy as np


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        # 数据格式为“img”的话：执行下列操作
        elif datatype == 'img':
            # get_paths_from_images 获取图片路径，返回的是各个数据集内包含所有文件路径的列表
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'npy':
            # 返回目录下所有图像文件的路径列表
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            # 获取数据集的长度
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        elif self.datatype == 'img':
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        # npy格式
        else:
            # 载入npy格式数据成numpy，shape=1024*1024
            img_HR = np.load(self.hr_path[index])
            # 扩增一个维度，shape=1*1024*1024
            img_HR = np.expand_dims(img_HR, axis=0)
            img_SR = np.load(self.sr_path[index])
            img_SR = np.expand_dims(img_SR, axis=0)
            if self.need_LR:
                img_LR = np.load(self.lr_path[index])
                img_LR = np.expand_dims(img_LR, axis=0)
        if self.need_LR:
            if self.datatype != 'npy':
                [img_LR, img_SR, img_HR] = Util.transform_augment(
                    [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
                return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
            else:
                [img_LR, img_SR, img_HR] = Util.transform_augment_npy(
                    [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
                return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            if self.datatype != 'npy':
                [img_SR, img_HR] = Util.transform_augment(
                    [img_SR, img_HR], split=self.split, min_max=(-1, 1))
                return {'HR': img_HR, 'SR': img_SR, 'Index': index}
            else:
                [img_SR, img_HR] = Util.transform_augment_npy(
                    [img_SR, img_HR], split=self.split, min_max=(-1, 1))
                return {'HR': img_HR, 'SR': img_SR, 'Index': index}