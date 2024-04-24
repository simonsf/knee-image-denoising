import os
import csv
import numpy as np
from torch.utils.data import Dataset
from md.utils.python.file_tools import readlines
from md.image3d.python.image3d import Image3d
import md.image3d.python.image3d_io as cio
import md.image3d.python.image3d_tools as ctools
from md.mdpytorch.utils.tensor_tools import ToTensor
import md.image3d.python.image3d_tools as imtools
from data.vsr_helpers import random_crop_with_data_augmentation, axes_wo_angle




def read_train_txt(imlist_file):
    """ read single-modality txt file

    In txt annotation mode, sampling and ground truth segmentation lists are the same.
    If these two need to be different, please use csv annotation mode.

    :param imlist_file: image list file path
    :return: a list of image path list, list of segmentation paths, list of sampling segmentation paths
    """
    lines = readlines(imlist_file)
    num_cases = int(lines[0])

    if len(lines)-1 < num_cases * 2:
        raise ValueError('too few lines in imlist file')

    lr_list, hr_list, sampling_list = [], [], []
    for i in range(num_cases):
        lr_path, hr_path = lines[1 + i * 2].strip(), lines[2 + i * 2].strip()
        assert os.path.exists(lr_path), 'image not exist: {}'.format(lr_path)
        assert os.path.exists(hr_path), 'mask not exist: {}'.format(hr_path)
        lr_list.append([lr_path])
        hr_list.append(hr_path)
        sampling_list.append(lr_path)

    return lr_list, hr_list, sampling_list


def read_train_csv(csv_file):
    """ read multi-modality csv file

    :param csv_file: csv file path
    :return: a list of image path list, list of segmentation paths, list of sampling segmentation paths
    """
    lr_list, hr_list, sampling_list = [], [], []
    with open(csv_file, 'r') as fp:
        reader = csv.reader(fp)
        headers = next(reader)

        assert headers[-1] == 'gt_path'
        assert headers[0] == 'image_path'

        for line in reader:
            for path in line:
                assert os.path.exists(path) or path == '', 'file not exist: {}'.format(path)

            lr_list.append(line[0].replace('/data_v2/', '/data_v2_2d_norm_reg/'))
            hr_list.append(line[1].replace('/data_v2/', '/data_v2_2d_norm_reg/'))
            sampling_list.append(line[0].replace('/data_v2/', '/data_v2_2d_norm_reg/'))

    return lr_list, hr_list, sampling_list


class SuperResolutionDataset(Dataset):
    """ training data set for volumetric super resolution """

    def __init__(self, imlist_file, spacing, crop_size, pad_t, default_values, sampling_method, interpolation, crop_normalizers,
                 aug_prob, shift_config, rot_config, flip_config, scale_config, cover_config, truncate_config,
                 brightness_config, gamma_contrast_config, gaussian_noise_config, gaussian_blur_config):
        """ constructor
        :param imlist_file: image-segmentation list file
        :param spacing:                the resolution, e.g., [1, 1, 1]
        :param crop_size:              crop size, e.g., [96, 96, 96]
        :param pad_t:                  re-sample padding type, 0 for zero-padding, 1 for edge-padding
        :param default_values:         default padding value list, e.g.,[0]
        :param sampling_method:        'GLOBAL', 'MASK'
        :param interpolation:          'LINEAR' for linear interpolation, 'NN' for nearest neighbor
        :param crop_normalizers:       used to normalize the image crops, one for one image modality
        :param aug_prob:               the probability to apply data augmentation
        :param shift_config:           the config of shift augmentation
        :param rot_config:             the config of rotate augmentation
        :param flip_config:            the config of flip augmentation
        :param scale_config:           the config of scale augmentation
        :param cover_config:           the config of cover augmentation
        :param truncate_config:        the config of truncate augmentation
        :param brightness_config:      the config of brightness augmentation
        :param gamma_contrast_config:  the config of gamma contrast augmentation
        :param gaussian_noise_config:  the config of gaussian noise augmentation
        :param gaussian_blur_config:   the config of gaussian blur augmentation
        """
        if imlist_file.endswith('txt'):
            self.lr_list, self.hr_list, self.sampling_list = read_train_txt(imlist_file)
        elif imlist_file.endswith('csv'):
            self.lr_list, self.hr_list, self.sampling_list = read_train_csv(imlist_file)
        else:
            raise ValueError('imseg_list must either be a txt file or a csv file')

        self.pad_t = pad_t
        self.default_value = default_values

        self.spacing = np.array(spacing, dtype=np.double)
        assert self.spacing.size == 2, 'only 2-element of spacing is supported'

        self.crop_size = np.array(crop_size, dtype=np.int32)
        assert self.crop_size.size == 2, 'only 2-element of crop size is supported'

        assert sampling_method in ('GLOBAL', 'MASK', 'MIXED', 'BOX'), 'sampling_method must either be GLOBAL, ' \
                                                                      'MASK , BOX or MIXED'
        self.sampling_method = sampling_method

        assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
        self.interpolation = interpolation

        self.crop_normalizers = crop_normalizers

        self.aug_prob = aug_prob
        self.shift_config = shift_config
        self.rot_config = rot_config
        self.flip_config = flip_config
        self.scale_config = scale_config
        self.cover_config = cover_config
        self.truncate_config = truncate_config
        self.brightness_config = brightness_config
        self.gamma_contrast_config = gamma_contrast_config
        self.gaussian_noise_config = gaussian_noise_config
        self.gaussian_blur_config = gaussian_blur_config

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.lr_list)

    def global_sample(self, image):
        """ random sample a position in the image
        :param image: a image3d object
        :return: a position in world coordinate
        """
        assert isinstance(image, Image3d)
        min_box, im_size_mm = image.world_box_full()
        crop_size_mm = self.crop_size * self.spacing

        sp = np.array(min_box, dtype=np.double)
        for i in range(3):
            if im_size_mm[i] > crop_size_mm[i]:
                sp[i] = min_box[i] + np.random.uniform(0, im_size_mm[i] - crop_size_mm[i])
        center = sp + crop_size_mm / 2
        return center

    def get_foreground(self, sampling):
        assert isinstance(sampling, Image3d)
        tmp = sampling.to_numpy()
        tmp[tmp > 100] = 1
        sampling.from_numpy(tmp)
        return sampling

    def mask_sample(self, sampling):
        seg = self.get_foreground(sampling)
        labels = imtools.count_labels(seg)
        if len(labels) == 0:
            # if no segmentation
            center = self.global_sample(seg)
        else:
            # if segmentation exists
            center = ctools.random_voxels_multi(seg, 1, labels.tolist())
            if len(center) > 0:
                center = seg.voxel_to_world(center[0])
            else:
                center = self.global_sample(seg)
        return center

    def __getitem__(self, index):
        """ get a training sample - image(s) and segmentation pair
        :param index:  the sample index
        :return cropped image, cropped mask, crop frame, case name
        """
        lr_path, hr_path, sampling_path = self.lr_list[index], self.hr_list[index], self.sampling_list[index]

        case_name = os.path.basename(os.path.dirname(lr_path))
        case_name += '_' + os.path.basename(lr_path)

        hr = cio.read_image(hr_path, dtype=np.float32)
        lr = cio.read_image(lr_path, dtype=np.float32)
        self.spacing = np.array([self.spacing[0], self.spacing[1], lr.spacing()[2]], dtype=np.double)
        self.crop_size = np.array([self.crop_size[0], self.crop_size[1], 1], dtype=np.int32)

        axes = axes_wo_angle(lr)
        lr.set_axes(axes)
        hr.set_axes(axes)

        if lr_path == sampling_path:
            sampling = lr.deep_copy()
        else:
            sampling = cio.read_image(sampling_path, dtype=np.float32)
            sampling.set_axes(axes)

        # sampling a crop center
        if self.sampling_method == 'GLOBAL':
            center = self.global_sample(sampling)
        elif self.sampling_method == 'MASK':
            center = self.mask_sample(sampling)
        else:
            raise ValueError('Only GLOBAL, MASK are supported as sampling_method')
        
        # sample crop from image with data augmentation
        voxel_center = lr.world_to_voxel(center)
        voxel_center = np.array([voxel_center[0], voxel_center[1], 0])
        center = lr.voxel_to_world(voxel_center)
        lr_img, hr_img = random_crop_with_data_augmentation(
            lr_path, center, self.spacing, self.crop_size, hr, self.aug_prob, self.shift_config,
            self.rot_config, self.flip_config, self.scale_config, self.cover_config, self.truncate_config,
            self.brightness_config, self.gamma_contrast_config, self.gaussian_noise_config, self.gaussian_blur_config,
            self.interpolation, pad_type=0, pad_values=self.default_value)

        # normalize crop image
#         for idx in range(len(images)):
#             if self.crop_normalizers[idx] is not None:
#                 self.crop_normalizers[idx](images[idx])
        # lr_img.from_numpy(lr_img.to_numpy()/100)
        # hr_img.from_numpy(hr_img.to_numpy()/100)

#         # set labels of not interest to zero
#         ctools.convert_labels_outside_to_zero(seg, 1, self.num_classes - 1)

        # image frame
        frame = lr_img.frame().to_numpy()

        # convert to tensors
        lr_img = ToTensor()(lr_img).squeeze(0)
        hr_img = ToTensor()(hr_img).squeeze(0)
        return lr_img, hr_img, frame, case_name
