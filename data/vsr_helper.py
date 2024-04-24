# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import os
import glob
import csv
import random
import math
import md

import md.image3d.python.image3d_tools as ctools
from md.image3d.python.frame3d import Frame3d
from md.image3d.python.image3d import Image3d
from md.utils.python.file_tools import readlines
from md.mdmath.python.math_tools import uniform_sample_point_from_unit_sphere
from md.mdmath.python.rotation3d import axis_angle_to_rotation_matrix
from imgaug import augmenters as iaa

def axes_wo_angle(img):
    prev_axes = img.frame().axes()
    max_ind = np.argmax(abs(prev_axes), 1)
    max_sign = np.sign(prev_axes[np.array([0, 1, 2]), max_ind])
    axes = np.zeros_like(prev_axes)
    axes[np.array([0, 1, 2]), max_ind] = max_sign
    return axes

def last_checkpoint(chk_root):
    """
    find the directory of last check point
    :param chk_root: the check point root directory, which may contain multiple checkpoints
    :return: the last check point directory
    """

    last_epoch = -1
    chk_folders = os.path.join(chk_root, 'chk_*')
    for folder in glob.glob(chk_folders):
        folder_name = os.path.basename(folder)
        tokens = folder_name.split('_')
        epoch = int(tokens[-1])
        if epoch > last_epoch:
            last_epoch = epoch

    if last_epoch == -1:
        raise OSError('No checkpoint folder found!')

    return os.path.join(chk_root, 'chk_{}'.format(last_epoch))


def get_checkpoint(chk_root, epoch=-1):
    if epoch > 0:
        chk_path = os.path.join(chk_root, 'chk_{}'.format(epoch))
        assert os.path.isdir(chk_path), "checkpoints not exist: " + chk_path
    else:
        chk_path = last_checkpoint(chk_root)

    return chk_path


def read_test_txt(txt_file):
    """ read single-modality txt file
    :param txt_file: image list txt file path
    :return: a list of image path list, list of image case names
    """
    lines = readlines(txt_file)
    case_num = int(lines[0])

    if len(lines) - 1 < case_num:
        raise ValueError('case num cannot be greater than path num!')

    file_list, name_list = [], []
    for i in range(case_num):
        im_msg = lines[1 + i]
        im_msg = im_msg.strip().split()
        im_name = im_msg[0]
        im_path = im_msg[1]
        if not os.path.exists(im_path):
            raise ValueError('image not exist: {}'.format(im_path))
        file_list.append([im_path])
        name_list.append(im_name)

    return file_list, name_list


def read_test_csv(csv_file):
    """ read multi-modality csv file
    :param csv_file: image list csv file path
    :return: a list of image path list, list of image case names
    """
    with open(csv_file, 'r') as fp:
        reader = csv.DictReader(fp)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        if 'image1' in reader.fieldnames:
            file_list, name_list, sampling_list = read_segment_csv(reader)
        elif 'image_path' in reader.fieldnames:
            file_list, name_list, sampling_list = read_detect_csv(reader)
        else:
            raise Exception(f"invalid csv format in {csv_file}")

    return file_list, name_list, sampling_list


def read_segment_csv(reader):
    """ read multi-modality csv file
    csv format: [case_name] image1 image2...imageN sampling
    :param reader: image list csv file reader
    :return: a list of image path list, list of image case names
    """
    file_list, name_list, sampling_list = [], [], []
    fieldnames = reader.fieldnames
    num_modality = 0
    for i in range(1, len(fieldnames)):
        if f'image{i}' in fieldnames:
            num_modality += 1
        else:
            break
    assert num_modality > 0, 'at least one image path in csv file'

    for line in reader:
        image_list = []
        for idx in range(1, num_modality+1):
            im_path = line[f'image{idx}']
            image_list.append(im_path)
            assert os.path.exists(im_path), 'file not exist: {}'.format(im_path)
        file_list.append(image_list)

        case_name = line['case_name'] if 'case_name' in fieldnames else os.path.basename(os.path.dirname(image_list[0]))
        name_list.append(case_name)

        if 'sampling' in fieldnames:
            sampling_path = line['sampling']
            assert os.path.exists(sampling_path), 'sampling path not exist: {}'.format(sampling_path)
            sampling_list.append(sampling_path)

    return file_list, name_list, sampling_list


def read_detect_csv(reader):
    """ read detect csv file
    csv format: [case_name] image_path mask_path
    :param reader: image list csv file reader
    :return: a list of image path list, list of image case names
    """
    file_list, name_list, sampling_list = [], [], []
    fieldnames = reader.fieldnames
    num_modality = 0
    if 'image_path' in fieldnames:
        num_modality += 1
    assert num_modality > 0, 'at least one image path in csv file'

    for line in reader:
        image_list = []
        for idx in range(1, num_modality+1):
            im_path = line['image_path']
            image_list.append(im_path)
            assert os.path.exists(im_path), 'file not exist: {}'.format(im_path)
        file_list.append(image_list)

        case_name = line['case_name'] if 'case_name' in fieldnames else os.path.basename(os.path.dirname(image_list[0]))
        name_list.append(case_name)

        if 'mask_path' in fieldnames:
            sampling_path = line['mask_path']
            assert os.path.exists(sampling_path), 'sampling path not exist: {}'.format(sampling_path)
            sampling_list.append(sampling_path)

    return file_list, name_list, sampling_list


def read_test_folder(folder_path):
    """ read single-modality input folder
    :param folder_path: image file folder path
    :return: a list of image path list, list of image case names
    """
    suffix = ['.mhd', '.nii', '.hdr', '.nii.gz', '.mha', '.image3d']
    file = []
    for suf in suffix:
        file += glob.glob(os.path.join(folder_path, '*' + suf))

    file_list, name_list = [], []
    for im_pth in sorted(file):
        _, im_name = os.path.split(im_pth)
        for suf in suffix:
            idx = im_name.find(suf)
            if idx != -1:
                im_name = im_name[:idx]
                break
        file_list.append([im_pth])
        name_list.append(im_name)

    return file_list, name_list


class FixedNormalizer(object):
    """
    use fixed mean and stddev to normalize image intensities
    intensity = (intensity - mean) / stddev
    if clip is enabled:
        intensity = np.clip((intensity - mean) / stddev, -1, 1)
    """
    def __init__(self, mean, stddev, clip=True):
        """ constructor """
        assert stddev > 0, 'stddev must be positive'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.mean = mean
        self.stddev = stddev
        self.clip = clip

    def __call__(self, image):
        """ normalize image """
        if isinstance(image, Image3d):
            ctools.intensity_normalize(image, self.mean, self.stddev, self.clip)
        elif isinstance(image, (list, tuple)):
            for im in image:
                assert isinstance(im, Image3d)
                ctools.intensity_normalize(im, self.mean, self.stddev, self.clip)
        else:
            raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

    def static_obj(self):
        """ get a static normalizer object by removing randomness """
        obj = FixedNormalizer(self.mean, self.stddev, self.clip)
        return obj

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 0, 'mean': self.mean, 'stddev': self.stddev, 'clip': self.clip}
        return obj


class AdaptiveNormalizer(object):
    """
    use the minimum and maximum percentiles to normalize image intensities
    """
    def __init__(self, min_p=0.001, max_p=0.999, clip=True, min_rand=0, max_rand=0):
        """
        constructor
        :param min_p: percentile for computing minimum value
        :param max_p: percentile for computing maximum value
        :param clip: whether to clip the intensity between min and max
        :param min_rand: the random perturbation (%) of minimum value (0-1)
        :param max_rand: the random perturbation (%) of maximum value (0-1)
        """
        assert 1 >= min_p >= 0, 'min_p must be between 0 and 1'
        assert 1 >= max_p >= 0, 'max_p must be between 0 and 1'
        assert max_p > min_p, 'max_p must be > min_p'
        assert 1 >= min_rand >= 0, 'min_rand must be between 0 and 1'
        assert 1 >= max_rand >= 0, 'max_rand must be between 0 and 1'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.min_p = min_p
        self.max_p = max_p
        self.clip = clip
        self.min_rand = min_rand
        self.max_rand = max_rand

    def normalize(self, single_image):

        assert isinstance(single_image, Image3d), 'image must be an image3d object'
        normalize_min, normalize_max = ctools.percentiles(single_image, [self.min_p, self.max_p])

        if self.min_rand > 0:
            offset = np.abs(normalize_min) * self.min_rand
            offset = np.random.uniform(-offset, offset)
            normalize_min += offset

        if self.max_rand > 0:
            offset = np.abs(normalize_max) * self.max_rand
            offset = np.random.uniform(-offset, offset)
            normalize_max += offset

        normalize_mean = (normalize_min + normalize_max) / 2.0
        normalize_stddev = (normalize_max - normalize_min) / 2.0
        ctools.intensity_normalize(single_image, normalize_mean, normalize_stddev, clip=self.clip)

    def __call__(self, image):
        """ normalize image """
        if isinstance(image, Image3d):
            self.normalize(image)
        elif isinstance(image, (list, tuple)):
            for im in image:
                assert isinstance(im, Image3d)
                self.normalize(im)
        else:
            raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

    def static_obj(self):
        """ get a static normalizer object by removing randomness """
        obj = AdaptiveNormalizer(self.min_p, self.max_p, self.clip, min_rand=0, max_rand=0)
        return obj

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 1, 'min_p': self.min_p, 'max_p': self.max_p, 'clip': self.clip}
        return obj


def image_crop(im, crop_center, crop_spacing, crop_size, crop_axes, method, pad_type, pad_value):
    """ crop an image region
    :param im               the input image
    :param crop_center      the crop center in world space
    :param crop_spacing     the crop spacing in mm
    :param crop_size        the crop size in voxels
    :param crop_axes        the crop axes (None if use RAI coordinate)
    :param method:          the interpolation method
    :param pad_type:        the padding type, 0 for value padding, 1 for edge padding
    :param pad_value:       the default padding value for value padding
    :return an image crop
    """
    axes = axes_wo_angle(im)
    im.set_axes(axes)

    # frame = Frame3d()
    frame = im.frame().deep_copy()
    frame.set_spacing(crop_spacing)
    # frame.set_origin(crop_center)

    if crop_axes is None:
        frame.set_axes_to_rai()
    else:
        frame.set_axes(crop_axes)

    # print(frame.world_to_voxel(crop_center))
    z_center_voxel = 0 # round(frame.world_to_voxel(crop_center)[2])
    voxel_center = np.array([frame.world_to_voxel(crop_center)[0], frame.world_to_voxel(crop_center)[1], z_center_voxel])
    crop_center = frame.voxel_to_world(voxel_center)
    frame.set_origin(crop_center)
    # print(voxel_center)

    crop_size = np.array(crop_size)
    crop_origin = frame.voxel_to_world(np.round(-crop_size / 2.0))
    crop_origin = np.array([crop_origin[0], crop_origin[1], im.origin()[2]])
    frame.set_origin(crop_origin)

    if method == 'NN':
        crop = ctools.resample_nn(im, frame, crop_size, pad_type, pad_value)
    elif method == 'LINEAR':
        crop = ctools.resample_trilinear(im, frame, crop_size, pad_type, pad_value)
    else:
        raise ValueError('Unsupported Interpolation Method')
    return crop


def read_crop_adaptive(im_or_path, crop_center, crop_spacing, crop_axes, crop_size, method, pad_type, pad_value):
    """ read image crop from disk adaptively, also support image3dd format
    :param im_or_path       the path or Image3d object of input image
    :param crop_center      the crop center in world space
    :param crop_spacing     the crop spacing in mm
    :param crop_size        the crop size in voxels
    :param crop_axes        the crop axes (None if use RAI coordinate)
    :param method:          the interpolation method
    :param pad_type:        the padding type, 0 for value padding, 1 for edge padding
    :param pad_value:       the default padding value for value padding
    :return an image crop
    """
    if isinstance(im_or_path, str):
        if im_or_path.endswith('.image3dd'):
            assert os.path.isdir(im_or_path), 'image3dd format expects a folder image path'
            crop = md.read_image3dd_crop(im_or_path, crop_center, crop_spacing, crop_axes, crop_size, dtype=np.float32)
        else:
            assert os.path.exists(im_or_path), 'image path does not exist: {}'.format(im_or_path)
            crop = md.read_image(im_or_path, dtype=np.float32)
        crop = image_crop(crop, crop_center, crop_spacing, crop_size, crop_axes, method, pad_type, pad_value)

    elif isinstance(im_or_path, Image3d):
        crop = image_crop(im_or_path, crop_center, crop_spacing, crop_size, crop_axes, method, pad_type, pad_value)

    else:
        raise ValueError('Only support Image3d object or image path string for input im')
    return crop


def random_crop_with_data_augmentation(lrpath, crop_center, crop_spacing, crop_size, hr_or_path,
                                       aug_prob, shift_config, rot_config, flip_config, scale_config, cover_config,
                                       truncate_config, brightness_config, gamma_contrast_config, gaussian_noise_config,
                                       gaussian_blur_config, method, pad_type, pad_values):
    """ random a crop from image with data augmentation enabled

    :param impath:                the images path with supported extension, e.g., nii.gz, mhd, image3d, image3dd
    :param crop_center:            crop center in world coordinate (before data augmentation)
    :param crop_spacing:           the crop spacing (array of size 3)
    :param crop_size:              the crop size (array of size 3)
    :param hr:                     the hr path or hr image3d object
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
    :param method:                 the interpolation method
    :param pad_type:               the padding type, 0 for value padding, 1 for edge padding
    :param pad_value:             the default padding values for value padding
    :return image crops, mask crop
    """
    rai_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.double)
    crop_center = np.array(crop_center, dtype=np.double)
    crop_spacing = np.array(crop_spacing, dtype=np.double)
    crop_size = np.array(crop_size, dtype=np.int32)
    # crop_axes = np.array([mask_or_path.axis(0), mask_or_path.axis(1), mask_or_path.axis(2)], dtype=np.double) #rai_axes
    crop_axes = rai_axes
    crop_spacing_origin = crop_spacing.copy()
    assert len(crop_center) == 3 and len(crop_spacing) == 3 and len(crop_size) == 3, 'wrong vector size'

    # do data augmentation by probabilitylr
    do_aug = np.random.choice([False, True], p=[1 - aug_prob, aug_prob]) if 0 < aug_prob <= 1 else False

    # read crop without data augmentation
    if not do_aug:
        # read crop
        lr = md.read_image(lrpath)
        lr_crop_axes = np.array([lr.axis(0), lr.axis(1), lr.axis(2)], dtype=np.double)
        crop_axes = axes_wo_angle(lr)
        crop_lr_image = read_crop_adaptive(
                lrpath, crop_center, crop_spacing, crop_axes, crop_size, method, pad_type, pad_values)

        if isinstance(hr_or_path, str):
            hr = md.read_image(hr_or_path)
        else:
            hr = hr_or_path
        hr_crop_axes = np.array([hr.axis(0), hr.axis(1), hr.axis(2)], dtype=np.double)
        crop_axes = axes_wo_angle(hr)
        crop_hr_image = read_crop_adaptive(
            hr_or_path, crop_center, crop_spacing, crop_axes, crop_size, method, pad_type, pad_values)

        return crop_lr_image, crop_hr_image

    # shift crop center by probability
    shift_prob = shift_config['shift_prob']
    shift_mm = shift_config['shift_mm']

    do_shift = np.random.choice([False, True], p=[1 - shift_prob, shift_prob]) if 0 < shift_prob <= 1 else False
    if do_shift:
        shift = np.random.uniform(-shift_mm, shift_mm, (3,))
        crop_center += np.array([shift[0], shift[1], 0])

    # rotate crop axes by probability
    rot_prob = rot_config['rot_prob']
    rot_axis = rot_config['rot_axis']
    rot_angle_degree = abs(rot_config['rot_angle_degree'])

    do_rot = np.random.choice([False, True], p=[1 - rot_prob, rot_prob]) if 0 < rot_prob <= 1 else False
    if do_rot:
        rot_axis = uniform_sample_point_from_unit_sphere() if rot_axis is None else np.array(rot_axis)
        rot_axis = rot_axis[0]
        angle = random.random() * rot_angle_degree * math.pi / 180.0
        crop_axes = axis_angle_to_rotation_matrix(rot_axis, angle)

    # scale crop spacing by probability
    scale_prob = scale_config['scale_prob']
    scale_min_ratio = abs(scale_config['scale_min_ratio'])
    scale_max_ratio = abs(scale_config['scale_max_ratio'])
    scale_isotropic = scale_config['scale_isotropic']
    assert 0 < scale_min_ratio <= scale_max_ratio, 'wrong config of scale augmentation'

    do_scale = np.random.choice([False, True], p=[1 - scale_prob, scale_prob]) if 0 < scale_prob <= 1 else False
    if do_scale:
        if scale_isotropic:
            scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio)
            scale_ratio = np.array([scale_ratio] * 3)
        else:
            scale_ratio = np.random.uniform(scale_min_ratio, scale_max_ratio, (3,))
        scale_ratio = np.array([scale_ratio[0], scale_ratio[1], 1.0])
        crop_spacing = crop_spacing / scale_ratio

    # read crop image
    lr = md.read_image(lrpath)
    if do_rot:
        lr_crop_axes = np.array([crop_axes[0], crop_axes[1], lr.axis(2)], dtype=np.double)
    else:
        lr_crop_axes = np.array([lr.axis(0), lr.axis(1), lr.axis(2)], dtype=np.double)
    crop_lr_image = read_crop_adaptive(lrpath, crop_center, crop_spacing, lr_crop_axes, crop_size, method, pad_type, pad_values)

    # read crop mask
    if isinstance(hr_or_path, str):
        hr = md.read_image(hr_or_path)
    else:
        hr = hr_or_path
    if do_rot:
        hr_crop_axes = np.array([crop_axes[0], crop_axes[1], hr.axis(2)], dtype=np.double)
    else:
        hr_crop_axes = np.array([hr.axis(0), hr.axis(1), hr.axis(2)], dtype=np.double)
    crop_hr_image = read_crop_adaptive(hr_or_path, crop_center, crop_spacing, hr_crop_axes, crop_size, method, pad_type, pad_values)

    # rotate the boxes and make crop RAI
    if do_rot:
        crop_lr_image.set_axes(lr_crop_axes) #(rai_axes)
        crop_hr_image.set_axes(hr_crop_axes) #(rai_axes)

    # scale the boxes and reset crop spacing
    if do_scale:
        crop_lr_image.set_spacing(crop_spacing_origin)
        crop_hr_image.set_spacing(crop_spacing_origin)

    # do flip by probability
    flip_x_prob = flip_config['flip_x_prob']
    flip_y_prob = flip_config['flip_y_prob']
    # flip_z_prob = flip_config['flip_z_prob']
    do_flip_x = np.random.choice([False, True], p=[1 - flip_x_prob, flip_x_prob]) if 0 <= flip_x_prob <= 1 else False
    do_flip_y = np.random.choice([False, True], p=[1 - flip_y_prob, flip_y_prob]) if 0 <= flip_y_prob <= 1 else False
    # do_flip_z = np.random.choice([False, True], p=[1 - flip_z_prob, flip_z_prob]) if 0 <= flip_z_prob <= 1 else False

    # flip along x dimension
    if do_flip_x:
        # flip crop image and mask
        ctools.imflip(crop_lr_image, 0)
        ctools.imflip(crop_hr_image, 0)

    # flip along y dimension
    if do_flip_y:
        # flip crop image and mask
        ctools.imflip(crop_lr_image, 1)
        ctools.imflip(crop_hr_image, 1)

    # flip along z dimension
    # if do_flip_z:
    #     # flip crop image and mask
    #     for idx in range(len(crop_images)):
    #         ctools.imflip(crop_images[idx], 2)
    #     ctools.imflip(crop_mask, 2)

    # cover crop image by probability
    cover_prob = cover_config['cover_prob']
    do_cover = np.random.choice([False, True], p=[1 - cover_prob, cover_prob]) if 0 < cover_prob <= 1 else False
    if do_cover:
        cover_ratio = cover_config['cover_ratio']
        mu_value = cover_config['mu_value']
        sigma_value = cover_config['sigma_value']
        assert 0 < sigma_value, 'wrong config of cover augmentation'

        block_size = int(cover_ratio * crop_size[0])
        np_lr = crop_lr_image.to_numpy()
        non_zero = np.nonzero(np_lr)
        if non_zero[0].size != 0:
            random_num = np.random.randint(0, len(non_zero[0]))
            endx = crop_size[0] if non_zero[0][random_num] + block_size > crop_size[0] else \
                non_zero[0][random_num] + block_size
            endy = crop_size[1] if non_zero[1][random_num] + block_size > crop_size[1] else \
                non_zero[1][random_num] + block_size
            endz = crop_size[2] if non_zero[2][random_num] + block_size > crop_size[2] else \
                non_zero[2][random_num] + block_size
            np_noise = np.random.normal(mu_value, sigma_value, (endx - non_zero[0][random_num],
                                                                endy - non_zero[1][random_num],
                                                                endz - non_zero[2][random_num]))
            np_lr = crop_lr_image.to_numpy()
            np_lr[non_zero[0][random_num]:endx, non_zero[1][random_num]:endy, non_zero[2][random_num]:endz] += np_noise
            crop_lr_image.from_numpy(np_lr)

    # # truncate crop image by probability -> 1)truncate from the bottom of the image
    # truncate_prob = truncate_config['trunc_bottom_prob']
    # do_truncate = np.random.choice([False, True], p=[1 - truncate_prob, truncate_prob]) if 0 < truncate_prob <= 1 else False
    # if do_truncate:
    #     truncate_min_ratio = truncate_config['trunc_ratio_range'][0]
    #     truncate_max_ratio = truncate_config['trunc_ratio_range'][1]
    #     assert 0 < truncate_min_ratio < 1 and 0 < truncate_max_ratio < 1, 'wrong config of truncate augmentation'

    #     truncate_ratio = np.random.uniform(truncate_min_ratio, truncate_max_ratio)
    #     truncate_slice = int(truncate_ratio * crop_size[0])
    #     np_lr = crop_lr_image.to_numpy()
    #     np_lr[:truncate_slice, :, :] = pad_values
    #     crop_lr_image.from_numpy(np_lr)
    #     np_hr = crop_hr_image.to_numpy()
    #     np_hr[:truncate_slice, :, :] = 0
    #     crop_hr_image.from_numpy(np_hr)

    # # truncate crop image by probability -> 2)truncate from the top of the image
    # if do_truncate == False:
    #     truncate_prob = truncate_config['trunc_top_prob']
    #     do_truncate = np.random.choice([False, True],
    #                                    p=[1 - truncate_prob, truncate_prob]) if 0 < truncate_prob <= 1 else False
    #     if do_truncate:
    #         truncate_min_ratio = truncate_config['trunc_ratio_range'][0]
    #         truncate_max_ratio = truncate_config['trunc_ratio_range'][1]
    #         assert 0 < truncate_min_ratio < 1 and 0 < truncate_max_ratio < 1, 'wrong config of truncate augmentation'

    #         truncate_ratio = np.random.uniform(truncate_min_ratio, truncate_max_ratio)
    #         truncate_slice = int(truncate_ratio * crop_size[2])
    #         for idx in range(len(crop_images)):
    #             np_img = crop_images[idx].to_numpy()
    #             np_img[truncate_slice:, :, :] = pad_values[idx]
    #             crop_images[idx].from_numpy(np_img)
    #         np_mask = crop_mask.to_numpy()
    #         np_mask[truncate_slice:, :, :] = 0
    #         crop_mask.from_numpy(np_mask)

    # brightness transform
    seq_list = []
    brightness_prob = brightness_config['brightness_prob']
    do_brightness = np.random.choice([False, True], p=[1 - brightness_prob,
                                                       brightness_prob]) if 0 < brightness_prob <= 1 else False
    if do_brightness:
        seq_list.append(iaa.Multiply(mul=brightness_config['mul_range'], per_channel=True))

    # gamma contrast transform
    gamma_contrast_prob = gamma_contrast_config['gamma_contrast_prob']
    do_gamma_contrast = np.random.choice([False, True], p=[1 - gamma_contrast_prob,
                                                           gamma_contrast_prob]) if 0 < gamma_contrast_prob <= 1 else False
    if do_gamma_contrast:
        gamma_range = gamma_contrast_config['gamma_range']
        gamma = np.random.uniform(gamma_range[0], gamma_range[0])
        np_lr = crop_lr_image.to_numpy()
        minm = np_lr.min()
        rnge = np_lr.max() - minm
        np_lr = np.power(((np_lr - minm) / float(rnge + 1e-8)), gamma) * float(rnge + 1e-8) + minm
        crop_lr_image.from_numpy(np_lr)

    # gaussian noise transform
    gaussian_noise_prob = gaussian_noise_config['gaussian_noise_prob']
    do_gaussian_noise = np.random.choice([False, True], p=[1 - gaussian_noise_prob,
                                                           gaussian_noise_prob]) if 0 < gaussian_noise_prob <= 1 else False
    if do_gaussian_noise:
        seq_list.append(iaa.AdditiveGaussianNoise(loc=0, scale=gaussian_noise_config['noise_scale'], per_channel=True))

    # gaussian blur transform
    gaussian_blur_prob = gaussian_blur_config['gaussian_blur_prob']
    do_gaussian_blur = np.random.choice([False, True], p=[1 - gaussian_blur_prob,
                                                          gaussian_blur_prob]) if 0 < gaussian_blur_prob <= 1 else False
    if do_gaussian_blur:
        seq_list.append(iaa.GaussianBlur(gaussian_blur_config['sigma_range']))

    if seq_list is not None:
        seq = iaa.Sequential(seq_list)
        np_lr = crop_lr_image.to_numpy()
        np_lr = seq.augment_images(np_lr)
        crop_lr_image.from_numpy(np_lr)

    return crop_lr_image, crop_hr_image
