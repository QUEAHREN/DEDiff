import os
import cv2
import numpy as np
import h5py
import shutil
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.mat']

# 制作新的测试集

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes

def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img

def main():

    save_path = "/gdata/linrj/Event_Deblur/GOPRO/GOPRO_total/test_ev"
    save_lq_path = os.path.join(save_path, "blur")
    save_gt_path = os.path.join(save_path, "sharp")
    save_h5_path = os.path.join(save_path, "H5")

    if not os.path.exists(save_lq_path):
        os.mkdir(save_lq_path)
    if not os.path.exists(save_gt_path):
        os.mkdir(save_gt_path)
    if not os.path.exists(save_h5_path):
        os.mkdir(save_h5_path)

    data_path = "/gdata/linrj/Event_Deblur/GOPRO/GOPRO_total/test"
    lq_path = os.path.join(data_path, "blur")
    gt_path = os.path.join(data_path, "sharp")
   
    paths_GT, sizes_GT = get_image_paths('img', gt_path)
    paths_LQ, sizes_LQ = get_image_paths('img', lq_path)
    # print(paths_LQ)

    v_flagx = []
    v_flag  = []
    for i in range(0,len(paths_LQ)-1):
        path_i = paths_LQ[i]
        path_i_n = paths_LQ[i+1]
        video_name_list_i = path_i.split('/')[len(path_i.split('/'))-1].split('_')
        video_name_list_i_n = path_i_n.split('/')[len(path_i_n.split('/'))-1].split('_')
        video_name_i = video_name_list_i[0] + video_name_list_i[1] + video_name_list_i[2]
        video_name_i_n = video_name_list_i_n[0] + video_name_list_i_n[1] + video_name_list_i_n[2]
        if video_name_i != video_name_i_n:
            v_flag.append(i)
    v_flagx = [x + 1 for x in v_flag]
    v_flagx.append(0)
    v_flag.append(len(paths_LQ)-1)
    v_flagx = sorted(v_flagx)
    v_flagx = v_flagx[::2]
    v_flagx = [item + i for item in v_flagx for i in range(0, 10)]

    print(len(v_flagx))
    h5_path = os.path.join(data_path, "H5", "test_blur.h5")
    event_h5f = h5py.File(h5_path,'r')

    new_index = 0
    with h5py.File(os.path.join(save_h5_path,'test_blur.h5'), 'w') as test_h5f:
        for index in v_flagx:
            shutil.copy(paths_LQ[index], save_lq_path)
            shutil.copy(paths_GT[index], save_gt_path)
            event_output = np.array(event_h5f[str(index)])  
            print(event_h5f[str(index)])
            test_h5f.create_dataset(str(new_index), data=event_output)
            new_index += 1

    

# python /ghome/linrj/XSY/SR3/preprocess/generate_testdata.py

if __name__ == '__main__':
    main()