import glob
import numpy as np
from os import walk
from PIL import Image
import imageio
from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter
import cv2
import scipy
from scipy import stats

shape = (480, 640, 3)
MED_KERNEL_SIZE = (10, 10)
MERGE_RATIO = 0.4
enh = 10
MED = 0.8


def set_to_white(image):
    h = image.shape[0]
    w = image.shape[1]
    for x in range(h):
        for y in range(w):
            if np.sqrt(((x - (h/2)) ** 2) + ((y - (w/2)) ** 2)) > (0.95 * w/2):
                image[x, y] = 1
    return image


data_dir = './Data/'
sub_folders = ['All', 'Normal']

names = []
for i in range(len(sub_folders)):
    my_path = data_dir + sub_folders[i]
    filenames = next(walk(my_path), (None, None, []))[2]
    names.append(filenames)

dr_names = sorted(list(set(names[0])))
normal_names = sorted(list(set(names[1])))

filelist_right = glob.glob(data_dir + sub_folders[0] + '/*.*')
col_dr = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist_right]
col_dr = np.array(col_dr)
col_dr = col_dr[:, :, :, 0]
filelist_right = glob.glob(data_dir + sub_folders[1] + '/*.*')
col_nrm = [np.array(Image.open(fname).resize((shape[1], shape[0])).__array__()) for fname in filelist_right]
col_nrm = np.array(col_nrm)
col_nrm = col_nrm[:, :, :, 0]

dr_ratios = np.array(col_dr)
normal_ratios = np.array(col_nrm)

avrg = np.concatenate((
    normal_ratios[0], normal_ratios[1], normal_ratios[2]), axis=0)
avrg = (avrg - np.min(avrg)) / np.ptp(avrg)
fcm_centers = 0
kernel = (10, 10)

for i in range(col_dr.shape[0]):
    print('Image Name:', list(dr_names)[i])
    tmp_dr_ = dr_ratios[i]
    tmp_dr_[450:480, 520:640] = 1
    tmp_dr = np.concatenate((tmp_dr_, avrg), axis=0)
    N_SIZE, M_SIZE = tmp_dr.shape[0], tmp_dr.shape[1]
    X = np.asarray(tmp_dr).reshape((N_SIZE * M_SIZE), 1)
    K = 6
    med_map = np.ones(10)
    N = 25
    gmm = GaussianMixture(n_components=K, n_init=N)
    gmm.fit(X)
    fcm_centers = np.empty(shape=(gmm.n_components, X.shape[1]))
    for j in range(gmm.n_components):
        density = scipy.stats.multivariate_normal(cov=gmm.covariances_[j], mean=gmm.means_[j]).logpdf(X)
        fcm_centers[j, :] = X[np.argmax(density)]
    print('Centers of Clusters:', np.array(fcm_centers).flatten())
    cnt = np.array(fcm_centers)
    cnt = cnt.flatten()
    N_SIZE, M_SIZE = tmp_dr_.shape[0], tmp_dr_.shape[1]
    X = np.asarray(tmp_dr_).reshape((N_SIZE * M_SIZE), 1)
    labeled_X = gmm.predict(X)
    labeled_X = np.asarray(labeled_X).reshape(N_SIZE, M_SIZE)
    smp_numbers = []
    for j in range(K):
        tmp = np.where(labeled_X == j)
        s = np.sum(tmp)
        smp_numbers.append(s)
    smp_numbers = np.sort(smp_numbers)
    smp_numbers = list(smp_numbers)
    max_cluster_1 = smp_numbers.index(smp_numbers[0])
    cnt_1 = np.where(labeled_X == max_cluster_1, 1, 1 / (labeled_X + 1))
    M = np.multiply(cnt_1, 255 * tmp_dr_)
    med_map = median_filter(M, size=MED_KERNEL_SIZE)
    med_map = (med_map - np.min(med_map)) / np.ptp(med_map)
    med_map = cv2.morphologyEx(255 * med_map, cv2.MORPH_CLOSE, kernel)
    med_map = (med_map - np.min(med_map)) / np.ptp(med_map)
    med_map[med_map == np.max(med_map)] = 1
    med_map_ = 1 - med_map
    med_map_[med_map_ < 0.96] = 0
    med_map_ = set_to_white(med_map_)
    med_map_ = np.array(255 * med_map_).astype('uint8')
    heatmap_img = cv2.applyColorMap(np.array(255 * med_map).astype('uint8'), cv2.COLORMAP_JET)
    tmp = (col_dr[i, :, :] - np.min(col_dr[i, :, :])) / np.ptp(col_dr[i, :, :])
    tmp = np.expand_dims(255 * tmp, axis=2).astype('uint8')
    tmp = np.concatenate((tmp, tmp, tmp), axis=2).astype('uint8')
    tmp_ = (col_dr[i, :, :] - np.min(col_dr[i, :, :])) / np.ptp(col_dr[i, :, :])
    tmp_ = np.expand_dims(255 * tmp_, axis=2).astype('uint8')
    tmp_ = np.concatenate((tmp_, tmp_, tmp_), axis=2).astype('uint8')
    super_imposed_img = cv2.addWeighted(heatmap_img, MERGE_RATIO, tmp, 1 - MERGE_RATIO, 0)
    super_imposed_img_ = cv2.addWeighted(heatmap_img, MERGE_RATIO, tmp_, 1 - MERGE_RATIO, 0)
    imageio.imwrite('./Outputs/Main Code V1.5/All Mask/' + list(dr_names)[i], med_map_)
    imageio.imwrite('./Outputs/Main Code V1.5/All Heatmap/' + list(dr_names)[i], heatmap_img)
    imageio.imwrite('./Outputs/Main Code V1.5/All Merged/' + list(dr_names)[i], super_imposed_img)
