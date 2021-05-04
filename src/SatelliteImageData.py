import csv
import cv2
import math
import numpy as np
import os
import sklearn
import tensorflow as tf
import tifffile as tiff

from pathlib import Path
from shapely.wkt import loads as wkt_loads
from tensorflow.keras.utils import Sequence


NUM_CLASSES = 10
CLASS_TYPES = {
    '001_MM_L2_LARGE_BUILDING':1,
    '001_MM_L3_RESIDENTIAL_BUILDING':1,
    '001_MM_L3_NON_RESIDENTIAL_BUILDING':1,
    '001_MM_L5_MISC_SMALL_STRUCTURE':2,
    '002_TR_L3_GOOD_ROADS':3,
    '002_TR_L4_POOR_DIRT_CART_TRACK':4,
    '002_TR_L6_FOOTPATH_TRAIL':4,
    '006_VEG_L2_WOODLAND':5,
    '006_VEG_L3_HEDGEROWS':5,
    '006_VEG_L5_GROUP_TREES':5,
    '006_VEG_L5_STANDALONE_TREES':5,
    '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND':6,
    '007_AGR_L6_ROW_CROP':6, 
    '008_WTR_L3_WATERWAY':7,
    '008_WTR_L2_STANDING_WATER':8,
    '003_VH_L4_LARGE_VEHICLE':9,
    '003_VH_L5_SMALL_VEHICLE':10,
    '003_VH_L6_MOTORBIKE':10
}


class SatelliteImageData:
    def __init__(self, data_root_dir):

        self.data_root_dir = data_root_dir
        self.temp_data_dir = data_root_dir + 'temp_data/'
        self.processed_data_dir = data_root_dir + 'processed_data/'

        self.three_band_img_dir = data_root_dir + 'three_band/'
        self.sixteen_band_img_dir = data_root_dir + 'sixteen_band/'

        self.wkt_file_path = data_root_dir + 'train_wkt_v4.csv'
        self.grid_sizes_file_path = data_root_dir + 'grid_sizes.csv'
        self.sample_submission_file_path = data_root_dir + 'sample_submission.csv'

        # Load grid sizes
        csv.field_size_limit(int(1e9))
        self.grid_sizes = {}
        grid_sizes_file_reader = csv.reader(open(self.grid_sizes_file_path))
        next(grid_sizes_file_reader)
        for img_id, x_max, y_min in grid_sizes_file_reader:
            self.grid_sizes[img_id] = (float(x_max), float(y_min))

        # Load polygons
        self.wkt_polygons = {}
        wkt_polygons_file_reader = csv.reader(open(self.wkt_file_path))
        next(wkt_polygons_file_reader)
        for img_id, class_type, polygons in wkt_polygons_file_reader:
            if img_id not in self.wkt_polygons.keys():
                self.wkt_polygons[img_id] = {}
            self.wkt_polygons[img_id][class_type] = wkt_loads(polygons)
        
        self.dataset_size = len(self.wkt_polygons)


    #
    # Multispectral bands image to RGB image
    #

    def stretch_n_bit(self, bands, lower_percent=2, higher_percent=98):
        '''
        Author: amaia
        Source: https://www.kaggle.com/aamaia/rgb-using-m-bands-example
        
        Scale the values of rgb channels to range 0 to 255.
        '''
        out = np.zeros_like(bands)
        for i in range(bands.shape[2]):
            a = 0
            b = 255
            c = np.percentile(bands[:,:,i], lower_percent)
            d = np.percentile(bands[:,:,i], higher_percent)
            t = a + (bands[:,:,i] - c) * (b - a) / (d - c)
            t[t<a] = a
            t[t>b] = b
            out[:,:,i] = t
        return out.astype(np.uint8)

    def get_m_band_img(self, img_id):
        filename = os.path.join(self.sixteen_band_img_dir, '{}_M.tif'.format(img_id))
        img = tiff.imread(filename)
        img = np.rollaxis(img, 0, 3)
        return img

    def m_band_to_rgb(self, img):
        '''
        Author: amaia
        Source: https://www.kaggle.com/aamaia/rgb-using-m-bands-example
        
        8 multispectral bands: [coastal, blue, green, yellow, red, red edge, near-IR1, near-IR2]
        '''
        rgb_img = np.zeros((img.shape[0], img.shape[1], 3))
        rgb_img[:,:,0] = img[:,:,4] #red
        rgb_img[:,:,1] = img[:,:,2] #green
        rgb_img[:,:,2] = img[:,:,1] #blue
        return rgb_img

    def get_visual_m_band_img(self, img_id):
        return self.stretch_n_bit(self.m_band_to_rgb(self.get_m_band_img(img_id)))

    def get_three_band_img(self, img_id):
        filename = os.path.join(self.three_band_img_dir, '{}.tif'.format(img_id))
        return np.rollaxis(tiff.imread(filename), 0, 3)

    def get_visual_three_band_img(self, img_id):
        return self.stretch_n_bit(self.get_three_band_img(img_id))


    #
    # Polygons to Masks
    #

    def _convert_coordinates_to_raster(self, coords, img_size, xymax):
        '''
        Author: visoft
        Source: https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        '''
        Xmax, Ymax = xymax
        H, W = img_size
        W1 = 1.0 * W * W / (W + 1)
        H1 = 1.0 * H * H / (H + 1)
        xf = W1 / Xmax
        yf = H1 / Ymax
        coords[:, 1] *= yf
        coords[:, 0] *= xf
        coords_int = np.round(coords).astype(np.int32)
        return coords_int

    def _get_and_convert_contours(self, polygon_list, raster_img_size, xymax):
        '''
        Author: visoft
        Source: https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        '''
        perim_list = []
        interior_list = []
        if polygon_list is None:
            return None
        for k in range(len(polygon_list)):
            poly = polygon_list[k]
            perim = np.array(list(poly.exterior.coords))
            perim_c = self._convert_coordinates_to_raster(perim, raster_img_size, xymax)
            perim_list.append(perim_c)
            for pi in poly.interiors:
                interior = np.array(list(pi.coords))
                interior_c = self._convert_coordinates_to_raster(interior, raster_img_size, xymax)
                interior_list.append(interior_c)
        return perim_list, interior_list

    def _plot_mask_from_contours(self, raster_img_size, contours, class_value=1):
        '''
        Author: visoft
        Source: https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        '''
        img_mask = np.zeros(raster_img_size, np.uint8)
        if contours is None:
            return img_mask
        perim_list, interior_list = contours
        cv2.fillPoly(img_mask, perim_list, class_value)
        cv2.fillPoly(img_mask, interior_list, 0)
        return img_mask

    def get_mask_of_class(self, raster_size, image_id, class_type):
        '''
        Author: visoft
        Source: https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
        ''' 
        xymax = self.grid_sizes[image_id]
        polygon_list = self.wkt_polygons[image_id][class_type]
        contours = self._get_and_convert_contours(polygon_list, raster_size, xymax)
        mask = self._plot_mask_from_contours(raster_size, contours, 1)
        return mask

    def get_stacked_masks(self, mask_size, img_id):
        stacked_masks = self.get_mask_of_class(mask_size, img_id, '1')
        for class_type in range(1, 10):
            stacked_masks = np.dstack([stacked_masks, self.get_mask_of_class(mask_size, img_id, str(class_type+1))])
        
        # stacked_masks[:,:,-2] += stacked_masks[:,:,-1]
        # stacked_masks = np.delete(stacked_masks, (-1), axis=-1)

        return stacked_masks


    #
    # Generate dataset by slicing images and masks into patches
    #

#     def generate_sliced_data_patches(self, image_size=(3000, 3000), patch_size=(300,300)):
#         Path(self.temp_data_dir).mkdir(parents=True, exist_ok=True)
#         np.save(self.temp_data_dir + 'image_size.npy', image_size)
#         np.save(self.temp_data_dir + 'patch_size.npy', patch_size)

#         for img_id in self.wkt_polygons.keys():
#             print(img_id)

#             loaded_image = self.stretch_n_bit(self.get_m_band_img(img_id))

#             loaded_image = cv2.resize(loaded_image, image_size, interpolation=cv2.INTER_LINEAR)
#             stacked_masks = self.get_stacked_masks(image_size, img_id)

#             temp_data_entry_dir = self.temp_data_dir + img_id + '/'
#             Path(temp_data_entry_dir).mkdir(parents=True, exist_ok=True)

#             x_cur_pos = 0
#             for i in range(math.floor(image_size[0]/patch_size[0])):
#                 y_cur_pos = 0
#                 x_next_pos = x_cur_pos + patch_size[0]
#                 for j in range(math.floor(image_size[1]/patch_size[1])):
#                     y_next_pos = y_cur_pos + patch_size[1]

#                     np.save(temp_data_entry_dir + 'img_{}_{}.npy'.format(i, j),
#                             loaded_image[x_cur_pos:x_next_pos,y_cur_pos:y_next_pos,:])
#                     np.save(temp_data_entry_dir + 'msks_{}_{}.npy'.format(i, j),
#                             stacked_masks[x_cur_pos:x_next_pos,y_cur_pos:y_next_pos,:])

#                     y_cur_pos = y_next_pos
#                 x_cur_pos = x_next_pos

    def generate_data_patches(self, image_size=(3000, 3000), patch_size=(300,300)):
        Path(self.processed_data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_dir+'imgs/').mkdir(parents=True, exist_ok=True)
        Path(self.processed_data_dir+'msks/').mkdir(parents=True, exist_ok=True)

        np.save(self.processed_data_dir + 'image_size.npy', image_size)
        np.save(self.processed_data_dir + 'patch_size.npy', patch_size)

        labelled_patch_ids = []
        unlabelled_patch_ids = []

        for idx, img_id in enumerate(self.grid_sizes.keys()):
            print('{}/{}: {}'.format(idx+1, len(self.grid_sizes.keys()), img_id))

            loaded_image = self.stretch_n_bit(self.get_m_band_img(img_id))
            loaded_image = cv2.resize(loaded_image, image_size, interpolation=cv2.INTER_LINEAR)

            stacked_masks = None
            if img_id in self.wkt_polygons.keys():
                stacked_masks = self.get_stacked_masks(image_size, img_id)

            x_cur_pos = 0
            for i in range(math.floor(image_size[0]/patch_size[0])):
                y_cur_pos = 0
                x_next_pos = x_cur_pos + patch_size[0]
                for j in range(math.floor(image_size[1]/patch_size[1])):
                    y_next_pos = y_cur_pos + patch_size[1]

                    np.save(self.processed_data_dir + 'imgs/{}_patch_{}_{}.npy'.format(img_id, i, j),
                            loaded_image[x_cur_pos:x_next_pos,y_cur_pos:y_next_pos,:])

                    if stacked_masks is not None:
                        np.save(self.processed_data_dir + 'msks/{}_patch_{}_{}.npy'.format(img_id, i, j),
                                stacked_masks[x_cur_pos:x_next_pos,y_cur_pos:y_next_pos,:])
                        labelled_patch_ids.append('{}_patch_{}_{}'.format(img_id, i, j))
                    else:
                        unlabelled_patch_ids.append('{}_patch_{}_{}'.format(img_id, i, j))

                    y_cur_pos = y_next_pos
                x_cur_pos = x_next_pos

        np.save(self.processed_data_dir + 'labelled_patch_ids.npy', labelled_patch_ids)
        np.save(self.processed_data_dir + 'unlabelled_patch_ids.npy', unlabelled_patch_ids)

    #
    # Dataset
    #

#     # Generate fake data
#     def _gen_fake_data(self, img, labels):
#         fake_data = []

#         for flip_mode in [-1,0,1]:
#             flipped_img = cv2.flip(img, flip_mode)
#             flipped_labels = cv2.flip(labels, flip_mode)
#             fake_data.append((flipped_img, flipped_labels))

#         return fake_data

#     def get_sliced_dataset(self, splits=(0.7, 0.2, 0.1), shuffle=True):
#         image_size = np.load(self.temp_data_dir + 'image_size.npy')
#         patch_size = np.load(self.temp_data_dir + 'patch_size.npy')

#         num_row = math.floor(image_size[0]/patch_size[0])
#         num_col = math.floor(image_size[1]/patch_size[1])

#         num_data_entry = int(self.dataset_size * num_row * num_col)

#         images = np.empty((num_data_entry, patch_size[0], patch_size[1], 8), dtype='float16')
#         labels = np.empty((num_data_entry, patch_size[0], patch_size[1], NUM_CLASSES), dtype='float16')

#         counter = 0
#         for img_id in self.wkt_polygons.keys():
#             for i in range(num_row):
#                 for j in range(num_col):
#                     images[counter] = np.load(self.temp_data_dir + '{}/img_{}_{}.npy'.format(img_id, i, j))
#                     labels[counter] = np.load(self.temp_data_dir + '{}/msks_{}_{}.npy'.format(img_id, i, j))[:,:,:NUM_CLASSES]
#                     counter += 1

#         print('counter:', counter)

#         split_idxs = []

#         idx_pos = 0
#         for split in splits[:-1]:
#             idx_pos += int(num_data_entry*split)
#             split_idxs.append(idx_pos)

#         images_splited = [images[i:j] for i, j in zip([0]+split_idxs, split_idxs+[None])]
#         labels_splited = [labels[i:j] for i, j in zip([0]+split_idxs, split_idxs+[None])]

#         if shuffle:
#             images_splited[0], labels_splited[0] = sklearn.utils.shuffle(np.array(images_splited[0]),
#                                                                          np.array(labels_splited[0]))

#         return tuple(tf.data.Dataset.from_tensor_slices(dataset) for dataset in zip(images_splited, labels_splited))

#     def ____get_balanced_sliced_dataset(self, splits=(0.7, 0.2, 0.1), shuffle=True):
#         image_size = np.load(self.temp_data_dir + 'image_size.npy')
#         patch_size = np.load(self.temp_data_dir + 'patch_size.npy')

#         num_row = math.floor(image_size[0]/patch_size[0])
#         num_col = math.floor(image_size[1]/patch_size[1])

#         # num_data_entry = int(self.dataset_size * num_row * num_col)
#         # num_data_entry = 400 + 81
#         num_data_entry = 81

#         images = np.empty((num_data_entry, patch_size[0], patch_size[1], 8), dtype='float16')
#         labels = np.empty((num_data_entry, patch_size[0], patch_size[1], NUM_CLASSES), dtype='float16')

#         counter = 0
#         counter_vehicle = 0
#         counter_others = 0
#         for img_id in self.wkt_polygons.keys():
#             for i in range(num_row):
#                 for j in range(num_col):
#                     loaded_img = np.load(self.temp_data_dir + '{}/img_{}_{}.npy'.format(img_id, i, j))
#                     loaded_lbs = np.load(self.temp_data_dir + '{}/msks_{}_{}.npy'.format(img_id, i, j))

#                     loaded_lbs[:,:,-2] += loaded_lbs[:,:,-1]
#                     loaded_lbs = np.delete(loaded_lbs, (-1), axis=-1)

#                     if np.sum(loaded_lbs[:,:,-1]) > 0:
#                         images[counter] = loaded_img
#                         labels[counter] = loaded_lbs
#                         counter += 1
#                         counter_vehicle += 1

#                         # fake_data = self._gen_fake_data(loaded_img, loaded_lbs)
#                         # for data in fake_data:
#                         #     counter += 1
#                         #     images[counter] = data[0]
#                         #     labels[counter] = data[1]

#                     # else:
#                     #     if counter_others < 400:
#                     #         images[counter] = loaded_img
#                     #         labels[counter] = loaded_lbs
#                     #         counter += 1
#                     #         counter_others += 1

#         print('counter_vehicle:', counter_vehicle)
#         print('counter_others:', counter_others)
#         print('counter:', counter)

#         split_idxs = []

#         idx_pos = 0
#         for split in splits[:-1]:
#             idx_pos += int(num_data_entry*split)
#             split_idxs.append(idx_pos)

#         images_splited = [images[i:j] for i, j in zip([0]+split_idxs, split_idxs+[None])]
#         labels_splited = [labels[i:j] for i, j in zip([0]+split_idxs, split_idxs+[None])]

#         if shuffle:
#             for i in range(len(splits)):
#                 images_splited[i], labels_splited[i] = sklearn.utils.shuffle(np.array(images_splited[i]),
#                                                                              np.array(labels_splited[i]))

#         return tuple(tf.data.Dataset.from_tensor_slices(dataset) for dataset in zip(images_splited, labels_splited))


class SlicedSatelliteDataGenerator(Sequence):
    def __init__(self,
                 root_path,
                 patch_ids,
                 input_shape=(256,256,8),
                 num_classes=10,
                 batch_size=20,
                 psudo_label_path=None):
        self.root_path = root_path
        self.patch_ids = patch_ids
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.psudo_label_path = psudo_label_path

    def __len__(self):
        return int(math.floor(len(self.patch_ids)/self.batch_size))

    def __getitem__(self, batch_id):
        batch_ids = np.arange(batch_id*self.batch_size,(batch_id+1)*self.batch_size)
        batch_patch_ids = [self.patch_ids[i] for i in batch_ids]

        images = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]),
                          dtype='float16')
        labels = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.num_classes),
                          dtype='float16')

        for i, img_id in enumerate(batch_patch_ids):
            images[i] = np.load(self.root_path + 'imgs/{}.npy'.format(img_id))

            if self.psudo_label_path:
                labels[i] = np.load(self.psudo_label_path + '/{}.npy'.format(img_id))
            else:
                labels[i] = np.load(self.root_path + 'msks/{}.npy'.format(img_id))

        return images, labels



