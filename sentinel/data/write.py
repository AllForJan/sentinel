# coding: utf-8
"""satellite image objects
* hierarchical class built structures:
    1. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
    2. Misc. Manmade structures
* hierarchical class pathways:
    3. Road 
    4. Track - poor/dirt/cart track, footpath/trail
* hierarchical class vegetation:
    5. Trees - woodland, hedgerows, groups of trees, standalone trees
    6. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
* hierarchical class water: 
    7. Waterway 
    8. Standing water
"""
import os
from glob import glob

import numpy as np
import tensorflow as tf
from tensor2tensor.data_generators.generator_utils import to_example

from joblib import Parallel, delayed as jl_delayed
from dask import delayed, compute
import dask.multiprocessing
from dask.diagnostics import ProgressBar

import pandas as pd
import geopandas as gpd

import geojson
from shapely.wkt import loads
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
import shapely.affinity

import tifffile
import cv2

import argparse

DATA_PATH = '/mnt/data/sentinel/dstl-satellite-imagery-feature-detection'
FILENAME2LABEL = {
    # bUIldINGS
    '001_MM_L2_LARGE_BUILDING': 1,
    '001_MM_L3_RESIDENTIAL_BUILDING': 1,
    '001_MM_L3_NON_RESIDENTIAL_BUILDING': 1,
    '001_MM_L5_MISC_SMALL_STRUCTURE': 2,
    # PATHWAYS
    '002_TR_L3_GOOD_ROADS': 3,
    '002_TR_L4_POOR_DIRT_CART_TRACK': 4,
    '002_TR_L6_FOOTPATH_TRAIL': 4,
    # VEGETATION
    '006_VEG_L2_WOODLAND': 5,
    '006_VEG_L3_HEDGEROWS': 5,
    '006_VEG_L5_GROUP_TREES': 5,
    '006_VEG_L5_STANDALONE_TREES': 5,
    # LAND
    '007_AGR_L2_CONTOUR_PLOUGHING_CROPLAND': 6,
    '007_AGR_L6_ROW_CROP': 6,
    # WATER
    '008_WTR_L3_WATERWAY': 7,
    '008_WTR_L2_STANDING_WATER': 8,
    # '003_VH_L4_LARGE_VEHICLE': 9,
    # '003_VH_L5_SMALL_VEHICLE': 10,
    # '003_VH_L6_MOTORBIKE': 10
}
LABEL_RANGE = np.arange(1, 9)
LABEL_COLUMNS = ['label_{}'.format(i) for i in LABEL_RANGE]
LABEL_MAP = {
    1: 'BUILDINGS',
    2: 'BUILDINGS',
    3: 'PATHWAYS',
    4: 'PATHWAYS',
    5: 'VEGETATION',
    6: 'CROP',
    7: 'WATER',
    8: 'WATER',
}

UNION_MAP = {
    'BUILDINGS': ['label_1', 'label_2'],
    'PATHWAYS': ['label_3', 'label_4'],
    'VEGETATION': ['label_5'],
    'CROP': ['label_6'],
    'WATER': ['label_7', 'label_8'],
}

HIERARCHIES = ['OTHER', 'BUILDINGS', 'PATHWAYS', 'VEGETATION', 'CROP', 'WATER']
HIERARCHY_LABELS = range(len(HIERARCHIES))
HIERARCHY_MAP = dict(zip(HIERARCHIES, HIERARCHY_LABELS))

get_file = lambda f, data_path=DATA_PATH: '{}/{}'.format(data_path, f)


### GEOMETRIES ###
@delayed
def load(filepath):
  gdf = gpd.read_file(filepath)
  filename = filepath.strip('.geojson').split('/')[-1]
  label = FILENAME2LABEL.get(filename, None)
  if label is not None:
    gdf['class_label'] = label
  return gdf


@delayed
def join(gpds):
  return pd.concat(gpds, axis=0)


def read_geojsons(file_pattern):
  files = glob(get_file(file_pattern))
  loaded = [load(i) for i in files]
  joined = join(loaded)
  with ProgressBar():
    return joined.compute(scheduler='processes')


def process_geojson_geometry(geojson_gdf):
  geojson_gdf = geojson_gdf.dropna()
  geojson_gdf.columns = geojson_gdf.columns.str.lower()
  geojson_gdf.class_label = geojson_gdf.class_label.astype('i4')
  wkt_cols = ['tile_name', 'class_label', 'geometry']
  columns = {'tile_name': 'image_id', 'class_label': 'label'}
  return geojson_gdf[wkt_cols].rename(columns=columns)


def load_geojson_geometry(file_pattern):
  geojson_gdf = read_geojsons(file_pattern)
  return process_geojson_geometry(geojson_gdf)


def load_wkt(filename='train_wkt_v4.csv', return_raw=False):
  wkt_gdf = pd.read_csv(get_file(filename))
  wkt_gdf = gpd.GeoDataFrame(wkt_gdf)
  wkt_gdf.columns = ['image_id', 'label', 'geometry']
  wkt_gdf.geometry = wkt_gdf.geometry.apply(loads)
  if return_raw: return wkt_gdf
  wkt_gdf = wkt_gdf[wkt_gdf.geometry.apply(len) > 0]
  return wkt_gdf[~wkt_gdf.label.isin([9, 10])]


def load_geometries(wkt_filename='train_wkt_v4.csv',
                    geojson_file_pattern='train_geojson_v3/*/00*.geojson'):
  wkt_gdf = load_wkt(wkt_filename)
  geojson_gdf = load_geojson_geometry(geojson_file_pattern)
  return pd.concat([wkt_gdf, geojson_gdf], axis=0)


def read_grid_sizes(filename='grid_sizes.csv'):
  return pd.read_csv(
      get_file(filename), names=['image_id', 'x_max', 'y_min'], skiprows=1)


def read_geojson_grids(file_pattern='train_geojson_v3/*/Grid*.geojson'):
  columns = {'fme_basena': 'image_id'}
  grids_gdf = read_geojsons(file_pattern).rename(columns=columns)
  grid_sizes_df = grids_gdf.set_index('image_id').bounds.reset_index()
  grid_sizes_df.drop(labels=['minx', 'maxy'], axis=1, inplace=True)
  return grid_sizes_df.rename(columns={'miny': 'y_min', 'maxx': 'x_max'})


def load_grid_sizes(filename='grid_sizes.csv',
                    file_pattern='train_geojson_v3/*/Grid*.geojson'):
  grid_sizes_df = read_grid_sizes(filename)
  geojson_grid_sizes_df = read_geojson_grids(file_pattern)
  return pd.concat(
      [grid_sizes_df, geojson_grid_sizes_df],
      axis=0).sort_values('image_id').reset_index(drop=True)


def group_per_image(geometries_gdf, agg=None):
  geometries_gdf.geometry = geometries_gdf.geometry.buffer(0)
  if agg is None: agg = {'geometry': unary_union}
  grouped_gdf = geometries_gdf.groupby(['image_id', 'label']).agg(agg)
  grouped_gdf = grouped_gdf.unstack('label').reset_index()
  grouped_gdf.columns = ['image_id'] + LABEL_COLUMNS
  return grouped_gdf


def load_targets():
  grid_sizes_df = load_grid_sizes().drop_duplicates(subset=['image_id'])
  geometries_gdf = load_geometries()
  grouped_gdf = group_per_image(geometries_gdf)
  return grouped_gdf.merge(grid_sizes_df, on='image_id', how='inner')
  # return grouped_gdf, grid_sizes_df


def load_dataset():
  targets_gdf = load_targets()
  get_image_path = lambda x: get_file('three_band/{}.tif').format(x)
  targets_gdf['image_path'] = targets_gdf.image_id.apply(get_image_path)
  return targets_gdf


## WRITE TF DATASET ##
def get_scalers(im_size, x_max, y_min):
  h, w = im_size  # they are flipped so that mask_for_polygons works correctly
  w_ = w * (w / (w + 1))
  h_ = h * (h / (h + 1))
  x_scaler = w_ / x_max
  y_scaler = h_ / y_min
  return x_scaler, y_scaler


def scale_to_mask(im_size, x_max, y_min, polygon):
  x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
  return shapely.affinity.scale(
      polygon, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def mask_for_polygons(mask, polygons, label, x_max, y_min):
  """ Return numpy mask for given polygons.
    polygons should already be converted to image coordinates.
    """
  class_mask = np.zeros_like(mask, np.uint8)
  image_size = mask.shape
  polygons = scale_to_mask(image_size, x_max, y_min, polygons)
  int_coords = lambda x: np.array(x).round().astype(np.int32)
  if isinstance(polygons, Polygon): polygons = [polygons]
  exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
  interiors = [
      int_coords(interior.coords)
      for poly in polygons
      for interior in poly.interiors
  ]
  cv2.fillPoly(class_mask, exteriors, 1)
  cv2.fillPoly(class_mask, interiors, 0)
  mask[np.nonzero(class_mask)] = label


def process_classes(image_size, df_row):
  label_notnull_idx = df_row[LABEL_COLUMNS].notnull()
  classes = LABEL_RANGE[label_notnull_idx.values.flatten()]
  print(len(classes))

  # mask = np.zeros(image_size, np.uint8)

  def polygons2mask(c):
    mask = np.zeros(image_size, np.uint8)  # new
    polygons = df_row[UNION_MAP[c]].values.flatten()
    polygons = polygons[polygons != None]
    if len(polygons) > 0:
      polygons = unary_union(polygons) if len(polygons) > 1 else polygons[0]
      label = HIERARCHY_MAP[c]
      mask_for_polygons(mask, polygons, label, df_row.x_max, df_row.y_min)
    return mask  # new

  new_classes = list(UNION_MAP.keys())[::-1]
  # [polygons2mask(c) for c in new_classes]
  # return mask
  return np.stack([polygons2mask(c) for c in new_classes], axis=-1)


def dense2sparse(mask):
  indices = np.nonzero(mask)
  return mask[indices], np.array(indices).T


def crop(image, mask, num_rows, num_cols):
  image_height, image_width = image.shape[:2]
  height, width = image_height // num_rows, image_width // num_cols
  for i in range(num_rows):
    for j in range(num_cols):
      if i == num_rows - 1:
        box = [slice(j * width, None), slice(i * height, None)]
      else:
        box = [
            slice(j * width, (j + 1) * width),
            slice(i * height, (i + 1) * height)
        ]
      yield image[box], mask[box]


def image2dict(image, mask, use_sparse):
  image_size = image.shape[:2]
  image_h, image_w = image_size
  if use_sparse:
    values, indices = dense2sparse(mask)
    return {
        'image_height': [image_h],
        'image_width': [image_w],
        'image': [image.astype('i4').tostring()],
        'mask_values': [values.astype('i4').tostring()],
        'mask_indices': [indices.astype('i4').tostring()]
    }
  else:
    return {
        'image_height': [image_h],
        'image_width': [image_w],
        'image': [image.astype('i4').tostring()],
        'mask': [mask.astype('i4').tostring()],
    }


def pandas2dict(df_row, num_rows, num_cols, use_sparse=False):
  image = tifffile.imread(df_row.image_path).transpose(1, 2, 0)
  image_size = image.shape[:2]
  mask = process_classes(image_size, df_row)

  # create TF example and write into TF record
  for image_patch, mask_patch in crop(image, mask, num_rows, num_cols):
    yield image2dict(image, mask, use_sparse)


def get_save_path(image_path,
                  save_folder,
                  use_sparse,
                  part_index,
                  data_folder='three_band'):
  tf_record_suffix = '-part-{}.tfrecord'.format(part_index)
  if use_sparse: tf_record_suffix = '-sparse{}'.format(tf_record_suffix)
  return image_path.replace(data_folder, save_folder).replace(
      '.tif', tf_record_suffix).replace('_', '-')


def save(df_row, save_folder, num_rows, num_cols, use_sparse=False):
  dictionaries = pandas2dict(df_row, num_rows, num_cols, use_sparse)

  for i, dictionary in enumerate(dictionaries):
    filename = get_save_path(df_row.image_path, save_folder, use_sparse, i + 1)
    print(filename)
    tf_example = to_example(dictionary).SerializeToString()
    writer = tf.python_io.TFRecordWriter(filename)
    writer.write(tf_example)
    writer.close()


def options():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-path', default=DATA_PATH, type=str)
  parser.add_argument('--save-folder', default='tfrecords', type=str)
  parser.add_argument('-nr', '--num-rows', default=1, type=int)
  parser.add_argument('-nc', '--num-cols', default=1, type=int)
  parser.add_argument('--use-sparse', default=False, action='store_true')
  return parser.parse_args()


if __name__ == '__main__':
  args = options()
  get_file = lambda f: '{}/{}'.format(args.data_path, f)
  sentinel_df = load_dataset()
  for i, df_row in sentinel_df.iterrows():
    save(
        df_row,
        save_folder=args.save_folder,
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        use_sparse=args.use_sparse)
