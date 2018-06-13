# coding: utf-8

import os
import json
from glob import glob
from datetime import date
import subprocess
from zipfile import ZipFile

import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def extract_files(filename, extract_dir):
  # filename = '../data/S2B_MSIL2A_20180408T095029_N0207_R079_T33UYP_20180408T115858.zip'
  zip_file = ZipFile(filename)

  _is_wave_band_img = lambda x: 'B' in x.filename and 'IMG_DATA' in x.filename
  band_files = [x for x in zip_file.infolist() if _is_wave_band_img(x)]
  zip_file.extractall(members=band_files, path=extract_dir)


def get_files(filename):
  # filename = '../data/S2B_MSIL2A_20180408T095029_N0207_R079_T33UYP_20180408T115858'
  filename += '.SAFE'
  files = glob(
      # '../data/S2B_MSIL2A_20180408T095029_N0207_R079_T33UYP_20180408T115858.SAFE/GRANULE/L2A_T33UYP_A005682_20180408T095239/IMG_DATA/*'
      os.path.join(filename, 'GRANULE', '*', 'IMG_DATA', '*'))
  files = sorted(files)
  return dict(
      zip(['10m', '20m', '60m'],
          [sorted(glob(os.path.join(x, '*'))) for x in files]))


def convert_to_geotiff(input_path, output_path, ot):
  cmd = "gdal_translate -ot {ot} -of GTiff -co TILED=YES {input} {output}"
  cmd = cmd.format(ot=ot, input=input_path, output=output_path)
  return cmd.split(" ")


def get_commands(vrt_file, tif_file, file_list, set_resolution=False,
                 ot='Byte'):
  # Merge bands
  # merge_cmd = "gdalbuildvrt -separate {0} {1}".format(out_vrt_file, " ".join(file_list))
  merge_cmd_list = "gdalbuildvrt -separate".split(" ")
  resolution_cmd_list = "-resolution user -tr 20 20".split(" ")
  if set_resolution: merge_cmd_list += resolution_cmd_list
  vrt_cmd = merge_cmd_list + [vrt_file] + file_list
  # Convert to uncompressed GeoTiff
  # trsl_cmd = "gdal_translate -ot Byte -of GTiff -co TILED=YES {0} {1}".format(out_vrt_file, out_tif_file)
  trsl_cmd_list = convert_to_geotiff(vrt_file, tif_file, ot)
  return vrt_cmd, trsl_cmd_list


def run_command(cmd_list):
  p = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
  print(p.communicate())


def read_tiff(filename):
  tiff_image = tiff.imread(filename)
  print(tiff_image.shape, tiff_image.min(), tiff_image.max())
  return tiff_image


def calculate_ndvi(image):
  rval = np.divide(image[..., bands['B8A']] - image[..., bands['B11']],
                   image[..., bands['B8A']] + image[..., bands['B11']])
  return np.where(np.isnan(rval), 0., rval)


def main(argv=None):
  out_path = '../data/S2B_MSIL2A_20180408T095029_N0207_R079_T33UYP_20180408T115858_PROCESSED/'
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_path = os.path.abspath(out_path)

  # RGB bands
  # out_tci_file = os.path.join(out_path, 'TCI.tif')
  # in_tci_file = file_dict['20m'][-2]
  # rgb_cmd_list = convert_to_geotiff(in_tci_file, out_tci_file, ot='Byte')
  # run_command(rgb_cmd_list)
  # tci_tif = read_tiff(out_tci_file)
  # plt.figure(figsize=[10, 10])
  # plt.imshow(tci_tif)

  # NDVI bands
  nvdi_vrt_file = os.path.join(out_path, 'nvdi.vrt')
  nvdi_tif_file = os.path.join(out_path, 'nvdi.tif')
  nvdi_files = [x for x in file_dict['20m'] if 'B11' in x or 'B8A' in x]
  nvdi_cmd_list = get_commands(
      nvdi_vrt_file,
      nvdi_tif_file,
      nvdi_files,
      ot='Int16',
      set_resolution=False)

  run_command(nvdi_cmd_list[0])  # merge to vrt
  run_command(nvdi_cmd_list[1])  # convert to tif

  # ndvi_bands = dict(zip(['B8A', 'B11'], [1, 0]))
  ndvi_tif = read_tiff(nvdi_tif_file)
  ndvi = calculate_ndvi(ndvi_tif)
  print(ndvi.min(), ndvi.max())

  if plot_ndvi:
    fig, axs = plt.subplots(1, 2, figsize=[15, 10])
    axs[0].imshow(ndvi)
    img = axs[1].imshow(np.float32(ndvi > 0.5))

    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(img, cax=cax)

    fig, ax = plt.subplots(1, 1, figsize=[10, 10])
    ax.imshow(tci_tif)
    ax.imshow(np.float32(ndvi > 0.3), alpha=0.6)


if __name__ == '__main__':
  extract_files(filename, extract_dir)
  get_files(filename)
  main()
