# coding: utf - 8
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from mpl_toolkits.basemap import Basemap

from shapely.geometry import Polygon, MultiLineString
from sentinelhub import CRS, transform_bbox


def show_area(area_shape, area_buffer=0.3):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  minx, miny, maxx, maxy = area_shape.bounds
  lng, lat = (minx + maxx) / 2, (miny + maxy) / 2

  m = Basemap(projection='ortho', lat_0=lat, lon_0=lng, resolution='l')
  m.drawcoastlines()
  m.bluemarble()

  if isinstance(area_shape, Polygon):
    area_shape = [area_shape]
  for polygon in area_shape:
    x, y = np.array(polygon.boundary)[0]
    m_poly = []
    for x, y in np.array(polygon.boundary):
      m_poly.append(m(x, y))
    ax.add_patch(
        plt_polygon(
            np.array(m_poly), closed=True, facecolor='red', edgecolor='red'))

  plt.tight_layout()
  plt.show()


def show_splitter(splitter, alpha=0.2, area_buffer=0.2, show_legend=False):
  area_bbox = splitter.get_area_bbox()
  minx, miny, maxx, maxy = area_bbox
  lng, lat = area_bbox.get_middle()
  w, h = maxx - minx, maxy - miny
  minx = minx - area_buffer * w
  miny = miny - area_buffer * h
  maxx = maxx + area_buffer * w
  maxy = maxy + area_buffer * h

  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(111)

  base_map = Basemap(
      projection='mill',
      lat_0=lat,
      lon_0=lng,
      llcrnrlon=minx,
      llcrnrlat=miny,
      urcrnrlon=maxx,
      urcrnrlat=maxy,
      resolution='l',
      epsg=4326)
  base_map.drawcoastlines(color=(0, 0, 0, 0))

  area_shape = splitter.get_area_shape()
  if isinstance(area_shape, Polygon):
    area_shape = [area_shape]
  for polygon in area_shape:
    if isinstance(polygon.boundary, MultiLineString):
      for linestring in polygon.boundary:
        ax.add_patch(
            plt_polygon(
                np.array(linestring),
                closed=True,
                facecolor=(0, 0, 0, 0),
                edgecolor='red'))
    else:
      ax.add_patch(
          plt_polygon(
              np.array(polygon.boundary),
              closed=True,
              facecolor=(0, 0, 0, 0),
              edgecolor='red'))

  bbox_list = splitter.get_bbox_list()
  info_list = splitter.get_info_list()

  cm = plt.get_cmap('jet', len(bbox_list))
  legend_shapes = []
  for i, (bbox, info) in enumerate(zip(bbox_list, info_list)):
    wgs84_bbox = transform_bbox(bbox, CRS.WGS84).get_polygon()

    tile_color = tuple(list(cm(i))[:3] + [alpha])
    ax.add_patch(
        plt_polygon(
            np.array(wgs84_bbox),
            closed=True,
            facecolor=tile_color,
            edgecolor='green'))

    if show_legend:
      legend_shapes.append(plt.Rectangle((0, 0), 1, 1, fc=cm(i)))

  if show_legend:
    plt.legend(legend_shapes, [
        '{},{}'.format(info['index_x'], info['index_y']) for info in info_list
    ])
  plt.tight_layout()
  plt.show()
