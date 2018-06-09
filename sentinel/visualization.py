# coding: utf - 8
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as plt_polygon
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def show_slices(batches, scan_indices, ns_slice, grid=True, **kwargs):
  """ Plot slice with number n_slice from scan with index given by scan_index from batch
    """
  font_caption = {
      'family': 'serif',
      'color': 'darkred',
      'weight': 'normal',
      'size': 18
  }
  font = {'family': 'serif', 'color': 'darkred', 'weight': 'normal', 'size': 15}

  # fetch some arguments, make iterables out of args
  def iterize(arg):
    return arg if isinstance(arg, (list, tuple)) else (arg,)

  components = kwargs.get('components', 'images')
  batches, scan_indices, ns_slice, components = [
      iterize(arg) for arg in (batches, scan_indices, ns_slice, components)
  ]
  clims = kwargs.get('clims', (-1200, 300))
  clims = clims if isinstance(clims[0], (tuple, list)) else (clims,)

  # lengthen args
  n_boxes = max(len(arg) for arg in (batches, scan_indices, ns_slice, clims))

  def lengthen(arg):
    return arg if len(arg) == n_boxes else arg * n_boxes

  batches, scan_indices, ns_slice, clims, components = [
      lengthen(arg)
      for arg in (batches, scan_indices, ns_slice, clims, components)
  ]

  # plot slices
  _, axes = plt.subplots(1, n_boxes, squeeze=False, figsize=(10, 4 * n_boxes))

  zipped = zip(
      range(n_boxes), batches, scan_indices, ns_slice, clims, components)

  for i, batch, scan_index, n_slice, clim, component in zipped:
    slc = batch.get(scan_index, component)[n_slice]
    axes[0][i].imshow(slc, cmap=plt.cm.gray, clim=clim)
    axes[0][i].set_xlabel('Shape: {}'.format(slc.shape[1]), fontdict=font)
    axes[0][i].set_ylabel('Shape: {}'.format(slc.shape[0]), fontdict=font)
    title = 'Scan' if component == 'images' else 'Mask'
    axes[0][i].set_title(
        '{} #{}, slice #{} \n \n'.format(title, scan_index, n_slice),
        fontdict=font_caption)
    axes[0][i].text(
        0.2,
        -0.25,
        'Total slices: {}'.format(len(batch.get(scan_index, component))),
        fontdict=font_caption,
        transform=axes[0][i].transAxes)

    # set inverse-spacing grid
    if grid:
      inv_spacing = 1 / batch.get(scan_index, 'spacing').reshape(-1)[1:]
      step_mult = 50
      xticks = np.arange(0, slc.shape[0], step_mult * inv_spacing[0])
      yticks = np.arange(0, slc.shape[1], step_mult * inv_spacing[1])
      axes[0][i].set_xticks(xticks, minor=True)
      axes[0][i].set_yticks(yticks, minor=True)
      axes[0][i].set_xticks([], minor=False)
      axes[0][i].set_yticks([], minor=False)

      axes[0][i].grid(color='r', linewidth=1.5, alpha=0.5, which='minor')

  plt.show()


def discrete_cmap(N, base_cmap=None):
  """Create an N-bin discrete colormap from the specified input map"""
  base = plt.cm.get_cmap(base_cmap)
  color_list = base(np.linspace(0, 1, N))
  cmap_name = base.name + str(N)
  return base.from_list(cmap_name, color_list, N)


def colorbar(mappable, ticks, labels=None):
  mappable.set_clim(-0.5, len(ticks) - 0.5)
  ax = mappable.axes
  fig = ax.figure
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="5%", pad=0.05)
  cbar = fig.colorbar(mappable, cax=cax, ticks=ticks)
  if labels is not None: cbar.ax.set_yticklabels(labels)
  return cbar