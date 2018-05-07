# coding: utf - 8
import os
import json
from datetime import date

import pandas as pd

from shapely.geometry import shape as geo_shape
from shapely.geometry import Polygon

from sentinelsat.sentinel import SentinelAPI, geojson_to_wkt

from sentinelhub.constants import CRS, DataSource
from sentinelhub.common import BBox
from sentinelhub.areas import AreaSplitter, BBoxSplitter

LOGIN_FILE = os.path.join(*__file__.split('/')[:-1])
LOGIN_FILE = os.path.join(LOGIN_FILE, 'sentinel_user_login.json')
LOGIN_FILE = os.path.abspath(LOGIN_FILE)
SENTINELHUB_URL = 'https://scihub.copernicus.eu/dhus'

L1C_PRODUCT, L2A_PRODUCT = 'S2MSI1C', 'S2MSI2A'
L1C_PERIOD = (date(2007, 1, 1), date(2017, 5, 2))  # Sentinel 2 Level 1C
L2A_PERIOD = (date(2017, 5, 2), 'NOW')
INTERSECTS, CONTAINS, WITHIN = 'Intersects', 'Contains', 'IsWithin'


def get_sentinel_products(footprint,
                          dates,
                          producttype=L1C_PRODUCT,
                          area_relation=INTERSECTS):
  """  area_relation='Intersects',  # area of interest intersects footprint
  area_relation='Contains',  # area of interest is inside footprint
  area_relation='IsWithin',  # footprint is inside area of interest
  """
  with open(LOGIN_FILE, 'r') as fp:
    login_info = json.load(fp)
  user_name, password = list(login_info.values())
  api = SentinelAPI(user_name, password, SENTINELHUB_URL)

  products = api.query(
      footprint,
      date=dates,
      platformname='Sentinel-2',
      producttype=producttype,
      area_relation=area_relation,
      cloudcoverpercentage=(0, 40))

  return api.to_geodataframe(products)


def get_time_intervals(start=date(2017, 5, 2), end='NOW', freq='1W-MON'):
  periods = pd.date_range(start=start, end=end, freq=freq).to_pydatetime()
  periods = [x.date() for x in periods]
  return list(zip(periods[:-1], periods[1:]))


class TileSplitter(AreaSplitter):
  """A tool that splits the given area into smaller parts. Given the area, time interval and data source it collects
    info from Sentinel Hub WFS service about all satellite tiles intersecting the area. For each of them it calculates
    bounding box and if specified it splits these bounding boxes into smaller bounding boxes. Then it filters out the
    ones that do not intersect the area. If specified by user it can also reduce the sizes of the remaining bounding
    boxes to best fit the area.
    :param shape_list: A list of geometrical shapes describing the area of interest
    :type shape_list: list(shapely.geometry.multipolygon.MultiPolygon or shapely.geometry.polygon.Polygon)
    :param crs: Coordinate reference system of the shapes in `shape_list`
    :type crs: sentinelhub.constants.CRS
    :param time_interval: Interval with start and end date of the form YYYY-MM-DDThh:mm:ss or YYYY-MM-DD
    :type time_interval: (str, str)
    :param tile_split_shape: Parameter that describes the shape in which the satellite tile bounding boxes will be
                             split. It can be a tuple of the form `(n, m)` which means the tile bounding boxes will be
                             split into `n` columns and `m` rows. It can also be a single integer `n` which is the same
                             as `(n, n)`.
    :type split_shape: int or (int, int)
    :param data_source: Source of requested satellite data. Default is Sentinel-2 L1C data.
    :type data_source: sentinelhub.constants.DataSource
    :param instance_id: User's Sentinel Hub instance id. If ``None`` the instance id is taken from the ``config.json``
                        configuration file.
    :type instance_id: str
    :param reduce_bbox_sizes: If True it will reduce the sizes of bounding boxes so that they will tightly fit the given
           area geometry from `shape_list`.
    :type reduce_bbox_sizes: bool
    """

  def __init__(self,
               shape_list,
               time_interval,
               crs=CRS.WGS84,
               tile_split_shape=1,
               data_source=DataSource.SENTINEL2_L1C,
               instance_id=None,
               **kwargs):
    super(TileSplitter, self).__init__(shape_list, crs, **kwargs)

    if data_source is DataSource.DEM:
      raise ValueError(
          'This splitter does not support splitting area by DEM tiles. Please specify some other '
          'DataSource')

    self.time_interval = time_interval
    self.tile_split_shape = tile_split_shape
    self.data_source = data_source
    self.instance_id = instance_id

    self.tile_dict = None

    self._make_split()

  def get_dates(self, gpd):
    """ Returns a list of acquisition times from tile info data
      :return: List of acquisition times in the order returned by WFS service.
      :rtype: list(datetime.datetime)
      """
    return [str(timestamp) for timestamp in gpd.ingestiondate]

  def get_geometries(self, gpd):
    """ Returns a list of geometries from tile info data
      :return: List of multipolygon geometries in the order returned by WFS service.
      :rtype: list(shapely.geometry.MultiPolygon)
      """
    return [geo_shape(geom) for geom in gpd.geometry]

  def _make_split(self):
    """This method makes the split
        """
    self.tile_dict = {}

    footprint = self.area_shape.envelope.boundary.to_wkt()
    products_gpd = get_sentinel_products(footprint, self.time_interval)

    geom_wgs84 = products_gpd.geometry.to_crs({'init': 'epsg:4326'})
    bboxes_df = geom_wgs84.envelope.bounds
    uuids = bboxes_df.index.values.tolist()
    bboxes = bboxes_df.values.tolist()

    date_list = self.get_dates(products_gpd)
    geometry_list = self.get_geometries(products_gpd)

    zipped = zip(uuids, bboxes, zip(date_list, geometry_list))
    for tile_name, bbox, (date, geometry) in zipped:
      if tile_name not in self.tile_dict:
        self.tile_dict[tile_name] = {
            'bbox': BBox(bbox, crs=self.crs),
            'times': [],
            'geometries': []
        }
      self.tile_dict[tile_name]['times'].append(date)
      self.tile_dict[tile_name]['geometries'].append(geometry)

    self.tile_dict = {
        tile_name: tile_props
        for tile_name, tile_props in self.tile_dict.items()
        if self._intersects_area(tile_props['bbox'])
    }

    self.bbox_list = []
    self.info_list = []

    for tile_name, tile_info in self.tile_dict.items():
      tile_bbox = tile_info['bbox']
      bbox_splitter = BBoxSplitter(
          [Polygon(tile_bbox.get_polygon())],
          tile_bbox.get_crs(),
          split_shape=self.tile_split_shape,
          reduce_bbox_sizes=self.reduce_bbox_sizes)

      for bbox, info in zip(bbox_splitter.get_bbox_list(),
                            bbox_splitter.get_info_list()):
        if self._intersects_area(bbox):
          info['tile'] = tile_name

          self.bbox_list.append(bbox)
          self.info_list.append(info)

  def get_tile_dict(self):
    """Returns the dictionary of satellite tiles intersecting the area geometry. For each tile they contain info
        about their bounding box and lists of acquisitions and geometries
        :return: Dictionary containing info about tiles intersecting the area
        :rtype: dict
        """
    return self.tile_dict
