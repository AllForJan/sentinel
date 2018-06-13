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

import folium
from geojson import Polygon
from sentinelsat.sentinel import SentinelAPI, read_geojson, geojson_to_wkt


def get_products(login_json, coordinates, date_start, date_end, download_dir):
  with open(login_json, 'r') as fp:
    LOGIN_INFO = json.load(fp)
  USER_NAME, PASSWORD = list(LOGIN_INFO.values())

  # connect to the API
  api = SentinelAPI(USER_NAME, PASSWORD, 'https://scihub.copernicus.eu/dhus')

  # define a map polygon
  geojson = Polygon(coordinates=coordinates)
  # search by polygon, time, and Hub query keywords
  footprint = geojson_to_wkt(geojson)
  dates = (date_start, date_end)  # (date(2018, 4, 1), date(2018, 4, 11))

  # June to July maps
  products = api.query(
      footprint,
      date=dates,
      platformname='Sentinel-2',
      # producttype='S2MSI2A',
      area_relation='Intersects',  # area of interest is inside footprint
      cloudcoverpercentage=(0, 40))

  # download all results from the search
  api.download_all(products, directory_path=download_dir)
  # product_id = list(products.keys())[0]
  # api.download(id=product_id, directory_path=download_dir)

  # GeoPandas GeoDataFrame with the metadata of the scenes and the footprints as geometries
  return api.to_geodataframe(products)


def add_choropleth(mapobj,
                   gdf,
                   fill_color='Blue',
                   fill_opacity=0.6,
                   line_opacity=0.2,
                   num_classes=5):
  # Convert the GeoDataFrame to WGS84 coordinate reference system
  gdf_wgs84 = gdf.to_crs({'init': 'epsg:4326'})

  # Call Folium choropleth function, specifying the geometry as a the WGS84 dataframe converted to GeoJSON,
  # the data as the GeoDataFrame, the columns as the user-specified id field and and value field.
  # key_on field refers to the id field within the GeoJSON string
  mapobj.choropleth(
      geo_data=gdf_wgs84.to_json(),
      fill_color=fill_color,
      fill_opacity=fill_opacity,
      line_opacity=line_opacity,
  )
  return mapobj


def plot_coordinates(geo_df, product_id):
  geo_df = geo_df.loc[[product_id]][['geometry']]
  folium_map = folium.Map(
      np.mean(coordinates[0], 0).tolist()[::-1], zoom_start=8)
  folium_map = add_choropleth(folium_map, geo_df)
  return folium_map


if __name__ == '__main__':
  get_products(login_json, coordinates, date_start, date_end, download_dir)
