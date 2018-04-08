# Analýza histórie povrchu pozemku

# Copenicus
* [Copernicus mapa, kde sa zobrazujú dostupné snímky pre územie/obdobie](https://scihub.copernicus.eu/dhus/#/home)
* [Popis data zdrojov zo satelitov Sentinel](https://sentinel.esa.int/web/sentinel/sentinel-data-access/typologies-and-services)



Odkazy:
* [QGis](https://www.qgis.org/en/site/)
* [Návod na Semi-Automatic Classification Plugin s vysvetlením vrstiev](https://fromgistors.blogspot.com/2015/10/search-download-sentinel-2.html)
* [Copernicus API návod](https://scihub.copernicus.eu/userguide/5APIsAndBatchScripting)
* [Návod na Semi-Automatic Classification Plugin - Crop monitoring](http://www.digital-geography.com/using-sentinel-2-for-crop-monitoring/)
* [Dokumentácia k Sentinel snímkom](https://earth.esa.int/web/sentinel/technical-guides/sentinel-2-msi/level-2a/product-formatting)
* [YouTube návod na Sentinel-2 Crop monitoring](https://www.youtube.com/watch?v=1T7oN5_BURA)
* [YouTube návod na Sentinel-2 WildFire Events](https://www.youtube.com/watch?v=f6O5A8RSM0c)
* [YouTube návod na Sentinel-2 - sťahovanie dát](https://www.youtube.com/watch?v=AFXfh7zCBxY)
* [NDVI výpočet v Pythone](http://neondataskills.org/HDF5/calc-ndvi-python/)


# Ďaľšie odkazy

* [Mapa zobrazujúca pokrytie územia zeleňou z roku 2015](https://land.copernicus.eu/pan-european/high-resolution-layers/forests/tree-cover-density/status-maps/2015/view), dostupné je aj z roku 2012




# Application Specific Data
* Spustenie výpočtu
```
("L2A_T33UYP_20180324T095031_B8A_20m" - "L2A_T33UYP_20180324T095031_B11_20m")/( "L2A_T33UYP_20180324T095031_B8A_20m" + "L2A_T33UYP_20180324T095031_B11_20m")
```

```
(B8A - B11)/( B8A +B11 )
```

* Príklad vyhľadávania v území Nitra s určením času, oblačnosti, satelitu

```
https://scihub.copernicus.eu/dhus/search?q=(%20footprint:%22Intersects(POLYGON((18.025318309126252%2048.25365988314252,17.945005267006067%2048.18825958525136,18.043165651819628%2048.18825958525136,18.025318309126252%2048.25365988314252,18.025318309126252%2048.25365988314252)))%22)%20AND%20(platformname:Sentinel-2%20AND%20filename:S2A_*%20AND%20producttype:S2MSI2Ap)%20AND%20cloudcoverpercentage:[0%20TO%2040]
```
