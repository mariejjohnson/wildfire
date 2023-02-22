import os
import subprocess

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

# gdalwarp -cutline INPUT.shp INPUT.tif OUTPUT.tif
GDAL = '/usr/bin/ogr2ogr'

def clip_files(shape, clips, out_dir):
    for i, clip in enumerate(clips):
        out = os.path.join(out_dir, '{}.shp'.format(i))
        command = [GDAL, '-clipsrc', clip, out, shape]
        subprocess.check_call(command)

def write_shp(csv, shp):
    df = pd.read_csv(csv)
    df['geometry'] = [np.nan for _ in range(df.shape[0])]
    lat_cols = [c for c in df.columns if 'Lat' in c]
    lon_cols = [c for c in df.columns if 'Lon' in c]
    for i, r in df.iterrows():
        lats = list(r[lat_cols])
        lons = list(r[lon_cols])
        poly = Polygon(zip(lons, lats))
        df.loc[i, 'geometry'] = poly
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326')
    gdf.to_file(shp)


def filter_shp(shape, out):
    df = gpd.read_file(shape)
    df = df[['Name', 'Date', 'geometry']]
    df['Date'] = [pd.to_datetime(x) for x in df['Date']]
    df.index = df['Name']
    for month in range(1, 12):
        matches = [i for i, r in df.iterrows() if r['Date'].month == month]
        mdf = df.loc[matches, ['Name', 'Date', 'geometry']].copy()
        mdf.to_csv(out)

def filter_date(shape, out): # This really isn't a filter it's just all the flights regardless of month
    df = gpd.read_file(shape)
    df = df[['Name', 'Date']]
    df['Date'] = [pd.to_datetime(x) for x in df['Date']]
    df.index = df['Name']
    for month in range(1, 12):
        matches = [i for i, r in df.iterrows() if r['Date'].month == month]
        mdf = df.loc[matches, ['Name', 'Date']].copy()
        mdf.to_csv(out)

if __name__ == '__main__':
    c = '/home/marie/alisal/aviris/data/aviris_flights_2006_2021.csv'
    all_flights = '/home/marie/alisal/aviris/data/aviris_flights_2006_2021.shp'
    # filter_shp(c, all_flights)


    clips = ['/home/marie/alisal/aviris/data/burns/ca3451712013120211011_20211003_20211122_burn_bndy_rprj.shp', # Alisal
'/home/marie/alisal/aviris/data/burns/caveGeo.shp'] # Cave fire
    # clip_files(all_flights, clips, '/home/marie/alisal/aviris/data')

    # shp = '/home/marie/alisal/aviris/data/1.shp' # Cave fire
    shp = '/home/marie/alisal/aviris/data/0.shp' # Alisal
    # oshp = '/home/marie/alisal/aviris/data'
    # icsv = '/home/marie/alisal/aviris/data/intersect_cave.csv' # matches exact geometry intersection
    icsv = '/home/marie/alisal/aviris/data/intersect_alisal.csv'  # matches exact geometry intersection
    # date_csv = '/home/marie/alisal/aviris/data/any_cave.csv'
    filter_shp(shp, icsv)# Matches exact geometry intersection
    # filter_date(shp, date_csv) # this is just all flights within the cave fire, I can also just covert the shapefile to df in R

