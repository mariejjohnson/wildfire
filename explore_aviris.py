import rasterio
from rasterio.mask import mask
import geopandas as gpd

inRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_refl/f150602t01p00r17_corr_v1'
inshp = '/home/marie/cave/ecostress/data/mtbs/reprojected/caveGeo.shp'
outRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_corr_v1_clip'
out_df = '/home/marie/alisal/aviris/data/flights/f_15.csv' # export as data frame?
vector = gpd.read_file(inshp)

with rasterio.open(inRas) as src:
    vector = vector.to_crs(src.crs)
    out_image, out_transform = mask(src, vector.geometry, crop=True)
    out_meta = src.meta.copy()

out_meta.update({
    "driver": "ENVI",
    "height": out_image.shape[1],  # height starts with shape[1]
    "width": out_image.shape[2],  # width starts with shape[2]
    "transform": out_transform
})

with rasterio.open(outRas, 'w', **out_meta) as dst:
    dst.write(out_image)

# I would like to plot the values at a single location from the multiband
# raster to explore these data.
# Thoughts
# 1. convert raster to data frame and plot that way (what I would do in R)
# 2. See if there is a function in rasterio that might do this? or geopandas
# 3. See if Dave has some slick code for doing this ...
# # It's hyperspectral data so you get a spectra (well technically
# you'd get multiple spectral signatures for one location and then do
# some unmixing ... I think).