import rasterio
from rasterio.mask import mask
import geopandas as gpd

# inRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_refl/f150602t01p00r17_corr_v1' # June 22 2015
inRas = '/media/marie/Data01/data/hyperspec/cave/f150416t01p00r11_refl/f150416t01p00r11_corr_v1'
inshp = '/home/marie/cave/ecostress/data/mtbs/reprojected/caveGeo.shp'
# outRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_corr_v1_clip_v2' # June 22 2015
outRas = '/media/marie/Data01/data/hyperspec/cave/f150416t01p00r11_refl/f150416t01p00r11_corr_v1_clip'
vector = gpd.read_file(inshp)

with rasterio.open(inRas) as src:
    vector = vector.to_crs(src.crs)
    out_image, out_transform = mask(src, vector.geometry, crop=True, nodata=-9999)
    out_meta = src.meta.copy()
    descriptions = src.descriptions
out_meta.update({
    "driver": "ENVI",
    "height": out_image.shape[1],  # height starts with shape[1]
    "width": out_image.shape[2],  # width starts with shape[2]
    "transform": out_transform,
    "nodata": -9999,
})

with rasterio.open(outRas, 'w', **out_meta) as dst:
    dst.descriptions = descriptions
    dst.write(out_image)
