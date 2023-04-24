import rasterio
from rasterio.mask import mask
import geopandas as gpd

# Objective: crop ENVI reflectance flight of interest to fire perimeter (changed to geotiff - doesn't seem to matter)

inRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_refl/f150602t01p00r17_corr_v1' # June 22 2015
# inRas = '/media/marie/Data01/data/hyperspec/cave/f150416t01p00r11_refl/f150416t01p00r11_corr_v1' # April 16th 2015
# inshp = '/home/marie/cave/ecostress/data/mtbs/reprojected/caveGeo.shp'

# inRas = '/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr'
# inshp = '/media/marie/Data01/data/MTBS/jesusita/mtbs/2009/ca3447411972520090505_20080623_20100613_burn_bndy_rprj.shp'
# inshp = '/home/marie/alisal/aviris/data/unburn_cave_sample.shp'
# inshp = '/home/marie/alisal/aviris/data/jesusita/burn_sample_jesusita.shp'
inshp = '/media/marie/Data01/data/MTBS/paint/ca3446911978419900627_19890705_19910828_burn_bndy_rprj.shp' # paint fire

# outRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_corr_clip_tif' # June 22 2015
# outRas = '/media/marie/Data01/data/hyperspec/cave/f150416t01p00r11_refl/f150416t01p00r11_corr_tif' # April 16th 2015
# outRas = '/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip' # Jesusita
# outRas = '/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip_cave' # Jesusita
# outRas = '/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip_unburn_cave_sample'
# outRas = '/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip_burn_sample_jesusita'
# outRas = '/media/marie/Data01/data/hyperspec/cave/f150416t01p00r11_refl/f150416t01p00r11_corr_v1_paint_fire'
outRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_refl/f150602t01p00r17_corr_v1_paint_fire'
vector = gpd.read_file(inshp)

# r = rasterio.open(inRas)

# foo = 1

with rasterio.open(inRas) as src:
    vector = vector.to_crs(src.crs)
    out_image, out_transform = mask(src, vector.geometry, crop=True, nodata=-9999) # Changing no data value that is within uint16 65535 or for int16 -9999
    out_meta = src.meta.copy()
    descriptions = src.descriptions
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],  # height starts with shape[1]
    "width": out_image.shape[2],  # width starts with shape[2]
    "transform": out_transform,
    "nodata": -9999,
})

with rasterio.open(outRas, 'w', **out_meta) as dst:
    dst.descriptions = descriptions
    dst.write(out_image)
