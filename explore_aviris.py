import rasterio
from rasterio.mask import mask
import geopandas as gpd

inRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_refl/f150602t01p00r17_corr_v1'
inshp = '/home/marie/cave/ecostress/data/mtbs/reprojected/caveGeo.shp'
outRas = '/home/marie/alisal/aviris/data/flights/f150602t01p00r17_corr_v1_clip'
vector = gpd.read_file(inshp)


# vector = vector[vector['HYBAS_ID'] == 6060122060]  # Subsetting to my AOI

with rasterio.open(inRas) as src:
    vector = vector.to_crs(src.crs)
    # print(Vector.crs)
    out_image, out_transform = mask(src, vector.geometry, crop=True)
    out_meta = src.meta.copy()  # copy the metadata of the source DEM

out_meta.update({
    "driver": "ENVI",
    "height": out_image.shape[1],  # height starts with shape[1]
    "width": out_image.shape[2],  # width starts with shape[2]
    "transform": out_transform
})

with rasterio.open(outRas, 'w', **out_meta) as dst:
    dst.write(out_image)

# foo = 1
# dataset = rasterio.open()
# cv = gpd.read_file(cshp)
# cv2 = cv.to_crs(epsg=32611)
# from rasterio.mask import mask
# out = mask(dataset, cv2.geometry)
# from rasterio.plot import show
# show((out, 1));
# out.write()
# foo = 1