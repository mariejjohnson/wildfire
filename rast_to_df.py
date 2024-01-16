import pandas as pd
import rasterio as rio
import rioxarray
import geopandas as gpd
rds = rioxarray.open_rasterio('/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip_burn_sample_jesusita')
rds.name = "data"
df = rds.squeeze().to_dataframe().reset_index()
geometry = gpd.points_from_xy(df.x, df.y)
gdf = gpd.GeoDataFrame(df, crs=rds.rio.crs, geometry=geometry)
df = pd.DataFrame(gdf)
df.to_csv('/media/marie/Data01/data/hyperspec/jesusita/f130606t01p00r07_refl/f130606t01p00r07rdn_refl_img_corr_clip_burn_sample_jesusita.csv')