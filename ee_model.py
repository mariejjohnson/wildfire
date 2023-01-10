import os
import sys
from copy import deepcopy
from pprint import pprint
from datetime import date

import ee
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from pandas import read_csv, concat, DataFrame
from sklearn.ensemble import RandomForestRegressor
from geopandas import GeoDataFrame
from shapely.geometry import Point

from ee_utils import get_world_climate # Is this actually necessary? Will I use climate

sys.path.insert(0, os.path.abspath('..'))
abspath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(abspath)


def prep_extracts(c, out_c): # features, dropna
    print(os.path.basename(c))
    df = pd.read_csv(c)
    print(df.shape)
    df.dropna(inplace=True)
    d = ['system:index', '.geo', 'id']
    target = ['b1']
    selected_features = ['aspect',
                         'awc',
                         'clay',
                         'elevation',
                         'ksat',
                         'lat',
                         'lon', 'sand',
                         'slope', 'tpi_1250',
                         'tpi_150',
                         'tpi_250']
    df.drop(columns=d, inplace=True)
    df = df[target + selected_features]
    df = df[df['b1'] > 0]
    print(df.shape)
    df.to_csv(out_c, index=False)


def random_forest(csv, n_estimators=150, out_shape=None, show_importance=False):
    df = read_csv(csv, engine='python').sample(frac=1.0).reset_index(drop=True)
    df = df.loc[df['b1'] > 0]
    split = int(df.shape[0] * 0.7)
    train = deepcopy(df.loc[:split, :])
    val = deepcopy(df.loc[split:, :])

    target = 'b1'
    train.dropna(axis=0, inplace=True)
    y = train[target].values.astype(float)
    drop = ['system:index', '.geo', 'b1', 'id']
    train.drop(columns=drop, inplace=True)
    x = train.values
    features = list(train.columns)
    geo = val.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
    val_df = deepcopy(train)

    val.dropna(axis=0, inplace=True)
    y_test = val[target].values
    val.drop(columns=drop, inplace=True)
    x_test = val.values

    rf = RandomForestRegressor(n_estimators=n_estimators,
                               n_jobs=-1,
                               bootstrap=True)

    rf.fit(x, y)

    if show_importance:
        _list = [(f, v) for f, v in zip(list(train.columns), rf.feature_importances_)]
        imp = sorted(_list, key=lambda x: x[1], reverse=True)
        print([f[0] for f in imp[:10]])

    y_pred = rf.predict(x_test)
    lr = linregress(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print('r: {:.3f}, fractional: {:.3f}, rmse: {:.3f}\n'.format(lr.rvalue, rmse / y_test.mean(), rmse))
    print('predicted {}'.format(target))
    print('predicted on {} features: '.format(len(features)))
    pprint(features)

    if out_shape:
        gdf = GeoDataFrame(val_df, geometry=geo, crs='EPSG:4326')
        gdf.to_file(out_shape)
    return


def export_prediction(out_name, table, asset_root, region, years,
                      export='asset', bag_fraction=0.5):
    """
    Trains a Random Forest classifier using a training table input, creates a stack of raster images of the same
    features, and classifies it.  I run this over a for-loop iterating state by state.
    :param region:
    :param asset_root:
    :param out_name:
    :param asset:
    :param export:
    :param bag_fraction:
    :return:
    """
    is_authorized()
    fc = ee.FeatureCollection(table)
    roi = ee.FeatureCollection(region)

    classifier = ee.Classifier.smileRandomForest(
        numberOfTrees=50,
        minLeafPopulation=5,
        bagFraction=bag_fraction).setOutputMode('REGRESSION')

    input_props = fc.first().propertyNames().remove('system:index')
    trained_model = classifier.train(fc, 'b1', input_props)
    print()

    for yr in years:
        input_bands = stack_bands(yr, roi)

        b, p = input_bands.bandNames().getInfo(), input_props.getInfo()
        check = [x for x in p if x not in b]
        if check:
            pprint(check)
            revised = [f for f in p if f not in check]
            input_props = ee.List(revised)
            trained_model = classifier.train(fc, 'b1', input_props)

        annual_stack = input_bands.select(input_props)
        classified_img = annual_stack.unmask().classify(trained_model).int().set({
            'system:index': ee.Date('{}-01-01'.format(yr)).format('YYYYMMdd'),
            'system:time_start': ee.Date('{}-01-01'.format(yr)).millis(),
            'system:time_end': ee.Date('{}-12-31'.format(yr)).millis(),
            'date_ingested': str(date.today()),
            'image_name': out_name,
            'training_data': table,
            'bag_fraction': bag_fraction})

        classified_img = classified_img.clip(roi.geometry())

        if export == 'asset':
            task = ee.batch.Export.image.toAsset(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                assetId=os.path.join(asset_root, '{}_{}'.format(out_name, yr)),
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)

        elif export == 'cloud':
            task = ee.batch.Export.image.toCloudStorage(
                image=classified_img,
                description='{}_{}'.format(out_name, yr),
                bucket='wudr',
                fileNamePrefix='{}_{}'.format(yr, out_name),
                scale=30,
                pyramidingPolicy={'.default': 'mode'},
                maxPixels=1e13)
        else:
            raise NotImplementedError('choose asset or cloud for export')

        task.start()
        print(os.path.join(asset_root, '{}_{}'.format(out_name, yr)))


def request_band_extract(file_prefix, points_layer, region, years, diagnose=False):
    """
    Extract raster values from a points kml file in Fusion Tables. Send annual extracts .csv to GCS wudr bucket.
    Concatenate them using map.tables.concatenate_band_extract().
    :param region:
    :param points_layer:
    :param file_prefix:
    :param filter_bounds: Restrict extract to within a geographic extent.
    :return:
    """
    # is_authorized()
    roi = ee.FeatureCollection(region)
    plots = ee.FeatureCollection(points_layer)
    for yr in years:
        stack = stack_bands(yr, roi)
        # if tables are coming out empty, use this to find missing bands
        if diagnose:
            filtered = ee.FeatureCollection([plots.first()])
            bad_ = []
            bands = stack.bandNames().getInfo()
            for b in bands:
                stack_ = stack.select([b])

                def sample_regions(i, points):
                    red = ee.Reducer.toCollection(i.bandNames())
                    reduced = i.reduceRegions(points, red, 30, stack_.select(b).projection())
                    fc = reduced.map(lambda f: ee.FeatureCollection(f.get('features'))
                                     .map(lambda q: q.copyProperties(f, None, ['features'])))
                    return fc.flatten()

                data = sample_regions(stack_, filtered)
                try:
                    print(b, data.getInfo()['features'][0]['properties'][b])
                except Exception as e:
                    print(b, 'not there', e)
                    bad_.append(b)
            print(bad_)
            return None

        desc = '{}_{}'.format(file_prefix, yr)
        plot_sample_regions = stack.sampleRegions(
            collection=plots,
            properties=['id'],
            scale=30,
            tileScale=16)

        task = ee.batch.Export.table.toCloudStorage(
            plot_sample_regions,
            description=desc,
            bucket='ggedi',
            fileNamePrefix=desc,
            fileFormat='CSV')

        task.start()
        print(desc)


def stack_bands(yr, roi):
    """
    Create a stack of bands for the year and region of interest specified.
    :param yr:
    :param southern
    :param roi:
    :return:
    """
    # is_authorized()
    input_bands = ee.Image.pixelLonLat().rename(['lon', 'lat'])
    ned = ee.Image('USGS/NED')
    terrain = ee.Terrain.products(ned).select('elevation', 'slope', 'aspect')

    elev = terrain.select('elevation')
    tpi_1250 = elev.subtract(elev.focal_mean(1250, 'circle', 'meters')).add(0.5).rename('tpi_1250')
    tpi_250 = elev.subtract(elev.focal_mean(250, 'circle', 'meters')).add(0.5).rename('tpi_250')
    tpi_150 = elev.subtract(elev.focal_mean(150, 'circle', 'meters')).add(0.5).rename('tpi_150')
    input_bands = input_bands.addBands([terrain, tpi_1250, tpi_250, tpi_150])

    water_year_start = '{}-10-01'.format(yr - 1)
    et_water_year_start, et_water_year_end = '{}-11-01'.format(yr - 1), '{}-11-01'.format(yr)
    spring_s, spring_e = '{}-03-01'.format(yr), '{}-05-01'.format(yr),
    late_spring_s, late_spring_e = '{}-05-01'.format(yr), '{}-07-15'.format(yr)
    summer_s, summer_e = '{}-07-15'.format(yr), '{}-09-30'.format(yr)
    winter_s, winter_e = '{}-01-01'.format(yr), '{}-03-01'.format(yr),
    fall_s, fall_e = '{}-09-30'.format(yr), '{}-12-31'.format(yr)

    for s, e, n, m in [(water_year_start, spring_e, 'wy_spr', (10, 5)),
                       (water_year_start, summer_e, 'wy', (10, 9)),
                       (et_water_year_start, et_water_year_end, 'wy_et', (11, 11)),
                       (spring_e, fall_s, 'gs', (5, 10)),
                       (spring_s, spring_e, 'spr', (3, 5)),
                       (late_spring_s, late_spring_e, 'lspr', (5, 7)),
                       (summer_s, summer_e, 'sum', (7, 9)),
                       (winter_s, winter_e, 'win', (1, 3))]:
        gridmet = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET").filterBounds(
            roi).filterDate(s, e).select('pr', 'etr', 'tmmn', 'tmmx')

        temp = ee.Image(gridmet.select('tmmn').mean().add(gridmet.select('tmmx').mean()
                                                          .divide(ee.Number(2))).rename('tmp_{}'.format(n)))

        ppt = gridmet.select('pr').reduce(ee.Reducer.sum()).rename('ppt_{}'.format(n))

        etr = gridmet.select('etr').reduce(ee.Reducer.sum()).rename('etr_{}'.format(n))

        wd_estimate = ppt.subtract(etr).rename('cwd_{}'.format(n))

        worldclim_prec = get_world_climate(months=m, param='prec')
        anom_prec = ppt.subtract(worldclim_prec).rename('anm_ppt_{}'.format(n))
        worldclim_temp = get_world_climate(months=m, param='tavg')
        anom_temp = temp.subtract(worldclim_temp).rename('anm_temp_{}'.format(n))

        input_bands = input_bands.addBands([temp, wd_estimate, ppt, etr, anom_prec, anom_temp])

    awc = ee.Image('users/dgketchum/soils/ssurgo_AWC_WTA_0to152cm_composite').rename('awc')
    clay = ee.Image('users/dgketchum/soils/ssurgo_Clay_WTA_0to152cm_composite').rename('clay')
    ksat = ee.Image('users/dgketchum/soils/ssurgo_Ksat_WTA_0to152cm_composite').rename('ksat')
    sand = ee.Image('users/dgketchum/soils/ssurgo_Sand_WTA_0to152cm_composite').rename('sand')

    gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater')
    occ_pos = gsw.select('occurrence').gt(0)
    water = occ_pos.unmask(0).rename('gsw')

    gedi = ee.Image('users/potapovpeter/GEDI_V27/GEDI_NAM_v27')

    input_bands = input_bands.addBands([awc, clay, ksat, sand, gedi])
    input_bands = input_bands.clip(roi)
    return input_bands


def is_authorized():
    try:
        ee.Initialize()  # investigate (use_cloud_api=True)
        print('Authorized')
        return True
    except Exception as e:
        print('You are not authorized: {}'.format(e))
        return False

# I don't understand how the if __name__ etc operates, I understand it is pulling
# functions from above - but I don't understand why you use: if __name__ == '__main__':
# This is step by step start with request_band_extract then comment out, then uncomment randomforest
# and so forth
# request_band_extract OUTPUT MUST BE MANUALLY DOWNLOADED FROM GCLOUD
# prep extracts needs to be manually uploaded to GEE
if __name__ == '__main__':
    is_authorized()
    points = 'users/mariejohnson22/inference/random_points'
    years_ = [2019]
    pref = 'bands_10JAN2023_Water' # file prefix
    roi = 'users/mariejohnson22/inference/mission_no_fire'
    # request_band_extract(pref, points, roi, years_) # comment out after running

    # THIS MUST BE MANUALLY DOWNLOADED FROM GCLOUD
    csv = '/home/marie/crazyHorse/gedi/extracts/bands_10JAN2023_Water_2019.csv'
    # random_forest(csv, show_importance=True)

    out_csv = '/home/marie/crazyHorse/gedi/extracts/prepped_10JAN2023_Water_2019.csv'
    # prep_extracts(csv, out_csv) # manually upload GEE - IS THIS TRAINING DATA?

    # Final steps
    # MAKE SURE PREPPED DATA HAVE BEEN UPLOADED TO GEE
    training_data = 'users/mariejohnson22/inference/prepped_10JAN2023_Water_2019' # I assume this is being generated for the missions unburned
    # I think I have to create this image collection manually on EE
    # image_coll = 'users/mariejohnson22/inference/canopy_height'  # if you want a different image collection you need to create one on EE
    image_coll = 'users/mariejohnson22/inference/canopy_height_water_included'
    # is clip how I get the burned area by changing the shapefile?
    # clip = 'users/mariejohnson22/inference/mission_no_fire' # shapefile of the missions without the burn
    clip = 'users/mariejohnson22/inference/wild_ch' # shapefile of the crazy horse fire

    out_img = 'canopy_height_crazy_horse_water_10JAN2023' # I don't know what this represents, I assume the predicted height for fire
    export_prediction(out_img, training_data, image_coll, clip, [2019])
# ========================= EOF =================================================
