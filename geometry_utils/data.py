import os
from io import BytesIO
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import py7zr
import requests


def load_france_borders(path=None):
    with TemporaryDirectory() as tempdir:
        path = path if isinstance(path, str) else tempdir
        path_file = os.path.join(path, 'FRA_ADM0.shp')
        if not os.path.exists(path_file):
            os.makedirs(os.path.dirname(path_file), exist_ok=True)
            response = requests.get('https://www.geoboundaries.org/data/1_3_3/zip/shapefile/FRA/FRA_ADM0.shp.zip')
            with ZipFile(BytesIO(response.content)) as zipfile:
                zipfile.extractall(path)
        gdf = gpd.read_file(path_file)
        gdf = gpd.clip(gdf, mask=(-5, 30, 20, 55))
    return gdf


def load_raster_population_france(path=None):
    with TemporaryDirectory() as tempdir:
        path = path if isinstance(path, str) else tempdir
        path_file = os.path.join(path, 'Filosofi2017_carreaux_1km_met.gpkg')
        if not os.path.exists(path_file):
            os.makedirs(os.path.dirname(path_file), exist_ok=True)
            response = requests.get('https://www.insee.fr/fr/statistiques/fichier/6215140/Filosofi2017_carreaux_1km_gpkg.zip')
            with ZipFile(BytesIO(response.content)) as zipfile:
                zipfile.extractall(path)
            with py7zr.SevenZipFile(os.path.join(path, 'Filosofi2017_carreaux_1km_gpkg.7z'), mode='r') as z:
                z.extractall(path)

        gdf = gpd.read_file(path_file, engine='pyogrio')
    gdf['x'] = gdf['Idcar_1km'].str.split('E').str[-1].astype(int)
    gdf['y'] = gdf['Idcar_1km'].str.split('N').str[-1].str.split('E').str[0].astype(int)
    return gdf[['x', 'y', 'Ind', 'geometry']].copy()
