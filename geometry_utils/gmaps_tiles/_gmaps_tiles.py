from io import BytesIO
from itertools import product

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
import rioxarray as rio
import xarray as xr
from PIL import Image
from shapely.geometry import box
from tqdm.auto import tqdm

from .. import encode_dataset, get_coordinates, proj


class GoogleMapsTiles:
    WORLD_SIZE = 2*6378137*np.pi
    XMIN = -6378137*np.pi
    YMIN = -6378137*np.pi
    XMAX = 6378137*np.pi
    YMAX = 6378137*np.pi
    CRS = 3857

    @classmethod
    def url_tile(cls, zoom, lon=None, lat=None, xtile=None, ytile=None):
        if isinstance(lon, float) and isinstance(lat, float):
            x, y = proj(lon, lat, 4326, cls.CRS)
            xtile_ = np.floor(2**zoom*(x-cls.XMIN)/cls.WORLD_SIZE).astype(int)
            ytile_ = np.floor(2**zoom*(cls.YMAX-y)/cls.WORLD_SIZE).astype(int)
        else:
            xtile_, ytile_ = xtile, ytile
        return xtile_, ytile_, f'https://mt1.google.com/vt/lyrs=s&x={xtile_}&y={ytile_}&z={zoom}'

    @staticmethod
    def url_to_numpy_array(url) -> np.ndarray:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        return image_array

    @classmethod
    def mesh_tile(cls, zoom, xtile, ytile):
        tile_size = cls.WORLD_SIZE/(2**zoom)
        mesh_size = tile_size/256
        xmin = cls.XMIN + xtile*tile_size
        ymin = cls.YMAX - (ytile+1)*tile_size
        x = xmin + mesh_size/2 + mesh_size*np.arange(256)
        y = ymin + mesh_size/2 + mesh_size*np.arange(256)
        return x, y

    @classmethod
    def download_raster(cls, zoom, place=None, lon=None, lat=None, xtile=None, ytile=None):
        if isinstance(place, str):
            lon_, lat_ = get_coordinates(place=place)
            xtile_, ytile_, url = cls.url_tile(zoom=zoom, lon=lon_, lat=lat_)
            tile = cls.url_to_numpy_array(url)
        elif isinstance(lon, float) and isinstance(lat, float):
            xtile_, ytile_, url = cls.url_tile(zoom=zoom, lon=lon, lat=lat)
            tile = cls.url_to_numpy_array(url)
        elif isinstance(xtile, np.integer) and isinstance(ytile, np.integer):
            xtile_, ytile_ = xtile, ytile
            tile = cls.url_to_numpy_array(cls.url_tile(zoom=zoom, xtile=xtile, ytile=ytile)[2])
        else:
            raise ValueError()
        x, y = cls.mesh_tile(zoom=zoom, xtile=xtile_, ytile=ytile_)

        ds = xr.Dataset({'red': (('y', 'x'), np.flip(tile[:, :, 0], axis=0)),
                         'green': (('y', 'x'), np.flip(tile[:, :, 1], axis=0)),
                         'blue': (('y', 'x'), np.flip(tile[:, :, 2], axis=0))},
                        coords={'y': ('y', y), 'x': ('x', x)})
        encode_dataset(ds, crs=cls.CRS)
        return ds

    @classmethod
    def download_numpy(cls, zoom, place=None, lon=None, lat=None, xtile=None, ytile=None):
        if isinstance(place, str):
            lon_, lat_ = get_coordinates(place=place)
            xtile_, ytile_, url = cls.url_tile(zoom=zoom, lon=lon_, lat=lat_)
            tile = cls.url_to_numpy_array(url)
        elif isinstance(lon, float) and isinstance(lat, float):
            xtile_, ytile_, url = cls.url_tile(zoom=zoom, lon=lon, lat=lat)
            tile = cls.url_to_numpy_array(url)
        elif isinstance(xtile, np.integer) and isinstance(ytile, np.integer):
            xtile_, ytile_ = xtile, ytile
            tile = cls.url_to_numpy_array(cls.url_tile(zoom=zoom, xtile=xtile, ytile=ytile)[2])
        else:
            raise ValueError()
        x, y = cls.mesh_tile(zoom=zoom, xtile=xtile_, ytile=ytile_)
        return x, y, tile

    @classmethod
    def plot_raster(cls, ds, figsize=None, gdf=None, show=True):
        x = ds.x.values
        y = ds.y.values
        dx, dy = np.diff(x)[0], np.diff(y)[0]
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        im = np.stack([ds['red'].values, ds['green'].values, ds['blue'].values], axis=-1)
        im = np.nan_to_num(im, nan=255).astype(int)
        plt.figure(figsize=figsize)
        plt.imshow(im, extent=(xmin-dx/2, xmax+dx/2, ymin-dy/2, ymax+dy/2), origin='lower')
        plt.xlim(xmin-dx/2, xmax+dx/2)
        plt.ylim(ymin-dy/2, ymax+dy/2)
        plt.gca().axis('off')
        if gdf is not None:
            gpd.clip(gdf.to_crs(cls.CRS), mask=(xmin-dx/2, ymin-dy/2, xmax+dx/2, ymax+dy/2)).plot(ax=plt.gca(), fc='none', lw=1)
        if show:
            plt.show()

    @classmethod
    def tiles_from_gdf(cls, zoom, gdf):
        union_gdf = gdf.to_crs(cls.CRS).unary_union
        lon_min, lat_min, lon_max, lat_max = gdf.to_crs(4326).total_bounds
        xtile_min, ytile_min, _ = cls.url_tile(zoom=zoom, lon=lon_min, lat=lat_max)
        xtile_max, ytile_max, _ = cls.url_tile(zoom=zoom, lon=lon_max, lat=lat_min)
        xx, yy = np.arange(xtile_min, xtile_max+1), np.arange(ytile_min, ytile_max+1)
        dataset = []
        for xtile, ytile in tqdm(product(xx, yy), total=len(xx)*len(yy)):
            ds = cls.download_raster(zoom, xtile=xtile, ytile=ytile)
            bounds_tile = box(ds.x.min(), ds.y.min(), ds.x.max(), ds.y.max())
            if bounds_tile.intersects(union_gdf):
                dataset.append(ds)
        dataset = xr.merge(dataset).fillna(255).astype(int)
        return dataset
