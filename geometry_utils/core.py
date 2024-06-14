from time import sleep
from typing import Iterable, Tuple, Union

import geopandas as gpd
import numpy as np
import osmnx as ox
import pyproj
from shapely.geometry import Polygon


def proj(x: Union[float, int, Iterable[float]],
         y: Union[float, int, Iterable[float]],
         proj_in: Union[str, int, pyproj.CRS],
         proj_out: Union[str, int, pyproj.CRS]) -> Tuple[Iterable[float], Iterable[float]]:
    """
    Projects coordinates from one coordinate system to another.

    Args:
        x (Union[float, int, Iterable[float]]): x-coordinates to be projected.
        y (Union[float, int, Iterable[float]]): y-coordinates to be projected.
        proj_in (Union[str, int, pyproj.CRS]): The input coordinate system.
        proj_out (Union[str, int, pyproj.CRS]): The output coordinate system.

    Returns:
        Tuple[Iterable[float], Iterable[float]]: The projected coordinates (x, y).
    """
    t = pyproj.Transformer.from_crs(crs_from=to_crs(proj_in), crs_to=to_crs(proj_out), always_xy=True)
    return t.transform(x, y)


def to_crs(proj: Union[str, int, pyproj.CRS, pyproj.Proj, None]) -> pyproj.CRS:
    """
    Converts a coordinate system to a pyproj.CRS object.

    Args:
        proj (Union[str, int, pyproj.CRS, pyproj.Proj, None]): The coordinate system to convert.

    Returns:
        pyproj.CRS: The pyproj.CRS object corresponding to the specified coordinate system.

    Example:
        .. code-block:: python

            to_crs('EPSG:4326')
            >>> <pyproj.CRS ...>

            to_crs(27572)
            >>> <pyproj.CRS ...>
    """
    if isinstance(proj, (int, str)):
        return pyproj.CRS(proj)
    if isinstance(proj, pyproj.CRS):
        return proj
    if isinstance(proj, pyproj.Proj):
        return proj.crs
    if proj is None:
        return None
    raise TypeError("`proj` type is not supported.")


def generate_grid(grid_x: np.ndarray, grid_y: np.ndarray, return_indices: bool = False, crs=None) -> gpd.GeoDataFrame:
    """
    Generates a grid of polygons from arrays of x and y coordinates.

    Parameters:
    - grid_x (np.ndarray): Array of x coordinates.
    - grid_y (np.ndarray): Array of y coordinates.
    - return_indices (bool, optional): Whether to return grid indices. Default is False.
    - crs: Coordinate reference system for the GeoDataFrame. Default is None.

    Returns:
    - gpd.GeoDataFrame: GeoDataFrame containing grid polygons.

    Raises:
    - ValueError: If the shapes of `grid_x` and `grid_y` are not compatible.

    """
    gx, gy = np.array(grid_x), np.array(grid_y)
    if (gx.shape != gy.shape) or (gx.ndim != 2):
        raise ValueError("`grid_x` and `grid_y` must be two 2D arrays of the same shape")

    def _interp_grid(X):
        dX = np.diff(X, axis=1)/2.
        X = np.hstack((X[:, [0]] - dX[:, [0]],
                       X[:, :-1] + dX,
                       X[:, [-1]] + dX[:, [-1]]))
        return X

    ny, nx = gx.shape

    x = _interp_grid(_interp_grid(gx).T).T
    y = _interp_grid(_interp_grid(gy).T).T

    gdf_grid = []

    for i in range(ny):
        for j in range(nx):
            p = Polygon([[x[i, j], y[i, j]], [x[i+1, j], y[i+1, j]], [x[i+1, j+1], y[i+1, j+1]], [x[i, j+1], y[i, j+1]]])
            gdf_grid.append(p)

    gdf_grid = gpd.GeoDataFrame(geometry=gdf_grid)
    if crs:
        gdf_grid = gdf_grid.set_crs(crs)

    if return_indices:
        iy, ix = np.indices(dimensions=(ny, nx))
        gdf_grid = gdf_grid.reset_index()
        gdf_grid['iy'] = iy.ravel()
        gdf_grid['ix'] = ix.ravel()

    return gdf_grid


def get_coordinates(place, crs=4326, retries=10, retry_delay=1, errors='raise'):
    """
    Obtains the geographical coordinates (longitude, latitude) of a given place.

    Args:
        place (Union[str, Iterable[str]]): Name of the place or list of place names.
        crs (Union[str, int, pyproj.CRS], optional): The projection to use for the coordinates.
                                                     Default: 4326 (WGS 84).
        retries (int, optional): The number of attempts in case of failure. Default: 5.
        retry_delay (int, optional): The delay between each attempt in seconds. Default: 1.

    Returns:
        Union[Tuple[float, float], List[Tuple[float, float]]]:
            - Tuple[float, float]: Geographical coordinates (longitude, latitude) of the place.
            - List[Tuple[float, float]]: List of geographical coordinates of each place.

    Example:
        .. code-block:: python

            get_coordinates("Paris")
            >>> (2.3488, 48.85341)

            places = ["Paris", "Lyon", "Marseille"]
            get_coordinates(places)
            >>> [(2.3488, 48.85341), (4.8357, 45.76404), (5.36978, 43.29695)]
    """
    from geopy.geocoders import Nominatim
    geolocator = Nominatim(user_agent='CyrilJl')
    results = []

    def get_coordinates_single(place):
        for _ in range(retries):
            try:
                location = geolocator.geocode(place)
                if location:
                    return proj(location.longitude, location.latitude, 4326, crs)
            except Exception:
                pass
            sleep(retry_delay)
        if errors == 'ignore':
            return (np.nan, np.nan)
        else:
            raise ValueError(f"Échec de la récupération des coordonnées de '{place}'")

    if isinstance(place, str):
        return get_coordinates_single(place)
    else:
        for p in place:
            results.append(get_coordinates_single(p))
        return results


def buildings_in_tile(bbox, crs=None, timeout=180) -> gpd.GeoDataFrame:
    ox.settings.log_console = False
    ox.settings.use_cache = False
    ox.settings.requests_timeout = timeout

    gdf = ox.features.features_from_bbox(bbox=bbox, tags={'building': True})
    if crs is not None:
        gdf = gdf.to_crs(crs)
    return gdf


def proj_bbox(bbox, crs_in, crs_out, buffer=0):
    b = gpd.GeoSeries([bbox.buffer(buffer)], crs=crs_in).to_crs(crs_out)
    return b.total_bounds
