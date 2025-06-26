import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely import wkt
from shapely.ops import transform
import pyproj
from scipy.spatial import cKDTree
from multiprocessing import Pool
from tqdm import tqdm

# ---------- Setup functions ----------
def wkt_to_linestring(wkt_str):
    return LineString(wkt.loads(wkt_str).coords)

def find_nearest(point):
    dist, idx = street_points_tree.query([point.x, point.y], k=1)
    nearest_point = gdf_street_points.iloc[idx].geometry
    return point.distance(nearest_point)

def process_route(row_tuple):
    idx, row = row_tuple
    line = row['Line']

    try:
        # Build geometry
        points = [Point(xy) for xy in line.coords]
        gdf_trasa = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326").to_crs(epsg=2180)

        # Validation: drop invalid, NaN or non-finite geometries
        def is_valid_and_finite(geom):
            if geom is None or not geom.is_valid or geom.is_empty:
                return False
            if hasattr(geom, "coords"):
                return all(np.isfinite(c[0]) and np.isfinite(c[1]) for c in geom.coords)
            return False

        if not all(gdf_trasa.geometry.apply(is_valid_and_finite)):
            print(f"Skipping idx={idx}: contains invalid or non-finite geometries")
            return idx, None

        # Main logic
        distances = [find_nearest(pt) for pt in gdf_trasa.geometry]
        return idx, np.array(distances)

    except Exception as e:
        print(f"Error processing idx={idx}: {e}")
        return None

# ---------- Global variables (used in workers) ----------
gdf_street_points = None
street_points_tree = None

def init_worker(shared_gdf, shared_tree):
    global gdf_street_points, street_points_tree
    gdf_street_points = shared_gdf
    street_points_tree = shared_tree

# ---------- Load input ----------
df = pd.read_parquet('dojazdy_df.parquet')
df['Line'] = df['Line'].apply(wkt_to_linestring)
gdf = gpd.GeoDataFrame(df, geometry='Line', crs="EPSG:4326")
gdf = gdf.reset_index(drop=True)

gdf_ulice = gpd.read_file('streets_małopolskie.shp').to_crs(epsg=2180)
gdf_streets_exploded = gdf_ulice.explode(index_parts=False)

print('Loaded data. Creating cKDTree...')

all_points = [Point(coord) for geom in gdf_streets_exploded.geometry for coord in geom.coords]
gdf_street_points = gpd.GeoDataFrame(geometry=all_points)
street_points_tree = cKDTree(np.c_[gdf_street_points.geometry.x, gdf_street_points.geometry.y])

# ---------- Parallel processing ----------
def parallel_process_routes(gdf, num_workers=8):
    with Pool(processes=num_workers, initializer=init_worker, initargs=(gdf_street_points, street_points_tree)) as pool:
        results = list(tqdm(pool.imap(process_route, gdf.iterrows()), total=len(gdf), desc="Processing in parallel"))
    distances = [None] * len(gdf)
    for idx, dists in results:
        distances[idx] = dists
    return distances

print('Starting to process')

df['distances'] = parallel_process_routes(gdf, num_workers=32)
df['Line'] = df['Line'].apply(lambda x: x.wkt)
df.to_parquet('dojazdy_df_X_2021_full_path_distances.parquet', index=False)
print("Done! File saved.")
