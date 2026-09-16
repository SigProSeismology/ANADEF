#!/usr/bin/env python3
"""
Build data/gis/hillshade.npz (grey hillshade + extent) for the map figures.

    python make_hillshade.py                       # fallback: NASA/basemap shaded-relief image (2 arcmin ~ 3 km)
    python make_hillshade.py dem.tif|dem.nc        # preferred: hillshade computed from a DEM (GEBCO, SRTM15+, GSI...)

A DEM GeoTIFF is read with rasterio, a netCDF (GEBCO/ETOPO style: lat, lon, elevation|z) with netCDF4.  The
hillshade is computed with matplotlib.colors.LightSource (azimuth 315, altitude 45, vertical exaggeration
chosen for the ~1 km cell) on land only; the sea is shaded from bathymetry at a lower contrast.
"""
import sys, os, json, numpy as np
BOX = (139.0, 148.0, 40.5, 46.8)            # lon_min, lon_max, lat_min, lat_max (slightly larger than the grid)
OUT = os.path.join('data', 'gis', 'hillshade.npz')

def from_dem(path):
    from matplotlib.colors import LightSource
    if path.lower().endswith(('.tif', '.tiff')):
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(path) as ds:
            win = from_bounds(BOX[0], BOX[2], BOX[1], BOX[3], ds.transform)
            z = ds.read(1, window=win).astype(float)
            t = ds.window_transform(win); dx = t.a; dy = -t.e
            extent = (t.c, t.c + dx * z.shape[1], t.f - dy * z.shape[0], t.f)
    else:
        import netCDF4
        ds = netCDF4.Dataset(path)
        lon = ds.variables['lon'][:] if 'lon' in ds.variables else ds.variables['x'][:]
        lat = ds.variables['lat'][:] if 'lat' in ds.variables else ds.variables['y'][:]
        zn = [v for v in ('elevation', 'z', 'Band1') if v in ds.variables][0]
        i = (lon >= BOX[0]) & (lon <= BOX[1]); j = (lat >= BOX[2]) & (lat <= BOX[3])
        z = np.array(ds.variables[zn][j, :][:, i], dtype=float)
        if lat[j][0] < lat[j][-1]:
            z = z[::-1]
        extent = (lon[i].min(), lon[i].max(), lat[j].min(), lat[j].max())
        dx = (extent[1] - extent[0]) / z.shape[1]; dy = (extent[3] - extent[2]) / z.shape[0]
    km_per_deg = 111.0 * np.cos(np.deg2rad(0.5 * (BOX[2] + BOX[3])))
    ls = LightSource(azdeg=315, altdeg=45)
    land = ls.hillshade(np.where(z > 0, z, 0), vert_exag=1.5, dx=dx * km_per_deg * 1000, dy=dy * 111000)
    sea = ls.hillshade(np.where(z <= 0, z, 0), vert_exag=1.5, dx=dx * km_per_deg * 1000, dy=dy * 111000)
    shade = np.where(z > 0, 0.15 + 0.85 * land, 0.45 + 0.55 * sea)             # sea lighter / lower contrast
    return shade.astype(np.float32), extent, f'hillshade from {os.path.basename(path)}'

def fallback():
    from PIL import Image
    import mpl_toolkits.basemap as b
    p = os.path.join(os.path.dirname(b.__file__), '..', 'basemap_data', 'shadedrelief.jpg')
    p = os.path.normpath(p)
    im = Image.open(p).convert('L'); W, H = im.size
    x0 = int((BOX[0] + 180) / 360 * W); x1 = int((BOX[1] + 180) / 360 * W)
    y0 = int((90 - BOX[3]) / 180 * H); y1 = int((90 - BOX[2]) / 180 * H)
    arr = np.asarray(im.crop((x0, y0, x1, y1)), dtype=np.float32) / 255.0
    from scipy.ndimage import gaussian_filter
    arr = gaussian_filter(arr, 1.2)                      # remove JPEG block artefacts of the 2-arcmin image
    arr = 0.35 + 0.65 * arr                               # lift the dark tones so overlays stay readable
    extent = ((x0 / W) * 360 - 180, (x1 / W) * 360 - 180, 90 - (y1 / H) * 180, 90 - (y0 / H) * 180)
    return arr, extent, 'NASA shaded relief via basemap-data (2 arcmin); replace with a DEM for publication'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        shade, extent, src = from_dem(sys.argv[1])
    else:
        shade, extent, src = fallback()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, shade=shade, extent=np.array(extent), source=np.array(src))
    print(f'wrote {OUT}: {shade.shape} px, extent {tuple(round(v, 3) for v in extent)}; {src}')
