import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import matplotlib.colors as mcol

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from chrono import chrono
from dataloader import add_coords
from model import TRUTH, PRED


def plot_map():
    # Coordinates of studied area boundaries (in °N and °E)
    lllat = 46.25   # lower left latitude
    urlat = 51.896  # upper right latitude
    lllon = -5.842  # lower left longitude
    urlon = 2       # upper right longitude
    extent = [lllon, urlon, lllat, urlat]
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"))
    ax.set_extent(extent)
    return ax


def plot_stations(datas, date, param):
    datas = datas[datas["date"] == date]

    plt.figure(figsize=(9, 5))
    ax = plot_map()

    plt.scatter(datas["lon"], datas["lat"], c=datas[param], cmap="jet")
    plt.colorbar(ax=ax)
    plt.title("%s / %s" % (date, param))
    plt.show()


@chrono
def make_anim(datas, param, *,
              name="out.mp4",
              start="2016-01-01 00:00:00",
              end="2017-12-31 23:00:00",
              freq="H",
              global_colorbar=False,
              cmap="jet"):

    def update(date):
        dots = datas[datas["date"] == date][["lon", "lat", param]].dropna()
        scat.set_offsets(np.array([dots["lon"], dots["lat"]]).T)
        title.set_text("%s / %s" % (date, param))

        if dots.shape[0] != 0:
            if global_colorbar:
                scat.set_facecolor(sm.to_rgba(dots[param]))
            else:
                cnorm.autoscale(dots[param])
                scat.set_facecolor(sm.to_rgba(dots[param]))
                cb.update_normal(sm)

    dates = pd.date_range(start, end, freq=freq)

    if global_colorbar:
        to_search = datas[datas["date"].isin(dates)][param]
        vmin = to_search.min()
        vmax = to_search.max()
    else:
        vmin = vmax = None

    fig = plt.figure(figsize=(9, 5))
    ax = plot_map()

    cm = mpl.cm.get_cmap(cmap)
    cnorm = mcol.Normalize(vmin, vmax)
    sm = mpl.cm.ScalarMappable(norm=cnorm, cmap=cm)
    cb = fig.colorbar(sm)

    scat = ax.scatter([], [], cmap=cm)
    title = ax.set_title("")

    anim = manim.FuncAnimation(fig, update, frames=dates)
    anim.save(name, writer="ffmpeg")


def prepare_prediction(pred):
    split = pred["Id"].str.split("_", expand=True).astype(int)
    pred["number_sta"] = split[0]
    pred["day_index"] = split[1]
    pred = add_coords(pred)
    dates = pd.date_range("20160102", periods=730)
    pred["date"] = dates[pred["day_index"]]
    return pred


def anim_prediction(pred, truth=None, **kwargs):
    toplot = pred.copy()
    toplot = prepare_prediction(toplot)
    if truth is not None:
        param = "Difference"
        toplot = toplot.merge(
            truth[["Id", TRUTH]], on="Id", how="left")
        toplot["Difference"] = np.abs(
            toplot[PRED] - toplot[TRUTH])
    else:
        param = PRED
    make_anim(toplot, param, freq="D", **kwargs)
    return toplot
