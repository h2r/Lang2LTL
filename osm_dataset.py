"""
OSM landmarks from https://github.com/jasonxyliu/magic-skydio/tree/master/language/data/map_info_dicts
Map ID to city name mapping from https://github.com/jasonxyliu/magic-skydio/blob/5b16d3f5151b250576ec1cdd513283f377368170/language/copynet/end2end_eval.py#L31
"""
import os
from pathlib import Path
import string
from pprint import pprint

from utils import load_from_file, save_to_file


def rename_map_files(osm_lmks_dpath):
    """
    Change file names for better interpretation.
    e.g. Map00.json -> new_york_1.json
    """
    mapid2cityname = {
        "Map00": "New York #1", "Map01": "New York #2", "Map02": "Los Angeles #1",
        "Map03": "Los Angeles #2", "Map04": "Chicago #1", "Map05": "Chicago #2",
        "Map07": "Houston", "Map10": "Philadelphia #1", "Map11": "Philadelphia #2",
        "Map12": "San Diego #1", "Map13": "San Diego #2", "Map14": "Jacksonville #1",
        "Map15": "Jacksonville #2", "Map16": "Columbus #1", "Map17": "Columbus #2",
        "Map18": "Charlotte #1", "Map19": "Charlotte #2", "Map20": "Indianapolis",
        "Map22": "Seattle", "Map24": "Denver #1", "Map25": "Denver #2", "Map29": "Boston"
    }
    map_fnames = [fname for fname in os.listdir(osm_lmks_dpath) if os.path.splitext(fname)[1] == ".json"]

    for map_fname in map_fnames:
        if "Map" in map_fname:
            map_fpath = os.path.join(osm_lmks_dpath, map_fname)
            map_id = Path(map_fpath).stem
            if map_id in mapid2cityname:
                city_name = mapid2cityname[map_id]
                if "#" in city_name:
                    city_name = city_name.translate(str.maketrans('', '', string.punctuation))  # remove #
                new_map_fname = "_".join(city_name.lower().split(" ")) + ".json"
                new_map_fpath = os.path.join(osm_lmks_dpath, new_map_fname)
                os.rename(map_fpath, new_map_fpath)


def construct_lmk2prop(osm_lmks_dpath, lmk2prop_fpath):
    """
    Convert landmark names to propositions
    """
    lmk_fnames = [fname for fname in os.listdir(osm_lmks_dpath) if os.path.splitext(fname)[1] == ".json"]
    city2map = {}
    for lmk_fname in lmk_fnames:
        city_name = os.path.splitext(lmk_fname)[0]
        lmk2prop = {}
        print(lmk_fname)
        lmk_fpath = os.path.join(osm_lmks_dpath, lmk_fname)
        lmk_json = load_from_file(lmk_fpath)
        lmk_names = lmk_json.keys()
        for lmk_name in lmk_names:
            prop = "_".join(lmk_name.translate(str.maketrans('', '', "'-")).lower().split())
            lmk2prop[lmk_name] = prop
        pprint(lmk2prop)
        city2map[city_name] = lmk2prop
        breakpoint()
    save_to_file(city2map, lmk2prop_fpath)

    city2map = load_from_file(lmk2prop_fpath)
    for city, lmk2prop in city2map.items():
        print(city)
        pprint(lmk2prop)


if __name__ == "__main__":
    osm_lmks_dpath = os.path.join("data", "osm", "osm_lmks")

    # rename_map_files(osm_lmks_dpath)

    lmk2prop_fpath = os.path.join("data", "osm", "osm_lmks", "lmk2prop.pkl")
    construct_lmk2prop(osm_lmks_dpath, lmk2prop_fpath)
