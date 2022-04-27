import meteoblue_dataset_sdk
import logging
import datetime as dt
import dateutil.parser
import pandas as pd
import numpy as np
import netCDF4
from scipy.spatial.distance import cdist
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import griddata


def meteoblue_timeinterval_to_timestamps(t):
    if len(t.timestrings) > 0:

        def map_ts(time):
            if "-" in time:
                return dateutil.parser.parse(time.partition("-")[0])
            return dateutil.parser.parse(time)

        return list(map(map_ts, t.timestrings))

    timerange = range(t.start, t.end, t.stride)
    return list(map(lambda t: dt.datetime.fromtimestamp(t), timerange))


def meteoblue_result_to_dataframe(geometry):
    t = geometry.timeIntervals[0]
    timestamps = meteoblue_timeinterval_to_timestamps(t)

    n_locations = len(geometry.lats)
    n_timesteps = len(timestamps)

    df = pd.DataFrame(
        {
            "TIMESTAMP": np.tile(timestamps, n_locations),
            "Longitude": np.repeat(geometry.lons, n_timesteps),
            "Latitude": np.repeat(geometry.lats, n_timesteps),
        }
    )

    for code in geometry.codes:
        name = str(code.code) + "_" + code.level + "_" + code.aggregation
        df[name] = list(code.timeIntervals[0].data)

    return df


def meteoblue_create_query_surface(cois, start, end, nwp):
    query_temperature = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "2 m above gnd"}],
            }
        ],
    }
    query_dtemperature = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 17,
                        "level": "2 m above gnd"
                    }
                ]
            }
        ]
    }
    query_cape = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 157,
                        "level": "180-0 mb above gnd"
                    }
                ]
            }
        ]
    }
    query_wdir = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "10 m above gnd"
                    }
                ]
            }
        ]
    }
    query_wspeed = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "10 m above gnd"
                    }
                ]
            }
        ]
    }

    return query_temperature, query_dtemperature, query_cape, query_wdir, query_wspeed


def meteoblue_create_query_sounding(cois, start, end, nwp):
    query_temperature_2m = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "2 m above gnd"}],
            }
        ],
    }
    query_temperature_1000mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "1000 mb"}],
            }
        ],
    }
    query_temperature_900mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "900 mb"}],
            }
        ],
    }
    query_temperature_850mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "850 mb"}],
            }
        ],
    }
    query_temperature_800mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "800 mb"}],
            }
        ],
    }
    query_temperature_700mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "700 mb"}],
            }
        ],
    }
    query_temperature_500mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "500 mb"}],
            }
        ],
    }
    query_temperature_250mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts",
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [{"code": 11, "level": "250 mb"}],
            }
        ],
    }
    query_wspeed_10m = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "10 m above gnd"
                    }
                ]
            }
        ]
    }
    query_wspeed_1000mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "1000 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_900mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "900 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_850mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "850 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_800mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "800 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_700mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "700 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_500mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "500 mb"
                    }
                ]
            }
        ]
    }
    query_wspeed_250mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 32,
                        "level": "250 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_10m = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "10 m above gnd"
                    }
                ]
            }
        ]
    }
    query_wdir_1000mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "1000 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_900mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "900 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_850mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "850 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_800mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "800 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_700mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "700 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_500mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "500 mb"
                    }
                ]
            }
        ]
    }
    query_wdir_250mb = {
        "units": {
            "temperature": "C",
            "velocity": "km/h",
            "length": "metric",
            "energy": "watts"
        },
        "geometry": {
            "type": "MultiPoint",
            "coordinates": cois,
            "locationNames": [""]*len(cois)
        },
        "format": "json",
        "timeIntervals": [
            start+"T+00:00/"+end+"T+00:00"
        ],
        "timeIntervalsAlignment": "none",
        "queries": [
            {
                "domain": nwp,
                "gapFillDomain": None,
                "timeResolution": "hourly",
                "codes": [
                    {
                        "code": 31,
                        "level": "250 mb"
                    }
                ]
            }
        ]
    }
    # query_dtemperature_2m = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "2 m above gnd"}],
    #         }
    #     ],
    # }
    # query_dtemperature_1000mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "1000 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_900mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "900 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_850mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "850 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_800mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "800 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_700mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "700 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_500mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "500 mb"}],
    #         }
    #     ],
    # }
    # query_dtemperature_250mb = {
    #     "units": {
    #         "temperature": "C",
    #         "velocity": "km/h",
    #         "length": "metric",
    #         "energy": "watts",
    #     },
    #     "geometry": {
    #         "type": "MultiPoint",
    #         "coordinates": cois,
    #         "locationNames": [""]*len(cois)
    #     },
    #     "format": "json",
    #     "timeIntervals": [
    #         start+"T+00:00/"+end+"T+00:00"
    #     ],
    #     "timeIntervalsAlignment": "none",
    #     "queries": [
    #         {
    #             "domain": nwp,
    #             "gapFillDomain": None,
    #             "timeResolution": "hourly",
    #             "codes": [{"code": 17, "level": "250 mb"}],
    #         }
    #     ],
    # }
    # return query_temperature_2m, query_temperature_1000mb, query_temperature_900mb, query_temperature_850mb, query_temperature_800mb, query_temperature_700mb, query_temperature_500mb, query_temperature_250mb, query_wspeed_10m, query_wspeed_1000mb, query_wspeed_900mb, query_wspeed_850mb, query_wspeed_800mb, query_wspeed_700mb, query_wspeed_500mb, query_wspeed_500mb, query_wspeed_250mb, query_wdir_10m, query_wdir_1000mb, query_wdir_900mb, query_wdir_850mb, query_wdir_800mb, query_wdir_700mb, query_wdir_500mb, query_wdir_500mb, query_wdir_250mb, query_dtemperature_2m, query_dtemperature_1000mb, query_dtemperature_900mb, query_dtemperature_850mb, query_dtemperature_800mb, query_dtemperature_700mb, query_dtemperature_500mb, query_dtemperature_250mb
    return query_temperature_2m, query_temperature_1000mb, query_temperature_900mb, query_temperature_850mb, query_temperature_800mb, query_temperature_700mb, query_temperature_500mb, query_temperature_250mb, query_wspeed_10m, query_wspeed_1000mb, query_wspeed_900mb, query_wspeed_850mb, query_wspeed_800mb, query_wspeed_700mb, query_wspeed_500mb, query_wspeed_500mb, query_wspeed_250mb, query_wdir_10m, query_wdir_1000mb, query_wdir_900mb, query_wdir_850mb, query_wdir_800mb, query_wdir_700mb, query_wdir_500mb, query_wdir_500mb, query_wdir_250mb

# Reading the NWP model output from the meteoblue dataset sdk
def get_surface_model_output(cois, start, end, nwp, apikey):
    # Display information about the current download state
    logging.basicConfig(level=logging.INFO)

    # Create queries
    query_temperature, query_dtemperature, query_cape, query_wdir, query_wspeed = meteoblue_create_query_surface(
        cois, start, end, nwp
    )

    client = meteoblue_dataset_sdk.Client(apikey)

    result = client.query_sync(query_temperature)
    df_temperature = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_dtemperature)
    df_dtemperature = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_cape)
    df_cape = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir)
    df_wdir = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed)
    df_wspeed = meteoblue_result_to_dataframe(result.geometries[0])

    data = pd.concat(
        [df_temperature, df_dtemperature.iloc[:, 3], df_cape.iloc[:, 3], df_wdir.iloc[:, 3], df_wspeed.iloc[:, 3]],
        axis=1,
    )

    data = data.rename({'11_2 m above gnd_none':'temperature','17_2 m above gnd_none': 'dewpoint', '157_180-0 mb above gnd_none': 'CAPE', '31_10 m above gnd_none': 'winddir', '32_10 m above gnd_none': 'windspeed'}, axis = 1)
    return data

# Reading the NWP model output from the meteoblue dataset sdk
def get_sounding_model_output(cois, start, end, nwp, apikey):
    # Display information about the current download state
    logging.basicConfig(level=logging.INFO)

    # # Create queries
    # query_temperature_2m, query_temperature_1000mb, query_temperature_900mb, query_temperature_850mb, query_temperature_800mb, query_temperature_700mb, query_temperature_500mb, query_temperature_250mb, query_wspeed_10m, query_wspeed_1000mb, query_wspeed_900mb, query_wspeed_850mb, query_wspeed_800mb, query_wspeed_700mb, query_wspeed_500mb, query_wspeed_500mb, query_wspeed_250mb, query_wdir_10m, query_wdir_1000mb, query_wdir_900mb, query_wdir_850mb, query_wdir_800mb, query_wdir_700mb, query_wdir_500mb, query_wdir_500mb, query_wdir_250mb, query_dtemperature_2m, query_dtemperature_1000mb, query_dtemperature_900mb, query_dtemperature_850mb, query_dtemperature_800mb, query_dtemperature_700mb, query_dtemperature_500mb, query_dtemperature_250mb = meteoblue_create_query_sounding(
    #     cois, start, end, nwp
    # )

        # Create queries
    query_temperature_2m, query_temperature_1000mb, query_temperature_900mb, query_temperature_850mb, query_temperature_800mb, query_temperature_700mb, query_temperature_500mb, query_temperature_250mb, query_wspeed_10m, query_wspeed_1000mb, query_wspeed_900mb, query_wspeed_850mb, query_wspeed_800mb, query_wspeed_700mb, query_wspeed_500mb, query_wspeed_500mb, query_wspeed_250mb, query_wdir_10m, query_wdir_1000mb, query_wdir_900mb, query_wdir_850mb, query_wdir_800mb, query_wdir_700mb, query_wdir_500mb, query_wdir_500mb, query_wdir_250mb= meteoblue_create_query_sounding(
        cois, start, end, nwp
    )

    client = meteoblue_dataset_sdk.Client(apikey)

    result = client.query_sync(query_temperature_2m)
    df_temperature_2m = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_1000mb)
    df_temperature_1000mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_900mb)
    df_temperature_900mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_850mb)
    df_temperature_850mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_800mb)
    df_temperature_800mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_700mb)
    df_temperature_700mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_500mb)
    df_temperature_500mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_temperature_250mb)
    df_temperature_250mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_2m)
    # df_dtemperature_2m = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_1000mb)
    # df_dtemperature_1000mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_900mb)
    # df_dtemperature_900mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_850mb)
    # df_dtemperature_850mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_800mb)
    # df_dtemperature_800mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_700mb)
    # df_dtemperature_700mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_500mb)
    # df_dtemperature_500mb = meteoblue_result_to_dataframe(result.geometries[0])

    # result = client.query_sync(query_dtemperature_250mb)
    # df_dtemperature_250mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_10m)
    df_wspeed_10m = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_1000mb)
    df_wspeed_1000mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_900mb)
    df_wspeed_900mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_850mb)
    df_wspeed_850mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_800mb)
    df_wspeed_800mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_700mb)
    df_wspeed_700mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_500mb)
    df_wspeed_500mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wspeed_250mb)
    df_wspeed_250mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_10m)
    df_wdir_10m = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_1000mb)
    df_wdir_1000mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_900mb)
    df_wdir_900mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_850mb)
    df_wdir_850mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_800mb)
    df_wdir_800mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_700mb)
    df_wdir_700mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_500mb)
    df_wdir_500mb = meteoblue_result_to_dataframe(result.geometries[0])

    result = client.query_sync(query_wdir_250mb)
    df_wdir_250mb = meteoblue_result_to_dataframe(result.geometries[0])

    # data = pd.concat(
    #     [df_temperature_2m, df_temperature_1000mb.iloc[:,3], df_temperature_900mb.iloc[:,3], df_temperature_850mb.iloc[:,3], df_temperature_800mb.iloc[:,3], df_temperature_700mb.iloc[:,3], df_temperature_500mb.iloc[:,3], df_temperature_250mb.iloc[:,3], df_dtemperature_2m.iloc[:,3], df_dtemperature_1000mb.iloc[:,3], df_dtemperature_900mb.iloc[:,3], df_dtemperature_850mb.iloc[:,3], df_dtemperature_800mb.iloc[:,3], df_dtemperature_700mb.iloc[:,3], df_dtemperature_500mb.iloc[:,3], df_dtemperature_250mb.iloc[:,3], df_wspeed_10m.iloc[:,3], df_wspeed_1000mb.iloc[:,3], df_wspeed_900mb.iloc[:,3], df_wspeed_850mb.iloc[:,3], df_wspeed_800mb.iloc[:,3], df_wspeed_700mb.iloc[:,3], df_wspeed_500mb.iloc[:,3], df_wspeed_250mb.iloc[:,3], df_wdir_10m.iloc[:,3], df_wdir_1000mb.iloc[:,3], df_wdir_900mb.iloc[:,3], df_wdir_850mb.iloc[:,3], df_wdir_800mb.iloc[:,3], df_wdir_700mb.iloc[:,3], df_wdir_500mb.iloc[:,3], df_wdir_250mb.iloc[:,3]],
    #     axis=1,
    # )

    data = pd.concat(
        [df_temperature_2m, df_temperature_1000mb.iloc[:,3], df_temperature_900mb.iloc[:,3], df_temperature_850mb.iloc[:,3], df_temperature_800mb.iloc[:,3], df_temperature_700mb.iloc[:,3], df_temperature_500mb.iloc[:,3], df_temperature_250mb.iloc[:,3], df_wspeed_10m.iloc[:,3], df_wspeed_1000mb.iloc[:,3], df_wspeed_900mb.iloc[:,3], df_wspeed_850mb.iloc[:,3], df_wspeed_800mb.iloc[:,3], df_wspeed_700mb.iloc[:,3], df_wspeed_500mb.iloc[:,3], df_wspeed_250mb.iloc[:,3], df_wdir_10m.iloc[:,3], df_wdir_1000mb.iloc[:,3], df_wdir_900mb.iloc[:,3], df_wdir_850mb.iloc[:,3], df_wdir_800mb.iloc[:,3], df_wdir_700mb.iloc[:,3], df_wdir_500mb.iloc[:,3], df_wdir_250mb.iloc[:,3]],
        axis=1,
    )

    #data = data.rename({'11_2 m above gnd_none':'temperature','17_2 m above gnd_none': 'dewpoint', '157_180-0 mb above gnd_none': 'CAPE', '31_10 m above gnd_none': 'winddir', '32_10 m above gnd_none': 'windspeed'}, axis = 1)
    return data