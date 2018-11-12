#!/usr/bin/env python3
#
# table_to_sureal.py
#
# Author: Werner Robitza
# License: MIT
#
# Converts a "normal" wide MOS score table to input for the "sureal" software.
#
# The input data must be a comma-separated CSV file with a header row containing
# at least the following columns, in this order:
#
#     PVS_ID, S1, S2, ...
#
# Where `PVS_ID` must be of the form `<db>_<src>_<hrc>`, where `<db>` can be
# any database identifier. `<src>` and `<hrc>` must contain at least one
# digit, which acts as an ID.
#
# An example of a PVS ID: `DB01_SRC01_HRC02`.
#
# S1, S2, etc. are columns giving the subjective rating for subject S1, S2,
# etc. for the PVS in each row. The columns may also be called "User1" etc.
#
# If there are further columns called `MOS`, `CI`, `N`, they will be ignored.
#
# The output will be JSON file printed to STDOUT, which may be further used
# for `sureal` input.

from __future__ import print_function

import os
import pandas as pd
import json
import argparse
import sys
import re


def log_error(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


def get_ids_from_pvs(pvs_id):
    pvs_id = pvs_id.strip()

    if len(pvs_id.split("_")) != 3:
        log_error("PVS IDs must be like 'XXX_XXX_XXX'!")

    db_id, src_id, hrc_id = pvs_id.split("_")
    return pvs_id, db_id, src_id, hrc_id


def extract_scores_from_record(record):
    """
    Extract subjective scores from a record
    """
    ratings = {}
    # go through each record to find actual ratings
    for key, val in record.items():
        # skip columns like MOS etc.
        if key.lower() in ["pvs_id", "mos", "ci", "n"]:
            continue
        # skip columns that do not start with "s" or "user", or "subject"
        if not any([key.lower().startswith(s) for s in ["user", "s", "subject"]]):
            print("Unkown column named " + str(key) + "!")
            continue
        ratings[key] = int(val)  # FIXME: should we allow fractional scores?

    return ratings


def convert_file(
    input_file,
    yuv_fmt="yuv420p",
    width=0,
    height=0,
    ref_score=5.0,
    ref_dir="",
    dis_dir="",
    def_ext=".yuv",
):
    input_data = pd.read_csv(input_file).to_dict()
    input_data_records = pd.read_csv(input_file).to_dict(orient="records")

    output_data = {
        "dataset_name": os.path.splitext(os.path.basename(input_file))[0],
        "yuv_fmt": yuv_fmt,
        "width": int(width),
        "height": int(height),
        "ref_score": float(ref_score),
        "ref_dir": "",
        "dis_dir": "",
        "ref_videos": [],
        "dis_videos": [],
    }

    if "PVS_ID" not in input_data:
        log_error("No 'PVS_ID' column in input file!")

    # construct empty sets for IDs
    src_ids = set()
    hrc_ids = set()
    pvs_ids = set()

    # gather sets of IDs
    for index, pvs_id in input_data["PVS_ID"].items():
        pvs_id, _, src_id, hrc_id = get_ids_from_pvs(pvs_id)
        pvs_ids.add(pvs_id)
        src_ids.add(src_id)
        hrc_ids.add(hrc_id)

    # gather list of reference contents
    for src_id in src_ids:
        output_data["ref_videos"].append(
            {
                "content_id": int(re.search(r"\d+", src_id).group(0)),
                "content_name": src_id,
                "path": os.path.join(ref_dir, src_id + def_ext),
            }
        )

    # gather list of "distorted" videos, aka PVSes
    for pvs_id in pvs_ids:
        pvs_id, _, src_id, hrc_id = get_ids_from_pvs(pvs_id)

        # check all records for the PVS ID
        ratings = None
        for record in input_data_records:
            record_pvs_id, _, record_src_id, record_hrc_id = get_ids_from_pvs(
                record["PVS_ID"]
            )
            if pvs_id == record_pvs_id:
                ratings = extract_scores_from_record(record)
                break

        if not ratings:
            log_error("Could not find ratings for PVS " + str(pvs_id))

        dis_video_data = {
            "content_id": int(re.search(r"\d+", src_id).group(0)),
            "asset_id": int(re.search(r"\d+", hrc_id).group(0)),
            "os": ratings,
            "path": os.path.join(dis_dir, pvs_id + def_ext),
        }
        output_data["dis_videos"].append(dis_video_data)

    print(json.dumps(output_data, indent=4))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("input", help="Path to MOS-score CSV file to convert")

    parser.add_argument(
        "-y", "--yuv-fmt", help="YUV format", type=str, default="yuv420p"
    )
    parser.add_argument("-rw", "--width", help="Reference width", type=int, default=0)
    parser.add_argument("-rh", "--height", help="Reference height", type=int, default=0)
    parser.add_argument(
        "-s", "--ref-score", help="Reference maximum score", type=float, default=5.0
    )
    parser.add_argument(
        "-r", "--ref-dir", help="Reference file directory", type=str, default=""
    )
    parser.add_argument(
        "-d", "--dis-dir", help="Distorted file directory", type=str, default=""
    )
    parser.add_argument(
        "-e", "--def-ext", help="Default file extension", type=str, default=".yuv"
    )

    args = parser.parse_args()
    convert_file(
        args.input,
        args.yuv_fmt,
        args.width,
        args.height,
        args.ref_score,
        args.ref_dir,
        args.dis_dir,
        args.def_ext,
    )


if __name__ == "__main__":
    main()
