"""Text comparison tool.

This module implements a functionality to extract the diffs of pairs of texts following a pre-built
python comparison algorithm. In addition, diffs are stored as HTML documents for better analysis.
"""

import difflib
import os
import shutil

import pandas as pd


diff = difflib.HtmlDiff(wrapcolumn=70)


def save_diff(diffs_path: str, raw_texts: list | pd.Series, clean_texts: list | pd.Series,
              idxs: list | pd.Series) -> None:
    """Saves the diff files to disk for a set of text pairs"""

    # Create the directory if it doesn´t exist
    os.makedirs(diffs_path, exist_ok=True)

    for (r, c, i) in zip(raw_texts, clean_texts, idxs):
        try:
            html = diff.make_file(r.splitlines(), c.splitlines())
            with open(f"{diffs_path}/{i.replace("/", "_")}.html", "w", encoding="utf-8") as f:
                f.write(html)
        except RecursionError:
            print("Error saving the diff file")


def clean_folder(diffs_path: str) -> None:
    """Clean the diffs folder"""
    if os.path.exists(diffs_path):
        shutil.rmtree(diffs_path)
