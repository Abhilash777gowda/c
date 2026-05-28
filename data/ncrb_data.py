"""
NCRB (National Crime Records Bureau) reference statistics generator.
Produces a CSV with synthetic but plausible yearly crime statistics for India.
"""

import os
import pandas as pd
from utils.helpers import setup_logging

logger = setup_logging()

CRIME_CATEGORIES = ['theft', 'assault', 'accident', 'drug_crime', 'cybercrime', 'non_crime']

# Approximate NCRB-style statistics (incidents per year, synthetic)
_NCRB_STATS = {
    'year':       [2018,   2019,   2020,   2021,   2022],
    'theft':      [347411, 331688, 279044, 264786, 298000],
    'assault':    [398582, 421148, 350290, 375000, 390000],
    'accident':   [467044, 449002, 373398, 412432, 461312],
    'drug_crime': [210000, 240000, 220000, 255000, 278000],
    'cybercrime': [ 27248,  44546,  50035,  52974,  65893],
    'non_crime':  [      0,      0,      0,      0,      0],
}


def generate_ncrb_csv(output_path: str = "data/ncrb_stats.csv") -> pd.DataFrame:
    """
    Generate a CSV containing NCRB-style yearly crime statistics.

    Parameters
    ----------
    output_path : str
        Destination path for the CSV file.

    Returns
    -------
    pd.DataFrame
        The generated dataframe.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(_NCRB_STATS)
    df.to_csv(output_path, index=False)
    logger.info(f"NCRB reference statistics saved to {output_path}")
    return df
