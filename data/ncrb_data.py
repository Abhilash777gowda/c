"""
NCRB (National Crime Records Bureau) mock statistics for India.
Based on publicly reported trends from NCRB Annual Crime in India Reports.
Categories aligned to CRIMSON-India classification scheme.

In production: replace with actual NCRB CSV data loaded directly.
Source reference: NCRB Crime in India Reports 2018-2022
https://ncrb.gov.in/en/crime-india
"""
import pandas as pd
import os

# Annual NCRB figures (approximate, in thousands of reported cases nationally)
# Source: NCRB Crime in India 2018–2022 published reports
NCRB_ANNUAL_DATA = {
    "year": [2018, 2019, 2020, 2021, 2022],
    # IPC registered theft cases (Sec 378-382) — in thousands
    "theft":      [302.1, 284.3, 213.4, 230.6, 245.3],
    # Assault / hurt / grievous hurt cases
    "assault":    [395.6, 405.2, 336.4, 379.1, 401.8],
    # Road accidents / accidental deaths (MORTH + NCRB)
    "accident":   [467.0, 449.0, 374.0, 412.0, 461.0],
    # Drug / narcotic substance cases (NDPS Act)
    "drug_crime": [ 60.8,  63.6,  59.8,  67.8,  74.0],
    # Cybercrime (IPC + IT Act)
    "cybercrime": [ 27.2,  44.5,  50.0,  52.9,  65.9],
    # Non-crime baseline (inverse proxy — arrests for non-cognisable offences)
    "non_crime":  [320.0, 310.0, 280.0, 295.0, 315.0],
}


def generate_ncrb_csv(output_path: str = "data/ncrb_stats.csv") -> pd.DataFrame:
    """
    Write the NCRB reference dataset to CSV and return as DataFrame.
    Monthly data is approximated by distributing annual totals across 12 months
    with seasonal variation (crimes peak in summer months Q2/Q3).
    """
    annual = pd.DataFrame(NCRB_ANNUAL_DATA)

    # Expand to monthly by distributing with seasonal weights
    seasonal_weights = [
        0.075, 0.075, 0.080, 0.085, 0.092, 0.095,  # Jan-Jun (rising toward summer)
        0.095, 0.090, 0.085, 0.083, 0.073, 0.072,  # Jul-Dec (peak Jul, drops Dec)
    ]

    rows = []
    categories = ["theft", "assault", "accident", "drug_crime", "cybercrime", "non_crime"]

    for _, row in annual.iterrows():
        year = int(row["year"])
        for month_idx, weight in enumerate(seasonal_weights):
            month = month_idx + 1
            record = {"date": pd.Timestamp(year=year, month=month, day=1)}
            for cat in categories:
                record[cat] = round(row[cat] * weight * 1000)  # scale to cases
            rows.append(record)

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_ncrb_csv()
    print(f"NCRB data generated: {len(df)} monthly rows")
    print(df.head(12).to_string())
