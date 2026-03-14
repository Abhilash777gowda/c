import os
import random
import pandas as pd
from scipy.stats import pearsonr
from utils.helpers import setup_logging

logger = setup_logging()

NCRB_CSV_PATH = "data/ncrb_stats.csv"


class CorrelationValidator:
    def __init__(self, categories):
        self.categories = categories

    # ── NCRB data loaders ────────────────────────────────────────────────────
    def _load_ncrb_csv(self) -> pd.DataFrame | None:
        """Load ground-truth NCRB statistics from CSV if available."""
        if not os.path.exists(NCRB_CSV_PATH):
            return None
        try:
            df = pd.read_csv(NCRB_CSV_PATH, parse_dates=["date"])
            df = df.set_index("date")
            logger.info(f"Real NCRB data loaded from {NCRB_CSV_PATH} ({len(df)} rows).")
            return df
        except Exception as e:
            logger.warning(f"Failed to load NCRB CSV: {e}")
            return None

    def _generate_mock_ncrb_data(self, predicted_trends: pd.DataFrame) -> pd.DataFrame:
        """Fallback: generate correlated mock data to demonstrate the module."""
        logger.info("Using mock NCRB data (no real CSV found at data/ncrb_stats.csv).")
        ncrb_data = {}
        for cat in self.categories:
            base = predicted_trends[cat].values
            noise = [random.randint(-2, 2) for _ in range(len(base))]
            mock = [max(0, v * random.uniform(0.8, 1.2) + n) for v, n in zip(base, noise)]
            ncrb_data[cat] = mock
        return pd.DataFrame(ncrb_data, index=predicted_trends.index)

    def _align_series(self, predicted_trends: pd.DataFrame,
                      ncrb_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align NCRB monthly data to the date range of predicted_trends.
        Uses resampling on the overlapping period.
        """
        try:
            # Resample NCRB to month-end to match predicted_trends index
            ncrb_monthly = ncrb_df.resample("ME").sum()
            # Find the overlapping date range
            start = max(predicted_trends.index.min(), ncrb_monthly.index.min())
            end   = min(predicted_trends.index.max(), ncrb_monthly.index.max())
            if start > end:
                logger.warning("No date overlap between news predictions and NCRB data. "
                               "Using full NCRB series trimmed to prediction length.")
                return ncrb_df.iloc[:len(predicted_trends)]
            aligned = ncrb_monthly.loc[start:end]
            logger.info(f"Date overlap for correlation: {start.date()} → {end.date()} "
                        f"({len(aligned)} months)")
            return aligned
        except Exception as e:
            logger.warning(f"Alignment failed: {e}. Using mock data.")
            return self._generate_mock_ncrb_data(predicted_trends)

    # ── Main API ─────────────────────────────────────────────────────────────
    def calculate_correlation(self, predicted_trends: pd.DataFrame,
                              ncrb_data: pd.DataFrame = None) -> dict:
        logger.info("Calculating Pearson Correlation between news predictions and official data...")

        if ncrb_data is None:
            ncrb_raw = self._load_ncrb_csv()
            if ncrb_raw is not None:
                ncrb_data = self._align_series(predicted_trends, ncrb_raw)
                data_source = "Real NCRB statistics (data/ncrb_stats.csv)"
            else:
                ncrb_data = self._generate_mock_ncrb_data(predicted_trends)
                data_source = "Mock NCRB data (install real CSV for production)"
        else:
            data_source = "Externally provided"

        correlations = {}
        print(f"\n--- Pearson Correlation Validation (News vs NCRB) ---")
        print(f"    Data source: {data_source}")
        print(f"    Predicted months: {len(predicted_trends)}  |  "
              f"NCRB months available: {len(ncrb_data)}")
        print()

        for cat in self.categories:
            pred_col = predicted_trends.get(cat)
            ncrb_col = ncrb_data.get(cat) if hasattr(ncrb_data, 'get') else None

            if pred_col is None or ncrb_col is None:
                print(f"{cat.replace('_', ' ').title():<15}: Column not found — skipped.")
                continue

            # Align lengths by taking the shorter series
            min_len = min(len(pred_col), len(ncrb_col))
            pred_arr = pred_col.values[:min_len]
            real_arr = (ncrb_col.values[:min_len]
                        if hasattr(ncrb_col, 'values') else list(ncrb_col)[:min_len])

            if len(pred_arr) > 1 and pred_arr.std() > 0 and real_arr.std() > 0:
                corr, p_val = pearsonr(pred_arr, real_arr)
                correlations[cat] = {"correlation": corr, "p_value": p_val}
                sig = "*" if p_val < 0.05 else ""
                print(f"{cat.replace('_', ' ').title():<15}: r={corr:+.4f} "
                      f"p={p_val:.4f} {sig}")
            else:
                correlations[cat] = {"correlation": None, "p_value": None}
                print(f"{cat.replace('_', ' ').title():<15}: Insufficient variance — skipped.")

        print("  (* p < 0.05 indicates statistically significant correlation)")
        return correlations
