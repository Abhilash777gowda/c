import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils.helpers import setup_logging

logger = setup_logging()

class TrendAnalyzer:
    def __init__(self, categories):
        self.categories = categories
        sns.set_theme(style="whitegrid")

    def generate_trends(self, df, date_col='date', output_path="plots/crime_trends.png"):
        logger.info("Generating crime trend analysis plots...")
        try:
            # Ensure output dir exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Make a copy and parse dates
            df_plot = df.copy()
            df_plot[date_col] = pd.to_datetime(df_plot[date_col])
            
            # Set index to date
            df_plot.set_index(date_col, inplace=True)
            
            # Resample by month (or 'W' for week) and sum the occurrences
            # We assume the columns exist and hold binary (0/1) indicators
            monthly_trends = df_plot[self.categories].resample('M').sum()
            
            plt.figure(figsize=(12, 6))
            for cat in self.categories:
                sns.lineplot(data=monthly_trends, x=monthly_trends.index, y=cat, label=cat.replace('_', ' ').title(), marker='o')
                
            plt.title('Crime and Accident Trends over Time (News Based)')
            plt.xlabel('Date')
            plt.ylabel('Number of Incidents Detected')
            plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Trend plot saved to {output_path}")
            return monthly_trends
        except Exception as e:
            logger.error(f"Failed to generate trends: {e}")
            raise
