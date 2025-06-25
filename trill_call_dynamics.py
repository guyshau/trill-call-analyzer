import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time
import os
from pathlib import Path


class MarmosetTrillAnalyzer:
    """
    Analyzer for temporal patterns of marmoset Trill calls using the MarmAudioDataset.
    """

    def __init__(self, annotations_path):
        """
        Initialize the analyzer with the path to the annotations file.

        Args:
            annotations_path (str): Path to the Annotations.tsv file
        """
        self.annotations_path = annotations_path
        self.df = None
        self.trill_calls = None

    def load_annotations(self):
        """
        Load the annotations file and filter for Trill calls.
        Assumes TSV format with columns that might include:
        - filename, call_type, start_time, end_time, timestamp, etc.
        """
        try:
            # Load the TSV file
            self.df = pd.read_csv(self.annotations_path, sep='\t')
            print(f"Loaded {len(self.df)} annotations")
            print(f"Columns: {list(self.df.columns)}")

            # Display first few rows to understand structure
            print("\nFirst 5 rows:")
            print(self.df.head())

            # Filter for Trill calls (case-insensitive)
            trill_mask = self.df.iloc[:, self._find_call_type_column()].str.lower().str.contains('trill', na=False)
            self.trill_calls = self.df[trill_mask].copy()

            print(f"\nFound {len(self.trill_calls)} Trill calls")

            return True

        except Exception as e:
            print(f"Error loading annotations: {e}")
            return False

    def _find_call_type_column(self):
        """
        Find the column that contains call type information.
        """
        possible_names = ['call_type', 'type', 'label', 'class', 'vocalization_type']
        for i, col in enumerate(self.df.columns):
            if any(name in col.lower() for name in possible_names):
                return i
        # If not found, assume it's the second column (common pattern)
        return 1 if len(self.df.columns) > 1 else 0

    def _find_timestamp_column(self):
        """
        Find the column that contains timestamp information.
        """
        possible_names = ['timestamp', 'time', 'datetime', 'date']
        for i, col in enumerate(self.df.columns):
            if any(name in col.lower() for name in possible_names):
                return i
        return None

    def extract_time_features(self):
        """
        Extract time-based features from existing date/time columns or filenames.
        Handles the MarmAudio dataset structure with separate year, month, day, hour columns.
        """
        if self.trill_calls is None:
            print("No trill calls data loaded. Run load_annotations() first.")
            return

        # Check if we have separate date/time columns (MarmAudio format)
        time_cols = ['year', 'month', 'day', 'hour', 'second', 'millisecond']
        if all(col in self.trill_calls.columns for col in time_cols):
            print("Using existing date/time columns from dataset...")
            self._extract_from_separate_columns()
        else:
            # Try to find timestamp column first
            timestamp_col_idx = self._find_timestamp_column()

            if timestamp_col_idx is not None:
                # Use timestamp column if available
                timestamp_col = self.df.columns[timestamp_col_idx]
                self.trill_calls['datetime'] = pd.to_datetime(self.trill_calls[timestamp_col])
            else:
                # Extract from filename (common pattern: includes date/time info)
                filename_col = 'file_name'  # Use the actual filename column
                self.trill_calls = self._extract_time_from_filename(self.trill_calls, filename_col)

        if 'datetime' in self.trill_calls.columns:
            # Extract time components if not already present
            if 'hour_extracted' not in self.trill_calls.columns:
                self.trill_calls['hour_extracted'] = self.trill_calls['datetime'].dt.hour
                self.trill_calls['minute_extracted'] = self.trill_calls['datetime'].dt.minute
                self.trill_calls['day_of_week'] = self.trill_calls['datetime'].dt.day_name()
                self.trill_calls['date'] = self.trill_calls['datetime'].dt.date

            # Use the hour column (either original or extracted)
            hour_col = 'hour' if 'hour' in self.trill_calls.columns and not self.trill_calls[
                'hour'].isna().all() else 'hour_extracted'

            # Create time periods
            if hour_col in self.trill_calls.columns:
                self.trill_calls['time_period'] = self.trill_calls[hour_col].apply(self._categorize_time_period)

            print("Time features extracted successfully")

            # Print summary of available time data
            if 'hour' in self.trill_calls.columns:
                valid_hours = self.trill_calls['hour'].notna().sum()
                print(f"Valid hour data available for {valid_hours}/{len(self.trill_calls)} Trill calls")
        else:
            print("Could not extract time information from data")

    def _extract_from_separate_columns(self):
        """
        Extract datetime from separate year, month, day, hour, etc. columns.
        """
        # Filter out rows with complete date information
        complete_date_mask = (
                self.trill_calls['year'].notna() &
                self.trill_calls['month'].notna() &
                self.trill_calls['day'].notna()
        )

        print(f"Rows with complete date info: {complete_date_mask.sum()}/{len(self.trill_calls)}")

        # Create datetime for rows with complete date info
        datetime_list = []
        for idx, row in self.trill_calls.iterrows():
            if complete_date_mask.loc[idx]:
                try:
                    year = int(row['year'])
                    month = int(row['month'])
                    day = int(row['day'])
                    hour = int(row['hour']) if pd.notna(row['hour']) else 0
                    second = int(row['second']) if pd.notna(row['second']) else 0
                    millisecond = int(row['millisecond']) if pd.notna(row['millisecond']) else 0

                    # Create datetime
                    dt = datetime(year, month, day, hour, 0, second, millisecond * 1000)
                    datetime_list.append(dt)
                except (ValueError, TypeError):
                    datetime_list.append(None)
            else:
                datetime_list.append(None)

        self.trill_calls['datetime'] = datetime_list

        # Also try to extract from parent_name which seems to contain date info
        if 'parent_name' in self.trill_calls.columns:
            self._extract_from_parent_name()

    def _extract_from_parent_name(self):
        """
        Extract additional date information from parent_name column (e.g., '2020_10_4').
        """

        def parse_parent_name(parent_name):
            if pd.isna(parent_name):
                return None
            try:
                # Handle parent_name format like '2020_10_4'
                parts = str(parent_name).split('_')
                if len(parts) >= 3:
                    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                    return datetime(year, month, day)
            except (ValueError, IndexError):
                pass
            return None

        # Fill missing datetime values using parent_name
        parent_datetime = self.trill_calls['parent_name'].apply(parse_parent_name)

        # Combine datetime from separate columns with parent_name datetime
        self.trill_calls['datetime'] = self.trill_calls['datetime'].fillna(parent_datetime)

        print(
            f"After combining sources: {self.trill_calls['datetime'].notna().sum()}/{len(self.trill_calls)} calls have datetime info")

    def _extract_time_from_filename(self, df, filename_col):
        """
        Extract datetime information from filename.
        Common patterns: YYYYMMDD_HHMMSS, timestamp_YYYYMMDD_HHMMSS, etc.
        Updated to handle non-string values.
        """
        import re

        def parse_filename_time(filename):
            # Handle non-string values
            if pd.isna(filename) or not isinstance(filename, str):
                return None

            # Pattern 1: YYYYMMDD_HHMMSS
            pattern1 = r'(\d{8})_(\d{6})'
            match1 = re.search(pattern1, filename)
            if match1:
                date_str, time_str = match1.groups()
                try:
                    return pd.to_datetime(date_str + time_str, format='%Y%m%d%H%M%S')
                except:
                    pass

            # Pattern 2: Unix timestamp
            pattern2 = r'(\d{10})'
            match2 = re.search(pattern2, filename)
            if match2:
                try:
                    return pd.to_datetime(int(match2.group(1)), unit='s')
                except:
                    pass

            return None

        df['datetime'] = df[filename_col].apply(parse_filename_time)
        return df

    def _categorize_time_period(self, hour):
        """
        Categorize hours into meaningful periods.
        """
        if 6 <= hour < 9:
            return 'Early Morning'
        elif 9 <= hour < 12:
            return 'Late Morning'
        elif 12 <= hour < 15:
            return 'Afternoon'
        elif 15 <= hour < 18:
            return 'Late Afternoon'
        elif 18 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    def analyze_daily_patterns(self):
        """
        Analyze daily patterns of Trill calls.
        """
        # Determine which hour column to use
        hour_col = None
        if 'hour' in self.trill_calls.columns and not self.trill_calls['hour'].isna().all():
            hour_col = 'hour'
        elif 'hour_extracted' in self.trill_calls.columns and not self.trill_calls['hour_extracted'].isna().all():
            hour_col = 'hour_extracted'

        if hour_col is None:
            print("No hour data available. Run extract_time_features() first.")
            return None, None

        print("=== DAILY TRILL CALL PATTERNS ===\n")

        # Filter out NaN values for analysis
        valid_hour_data = self.trill_calls[self.trill_calls[hour_col].notna()]
        print(f"Analyzing {len(valid_hour_data)} Trill calls with valid hour data")

        # Hourly distribution
        hourly_counts = valid_hour_data[hour_col].value_counts().sort_index()
        print("Hourly distribution of Trill calls:")
        for hour, count in hourly_counts.items():
            print(f"  {int(hour):02d}:00 - {count} calls")

        # Time period distribution
        if 'time_period' in valid_hour_data.columns:
            period_counts = valid_hour_data['time_period'].value_counts()
            print(f"\nTime period distribution:")
            for period, count in period_counts.items():
                percentage = (count / len(valid_hour_data)) * 100
                print(f"  {period}: {count} calls ({percentage:.1f}%)")
        else:
            period_counts = None

        # Peak hours
        if len(hourly_counts) > 0:
            peak_hour = hourly_counts.idxmax()
            peak_count = hourly_counts.max()
            print(f"\nPeak hour: {int(peak_hour):02d}:00 with {peak_count} calls")

        return hourly_counts, period_counts

    def plot_temporal_patterns(self, save_plots=True):
        """
        Create visualizations of temporal patterns.
        """
        # Determine which hour column to use
        hour_col = None
        if 'hour' in self.trill_calls.columns and not self.trill_calls['hour'].isna().all():
            hour_col = 'hour'
        elif 'hour_extracted' in self.trill_calls.columns and not self.trill_calls['hour_extracted'].isna().all():
            hour_col = 'hour_extracted'

        if hour_col is None:
            print("No hour data available. Run extract_time_features() first.")
            return

        # Filter out NaN values for plotting
        valid_data = self.trill_calls[self.trill_calls[hour_col].notna()]

        if len(valid_data) == 0:
            print("No valid hour data available for plotting.")
            return

        print(f"Plotting temporal patterns for {len(valid_data)} Trill calls with valid time data")

        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Marmoset Trill Call Temporal Patterns', fontsize=16, fontweight='bold')

        # 1. Hourly distribution
        hourly_counts = valid_data[hour_col].value_counts().sort_index()
        axes[0, 0].bar(hourly_counts.index, hourly_counts.values, color='skyblue', alpha=0.7)
        axes[0, 0].set_xlabel('Hour of Day')
        axes[0, 0].set_ylabel('Number of Trill Calls')
        axes[0, 0].set_title('Hourly Distribution of Trill Calls')
        axes[0, 0].set_xticks(range(int(hourly_counts.index.min()), int(hourly_counts.index.max()) + 1, 2))
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Time period distribution
        if 'time_period' in valid_data.columns:
            period_counts = valid_data['time_period'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(period_counts)))
            axes[0, 1].pie(period_counts.values, labels=period_counts.index, autopct='%1.1f%%',
                           colors=colors, startangle=90)
            axes[0, 1].set_title('Distribution by Time Period')
        else:
            axes[0, 1].text(0.5, 0.5, 'Time period data\nnot available',
                            ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Time Period Distribution (N/A)')

        # 3. Daily pattern (if multiple days available)
        if 'date' in valid_data.columns and len(valid_data['date'].unique()) > 1:
            daily_counts = valid_data['date'].value_counts().sort_index()
            axes[1, 0].plot(daily_counts.index, daily_counts.values, marker='o', linewidth=2)
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Number of Trill Calls')
            axes[1, 0].set_title('Daily Variation in Trill Calls')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Multiple days data\nnot available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Daily Variation (N/A)')

        # 4. Heatmap of hour vs day of week
        if 'day_of_week' in valid_data.columns and len(valid_data['day_of_week'].unique()) > 1:
            pivot_table = pd.crosstab(valid_data['day_of_week'], valid_data[hour_col])
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            available_days = [day for day in day_order if day in pivot_table.index]
            if available_days:
                pivot_table = pivot_table.reindex(available_days, fill_value=0)

                sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
                axes[1, 1].set_xlabel('Hour of Day')
                axes[1, 1].set_ylabel('Day of Week')
                axes[1, 1].set_title('Trill Calls: Hour vs Day of Week')
            else:
                axes[1, 1].text(0.5, 0.5, 'Day of week data\nnot sufficient',
                                ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Hour vs Day Heatmap (N/A)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Day of week data\nnot available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Hour vs Day Heatmap (N/A)')

        plt.tight_layout()

        if save_plots:
            plt.savefig('trill_temporal_patterns.png', dpi=300, bbox_inches='tight')
            print("Plot saved as 'trill_temporal_patterns.png'")

        plt.show()

    def generate_report(self):
        """
        Generate a comprehensive report of findings.
        """
        if self.trill_calls is None:
            print("No data loaded. Run load_annotations() first.")
            return

        print("=" * 60)
        print("MARMOSET TRILL CALL TEMPORAL ANALYSIS REPORT")
        print("=" * 60)

        print(f"\nDataset Overview:")
        print(f"- Total annotations: {len(self.df)}")
        print(f"- Total Trill calls: {len(self.trill_calls)}")
        print(f"- Percentage of Trill calls: {(len(self.trill_calls) / len(self.df) * 100):.2f}%")

        if 'hour' in self.trill_calls.columns:
            hourly_counts, period_counts = self.analyze_daily_patterns()

            print(f"\nKey Findings:")
            peak_hour = hourly_counts.idxmax()
            print(f"- Peak activity hour: {int(peak_hour):02d}:00")

            # Find morning vs afternoon preference
            morning_calls = sum(hourly_counts[6:12])
            afternoon_calls = sum(hourly_counts[12:18])
            evening_calls = sum(hourly_counts[18:24])
            night_calls = sum(hourly_counts[0:6])

            print(f"- Morning calls (6-12h): {morning_calls} ({morning_calls / len(self.trill_calls) * 100:.1f}%)")
            print(f"- Afternoon calls (12-18h): {afternoon_calls} ({afternoon_calls / len(self.trill_calls) * 100:.1f}%)")
            print(f"- Evening calls (18-24h): {evening_calls} ({evening_calls / len(self.trill_calls) * 100:.1f}%)")
            print(f"- Night calls (0-6h): {night_calls} ({night_calls / len(self.trill_calls) * 100:.1f}%)")

            # Identify patterns
            if morning_calls > afternoon_calls and morning_calls > evening_calls:
                print(f"\n✓ PATTERN IDENTIFIED: Morning preference detected!")
                print(f"  Trill calls are most frequent in the morning hours (6-12h)")
            elif afternoon_calls > morning_calls and afternoon_calls > evening_calls:
                print(f"\n✓ PATTERN IDENTIFIED: Afternoon preference detected!")
                print(f"  Trill calls are most frequent in the afternoon hours (12-18h)")
            else:
                print(f"\n- No clear time-of-day preference detected")

        print("\n" + "=" * 60)


# Usage example and main execution
def main():
    """
    Main function to run the analysis.
    """
    # Adjust this path to your actual annotations file
    annotations_path = "Annotations.tsv"  # Change this to your actual path

    # Check if file exists
    if not os.path.exists(annotations_path):
        print(f"Annotations file not found at: {annotations_path}")
        print("Please update the path to your Annotations.tsv file")

        # Create a sample dataset for demonstration
        print("\nCreating sample data for demonstration...")
        create_sample_data()
        annotations_path = "sample_annotations.tsv"

    # Initialize analyzer
    analyzer = MarmosetTrillAnalyzer(annotations_path)

    # Load and analyze data
    if analyzer.load_annotations():
        analyzer.extract_time_features()
        analyzer.plot_temporal_patterns()
        analyzer.generate_report()


def create_sample_data():
    """
    Create sample data for demonstration purposes.
    """
    import random
    from datetime import datetime, timedelta

    # Create sample data
    np.random.seed(42)

    call_types = ['Trill', 'Twitter', 'Chatter', 'Phee', 'Trill', 'Twitter', 'Trill']

    # Generate more trills in morning hours (simulate feeding time pattern)
    sample_data = []
    base_date = datetime(2023, 6, 1)

    for day in range(30):  # 30 days of data
        current_date = base_date + timedelta(days=day)

        # Generate calls throughout the day with morning bias for trills
        for hour in range(6, 20):  # Active hours 6 AM to 8 PM
            # Morning hours (6-10) have higher probability of trills
            if 6 <= hour <= 10:
                n_calls = np.random.poisson(8)  # More calls in morning
                trill_prob = 0.4  # Higher trill probability
            elif 11 <= hour <= 14:  # Post-feeding period
                n_calls = np.random.poisson(6)
                trill_prob = 0.3
            else:
                n_calls = np.random.poisson(3)
                trill_prob = 0.2

            for _ in range(n_calls):
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)

                call_time = current_date.replace(hour=hour, minute=minute, second=second)

                # Choose call type based on probability
                if np.random.random() < trill_prob:
                    call_type = 'Trill'
                else:
                    call_type = np.random.choice(['Twitter', 'Chatter', 'Phee'])

                filename = f"recording_{call_time.strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}.flac"

                sample_data.append({
                    'filename': filename,
                    'call_type': call_type,
                    'timestamp': call_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'confidence': np.random.uniform(0.7, 1.0),
                    'duration': np.random.uniform(0.5, 3.0)
                })

    # Create DataFrame and save
    df_sample = pd.DataFrame(sample_data)
    df_sample.to_csv('sample_annotations.tsv', sep='\t', index=False)
    print(f"Created sample dataset with {len(df_sample)} annotations")
    print(f"Trill calls in sample: {len(df_sample[df_sample['call_type'] == 'Trill'])}")


if __name__ == "__main__":
    main()