import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, time, timedelta
import os
from pathlib import Path
from collections import Counter, defaultdict
import networkx as nx
from scipy.stats import chi2_contingency, fisher_exact
import warnings

warnings.filterwarnings('ignore')


class MarmosetSequenceAnalyzer:
    """
    Advanced analyzer for marmoset call sequences and response patterns.
    Focuses on call transitions, especially patterns involving Trill calls.
    """

    def __init__(self, annotations_path, metadata_path=None):
        """
        Initialize the analyzer with paths to annotations and optional metadata.

        Args:
            annotations_path (str): Path to the Annotations.tsv file
            metadata_path (str): Path to the Metadata.tsv file (optional)
        """
        self.annotations_path = annotations_path
        self.metadata_path = metadata_path
        self.df = None
        self.metadata = None
        self.trill_calls = None
        self.call_sequences = None
        self.transition_matrix = None
        self.call_types = []

    def load_data(self):
        """
        Load annotations and metadata files.
        """
        try:
            # Load annotations
            self.df = pd.read_csv(self.annotations_path, sep='\t')
            print(f"Loaded {len(self.df)} annotations")
            print(f"Annotation columns: {list(self.df.columns)}")

            # Load metadata if available
            if self.metadata_path and os.path.exists(self.metadata_path):
                self.metadata = pd.read_csv(self.metadata_path, sep='\t')
                print(f"Loaded metadata with {len(self.metadata)} entries")
                print(f"Metadata columns: {list(self.metadata.columns)}")
            else:
                print("No metadata file provided or found")

            # Display sample data
            print("\nFirst 5 annotation rows:")
            print(self.df.head())

            # Identify call type column
            call_type_col_idx = self._find_call_type_column()
            call_type_col = self.df.columns[call_type_col_idx]

            # Get unique call types
            self.call_types = self.df[call_type_col].dropna().unique()
            print(f"\nFound call types: {self.call_types}")

            # Filter for Trill calls
            trill_mask = self.df[call_type_col].str.lower().str.contains('trill', na=False)
            self.trill_calls = self.df[trill_mask].copy()
            print(f"Found {len(self.trill_calls)} Trill calls")

            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def _find_call_type_column(self):
        """Find the column containing call type information."""
        possible_names = ['call_type', 'type', 'label', 'class', 'vocalization_type', 'voc_type']
        for i, col in enumerate(self.df.columns):
            if any(name in col.lower() for name in possible_names):
                return i
        return 1 if len(self.df.columns) > 1 else 0

    def _find_time_column(self):
        """Find the column containing time information."""
        possible_names = ['start_time', 'timestamp', 'time', 'start', 'begin_time']
        for i, col in enumerate(self.df.columns):
            if any(name in col.lower() for name in possible_names):
                return i
        return None

    def _find_file_column(self):
        """Find the column containing file information."""
        possible_names = ['filename', 'file_name', 'file', 'recording', 'audio_file']
        for i, col in enumerate(self.df.columns):
            if any(name in col.lower() for name in possible_names):
                return i
        return 0

    def prepare_sequences(self, max_gap_seconds=30):
        """
        Prepare call sequences by grouping calls within time windows.

        Args:
            max_gap_seconds (int): Maximum gap between calls to consider them in sequence
        """
        if self.df is None:
            print("No data loaded. Run load_data() first.")
            return False

        try:
            call_type_col = self.df.columns[self._find_call_type_column()]
            time_col_idx = self._find_time_column()
            file_col_idx = self._find_file_column()

            # Create working dataframe
            work_df = self.df.copy()

            # Handle time information
            if time_col_idx is not None:
                time_col = self.df.columns[time_col_idx]
                # Try to parse time column
                try:
                    work_df['timestamp'] = pd.to_datetime(work_df[time_col], errors='coerce')
                except:
                    # If direct parsing fails, try to extract from various formats
                    work_df['timestamp'] = work_df[time_col].apply(self._parse_time_flexible)
            else:
                # Extract time from filename or other columns
                work_df['timestamp'] = self._extract_time_from_available_data(work_df)

            # If we have file information, group by file first
            file_col = 'parent_name'

            # Group calls into sequences
            sequences = []

            if 'timestamp' in work_df.columns and work_df['timestamp'].notna().any():
                # Time-based sequencing
                work_df = work_df.sort_values(['timestamp']).reset_index(drop=True)

                current_sequence = []
                last_time = None

                for idx, row in work_df.iterrows():
                    if pd.isna(row['timestamp']):
                        continue

                    call_type = row[call_type_col]
                    if pd.isna(call_type):
                        continue

                    current_time = row['timestamp']

                    # Check if this call belongs to current sequence
                    if (last_time is None or
                            (current_time - last_time).total_seconds() <= max_gap_seconds):
                        current_sequence.append({
                            'call_type': call_type,
                            'timestamp': current_time,
                            'index': idx
                        })
                    else:
                        # Start new sequence
                        if len(current_sequence) > 1:
                            sequences.append(current_sequence)
                        current_sequence = [{
                            'call_type': call_type,
                            'timestamp': current_time,
                            'index': idx
                        }]

                    last_time = current_time

                # Add final sequence
                if len(current_sequence) > 1:
                    sequences.append(current_sequence)

            else:
                # Parent-based sequencing (group by parent)
                print("Using file-based sequencing...")

                file_col = 'parent_name'
                sequences = []
                grouped = work_df.groupby(file_col)

                for i, (parent_name, group_df) in enumerate(grouped):
                    if i % 1000 == 0:
                        print(f"Processing group {i + 1}/{len(grouped)}...")

                    # Filter valid calls, excluding 'Vocalization'
                    valid_calls = group_df[
                        (~group_df[call_type_col].isna()) &
                        (group_df[call_type_col].str.lower() != 'vocalization')
                        ]

                    if len(valid_calls) > 1:
                        sequence = [
                            {
                                'call_type': row[call_type_col],
                                'file': row[file_col],
                                'index': idx
                            }
                            for idx, row in valid_calls.iterrows()
                        ]
                        sequences.append(sequence)

                self.call_sequences = sequences
                print(f"Created {len(sequences)} call sequences")

            # Print sequence statistics
            if sequences:
                seq_lengths = [len(seq) for seq in sequences]
                print(
                    f"Sequence length stats: min={min(seq_lengths)}, max={max(seq_lengths)}, mean={np.mean(seq_lengths):.1f}")

                # Count sequences containing trills
                trill_sequences = [seq for seq in sequences if
                                   any('trill' in call['call_type'].lower() for call in seq)]
                print(f"Sequences containing Trill calls: {len(trill_sequences)}")

            return True

        except Exception as e:
            print(f"Error preparing sequences: {e}")
            return False

    def _parse_time_flexible(self, time_str):
        """Flexible time parsing for various formats."""
        if pd.isna(time_str):
            return None

        try:
            # Try common formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%d/%m/%Y %H:%M:%S',
                '%H:%M:%S',
                '%H:%M:%S.%f'
            ]

            for fmt in formats:
                try:
                    return pd.to_datetime(time_str, format=fmt)
                except:
                    continue

            # Try pandas auto-parsing
            return pd.to_datetime(time_str, errors='coerce')
        except:
            return None

    def _extract_time_from_available_data(self, df):
        """Extract time information from available columns."""
        # Check for separate time columns
        time_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']
        if all(col in df.columns for col in time_columns[:4]):  # At least year, month, day, hour
            timestamps = []
            for idx, row in df.iterrows():
                try:
                    year = int(row['year']) if pd.notna(row['year']) else 2023
                    month = int(row['month']) if pd.notna(row['month']) else 1
                    day = int(row['day']) if pd.notna(row['day']) else 1
                    hour = int(row['hour']) if pd.notna(row['hour']) else 0
                    minute = int(row['minute']) if pd.notna(row['minute']) else 0
                    second = int(row['second']) if pd.notna(row['second']) else 0

                    timestamps.append(datetime(year, month, day, hour, minute, second))
                except:
                    timestamps.append(None)
            return timestamps

        return [None] * len(df)

    def analyze_transition_patterns(self):
        """
        Analyze call transition patterns and create transition matrices.
        """
        if not self.call_sequences:
            print("No sequences prepared. Run prepare_sequences() first.")
            return None

        # Create transition matrix
        transitions = defaultdict(lambda: defaultdict(int))
        all_call_types = set()

        # Collect all transitions
        for sequence in self.call_sequences:
            for i in range(len(sequence) - 1):
                from_call = sequence[i]['call_type']
                to_call = sequence[i + 1]['call_type']

                # Normalize call types (handle case variations)
                from_call = from_call.strip().title()
                to_call = to_call.strip().title()

                transitions[from_call][to_call] += 1
                all_call_types.add(from_call)
                all_call_types.add(to_call)

        # Convert to matrix format
        call_types_list = sorted(list(all_call_types))
        n_types = len(call_types_list)

        transition_matrix = np.zeros((n_types, n_types))

        for i, from_call in enumerate(call_types_list):
            for j, to_call in enumerate(call_types_list):
                transition_matrix[i, j] = transitions[from_call][to_call]

        self.transition_matrix = transition_matrix
        self.call_types_ordered = call_types_list

        print("=== CALL TRANSITION ANALYSIS ===\n")

        # Analyze Trill-specific patterns
        self._analyze_trill_patterns(transitions)

        # Calculate transition probabilities
        self._calculate_transition_probabilities(transition_matrix, call_types_list)

        return transitions

    def _analyze_trill_patterns(self, transitions):
        """Analyze patterns specifically involving Trill calls."""

        # Find Trill variations in the data
        trill_variants = [call for call in self.call_types_ordered if 'trill' in call.lower()]

        if not trill_variants:
            print("No Trill calls found in sequences.")
            return

        print(f"Found Trill variants: {trill_variants}")

        for trill_type in trill_variants:
            print(f"\n--- Analysis for {trill_type} ---")

            # What follows Trill calls?
            following_calls = transitions[trill_type]
            if following_calls:
                print(f"Calls following {trill_type}:")
                sorted_following = sorted(following_calls.items(), key=lambda x: x[1], reverse=True)
                total_following = sum(following_calls.values())

                for call_type, count in sorted_following:
                    percentage = (count / total_following) * 100
                    print(f"  {call_type}: {count} times ({percentage:.1f}%)")

                # Check for self-transitions (Trill -> Trill)
                if trill_type in following_calls:
                    self_transitions = following_calls[trill_type]
                    self_percentage = (self_transitions / total_following) * 100
                    print(
                        f"  >>> {trill_type} -> {trill_type}: {self_transitions} times ({self_percentage:.1f}%) - SELF-REPETITION")

            # What precedes Trill calls?
            preceding_calls = {}
            for from_call, to_calls in transitions.items():
                if trill_type in to_calls:
                    preceding_calls[from_call] = to_calls[trill_type]

            if preceding_calls:
                print(f"\nCalls preceding {trill_type}:")
                sorted_preceding = sorted(preceding_calls.items(), key=lambda x: x[1], reverse=True)
                total_preceding = sum(preceding_calls.values())

                for call_type, count in sorted_preceding:
                    percentage = (count / total_preceding) * 100
                    print(f"  {call_type}: {count} times ({percentage:.1f}%)")

    def _calculate_transition_probabilities(self, matrix, call_types):
        """Calculate and display transition probabilities."""
        print(f"\n=== TRANSITION PROBABILITY MATRIX ===")

        # Normalize by rows to get probabilities
        row_sums = matrix.sum(axis=1)
        prob_matrix = np.divide(matrix, row_sums[:, np.newaxis],
                                out=np.zeros_like(matrix), where=row_sums[:, np.newaxis] != 0)

        # Create DataFrame for better display
        prob_df = pd.DataFrame(prob_matrix, index=call_types, columns=call_types)

        # Display high-probability transitions
        print("\nHigh-probability transitions (>20%):")
        for i, from_call in enumerate(call_types):
            for j, to_call in enumerate(call_types):
                prob = prob_matrix[i, j]
                if prob > 0.2:  # More than 20%
                    count = int(matrix[i, j])
                    print(f"  {from_call} -> {to_call}: {prob:.3f} ({count} occurrences)")

    def plot_transition_network(self, min_transitions=3, save_plot=True):
        """
        Create a network visualization of call transitions.

        Args:
            min_transitions (int): Minimum number of transitions to include in plot
            save_plot (bool): Whether to save the plot
        """
        if self.transition_matrix is None:
            print("No transition matrix available. Run analyze_transition_patterns() first.")
            return

        # Create network graph
        G = nx.DiGraph()

        # Add nodes
        for call_type in self.call_types_ordered:
            G.add_node(call_type)

        # Add edges with weights
        for i, from_call in enumerate(self.call_types_ordered):
            for j, to_call in enumerate(self.call_types_ordered):
                weight = self.transition_matrix[i, j]
                if weight >= min_transitions:
                    G.add_edge(from_call, to_call, weight=weight)

        # Create plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=3, iterations=50)

        # Draw nodes
        node_colors = ['red' if 'trill' in node.lower() else 'lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.7)

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        # Normalize weights for line thickness
        normalized_weights = [w / max_weight * 5 + 0.5 for w in weights]

        nx.draw_networkx_edges(G, pos, width=normalized_weights, alpha=0.6,
                               edge_color='gray', arrows=True, arrowsize=20)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        # Add edge labels for significant transitions
        edge_labels = {}
        for u, v in edges:
            weight = G[u][v]['weight']
            if weight >= min_transitions * 2:  # Only show labels for strong transitions
                edge_labels[(u, v)] = str(weight)

        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)

        plt.title(f'Marmoset Call Transition Network\n(Minimum {min_transitions} transitions shown)',
                  fontsize=14, fontweight='bold')
        plt.legend(['Trill calls', 'Other calls'], loc='upper right')
        plt.axis('off')
        plt.tight_layout()

        if save_plot:
            plt.savefig('call_transition_network.png', dpi=300, bbox_inches='tight')
            print("Network plot saved as 'call_transition_network.png'")

        plt.show()

    def plot_transition_heatmap(self, save_plot=True):
        """
        Create a heatmap visualization of transition probabilities.
        """
        if self.transition_matrix is None:
            print("No transition matrix available. Run analyze_transition_patterns() first.")
            return

        # Calculate probability matrix
        row_sums = self.transition_matrix.sum(axis=1)
        prob_matrix = np.divide(self.transition_matrix, row_sums[:, np.newaxis],
                                out=np.zeros_like(self.transition_matrix),
                                where=row_sums[:, np.newaxis] != 0)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(prob_matrix,
                    xticklabels=self.call_types_ordered,
                    yticklabels=self.call_types_ordered,
                    annot=True, fmt='.2f', cmap='YlOrRd',
                    cbar_kws={'label': 'Transition Probability'})

        plt.title('Call-to-Call Transition Probability Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('To Call Type')
        plt.ylabel('From Call Type')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_plot:
            plt.savefig('transition_probability_heatmap.png', dpi=300, bbox_inches='tight')
            print("Heatmap saved as 'transition_probability_heatmap.png'")

        plt.show()

    def statistical_analysis(self):
        """
        Perform statistical tests on transition patterns.
        """
        if self.transition_matrix is None:
            print("No transition matrix available. Run analyze_transition_patterns() first.")
            return

        print("=== STATISTICAL ANALYSIS OF TRANSITIONS ===\n")

        # Chi-square test for independence
        try:
            chi2, p_value, dof, expected = chi2_contingency(self.transition_matrix)
            print(f"Chi-square test for independence of call transitions:")
            print(f"  Chi-square statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.6f}")
            print(f"  Degrees of freedom: {dof}")

            if p_value < 0.05:
                print("  Result: Call transitions are NOT random (p < 0.05)")
                print("  This suggests structured communication patterns!")
            else:
                print("  Result: Call transitions appear random (p >= 0.05)")
        except Exception as e:
            print(f"Could not perform chi-square test: {e}")

        # Analyze self-transitions (same call type following itself)
        print(f"\n=== SELF-TRANSITION ANALYSIS ===")

        for i, call_type in enumerate(self.call_types_ordered):
            self_transitions = self.transition_matrix[i, i]
            total_from_this_type = self.transition_matrix[i, :].sum()

            if total_from_this_type > 0:
                self_prob = self_transitions / total_from_this_type
                print(
                    f"{call_type}: {self_transitions:.0f}/{total_from_this_type:.0f} = {self_prob:.3f} probability of repetition")

    def generate_comprehensive_report(self):
        """
        Generate a comprehensive report of all findings.
        """
        print("=" * 80)
        print("COMPREHENSIVE MARMOSET CALL SEQUENCE ANALYSIS REPORT")
        print("=" * 80)

        if self.df is not None:
            print(f"\nDataset Overview:")
            print(f"- Total annotations: {len(self.df)}")
            if self.trill_calls is not None:
                print(f"- Total Trill calls: {len(self.trill_calls)}")
                print(f"- Percentage of Trill calls: {(len(self.trill_calls) / len(self.df) * 100):.2f}%")

        if self.call_sequences:
            print(f"\nSequence Analysis:")
            print(f"- Total sequences identified: {len(self.call_sequences)}")

            seq_lengths = [len(seq) for seq in self.call_sequences]
            print(f"- Average sequence length: {np.mean(seq_lengths):.1f} calls")
            print(f"- Longest sequence: {max(seq_lengths)} calls")

            trill_sequences = [seq for seq in self.call_sequences if
                               any('trill' in call['call_type'].lower() for call in seq)]
            print(
                f"- Sequences containing Trill calls: {len(trill_sequences)} ({len(trill_sequences) / len(self.call_sequences) * 100:.1f}%)")

        if self.transition_matrix is not None:
            print(f"\nTransition Pattern Summary:")
            total_transitions = self.transition_matrix.sum()
            print(f"- Total call transitions analyzed: {total_transitions:.0f}")

            # Find most common transitions
            max_idx = np.unravel_index(np.argmax(self.transition_matrix), self.transition_matrix.shape)
            most_common_from = self.call_types_ordered[max_idx[0]]
            most_common_to = self.call_types_ordered[max_idx[1]]
            most_common_count = self.transition_matrix[max_idx]

            print(f"- Most common transition: {most_common_from} -> {most_common_to} ({most_common_count:.0f} times)")

            # Check for Trill patterns
            trill_indices = [i for i, call in enumerate(self.call_types_ordered) if 'trill' in call.lower()]
            if trill_indices:
                print(f"\nTrill Call Insights:")
                for trill_idx in trill_indices:
                    trill_name = self.call_types_ordered[trill_idx]

                    # Self-transitions
                    self_transitions = self.transition_matrix[trill_idx, trill_idx]
                    total_from_trill = self.transition_matrix[trill_idx, :].sum()
                    if total_from_trill > 0:
                        self_prob = self_transitions / total_from_trill
                        print(f"- {trill_name} repetition rate: {self_prob:.1%}")

                    # What follows trills most often
                    following_idx = np.argmax(self.transition_matrix[trill_idx, :])
                    if following_idx != trill_idx:  # If not self-transition
                        most_following = self.call_types_ordered[following_idx]
                        following_count = self.transition_matrix[trill_idx, following_idx]
                        if total_from_trill > 0:
                            following_prob = following_count / total_from_trill
                            print(f"- Most common call after {trill_name}: {most_following} ({following_prob:.1%})")

        print("\n" + "=" * 80)


# Enhanced main function with sequence analysis
def main():
    """
    Main function to run the enhanced sequence analysis.
    """
    # File paths - adjust these to your actual file locations
    annotations_path = "Annotations.tsv"
    metadata_path = "Metadata.tsv"  # Optional

    # Check if files exist
    if not os.path.exists(annotations_path):
        print(f"Annotations file not found at: {annotations_path}")
        print("Creating sample data for demonstration...")
        create_enhanced_sample_data()
        annotations_path = "sample_annotations_sequences.tsv"

    # Initialize analyzer
    analyzer = MarmosetSequenceAnalyzer(annotations_path, metadata_path)

    # Run analysis pipeline
    if analyzer.load_data():
        print("\n" + "=" * 50)
        print("PREPARING CALL SEQUENCES...")
        print("=" * 50)

        if analyzer.prepare_sequences(max_gap_seconds=30):
            print("\n" + "=" * 50)
            print("ANALYZING TRANSITION PATTERNS...")
            print("=" * 50)

            analyzer.analyze_transition_patterns()

            print("\n" + "=" * 50)
            print("CREATING VISUALIZATIONS...")
            print("=" * 50)

            analyzer.plot_transition_network(min_transitions=2)
            analyzer.plot_transition_heatmap()

            print("\n" + "=" * 50)
            print("STATISTICAL ANALYSIS...")
            print("=" * 50)

            analyzer.statistical_analysis()

            print("\n" + "=" * 50)
            print("GENERATING COMPREHENSIVE REPORT...")
            print("=" * 50)

            analyzer.generate_comprehensive_report()


def create_enhanced_sample_data():
    """
    Create enhanced sample data with realistic call sequences.
    """
    import random
    from datetime import datetime, timedelta

    np.random.seed(42)
    random.seed(42)

    # Define call types and their transition probabilities
    call_types = ['Trill', 'Twitter', 'Phee', 'Chatter', 'Chirp', 'Whistle']

    # Realistic transition probabilities based on research
    # Higher probability of Trill->Trill, Phee->Phee (self-repetition)
    # Trills often followed by Twitter or Chatter
    transition_probs = {
        'Trill': {'Trill': 0.3, 'Twitter': 0.25, 'Chatter': 0.2, 'Phee': 0.15, 'Chirp': 0.05, 'Whistle': 0.05},
        'Twitter': {'Twitter': 0.2, 'Trill': 0.15, 'Chatter': 0.3, 'Phee': 0.2, 'Chirp': 0.1, 'Whistle': 0.05},
        'Phee': {'Phee': 0.4, 'Trill': 0.2, 'Twitter': 0.15, 'Chatter': 0.15, 'Chirp': 0.05, 'Whistle': 0.05},
        'Chatter': {'Chatter': 0.25, 'Twitter': 0.3, 'Trill': 0.2, 'Phee': 0.15, 'Chirp': 0.05, 'Whistle': 0.05},
        'Chirp': {'Chirp': 0.3, 'Twitter': 0.25, 'Trill': 0.2, 'Chatter': 0.15, 'Phee': 0.05, 'Whistle': 0.05},
        'Whistle': {'Whistle': 0.2, 'Phee': 0.3, 'Trill': 0.2, 'Twitter': 0.15, 'Chatter': 0.1, 'Chirp': 0.05}
    }

    sample_data = []
    base_date = datetime(2023, 6, 1)

    # Generate sequences over multiple days
    for day in range(30):
        current_date = base_date + timedelta(days=day)

        # Generate 5-15 sequences per day
        n_sequences = np.random.randint(5, 16)

        for seq_num in range(n_sequences):
            # Start time for this sequence (random hour between 6-20)
            start_hour = np.random.randint(6, 21)
            start_minute = np.random.randint(0, 60)
            sequence_start = current_date.replace(hour=start_hour, minute=start_minute, second=0)

            # Generate a sequence of 2-8 calls
            sequence_length = np.random.randint(2, 9)

            # Choose starting call type (Trill more likely in morning)
            if 6 <= start_hour <= 10:
                start_call = np.random.choice(call_types, p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05])
            else:
                start_call = np.random.choice(call_types, p=[0.2, 0.25, 0.25, 0.15, 0.1, 0.05])

            current_call = start_call
            current_time = sequence_start

            for call_idx in range(sequence_length):
                # Add current call to data
                filename = f"recording_{current_time.strftime('%Y%m%d_%H%M%S')}_{seq_num:03d}_{call_idx:02d}.flac"

                sample_data.append({
                    'filename': filename,
                    'call_type': current_call,
                    'start_time': current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'end_time': (current_time + timedelta(seconds=np.random.uniform(0.5, 3.0))).strftime(
                        '%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'confidence': np.random.uniform(0.7, 1.0),
                    'duration': np.random.uniform(0.5, 3.0),
                    'sequence_id': f"{day:02d}_{seq_num:03d}",
                    'year': current_time.year,
                    'month': current_time.month,
                    'day': current_time.day,
                    'hour': current_time.hour,
                    'minute': current_time.minute,
                    'second': current_time.second
                })

                # Choose next call based on transition probabilities
                if call_idx < sequence_length - 1:  # Not the last call
                    probs = transition_probs[current_call]
                    next_call = np.random.choice(list(probs.keys()), p=list(probs.values()))
                    current_call = next_call

                    # Next call occurs 1-10 seconds later
                    current_time += timedelta(seconds=np.random.uniform(1, 10))

    # Create DataFrame and save
    df_sample = pd.DataFrame(sample_data)
    df_sample = df_sample.sort_values(['year', 'month', 'day', 'hour', 'minute', 'second']).reset_index(drop=True)
    df_sample.to_csv('sample_annotations_sequences.tsv', sep='\t', index=False)

    print(f"Created enhanced sample dataset with {len(df_sample)} annotations")
    print(f"Call type distribution:")
    for call_type, count in df_sample['call_type'].value_counts().items():
        print(f"  {call_type}: {count}")

    # Also create a simple metadata file
    metadata_sample = []
    for day in range(30):
        date = base_date + timedelta(days=day)
        metadata_sample.append({
            'date': date.strftime('%Y-%m-%d'),
            'recording_session': f"session_{day:02d}",
            'weather': np.random.choice(['sunny', 'cloudy', 'rainy']),
            'temperature': np.random.randint(20, 35),
            'group_size': np.random.randint(3, 8),
            'dominant_individual': f"individual_{np.random.randint(1, 6)}"
        })

    df_metadata = pd.DataFrame(metadata_sample)
    df_metadata.to_csv('sample_metadata.tsv', sep='\t', index=False)
    print(f"Created sample metadata with {len(df_metadata)} entries")


if __name__ == "__main__":
    main()