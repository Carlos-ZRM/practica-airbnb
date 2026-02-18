"""
Example usage of the graph_info class for comprehensive data analysis and visualization
"""

import pandas as pd
import numpy as np
from class.class_graph_info import graph_info

# ============================================================================
# Example 1: Create Sample DataFrame (with mixed variable types)
# ============================================================================

# Create sample data
np.random.seed(42)
sample_data = {
    'Age': np.random.randint(18, 80, 200),
    'Salary': np.random.normal(50000, 20000, 200),
    'Score': np.random.uniform(0, 100, 200),
    'Department': np.random.choice(['HR', 'IT', 'Sales', 'Finance'], 200),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 200),
    'Is_Active': np.random.choice([True, False], 200),
    'Is_Manager': np.random.choice([True, False], 200),
}

df = pd.DataFrame(sample_data)

# ============================================================================
# Example 2: Initialize graph_info class
# ============================================================================

# Create instance - automatically detects variable types
analyzer = graph_info(df)

# ============================================================================
# Example 3: Get Summary Information
# ============================================================================

summary = analyzer.get_summary()
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Total Rows: {summary['total_rows']}")
print(f"Total Columns: {summary['total_columns']}")
print(f"Numerical Variables: {summary['numerical_vars']}")
print(f"Categorical Variables: {summary['categorical_vars']}")
print(f"Boolean Variables: {summary['boolean_vars']}")
print("="*60 + "\n")

# ============================================================================
# Example 4: Generate Individual Plots (Optional)
# ============================================================================

# Get numerical plots as base64 (without explanation variable)
# num_plots = analyzer.plot_num()

# Get numerical plots grouped by a categorical variable
# num_plots = analyzer.plot_num(explicativa='Department')

# Get categorical plots
# cat_plots = analyzer.plot_categorical()

# Get boolean plots
# bool_plots = analyzer.plot_boolean()

# ============================================================================
# Example 5: Generate Complete HTML Report (MAIN OUTPUT)
# ============================================================================

# Generate comprehensive HTML report
# Option 1: Without grouping variable
html_file = analyzer.generate_html_report(filename='analysis_report.html')

# Option 2: With grouping variable (groups numerical variables by categorical)
# html_file = analyzer.generate_html_report(
#     filename='analysis_report_grouped.html',
#     explicativa='Department'
# )

print(f"\n✓ Report generated successfully!")
print(f"✓ Open '{html_file}' in your browser to view the analysis")

# ============================================================================
# HOW TO USE WITH YOUR OWN DATA
# ============================================================================

"""
# Load your data
df_filtered = pd.read_csv('your_data.csv')

# Create analyzer instance (automatically identifies variable types)
analyzer = graph_info(df_filtered)

# Generate HTML report
analyzer.generate_html_report(filename='my_analysis.html')

# If you want to group numerical variables by a categorical variable:
analyzer.generate_html_report(
    filename='my_analysis_grouped.html',
    explicativa='YourCategoricalColumn'
)

# Get summary statistics
summary = analyzer.get_summary()
print(summary)
"""
