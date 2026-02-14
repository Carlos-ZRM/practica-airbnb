import matplotlib.pyplot as plt
import pandas as pd


df_calendar = pd.read_csv('calendar.csv')
# Create side-by-side histograms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram for minimum_nights
axes[0].hist(df_calendar['minimum_nights'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Minimum Nights')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Minimum Nights')
axes[0].grid(axis='y', alpha=0.3)

# Histogram for maximum_nights
axes[1].hist(df_calendar['maximum_nights'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Maximum Nights')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Maximum Nights')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()