import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = '/home/rlinux/VS Code Projects/Lab 12/co2_mm_mlo.csv'

data = pd.read_csv(
    file_path,
    skiprows=52,
    delimiter=',',
    header=None,
    names=['year', 'month', 'decimal date', 'average', 'interpolated', 'trend', 'ndays'],
    engine='python',
)

data['actual_year'] = data['month'].astype(str).str.split('.').str[0].astype(int)

filtered_data = data[(data['actual_year'] >= 1981) & (data['actual_year'] <= 1990)]

if not filtered_data.empty:
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_data['decimal date'], filtered_data['average'], label='Filtered Data (1981-1990)', marker='o')
    plt.xlabel('Decimal Year')
    plt.ylabel('CO2 Concentration (ppm)')
    plt.title('Atmospheric CO2 (1981-1990)')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("\nNo data available for the selected range.")




