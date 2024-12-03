import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

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

filtered_data.loc[:, 'trend_change'] = filtered_data['trend'].diff()



plt.figure(figsize=(10, 5))
plt.plot(filtered_data['decimal date'], filtered_data['trend'], label='CO2 Trend (1981-1990)', marker='o')
plt.xlabel('Decimal Year')
plt.ylabel('CO2 Trend (ppm)')
plt.title('Monthly CO2 Trend (1981-1990)')
plt.legend()
plt.grid()
plt.show()


yearly_averages = filtered_data.groupby('actual_year')['average'].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.bar(yearly_averages['actual_year'], yearly_averages['average'])
plt.xlabel('Year')
plt.ylabel('Average CO2 Concentration (ppm)')
plt.title('Yearly Average CO2 Concentration (1981-1990)')
plt.grid(axis='y')
plt.show()


x = filtered_data['decimal date']
y = filtered_data['average']

slope, intercept, r_value, p_value, std_err = linregress(x, y)

plt.figure(figsize=(10, 5))
plt.plot(x, y, label='CO2 Concentration (1981-1990)', marker='o')
plt.plot(x, slope * x + intercept, label=f'Linear Fit: y={slope:.2f}x+{intercept:.2f}', linestyle='--')
plt.xlabel('Decimal Year')
plt.ylabel('CO2 Concentration (ppm)')
plt.title('CO2 Linear Trend (1981-1990)')
plt.legend()
plt.grid()
plt.show()

