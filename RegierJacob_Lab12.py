import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.fft import fft, fftfreq

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

print(f"Slope: {slope:.2f} ppm/year")
print(f"R-squared: {r_value**2:.3f}")

monthly_averages = filtered_data.groupby('month')['average'].mean()
plt.figure(figsize=(10, 6))
plt.plot(monthly_averages.index, monthly_averages.values, marker='o', label='Monthly Average CO2 (1981-1990)')
plt.title('Seasonal Variation in CO2 (1981-1990)')
plt.xlabel('Month')
plt.ylabel('Average CO2 Concentration (ppm)')
plt.legend()
plt.grid(True)
plt.show()


polynomial_coefficients = np.polyfit(filtered_data['decimal date'], filtered_data['average'], 2)
polynomial_fit = np.polyval(polynomial_coefficients, filtered_data['decimal date'])
filtered_data['residuals'] = filtered_data['average'] - polynomial_fit

plt.figure(figsize=(10, 10))
plt.subplot(2, 1, 1)
plt.plot(filtered_data['decimal date'], filtered_data['average'], label='Original Data')
plt.plot(filtered_data['decimal date'], polynomial_fit, label='Polynomial Fit', linestyle='--')
plt.title('CO2 Concentration and Polynomial Fit (1981-1990)')
plt.xlabel('Decimal Year')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(filtered_data['decimal date'], filtered_data['residuals'], label='Residuals', color='orange')
plt.title('Residuals After Polynomial Fit')
plt.xlabel('Decimal Year')
plt.ylabel('Residuals (ppm)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('LastnameFirstname_Lab12_Fig1.png')
plt.show()



residuals = filtered_data['residuals'].values
N = len(residuals)
T = filtered_data['decimal date'].diff().mean()  
frequencies = fftfreq(N, d=T)[:N // 2]
fft_values = np.abs(fft(residuals))[:N // 2]
dom_frequency = frequencies[np.argmax(fft_values)]

T_est = 1 / dom_frequency
A_est = residuals.max() - residuals.min()
phi_estimated = 0 

filtered_data['sinusoidal_fit'] = A_est * np.sin(2 * np.pi * (filtered_data['decimal date'] / T_est) + phi_estimated)
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['decimal date'], filtered_data['residuals'], label='Residuals', color='orange')
plt.plot(filtered_data['decimal date'], filtered_data['sinusoidal_fit'], label='Sinusoidal Fit', linestyle='--')
plt.title('Residuals and Sinusoidal Fit (Using Fourier Analysis)')
plt.xlabel('Decimal Year')
plt.ylabel('Residuals (ppm)')
plt.legend()
plt.grid()
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(frequencies, fft_values, label='FFT Magnitude')
plt.title('Frequency Spectrum of Residuals')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.grid()
plt.legend()
plt.show()
