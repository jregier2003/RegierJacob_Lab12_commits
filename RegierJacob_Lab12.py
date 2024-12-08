import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit

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


fft_values = fft(filtered_data['residuals'].fillna(0))
frequencies = fftfreq(len(filtered_data), d=(filtered_data['decimal date'].iloc[1] - filtered_data['decimal date'].iloc[0]))
fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
dominant_frequency = frequencies[np.argmax(fft_magnitude[:len(frequencies)//2])]
T_est = 1 / dominant_frequency

def sinusoidal_model(x, A, T, phi):
    return A * np.sin(2 * np.pi * x / T + phi)

popt, _ = curve_fit(sinusoidal_model, filtered_data['decimal date'], filtered_data['residuals'], p0=[2, T_est, 0])
A_fit, T_fit, phi_fit = popt

filtered_data['sinusoidal_fit'] = sinusoidal_model(filtered_data['decimal date'], A_fit, T_fit, phi_fit)
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['decimal date'], filtered_data['residuals'], label='Residuals', color='orange')
plt.plot(filtered_data['decimal date'], filtered_data['sinusoidal_fit'], label='Sinusoidal Fit (Updated)', linestyle='--', color='blue')
plt.title('Residuals and Updated Sinusoidal Fit')
plt.xlabel('Decimal Year')
plt.ylabel('Residuals (ppm)')
plt.legend()
plt.grid()
plt.show()


extended_years = np.arange(1981, 2025, 1/12)  # Monthly steps from 1981 to 2025
extended_polynomial_fit = np.polyval(polynomial_coefficients, extended_years)
extended_sinusoidal_fit = sinusoidal_model(extended_years, A_fit, T_fit, phi_fit)
combined_model = extended_polynomial_fit + extended_sinusoidal_fit
predicted_400_ppm_index = np.where(combined_model >= 400)[0][0]
predicted_400_ppm_date = extended_years[predicted_400_ppm_index]
print(f"Predicted date for CO2 reaching 400 ppm: {predicted_400_ppm_date:.2f}")
actual_400_ppm_date = data[data['average'] >= 400]['decimal date'].min()
print(f"Actual observed date for CO2 reaching 400 ppm: {actual_400_ppm_date:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(extended_years, combined_model, label='Combined Model')
plt.axhline(400, color='red', linestyle='--', label='400 ppm Threshold')
plt.scatter([predicted_400_ppm_date], [400], color='green', label=f'Predicted: {predicted_400_ppm_date:.2f}', zorder=5)
plt.scatter([actual_400_ppm_date], [400], color='blue', label=f'Actual: {actual_400_ppm_date:.2f}', zorder=5)
plt.title('Prediction of 400 ppm CO2 Level')
plt.xlabel('Year')
plt.ylabel('CO2 Concentration (ppm)')
plt.legend()
plt.grid()
plt.show()



