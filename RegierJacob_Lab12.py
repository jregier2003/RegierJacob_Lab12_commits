import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'co2_mm_mlo.csv'
data = pd.read_csv(file_path, skiprows=51, delimiter=',', header=0)

print(data.head())
