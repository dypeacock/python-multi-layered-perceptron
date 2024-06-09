import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data from Excel file
file = pd.ExcelFile('FEHDataStudent.xlsx')
with pd.ExcelFile('FEHDataStudent.xlsx') as xls:
    df1 = pd.read_excel(xls, "Sheet1", usecols=[0,1,2,3,4,5,6,7,8])

# Convert non-numeric values to NaN
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Calculate correlation
correlation_matrix = df1.corr()

# Set pandas display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Create heatmap with seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='Greens', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
