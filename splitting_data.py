import pandas as pd
import numpy as np

# Read data from Excel file
file = pd.ExcelFile('FEHDataStudent.xlsx')
with pd.ExcelFile('FEHDataStudent.xlsx') as xls:
    df1 = pd.read_excel(xls, "Sheet1", usecols=[0,1,2,3,4,5,6,7,8])

# Shuffle the data
df1_shuffled = df1.sample(frac=1, random_state=42)  # Using a fixed random_state for reproducibility

# Calculate the size of each split
total_rows = len(df1_shuffled)
split1_size = int(total_rows * 0.6)
split2_size = int(total_rows * 0.2)

# Split the data into three parts
split1 = df1_shuffled[:split1_size]
split2 = df1_shuffled[split1_size:split1_size + split2_size]
split3 = df1_shuffled[split1_size + split2_size:]

# Write each part to a separate file

split1.to_excel('training.xlsx', index=False)
split2.to_excel('validation.xlsx', index=False)
split3.to_excel('test.xlsx', index=False)
