# %%

import pandas as pd

# Define the path to the input .dat file and the output .csv file
DATA_FOLDER = "../../Data/Theory/"
input_file_path = f"{DATA_FOLDER}FRDM2012.dat"
output_file_path = f"{DATA_FOLDER}FRDM2012.csv"

# Define the column names based on your description
column_names = [
    "N",
    "A",
    "ε2",
    "ε3",
    "ε4",
    "ε6",
    "β2",
    "β3",
    "β4",
    "β6",
    "Es+p",
    "Emic",
    "Ebind",
    "Mth",
    "Mexp",
    "σexp",
    "EFLmic",
    "MFLth",
]

# Load the .dat file into a DataFrame
df = pd.read_csv(input_file_path, delim_whitespace=True, names=column_names)

df["Z"] = df["A"] - df["N"]
# rename
df["BE"] = df["Ebind"]


# Save the DataFrame to a CSV file
df[["Z", "N", "BE"]].to_csv(output_file_path, index=False)

# %%
