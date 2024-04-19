import pandas as pd
import sys
import os
import pandas as pd
import glob

# path to tis file
FILEPATH = os.path.abspath(__file__)
PATH_2_DATA = "../../Data/Theory/FRDM2012.csv"
print(glob.glob(PATH_2_DATA))


class FRDM:
    def __init__(self, verbose=False) -> None:
        self.__name__ = "FRDM"
        self.precomputed_path = os.path.dirname(FILEPATH)
        self.df_precomputed = pd.read_csv(PATH_2_DATA)
        self.verbose = verbose

    def __call__(self, Z, N):
        try:
            row = self.df_precomputed.query(f"Z == {Z} and N == {N}")
            return row["BE"].values[0]
        except IndexError:
            return None


frdm_be = FRDM()
