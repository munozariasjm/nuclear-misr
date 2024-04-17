from nuclearpy_models.models import sr_be, sr_rc, dz_be, mnp_rc
import streamlit as st
import pandas as pd
from openpyxl import load_workbook
import plotly.express as px

PATH2DATA = pd.read_excel("Data/Theory/MasterNuclei.xlsx")
exp_data = pd.read_csv("Data/Experiment/AME2020.csv")
#### Create a class


def get_sheets_name(filename):
    workbook = load_workbook(filename)
    return workbook.sheetnames


th_datastes = get_sheets_name(PATH2DATA)


def get_datasets(used_dfs):
    th_datastes = [th_dataste for th_dataste in used_dfs if th_dataste != "AME2020"]
    thdfs = {
        th_dataste: pd.read_excel(PATH2DATA, sheet_name=th_dataste)
        .query("Z >= 12")
        .query("Z<50")
        for th_dataste in th_datastes
    }
    return thdfs


def inference_be(Z, N, index, thdfs, data):

    data = pd.DataFrame(data, index=[0])
    be_misr_pred = sr_be(Z, N, index)
    binding_energy_preds = {
        "Z": Z,
        "N": N,
        "MISR": be_misr_pred[0],
        "unc_MISR": be_misr_pred[1],
        "DZ": dz_be(Z, N),
        "Exp": data["BE"].values[0],
    }
    for th_dataste in thdfs:
        thdf = thdfs[th_dataste]
        thdf = thdf.query("Z == @Z").query("N == @N")
        if thdf.empty:
            continue
        else:
            binding_energy_preds[th_dataste] = thdf["BE"].values[0]

    return pd.DataFrame(binding_energy_preds)


def inference_rc(Z, N, index, thdfs, data):
    data = pd.DataFrame(data, index=[0])
    rc_misr_pred = sr_rc(Z, N, index)
    charge_radii_preds = {
        "Z": Z,
        "N": N,
        "MISR": rc_misr_pred[0],
        "uncertainty": rc_misr_pred[1],
        "MNP": mnp_rc(Z, N),
    }
    for th_dataste in thdfs:
        thdf = thdfs[th_dataste]
        thdf = thdf.query("Z == @Z").query("N == @N")
        if thdf.empty:
            continue
        else:
            charge_radii_preds[th_dataste] = thdf["RC"].values[0]

    return pd.DataFrame(charge_radii_preds)


def plot_be(Z, N, index, thdfs, data):


def main():
    st.title("Discovering Nuclear Models from Symbolic Machine Learning")

    st.write(
        "This is a web application that allows you to discover nuclear models from symbolic machine learning."
    )
    st.sidebar.title("Models")

    Z = st.number_input("Enter the number of protons (Z)", 12, 50, 18)
    N = st.number_input("Enter the number of neutrons (N)", 12, 50, 20)
    # select an index
    index = st.number_input("Enter the index", 0, 10, 0)
    if Z < 12 or Z > 50:
        st.warning("Z must be between 12 and 50. Models are trained for this range.")
        return

    # Now display the models
    st.sidebar.title("Models")
    possible_theory_models = [
    "DD-ME2",
    "SKMS",
    "NL3S",
    "UNEDF1",
    ]
    used_dfs = st.sidebar.multiselect(
        "Select the datasets to use", th_datastes, th_datastes
    )
    thdfs = get_datasets(used_dfs)
    st.sidebar.write("The following datasets are used:")
    st.sidebar.write(used_dfs)

    be_pred = inference_be(Z, N, index, thdfs, exp_data)
    rc_pred = inference_rc(Z, N, index, thdfs, exp_data)

    cols =  st.columns(2)
    with cols[0]:
        st.write("Binding Energy")
        st.write(be_pred)

    with cols[1]:
        st.write("Charge Radii")
        st.write(rc_pred)


