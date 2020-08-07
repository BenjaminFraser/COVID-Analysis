#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import dependencies and external libs
import io
import numpy as np
import os
import pandas as pd
import requests
import sys

# set save directory paths
SAVE_DIR = os.path.join(os.getcwd(), 'Preprocessed_data')

# start date for majority of datasets
DATA_START = '2020-03-10'

# load population mappings for all uk towns / regions
POPULATION_MAPPINGS = pd.read_csv('UK-population-mappings.csv', 
                                    index_col='name').to_dict()
POPULATION_MAPPINGS = POPULATION_MAPPINGS['Population']

# scrape URLs for latest data - ensure these are correct and checked regularly
ENGLAND_URL = 'https://coronavirus.data.gov.uk/downloads/csv/coronavirus-cases_latest.csv'

#SCOTLAND_URL = ('https://statistics.gov.scot/slice/observations.csv?&dataset=http%3A%2F%2F' +
#                'statistics.gov.scot%2Fdata%2Fcoronavirus-covid-19-management-information&' +
#                'http%3A%2F%2Fpurl.org%2Flinked-data%2Fcube%23measureType=http%3A%2F%2F' +
#                'statistics.gov.scot%2Fdef%2Fmeasure-properties%2Fcount&http%3A%2F%2F' +
#                'statistics.gov.scot%2Fdef%2Fdimension%2Fvariable=http%3A%2F%2F' +
#                'statistics.gov.scot%2Fdef%2Fconcept%2Fvariable%2F' +
#                'cumulative-people-tested-for-covid-19-positive')

SCOTLAND_URL = ('https://statistics.gov.scot/slice/observations.csv?&dataset=http%3A%2F%2F' + 
                'statistics.gov.scot%2Fdata%2Fcoronavirus-covid-19-management-information&' + 
                'http%3A%2F%2Fpurl.org%2Flinked-data%2Fcube%23measureType=http%3A%2F%2F' + 
                'statistics.gov.scot%2Fdef%2Fmeasure-properties%2Fcount&http%3A%2F%2F' + 
                'statistics.gov.scot%2Fdef%2Fdimension%2Fvariable=http%3A%2F%2F' +
                'statistics.gov.scot%2Fdef%2Fconcept%2Fvariable%2Ftesting-cumulative-' + 
                'people-tested-for-covid-19-positive')

WALES_URL = ('http://www2.nphs.wales.nhs.uk:8080/CommunitySurveillanceDocs.nsf/' + 
             '3dc04669c9e1eaa880257062003b246b/77fdb9a33544aee88025855100300cab/' + 
             '$FILE/Rapid%20COVID-19%20surveillance%20data.xlsx')

IRELAND_URL = ('http://opendata-geohive.hub.arcgis.com/datasets/d9be85b30d7748b5b7c09450b' + 
               '8aede63_0.csv?outSR={%22latestWkid%22:3857,%22wkid%22:102100}')

# filename for manually collected locations (Northern ireland, Jersey, Isle of Man, Guernsey)
MISC_GATHERED_AREAS = 'Misc_scraped_data.csv'


def cumulative_to_daily(data_df):
    """ Convert a given cumulative df to a daily one 

    Parameters
    ----------
    data_df : Pandas DataFrame 
        Dataframe of cumulative values to convert into daily.

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with daily values.

    """
    daily_df = data_df.copy()
    for col in daily_df.columns:
        daily_df[f'{col} Lag'] = daily_df[col].shift(1).fillna(0)
        daily_df[col] = daily_df[col] - daily_df[f'{col} Lag']
        daily_df.drop(col + ' Lag', axis=1, inplace=True)
    return daily_df


def population_conversions(dataframe, mapping):
    """ Convert figures to per population, using mapping values. 

    Parameters
    ----------
    dataframe : Pandas DataFrame 
        Dataframe of time-series data (index as dates/time) to convert.

    mapping : Python Dict
        Python dict with towns/regions as keys, and values as populations.

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with converted values.

    """ 
    new_df = dataframe.copy()
    for column in dataframe.columns:
        new_df[column] = (dataframe[column] / mapping[column]) * 100000
    return new_df


def load_preprocess_england(regional=False, start_date=DATA_START, 
                            cumulative=False, per_pop=True):
    """ Download and pre-process England Regional COVID-19 Cases 

    Parameters
    ----------
    regional : bool
        If regional is true, get only top-level England regions (e.g. South-West),
        else obtain all UTLA areas.

    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.

    """

    # download live data using requests for england cases data
    data_request = requests.get(ENGLAND_URL, allow_redirects=True).text
    raw_csv = io.StringIO(data_request)
    eng_cases = pd.read_csv(raw_csv)

    # columns to keep
    cols = ['Area name', 'Specimen date', 
            'Cumulative lab-confirmed cases']

    # if regional selected only filter by regions
    if regional:
        england_df = eng_cases[eng_cases['Area type'] == 'region'][cols].copy()

    # else filter by all UTLA
    else:
        england_df = eng_cases[eng_cases['Area type'] == 
                                         'utla'][cols].copy()


    # format dataframe, columns and index into selected form
    england_df = england_df.set_index(['Area name', 'Specimen date']).unstack('Area name')
    england_df.columns = england_df.columns.droplevel(0)

    # backfill any null values, and filter data only beyond start_date
    england_df.fillna(method='bfill', inplace=True)
    england_df = england_df.loc[(england_df.index > start_date)]

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return england_df
    else:
        england_df = cumulative_to_daily(england_df)

        # if per pop - convert to per 100,000 figures
        if per_pop:
            return population_conversions(england_df, POPULATION_MAPPINGS)
        else:
            return england_df


def load_preprocess_scotland(start_date=DATA_START, 
                             cumulative=False, per_pop=True):
    """ Download and pre-process selected Scotland data 

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.
    """

    scotland_cases = pd.read_csv(SCOTLAND_URL, skiprows=[0, 1, 2, 3, 4, 5, 6])

    scotland_cases.columns = scotland_cases.columns.astype(str)

    # drop reference col 
    drop_col = 'http://purl.org/linked-data/sdmx/2009/dimension#refArea'
    scotland_cases.drop(drop_col, axis=1, inplace=True)

    # replace asterisks with zero - they represent less than 5 counts
    scotland_cases = scotland_cases.replace('*', 0)

    # set area as index, drop original index, and transpose table
    scotland_cases.set_index('Reference Area', drop=True, inplace=True)
    scotland_cases = scotland_cases.T
    scotland_cases = scotland_cases.astype(float)

    # drop total scotland cases - we can work this out later if req'd
    scotland_cases.drop('Scotland', axis=1, inplace=True)

    # only select data beyond chosen start date
    scotland_df = scotland_cases.loc[(scotland_cases.index > start_date)].copy()
    scotland_df.index.name = 'Date'

    # drop rows with NaN values -  can occur with new entries occassionally
    scotland_df.dropna(axis=0, inplace=True)

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return scotland_df
    else:
        scotland_df = cumulative_to_daily(scotland_df)

        # if per pop - convert to per 100,000 figures
        if per_pop:
            return population_conversions(scotland_df, POPULATION_MAPPINGS)
        else:
            return scotland_df


def load_preprocess_wales(start_date=DATA_START, cumulative=False):
    """ Download and pre-process Wales COVID-19 Cases as per 100,000 figures.

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.

    """

    # download latest wales data as excel file - read into Pandas
    wales_cases = pd.read_excel(WALES_URL, sheet_name='Tests by specimen date')

    # chosen cols to keep - we want per 100,000 pop
    keep_cols = ['Local Authority', 'Specimen date', 
                 'Cumulative incidence per 100,000 population']

    wales_df = wales_cases[keep_cols].copy()

    # format index and column levels as appropriate
    wales_df = wales_df.set_index(['Local Authority', 
                                   'Specimen date']).unstack('Local Authority')
    wales_df.columns = wales_df.columns.droplevel(0)

    # remove columns outside wales and unknown from the data
    wales_df.drop(['Outside Wales', 'Unknown'], inplace=True, axis=1)

    # only select data beyond the chosen start date
    wales_df = wales_df.loc[(wales_df.index > DATA_START)]

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return wales_df
    else:
        return cumulative_to_daily(wales_df)


def load_preprocess_ireland(start_date=DATA_START, cumulative=False, per_pop=True):
    """ Download and pre-process Ireland Regional COVID-19 Cases 

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.
    """

    ireland_cases = pd.read_csv(IRELAND_URL, parse_dates=['TimeStamp'])

    ireland_cases.columns = ireland_cases.columns.astype(str)

    keep_cols = ['TimeStamp', 'ConfirmedCovidCases', 'CountyName']
    ireland_df = ireland_cases[keep_cols].copy()

    # drop complete duplicates of rows if they exsit
    ireland_df.drop_duplicates(inplace=True)

    # format dataframe, columns and index into selected form
    ireland_df = ireland_df.set_index(['CountyName', 'TimeStamp']).unstack('CountyName')
    ireland_df.columns = ireland_df.columns.droplevel(0)
    ireland_df.index.name = 'Date'

    # only keep dates beyond chosen start date
    ireland_df = ireland_df.loc[(ireland_df.index > start_date)].copy()

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return ireland_df
    else:
        ireland_df = cumulative_to_daily(ireland_df)

        # if per pop - convert to per 100,000 figures
        if per_pop:
            return population_conversions(ireland_df, POPULATION_MAPPINGS)
        else:
            return ireland_df


def load_preprocess_NI(start_date=DATA_START, cumulative=False, per_pop=True):
    """ Download and pre-process Northern Ireland Area COVID-19 Cases 

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.
    """

    ni_cases = pd.read_csv(MISC_GATHERED_AREAS, index_col='name')
    ni_cases.columns = ni_cases.columns.astype(str)

    # keep only NI data
    ni_cases = ni_cases[ni_cases['code'].str.startswith('N')].copy()

    # drop unwanted cols
    ni_df = ni_cases.drop(['code', 'Population'], axis=1).T
    ni_df.index = pd.to_datetime(ni_df.index)
    ni_df.index.name = 'Date'

    # We have many missing values - interpolate using linear function
    ni_df.interpolate(method='slinear', inplace=True)

    # only select data beyond the chosen start date
    ni_df = ni_df.loc[(ni_df.index > start_date)]

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return ni_df
    else:
        ni_df = cumulative_to_daily(ni_df)

        # if per pop - convert to per 100,000 figures
        if per_pop:
            return population_conversions(ni_df, POPULATION_MAPPINGS)
        else:
            return ni_df

# Backup Wales function should the existing web URL not be updated
#def load_preprocess_wales(start_date=DATA_START, cumulative=False, per_pop=True):
    """ Import and pre-process Wales COVID-19 Cases from manually updated figures.

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.
    """

    #wales_cases = pd.read_excel(MISC_GATHERED_AREAS, 
     #                           sheet_name='Daily Cases Extended', index_col='name')
    #wales_cases.columns = wales_cases.columns.astype(str)

    # keep only Wales data
    #wales_cases = wales_cases[wales_cases['code'].str.startswith('W')].copy()

    # drop unwanted cols
    #wales_df = wales_cases.drop(['code', 'Population'], axis=1).T
    #wales_df.index = pd.to_datetime(wales_df.index)
    #wales_df.index.name = 'Date'

    # only select data beyond the chosen start date
    #wales_df = wales_df.loc[(wales_df.index > start_date)]

    # if cumulative, data is already in that form, else convert to daily
    #if cumulative:
    #    return wales_df
    #else:
    #    wales_df = cumulative_to_daily(wales_df)

        # if per pop - convert to per 100,000 figures
    #    if per_pop:
    #        return population_conversions(wales_df, POPULATION_MAPPINGS)
     #   else:
     #       return wales_df


def load_preprocess_other(start_date=DATA_START, cumulative=False, per_pop=True):
    """ Download and pre-process Other Area COVID-19 Cases, including Isle of Man,
        Jersey, and Guernsey.

    Parameters
    ----------
    start_date : string
        Date string of format 'Year-Month-Day', e.g. (2020-03-10) to start at.

    cumulative : bool
        If cumulative true, return cumulative figures, else get daily figures

    per_pop : bool
        If per_pop true, calculate and return per 100,000 population figures

    Returns
    -------
    Pandas DataFrame
        Pandas DataFrame object with COVID-19 figures in selected format.
    """

    other_cases = pd.read_csv(MISC_GATHERED_AREAS, index_col='name')
    other_cases.columns = other_cases.columns.astype(str)

    # keep only non Northern Ireland or non Wales data
    other_cases = other_cases[~other_cases['code'].str.startswith('N')].copy()
    other_cases = other_cases[~other_cases['code'].str.startswith('W')].copy()

    # drop unwanted cols
    other_df = other_cases.drop(['code', 'Population'], axis=1).T
    other_df.index = pd.to_datetime(other_df.index)
    other_df.index.name = 'Date'

    # only select data beyond the chosen start date
    other_df = other_df.loc[(other_df.index > start_date)]

    # if cumulative, data is already in that form, else convert to daily
    if cumulative:
        return other_df
    else:
        other_df = cumulative_to_daily(other_df)

        # if per pop - convert to per 100,000 figures
        if per_pop:
            return population_conversions(other_df, POPULATION_MAPPINGS)
        else:
            return other_df


def save_data(dataframe, save_dir, file_name):
    """ Save given dataframe to save dir with selected file name 

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Chosen Pandas DataFrame to save.

    save_dir : string
        String containing absolute path to save directory.

    save_dir : string
        Chosen name to save file as, including file extension.

    Returns
    -------
    None
        DataFrame saved as csv file, no values returned from function.
    """
    dataframe.to_csv(os.path.join(save_dir, file_name))


# sequence to run if script manually run using python command line
if __name__ == "__main__":

    # create save dir if not already existing
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # obtain and save England data
    england_data = load_preprocess_england()
    print(f"England Data - Shape: {england_data.shape}. No. of Nulls / NaN's: "
          f"{england_data.isna().sum().sum()}. Latest Date: {england_data.index.values[-1]}.")
    save_data(england_data, save_dir=SAVE_DIR, 
              file_name='england-areas-lab-confirmed-per-pop.csv')
    print("England data preprocessed and saved.\n")


    # obtain and save Scotland data
    scotland_data = load_preprocess_scotland()
    print(f"Scotland Data - Shape: {scotland_data.shape}. No. of Nulls / NaN's: "
          f"{scotland_data.isna().sum().sum()}. Latest Date: {scotland_data.index.values[-1]}.")
    save_data(scotland_data, save_dir=SAVE_DIR, 
              file_name='scotland-regions-confirmed-per-pop.csv')
    print("Scotland data preprocessed and saved.\n")


    # obtain and save Wales data
    wales_data = load_preprocess_wales()
    print(f"Wales Data - Shape: {wales_data.shape}. No. of Nulls / NaN's: "
          f"{wales_data.isna().sum().sum()}. Latest Date: {wales_data.index.values[-1]}.")
    save_data(wales_data, save_dir=SAVE_DIR, 
              file_name='wales-regions-lab-confirmed-per-pop.csv')
    print("Wales data preprocessed and saved.\n")


    # obtain and save Ireland data
    ireland_data = load_preprocess_ireland()
    print(f"Ireland Data - Shape: {ireland_data.shape}. No. of Nulls / NaN's: "
          f"{ireland_data.isna().sum().sum()}. Latest Date: {ireland_data.index.values[-1]}.")
    save_data(ireland_data, save_dir=SAVE_DIR, 
              file_name='ireland-regions-confirmed-per-pop.csv')
    print("Ireland data preprocessed and saved.\n")


    # obtain and save Northern Ireland data
    ni_data = load_preprocess_NI(start_date='2020-03-19')
    print(f"Northern Ireland Data - Shape: {ni_data.shape}. No. of Nulls / NaN's: "
          f"{ni_data.isna().sum().sum()}. Latest Date: {ni_data.index.values[-1]}.")
    save_data(ni_data, save_dir=SAVE_DIR, 
              file_name='ni-regions-confirmed-per-pop.csv')
    print("Northern Ireland data preprocessed and saved.\n")


    # obtain and save Other location data (Isle of Man, Jersey, Guernsey)
    other_data = load_preprocess_other(start_date='2020-03-19')
    print(f"Other Data - Shape: {other_data.shape}. No. of Nulls / NaN's: "
          f"{other_data.isna().sum().sum()}. Latest Date: {other_data.index.values[-1]}.")
    save_data(other_data, save_dir=SAVE_DIR, 
              file_name='other-regions-confirmed-per-pop.csv')
    print("Other data preprocessed and saved.\n")


    print("Completed all preprocessing and saving of datasets.")