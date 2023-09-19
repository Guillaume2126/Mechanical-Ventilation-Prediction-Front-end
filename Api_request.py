import streamlit as st
import pandas as pd
import requests
import json

def make_api_request_with_features(R, C, u_in, u_out):
    # Create a dictionary to represent the data in the expected format
    params = dict(R=R, C=C, u_in=u_in, u_out=u_out)

    # Define the API URL
    api_url = 'https://mvpapi-azdjuqy4ca-ew.a.run.app/predict_from_csv'

    try:
        # Send a GET request to the API and parse the JSON response
        api_response = requests.get(api_url, params=params).json()
        return api_response
    except json.JSONDecodeError as e:
        # Handle the JSON decoding error (e.g., print an error message)
        print(f"JSONDecodeError: {e}")
        return None

def make_api_request_with_id(idx):
    params = dict(idx=idx)
    #API call
    api_url = 'https://mvpapi-azdjuqy4ca-ew.a.run.app/predict_series_with_id'
    api_response = requests.get(api_url, params=params).json()
    return api_response
