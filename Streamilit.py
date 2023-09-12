import pandas as pd
import numpy as np
import streamlit as st
import requests
from Api_request import make_api_request_with_features, make_api_request_with_id
import time
import matplotlib.pyplot as plt
from Calculate_mae import mean_absolute_error
from matplotlib.patches import Rectangle
from streamlit_lottie import st_lottie_spinner
from streamlit_extras.stoggle import stoggle

#Config name of the page
st.set_page_config (page_title='Mechanical ventilation')


# ------- 1 - Title and info session ---------
#Title
st.title('Ventilation Pressure Predictor')
#Subtitle
st.subheader('Please provide your features to have access to the prediction')


#------- 2 - Choose kind of features --------
#Title
st.info('1️⃣ Select the kind of data you want to provide')
#List of three choices
button_data_provide = st.selectbox('Pick one:', ["Please Select:","I have a breath_id",
                                            "I don't have a breath_id but I have all the features",
                                            "I don't have neither one or the other"],
                            )
#Conditions depending of the choices of kind of features:
if button_data_provide =="I don't have neither one or the other":
    st.warning("Please collect some features and come back later to have a prediction")



#------- 4- General - Select the way to provide data ---------
#Useful function to have the gif of lungs
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#-------- 4-A- If the user choose "I have a breath_id", add a text field to fill and do API call --------
if button_data_provide == "I have a breath_id":
    st.info('2️⃣ Please provide a breath_id') #Title
    breath_ids = st.multiselect('Multiselect', list(range(1, 201)), max_selections=5) #Input field
    predict_with_breath_id = st.button(":blue[Get prediction]") #Button to get prediction
    if predict_with_breath_id:
        if breath_ids:
            #Waiting animation (lungs + bar)
            col1, col2, col3 = st.columns([1,1, 1])
            lottie_json = load_lottieurl("https://lottie.host/190f6b9e-80da-496f-a5b7-7374254d7634/TF29EiWHw9.json")
            with col2:
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(2):
                    with col2:
                        with st_lottie_spinner(lottie_json, height = 200, width = 200, key=percent_complete):
                            time.sleep(2.5)
                            my_bar.progress((percent_complete+1)*50, text=progress_text)
            my_bar.empty() #Remove waiting bar
            st.success('Here are your results 🔽') #Success message

            #Start API call
            for breath_id in breath_ids:
                pressure = make_api_request_with_id(breath_id)
                time_step=[ 0.0, 0.0331871509552001, 0.0663647651672363, 0.0997838973999023,
                    0.1331243515014648, 0.1665058135986328, 0.1999211311340332, 0.233269453048706,
                    0.2667148113250732, 0.3001444339752197, 0.3334481716156006, 0.3667137622833252,
                    0.4000871181488037, 0.4334573745727539, 0.4668083190917969, 0.5001921653747559,
                    0.5335805416107178, 0.5669963359832764, 0.6003098487854004, 0.6336038112640381,
                    0.667017936706543, 0.7003989219665527, 0.7338323593139648, 0.7672531604766846,
                    0.8007259368896484, 0.8341472148895264, 0.8675739765167236, 0.9009172916412354,
                    0.9343087673187256, 0.967742681503296, 1.0011558532714844, 1.0346879959106443,
                    1.0681016445159912, 1.1015379428863523, 1.1348886489868164, 1.168378829956055,
                    1.2017686367034912, 1.235328197479248, 1.2686767578125, 1.3019189834594729,
                    1.335435390472412, 1.3688392639160156, 1.4022314548492432, 1.4356489181518557,
                    1.4690682888031006, 1.5024497509002686, 1.5358901023864746, 1.5694541931152344,
                    1.602830410003662, 1.636289119720459, 1.6696226596832275, 1.7029592990875244,
                    1.7363479137420654, 1.7697343826293943, 1.803203582763672, 1.8365991115570068,
                    1.869977235794068, 1.903436183929444, 1.9368293285369875, 1.970158576965332,
                    2.0035817623138428, 2.0370094776153564, 2.0702223777771, 2.1036837100982666,
                    2.1370668411254883, 2.170450448989868, 2.203945636749268, 2.23746919631958,
                    2.270882368087769, 2.304311990737915, 2.3376832008361816, 2.371119737625122,
                    2.4044580459594727, 2.4377858638763428, 2.471191644668579, 2.504603147506714,
                    2.537960767745972, 2.571407556533813, 2.604744434356689, 2.638017416000366]
                df = pd.DataFrame(pressure)
                df["time_step"]=time_step
                mae = mean_absolute_error(df["actual_pressure"], df["predicted_pressure"])

                # Create graph using matplotlib
                fig, ax = plt.subplots()
                ax.plot(df["time_step"], df["actual_pressure"], label="Actual Pressure", color='#3c7dc2')
                #ax.plot(df["time_step"], df["predicted_pressure"], label="Predicted Pressure", color='#eb8634')
                ax.plot(df["time_step"], df["predicted_pressure"], label="Predicted Pressure", color='#eb8634')

                # set y and x label
                ax.set_ylabel("Pressure")
                ax.set_xlabel("Time step")

                # Add legend, other and title
                ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1))
                ax.grid(alpha=0.15) #Improve transparency of grid
                ax.spines['top'].set_visible(False) #Remove top bar
                ax.spines['right'].set_visible(False) #Remove right bar
                fig.set_size_inches(10, 5) #Range 10, 5
                plt.title(f"Mechanical Ventilation Prediction - Breath ID={breath_id}")

                #Add MAE
                ratio_max_min = df["actual_pressure"].max()-(df["actual_pressure"].min())
                same_size_rectangle = ((df["actual_pressure"].max())-(df["actual_pressure"].min()))/(16.7343242500-4.853261668752088)
                rectangle = Rectangle((1.53, df["actual_pressure"].min()+ratio_max_min*0.6), 0.5, 0.8*same_size_rectangle, fill=True, color='red', alpha=0.2)
                ax.add_patch(rectangle)
                plt.annotate(f'MAE* = {mae}', xy=(1.6, df["actual_pressure"].min()+ratio_max_min*0.615), fontsize=12, color='black')

                # Add in Streamlit
                st.pyplot(fig)
                st.write("\* **MAE=Mean Absolute Error** is defined as the average variance between the significant values in the dataset and the projected values in the same dataset")
                st.write(" ")
        else:
            st.error("Please, don't forget to enter at least one breath_id")

#-------- 4-B- If the user choose "I don't have a breath_id but I have all the features", add some field to fill and do API call --------
if button_data_provide == "I don't have a breath_id but I have all the features":
    st.info('2️⃣ Please provide your features as CSV file:') #Title

    up_file = st.file_uploader("Please upload a file with 5 columns: 'R', 'C', 'u_in', 'u_out' and 'pressure'",
                         type=["csv"]) #Add an if condition if the file is not a csv
    if up_file:
        st.success("File uploaded successfully!")

        get_prediction_using_csv = st.button(":blue[Get prediction]")

        if get_prediction_using_csv:
            #waiting animation(lungs and bar)
            col1, col2, col3 = st.columns([1,1, 1])
            lottie_json = load_lottieurl("https://lottie.host/190f6b9e-80da-496f-a5b7-7374254d7634/TF29EiWHw9.json")
            with col2:
                progress_text = "Operation in progress. Please wait."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(4):
                    with col2:
                        with st_lottie_spinner(lottie_json, height = 200, width = 200, key=percent_complete):
                            time.sleep(2.5)
                        time.sleep(0.01)
                        my_bar.progress((percent_complete+1)*25, text=progress_text)
            my_bar.empty()

            st.success('Here are your results 🔽')

        #Read csv
        df_to_predict = pd.read_csv(up_file)

        #API call
        list_of_pressure = []
        list_of_time_step = []
        for i in range(len(df_to_predict)):
            pressure = make_api_request_with_features(df_to_predict["R"][i],
                            df_to_predict["C"][i],
                            df_to_predict["u_in"][i],
                            df_to_predict["u_out"][i])
            if pressure is not None:
                list_of_pressure.append(pressure)
                list_of_time_step.append(df_to_predict["time_step"][i])
            else:
                list_of_pressure.append("API doesnt work")

        df = pd.DataFrame({'actual_pressure':df_to_predict["pressure"], "predicted_pressure":list_of_pressure, 'time_step': list_of_time_step})
        mae = mean_absolute_error(df["actual_pressure"], df["predicted_pressure"])

        # Create graph using matplotlib
        fig, ax = plt.subplots()
        ax.plot(df["time_step"], df["actual_pressure"], label="Actual Pressure", color='#3c7dc2')
        #ax.plot(df["time_step"], df["predicted_pressure"], label="Predicted Pressure", color='#eb8634')
        ax.plot(df["time_step"], df["predicted_pressure"], "--", label="Predicted Pressure", color='#eb8634')

        # set y and x label
        ax.set_ylabel("Pressure")
        ax.set_xlabel("Time step")

        # Add legend, other and title
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, -0.1))
        ax.grid(alpha=0.15) #Improve transparency of grid
        ax.spines['top'].set_visible(False) #Remove top bar
        ax.spines['right'].set_visible(False) #Remove right bar
        fig.set_size_inches(10, 5) #Range 10, 5
        plt.title(f"Mechanical Ventilation Prediction")

        #Add MAE
        ratio_max_min = df["actual_pressure"].max()-(df["actual_pressure"].min())
        same_size_rectangle = ((df["actual_pressure"].max())-(df["actual_pressure"].min()))/(16.7343242500-4.853261668752088)
        rectangle = Rectangle((1.53, df["actual_pressure"].min()+ratio_max_min*0.6), 0.5, 0.8*same_size_rectangle, fill=True, color='red', alpha=0.2)
        ax.add_patch(rectangle)
        plt.annotate(f'MAE = {mae}', xy=(1.6, df["actual_pressure"].min()+ratio_max_min*0.615), fontsize=12, color='black')
        # Add in Streamlit
        st.pyplot(fig)
