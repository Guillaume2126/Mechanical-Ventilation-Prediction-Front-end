import pandas as pd
import numpy as np
import streamlit as st
import requests
import json
from Api_request import make_api_request_with_features, make_api_request_with_id
import time
from streamlit_extras.stoggle import stoggle
from streamlit_extras.tags import tagger_component

# ------- 1 - Title and info session ---------
#Title
st.title('Mechanical Ventilation Predictor')
#Subtitle
st.subheader('How does this predictor works ?')
#Aim of the predictor
st.write('The aim of this predictor is to forecast the pressure inside lungs :lungs: using differents features. Here is how it works:' )
st.write("1️⃣ First, **have a look** at the different features required")
st.write("2️⃣ Then, **select** the kind of feature that you want to provide")
st.write("3️⃣ **Provide the data** using the method of your choice")
st.write("4️⃣ Click on **'Get prediction'**	 ")
#Let's start text
"""Let's start 👇"""


#------- 2 - Explanation of Data --------
#Title
st.info('1️⃣ In order to use this predictor, you will need the following features:')
#Sentence
tagger_component(
    "",
    ["Here are the list of features used for the calculation:"],
    color_name=["blue"],
)
"""Here are the list of features used for the calculation:"""
# Making two columns with different widths
col1, col2 = st.columns([1,6])
col3, col4 = st.columns([1,1])
#Filling the column 1 with the different features
with col1:
     st.write('	:small_blue_diamond: **R**')

stoggle("More explanations about R ?",
    """put a beautiful picture of what R is""",
)
with col1:
     st.write('	:small_blue_diamond: **C**')
     st.write('	:small_blue_diamond: **u_in**')
     st.write('	:small_blue_diamond: **u_out**')
#TODO Filling the column 2 with the description of the different features
with col2:
    st.caption("Change in pressure per change in flow (air volume per time)")

    st.caption("Change in volume per change in pressure")
    st.caption("Explanation of u_in")
    st.caption("Explanation of u_out")
#Sentence
"""	:point_right: If you don't have these features, you can also use the breath_id:"""
# Making two columns with different widths
col1, col2 = st.columns([1,6])
#Filling the column 1 with breath_id
with col1:
     st.write(':small_blue_diamond: **breath_id**')
#TODO Filling the column 2 with description of breath_id
with col2:
    st.caption("Explanation of breath_id")



#------- 3 - Choose kind of features --------
#Title
st.info('2️⃣ Select the kind of data you want to provide:')

#List of three choices
button_data_provide = st.radio('Pick one:', ["I have a breath_id",
                                            "I don't have a breath_id but I have all the features",
                                            ":grey[***I don't have neither one or the other***]"],
                            )
#Conditions depending of the choices of kind of features:
if button_data_provide ==":grey[***I don't have neither one or the other***]":
    st.warning("I am sorry, you can't use this predictor")



#------- 4- General - Select the way to provide data ---------

#-------- 4-A- If the user choose "I have a breath_id", add a text field to fill and do API call --------
if button_data_provide == "I have a breath_id":
    st.info('3️⃣ Please provide your breath_id') #Title
    breath_id = st.text_input('Enter your breath_id') #Input field
    predict_with_breath_id = st.button(":blue[Get prediction]") #Button to get prediction
    if predict_with_breath_id:
        #Loading bar
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        #Waiting circle
        with st.spinner('Wait for it...'):
            time.sleep(5)
        with st.spinner('Your results are coming...'):
            time.sleep(5)
        with st.spinner('Sorry, I am slow to load but the prediction will be perfect !!'):
            time.sleep(5)
        st.success('Done!')

        #API call
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
        st.area_chart(df,
                    x="time_step",
                    y=["actual_pressure", "predicted_pressure"],
                    color=['#FF0000','#CCEEFF']  # Optional
                )
        st.line_chart(df,
                    x="time_step",
                    y=["actual_pressure", "predicted_pressure"],
                    color=['#FF0000','#CCEEFF']  # Optional
                )

#-------- 4-B- If the user choose "I don't have a breath_id but I have all the features", add some field to fill and do API call --------
if button_data_provide == "I don't have a breath_id but I have all the features":
    st.info('3️⃣ Please provide your features') #Title
    #Options to provide data
    data_selection = st.radio('Choose a way to provide data:', ["Provide features manually", "Import CSV"])

    # -------- 4-B-1- If user choose : provide data manually ---------
    if data_selection=="Provide features manually": #TODO: Most important, do we predict value one by one or by 80, if 80, I need time_step
        #Creation of three columns
        col1, col2, col3 = st.columns([5,1,5])
        #Selection of R and C in two different columns
        with col1:
            R = st.slider('R value', min_value=0, max_value=50)
        with col3:
            C = st.slider('C value', min_value=0, max_value=50)
        #Creation of three columns, only for the display of dataframe
        col1, col2, col3 = st.columns([1,1,1])
        #Dataframe to fill by the user, with u_in and u_out (80 rows)
        with col2:
            df = pd.DataFrame(
            [
            {"u_in": 0, "u_out": 0},
        ]
        ) #Dataframe with only one row
            df = pd.concat([df] * 80, ignore_index=True) #We multiplicate by 80 to have 80 rows
            edited_df = st.data_editor(df) #This command will allow the user to fill the table with new values
            #Once the user click on "Get prediction", give an error if the table is empty, prediction otherwise
            if st.button("Get prediction"):
                if df.equals(edited_df): #If no modification of the table...
                    st.warning("You didn't modify the table !") # ... then we have a warning message
                else:
                    st.success('Here are your results:')
                    df = edited_df #Update the dataframe with the information provide by user
                    st.balloons() #Animation -TODO Change the animation, but maybe we can put something ?
                    u_in = df["u_in"][0] #TODO #Assign varialbe for API call (only first row here)
                    u_out = df["u_out"][0] #TODO #Assign varialbe for API call (only first row here)
                    #Prams for API
                    params = dict(
                    R=R,
                    C=C,
                    u_in=u_in,
                    u_out=u_out)
                    #API call
                    api_url = 'https://mvpapi-azdjuqy4ca-ew.a.run.app/predict' #Take one argument of u_in and u_out TODO: create a dataframe ?
                    api_response = requests.get(api_url, params=params)
                    response_text = api_response.text
                    #End of API call and display of the answer
                    try:
                        response_data = json.loads(response_text)
                        pressure = response_data.get("pressure", "")
                        st.write(f"The predicted pressure is {pressure}")
                    except json.JSONDecodeError:
                        st.warning("Unable to decode JSON response.")



    #-------- 4-B-2- Provide data using a CSV file ---------
    if data_selection=="Import CSV":
        #Tool to upload the file
        up_file = st.file_uploader("Please upload a file with 4 columns: 'R', 'C', 'u_in' and 'u_out'",
                         help="Please provide a file with 4 columns: 'R', 'C', 'u_in' and 'u_out'",
                         type=["csv"]) #Add an if condition if the file is not a csv
        if up_file:
            st.success("File uploaded successfully!")

        get_prediction_using_csv = st.button(":blue[Get prediction]")

        if get_prediction_using_csv:
            df_to_predict = pd.read_csv(up_file)

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

            df_predict_api = pd.DataFrame({'pressure_api': list_of_pressure, 'time_step_api': list_of_time_step})
            merged_df = pd.concat([df_predict_api, df_to_predict], axis=1)

            st.area_chart(
                    merged_df,
                    x='time_step',
                    y=["pressure", "pressure_api"], color=['#FF0000','#CCEEFF']  # Optional
                )

            st.line_chart(
                    merged_df,
                    x='time_step',
                    y=["pressure", "pressure_api"], color=['#FF0000','#CCEEFF']  # Optional
                )


        #TODO Command to read csv, a treatment
        #TODO Put a success message if its correctly load
        #TODO API call, it should look like this:
        #TODO Change the color and template of the graphe


# TODO: Take the answer of the API and plot a graph
# TODO: Take the answer of the API and display a dataframe (only if the user want, add a checkbox to ask the user)
# TODO: Put a download button (I know its possible, see doc) to download either the graph, the dataframe, or both
# TODO: Put it on internet
# TODO: Mettre quelque chose pour afficher les parties une a une
# TODO: Mettre des gifs/videos pour expliquer
# TODO: Changer le nom des variables
# TODO: Under variables, put a click image to describe on click(arrow point below)


#Creer un bouton pour telecharger les données
# If we want to add an image : st.image('./header.png')

# Use this command to plot the graphe: plot_graph = st.line_chart(df1)

# Insert out of order.



# Error message : st.error('Error message')
# Warning message : st.warning('Warning message')


#Ensuite, pour la sortie des informations, mettre un graphe, et mettre l'option "Afficher la table de prediction"
