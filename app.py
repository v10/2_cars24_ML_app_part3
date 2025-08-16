import pandas as pd
import streamlit as st
import pickle


cars_df: pd.DataFrame = pd.read_csv("./cars24-car-price.csv")

st.write("""
         # Cars24 Used car price prediction
         """)
st.dataframe(cars_df.head())

tasks = """
    few things about task
    - model file for car price prediction is given car_prd_model.pkl
    - our target variable Y = selling_price , X = rest of the variables/columns
    - we must have done some encoding while training the model we need to same here as well.
"""

## Encoding Categorical features - use the same encodings you have used while training your data
encode_dict: dict[str, dict[str, int]] = {
    "fuel_type": {'Diesel': 1, 'Petrol': 2, 'CNG': 3, 'LPG': 4, 'Electric': 5},
    "seller_type": {'Dealer': 1, 'Individual': 2, 'Trustmark Dealer': 3},
    "transmission_type": {'Manual': 1, 'Automatic': 2}
}

# we are doing it as user will enter petrol, diesel or cng or lpg 
# but we encoded it as numeric number while training the model.. so we need to convert
# from petrol to 2 .. diesel to 1 etc..


# issue : 
# if there was a complex encoding then we would have stored that encoding in 
# sql or csv format , loaded it here - this is the simplest form of encoding. 


# from user we'll take following inputs
# fuel_type , transmission_type , engine_power, num_of_seats 
# we are ignoring other column values. 

col1, col2, col3 , col4 = st.columns(4)

with col1:
    fuel_type= st.selectbox(
        "Fuel type",
        ("Diesel", "Petrol","CNG","LPG","Electric"),
    )
with col2:
    num_of_seats = st.selectbox(
        "Number of seats",
        ("4","5","7","9","11"),
    )
with col3:
    engine_power  = st.slider(
        "set engine power",
        500 , 5000 , step = 100)
with col4:
    transmission_type = st.selectbox(
        "Transmission type",
        ("Automatic", "Manual"),
    )
    
# we took the input from user
# now we need some sort of button that when user click that button 
# all the taken values will go to model in proper format
# and then model will return the output.


    
if st.button("Predict Price", type="primary"):
    # 
    decoded_fuel_type = int(encode_dict["fuel_type"][fuel_type])
    decoded_transmission_type = int(encode_dict["transmission_type"][transmission_type])
    e_power = int(engine_power)
    seats= int(num_of_seats)
    print("printing..")
    print(decoded_fuel_type, decoded_transmission_type,e_power,seats)
    # input format (some values we are taking from user , some values we are hardcoding)
    # [year,dealer_type, km_driven,fuel_type, transmission_type, milage,engine_power, max_power, seats]
    
    
    # now we need to read pickle file (model)
    with open("car_prd_model","rb") as f:
        model = pickle.load(f)
        my_input_data = [[2012,2,10000,decoded_fuel_type,decoded_transmission_type,19.7,e_power,33.2,seats]]
        
        price =model.predict(my_input_data)
    st.text(price)    
    
    
# Deployment of this app:
#first push this repo on github
#click on deploy button on streamlit
# it will ask to login and then fill the details.
# then it will generate the link 
# you can click on that link in browser and open it.
# But when you run the model it will not run 
# as it needs sklearn library and other dependencies

# so now you need to tell steramlit that before running my app please install the dependencies
# those dependencies you will store into the requirments.txt file
# pip freeze > requirements.txt 
# above command will dump all dependencies to your requirements.txt

# after this change you will agian push your changes to github
# now you mostly don't need to deploy again on streamlit
# just refresh the link you created from streamlit and then it should take the latest changes you pushed
# and it should run.


# another task 
# how to use database here - ask chatGPT
# how to use the mongoDB here, sqlite, SQL here..