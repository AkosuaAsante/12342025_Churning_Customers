import streamlit as st
import pickle 
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Dropout

def create_model(neurons=64, dropout_rate=0.2, activation='relu'):
    inputs = Input(shape=(X.shape[1],))
    x = Dense(neurons, activation=activation)(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(neurons, activation=activation)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model



model=pickle.load(open('telmomodel.pkl','rb'))







def preprocess_input(montlycharges, totalcharges, Onlinesecurity, tenure, contract_type, payment_type):
    # Perform label encoding for categorical variables
    onlinesecurity_values = ['No', 'Yes', 'No internet service']
    contract_values = ['Month-to-month', 'One year', 'Two year']
    paymentmethod_values = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']

    le_OnlineSecurity = LabelEncoder()
    le_contract = LabelEncoder()
    le_paymentmethod = LabelEncoder()

    le_OnlineSecurity.fit(onlinesecurity_values)
    le_contract.fit(contract_values)
    le_paymentmethod.fit(paymentmethod_values)

    Onlinesecurity_encoded = le_OnlineSecurity.transform([Onlinesecurity])[0]
    contract_type_encoded = le_contract.transform([contract_type])[0]
    payment_type_encoded = le_paymentmethod.transform([payment_type])[0]

    # Create a NumPy array with the preprocessed features
    input_features = np.array([[montlycharges, totalcharges, tenure, Onlinesecurity_encoded, contract_type_encoded, payment_type_encoded]])

    return input_features





def show_predict():
    st.title("Customer Churing Predictor")

    onlinesecurity=('No','Yes','No internet service')
    contract=('Month-to-month' ,'One year' ,'Two year')
    paymentmethod=('Electronic check' ,'Mailed check' ,'Bank transfer (automatic)',
                   'Credit card (automatic)')
   
    montlycharges = st.number_input("Enter Monthly charges:", min_value=0.0, max_value=1000000.0, value=50.0, step=0.1)
    totalcharges = st.number_input("Enter Total charges:", min_value=0.0, max_value=1000000.0, value=50.0, step=0.1)
    Onlinesecurity=st.selectbox("online security",onlinesecurity)
    tenure=st.slider("Tenure", 1,100,3)
    contract=st.selectbox("Contract",contract)
    payment=st.selectbox('Payment Method',paymentmethod)

    input_features = preprocess_input(montlycharges, totalcharges, Onlinesecurity, tenure, contract, payment)

    
    scaler = StandardScaler()
    input_features = scaler.fit_transform(input_features)
    prediction= model.predict(input_features)

    possibility=model.predict_proba(input_features)

    if prediction < 0:
        prediction = "No"
    else:
        prediction = 'Yes'

    st.write("Customer will Churn?:", prediction)
    st.write("Confidence:", f"{possibility[0] * 100:.2f}%")

   


    
   

show_predict()