import streamlit as st
import pickle
import numpy as np
from streamlit_option_menu import option_menu

# Cache the model loading process using st.cache_resource
@st.cache_resource
def load_model(model_path):
    try:
        model = pickle.load(open(model_path, 'rb'))
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()

# Load models
parkinsons_model = load_model(r"F:\Project_3\parkinson_random.pkl")
kidney_model = load_model(r"F:\Project_3\best_model.pkl")
liver_model = load_model(r"F:\Project_3\Random_Forest_model.pkl")

# Sidebar for Multiple Disease Prediction
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction',
        ['Parkinson Prediction', 'Liver Prediction', 'Kidney Prediction'],
        icons=['activity', 'heart', 'droplet'],
        default_index=0
    )

# Main App Title
st.title("Multi-Disease Prediction App")
st.write("This application predicts diseases based on user input. Use the sidebar to select a prediction type.")

# -------------------- PARKINSON'S DISEASE PREDICTION --------------------
if selected == "Parkinson Prediction":
    st.title("Parkinson's Disease Prediction using Machine Learning")

    # User Input Fields (Formatted into columns)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
    with col3:
        DDP = st.text_input('Jitter:DDP')
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
    with col3:
        APQ = st.text_input('MDVP:APQ')
    with col4:
        DDA = st.text_input('Shimmer:DDA')
    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')
    with col2:
        RPDE = st.text_input('RPDE')
    with col3:
        DFA = st.text_input('DFA')
    with col4:
        spread1 = st.text_input('spread1')
    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')
    with col2:
        PPE = st.text_input('PPE')

    # Prediction Logic
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, 
                          Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, 
                          DFA, spread1, spread2, D2, PPE]

            # Convert all inputs to float
            user_input = [float(x) for x in user_input]

            # Make prediction
            parkinsons_prediction = parkinsons_model.predict([user_input])

            # Display result
            if parkinsons_prediction[0] == 1:
                st.success("The person has Parkinson's disease")
            else:
                st.success("The person does not have Parkinson's disease")

        except ValueError:
            st.error("Please enter valid numerical values for all input fields.")

# -------------------- KIDNEY DISEASE PREDICTION --------------------
elif selected == "Kidney Prediction":
    st.title("Kidney Disease Prediction")


    # Column Inputs
    col1, col2, col3, col4, col5 = st.columns(5)

    # Column 1: Basic patient details
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, step=1, value=30)
        bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=200, step=1, value=80)
        sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.050, step=0.001, value=1.020)
        al = st.number_input("Albumin", min_value=0, max_value=5, step=1, value=1)
        su = st.number_input("Sugar", min_value=0, max_value=5, step=1, value=1)

    # Column 2: Blood and urine test results
    with col2:
        bgr = st.number_input("Blood Glucose Random", min_value=50, max_value=500, step=1, value=150)
        bu = st.number_input("Blood Urea", min_value=0, max_value=300, step=1, value=50)
        sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=15.0, step=0.1, value=1.2)
        pot = st.number_input("Potassium Level", min_value=1.0, max_value=10.0, step=0.1, value=4.0)

    # Column 3: Blood cell counts
    with col3:
        wc = st.number_input("White Blood Cell Count", min_value=2000, max_value=20000, step=100, value=9600)
        rbc = st.radio("Red Blood Cells", ['abnormal', 'normal'])
        pc = st.radio("Pus Cells", ['abnormal', 'normal'])
        pcc = st.radio("Pus Cell Clumps", ['present', 'not present'])
        ba = st.radio("Bacteria", ['present', 'not present'])

    # Column 4: Clinical conditions
    with col4:
        htn = st.radio("Hypertension", ['yes', 'no'])
        dm = st.radio("Diabetes Mellitus", ['yes', 'no'])
        cad = st.radio("Coronary Artery Disease", ['yes', 'no'])
        pe = st.radio("Pedal Edema", ['yes', 'no'])
        ane = st.radio("Anemia", ['yes', 'no'])

    # Convert categorical inputs to numeric values
    rbc_value = 1 if rbc == 'abnormal' else 0
    pc_value = 1 if pc == 'abnormal' else 0
    pcc_value = 1 if pcc == 'present' else 0
    ba_value = 1 if ba == 'present' else 0
    htn_value = 1 if htn == 'yes' else 0
    dm_value = 1 if dm == 'yes' else 0
    cad_value = 1 if cad == 'yes' else 0
    pe_value = 1 if pe == 'yes' else 0
    ane_value = 1 if ane == 'yes' else 0

    # Prepare input data with all 19 features
    input_features = np.array([[age, bp, sg, al, su, bgr, bu, sc, pot, wc, 
                                rbc_value, pc_value, pcc_value, ba_value,
                                htn_value, dm_value, cad_value, pe_value, ane_value]])

    # Prediction Button
    if st.button("Get Kidney Test Result"):
        try:
            kidney_prediction = kidney_model.predict(input_features)

            # Display result
            if kidney_prediction[0] == 1:
                st.warning("You are likely to have kidney disease. Please see a doctor.")
            else:
                st.success("Your kidney is healthy.")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# -------------------- LIVER DISEASE PREDICTION --------------------
elif selected == "Liver Prediction":
    st.title("Liver Disease Prediction")
    st.write("This app predicts whether a person has liver disease or not based on various health metrics.")

    # Collecting user inputs
    age = st.number_input("Age", min_value=1, max_value=100, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])  # Using selectbox for gender
    total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=20.0, step=0.1)
    direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=10.0, step=0.1)
    alkaline_phosphotase = st.number_input("Alkaline Phosphotase (U/L)", min_value=0, max_value=1000, step=1)
    alamine_aminotransferase = st.number_input("Alamine Aminotransferase (U/L)", min_value=0, max_value=1000, step=1)
    aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (U/L)", min_value=0, max_value=1000, step=1)
    total_protiens = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=10.0, step=0.1)
    albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, step=0.1)
    albumin_and_globulin_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=5.0, step=0.1)

    # Collecting features into a list
    user_input = [
        age,
        1 if gender == "Male" else 0,  # Encoding gender as 1 (Male) or 0 (Female)
        total_bilirubin,
        direct_bilirubin,
        alkaline_phosphotase,
        alamine_aminotransferase,
        aspartate_aminotransferase,
        total_protiens,
        albumin,
        albumin_and_globulin_ratio
    ]

    # When user clicks "Predict"
    if st.button("Predict"):
        try:
            # Reshape input data to 2D array for prediction
            prediction = liver_model.predict([user_input])  # Here we pass a 2D array

            # Show the result
            if prediction[0] == 1:
                st.write("The prediction is: **Liver Disease Present**")
            else:
                st.write("The prediction is: **No Liver Disease**")
        except Exception as e:
            st.error(f"Error making prediction: {e}")