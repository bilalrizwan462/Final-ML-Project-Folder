import numpy as np
import pickle
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# loading the model:
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a prediction function:
def heart_disease_prediction(input_data):



    # Creating a numpy array using input data
    input_data_as_numpy_array = np.asarray(input_data)

    # reshaping the array to predict only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print("Model Prediction is: ", prediction)

    if prediction[0] == 0:
        return 'This person does not have a heart disease'
    else:
        return 'This person has a heart disease'

def main():

    # Giving title
    st.title('Heart Disease Prediction')

    #Introduction to Web app
    st.write('This is a web app created to predict whether a person is suffering from heart disease\
             Please provide accurate information with the best of your knowledge\
             Then click on Heart Disease Prediction Results to see the prediction.\
             Model\'s ROC-AUC score is 0.8422 with 78.3% Recall and 22.5% Precision')


    Smoking_Yes = 0
    AlcoholDrinking_Yes = 0
    Stroke_Yes = 0
    DiffWalking_Yes = 0
    Sex_Male = 0
    AgeCategory_25_29 = 0
    AgeCategory_30_34 = 0
    AgeCategory_35_39 = 0
    AgeCategory_40_44 = 0
    AgeCategory_45_49 = 0
    AgeCategory_50_54 = 0
    AgeCategory_55_59 = 0
    AgeCategory_60_64 = 0
    AgeCategory_65_69 = 0
    AgeCategory_70_74 = 0
    AgeCategory_75_79 = 0
    AgeCategory_80_or_older = 0
    Race_Asian = 0
    Race_Black = 0
    Race_Hispanic = 0
    Race_Other = 0
    Race_White = 0
    Diabetic_No__borderline_diabetes = 0
    Diabetic_Yes = 0
    Diabetic_Yes__during_pregnancy_ = 0
    PhysicalActivity_Yes = 0
    GenHealth_Fair = 0
    GenHealth_Good = 0
    GenHealth_Poor = 0
    GenHealth_Very_good = 0
    Asthma_Yes = 0
    KidneyDisease_Yes = 0
    SkinCancer_Yes = 0

    #Getting User Input Data

    BMI = st.number_input('Body Mass Index (BMI)')

    PhysicalHealth = st.slider("For how many days during the past 30 days were you sick?", 0, 30)

    MentalHealth = st.slider('For how many days during past 30 days wes your mental health not good?', 0, 30)

    SleepTime = st.number_input('On average, how many hours of sleep do you get?')

    Smoking = st.radio('Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 ciggarettes]', ['Yes', 'No'])
    if(Smoking == 'Yes'):
        Smoking_Yes = 1
    else:
        Smoking_Yes = 0

    AlcoholDrinking = st.radio('Are you a heavy drinker? (Adult men having more than 14 drinks per week and adult women having more than 7 drinks per week', ['Yes', 'No'])
    if (AlcoholDrinking == 'Yes'):
        AlcoholDrinking_Yes = 1
    else:
        AlcoholDrinking_Yes = 0

    Stroke = st.radio('Did you ever suffer from a stroke ?', ['Yes', 'No'])
    if (Stroke == 'Yes'):
        Stroke_Yes = 1
    else:
        Stroke_Yes = 0



    DiffWalking = st.radio('Do you have a serious difficulty walking or climbing stairs?', ['Yes', 'No'])
    if (DiffWalking == 'Yes'):
        DiffWalking_Yes = 1
    else:
        DiffWalking_Yes = 0


    Sex = st.radio('Are you male or female?', ['Male', 'Female'])

    if (Sex == 'Male'):
        Sex_Male = 1
    else:
        Sex_Male = 0

    AgeCategory = st.radio('Which age category do you belong to ?', ['18-24', '25-29', '30-34', '35-39', '40-44',
                                                          '45-49', '50-54', '55-59', '60-64', '65-69',
                                                                     '70-74', '75-79', '80 or older'])


    if (AgeCategory == '25-29'):
        AgeCategory_25_29 = 1
    elif (AgeCategory == '30-34'):
        AgeCategory_30_34 = 1
    elif (AgeCategory == '35-39'):
        AgeCategory_35_39 = 1
    elif (AgeCategory == '40-44'):
        AgeCategory_40_44 = 1
    elif (AgeCategory == '45-49'):
        AgeCategory_45_49 = 1
    elif (AgeCategory == '50-54'):
        AgeCategory_50_54 = 1
    elif (AgeCategory == '55-59'):
        AgeCategory_55_59 = 1
    elif (AgeCategory == '60-64'):
        AgeCategory_60_64 = 1
    elif (AgeCategory == '65-69'):
        AgeCategory_65_69 = 1
    elif (AgeCategory == '70-74'):
        AgeCategory_70_74 = 1
    elif (AgeCategory == '75-79'):
        AgeCategory_75_79 = 1
    elif (AgeCategory == '80 or older'):
        AgeCategory_80_or_older = 1

    Race = st.radio('Which race do you belong to ?', ['White', 'Black', 'Asian', 'American Indian/Alaskan Native',
                                                      'Hispanic', 'Other'])
    if (Race == 'Asian'):
        Race_Asian = 1
    elif (Race == 'Black'):
        Race_Black = 1
    elif (Race == 'Hispanic'):
        Race_Hispanic = 1
    elif (Race == 'Other'):
        Race_Other = 1
    elif (Race == 'White'):
        Race_White = 1

    Diabetic = st.radio('Are you diabetic?', ['Yes', 'No', 'borderline diabetes', 'Yes (during pregnancy)'])
    if (Diabetic == 'Yes'):
        Diabetic_Yes = 1
    elif (Diabetic == 'borderline diabetes'):
        Diabetic_No__borderline_diabetes = 1
    elif (Diabetic == 'Yes (during pregnancy)'):
        Diabetic_Yes__during_pregnancy_ = 1

    PhysicalActivity = st.radio('Did you perform any routine physical activity apart from your job?', ['Yes', 'No'])
    PhysicalActivity_Yes = 0
    if (PhysicalActivity == 'Yes'):
        PhysicalActivity_Yes = 1
    else:
        PhysicalActivity_Yes = 0


    GenHealth = st.radio('How would you describe your general health?', ['Very good', 'Fair', 'Good', 'Poor', 'Excellent'])
    if (GenHealth == 'Very good'):
        GenHealth_Very_good = 1
    elif (GenHealth == 'Fair'):
        GenHealth_Fair = 1
    elif (GenHealth == 'Good'):
        GenHealth_Good = 1
    elif (GenHealth == 'Poor'):
        GenHealth_Poor = 1


    Asthma = st.radio('Are you Asthmatic?', ['Yes', 'No'])
    if (Asthma == 'Yes'):
        Asthma_Yes = 1
    else:
        Asthma_Yes = 0

    KidneyDisease = st.radio('Not including kidney stones, bladder infection or incontinence,'
                             ' were you ever told you had kidney disease?', ['Yes', 'No'])
    if (KidneyDisease == 'Yes'):
        KidneyDisease_Yes = 1
    else:
        KidneyDisease_Yes = 0


    SkinCancer = st.radio('Did you ever had (or have) skin cancer?', ['Yes', 'No'])
    if (SkinCancer == 'Yes'):
        SkinCancer_Yes = 1
    else:
        SkinCancer_Yes = 0


    # code for Prediction
    diagnosis = ''

    # Creating a button for prediction

    if st.button('Heart Disease Prediction Results'):

        diagnosis = heart_disease_prediction([BMI, PhysicalHealth, MentalHealth, SleepTime, Smoking_Yes,
       AlcoholDrinking_Yes, Stroke_Yes, DiffWalking_Yes, Sex_Male,
       AgeCategory_25_29, AgeCategory_30_34, AgeCategory_35_39,
       AgeCategory_40_44, AgeCategory_45_49, AgeCategory_50_54,
       AgeCategory_55_59, AgeCategory_60_64, AgeCategory_65_69,
       AgeCategory_70_74, AgeCategory_75_79, AgeCategory_80_or_older,
       Race_Asian, Race_Black, Race_Hispanic, Race_Other, Race_White,
       Diabetic_No__borderline_diabetes, Diabetic_Yes,
       Diabetic_Yes__during_pregnancy_, PhysicalActivity_Yes,
       GenHealth_Fair, GenHealth_Good, GenHealth_Poor,
       GenHealth_Very_good, Asthma_Yes, KidneyDisease_Yes,
       SkinCancer_Yes])

    st.success(diagnosis)

    st.write('Disclaimer: Please note that these results are not to be used for \
    diagnosing heart conditions. In case of any symptoms, please consult a cardiologist!')

if __name__ == '__main__':
    main()