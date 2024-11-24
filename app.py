if page == "AI Project":
    st.title("My AI Project!")
    # Load the trained model
    model = joblib.load("titanic_model.pkl")

    # Input fields for user to provide passenger details
    st.header("Enter Passenger Details:")

    Pclass = st.selectbox("Passenger Class (Pclass):", [1, 2, 3], index=2)
    Sex = st.selectbox("Gender (Sex):", ["Male", "Female"])
    Sex = 0 if Sex == "Male" else 1
    Age = st.slider("Age:", min_value=0, max_value=100, value=30)
    SibSp = st.number_input("Number of Siblings/Spouses Aboard (SibSp):", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard (Parch):", min_value=0, max_value=10, value=0)
    Fare = st.number_input("Ticket Fare (Fare):", min_value=0.0, value=10.0, step=1.0)
    Embarked = st.selectbox("Port of Embarkation (Embarked):", ["Cherbourg", "Queenstown", "Southampton"])
    Embarked = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}[Embarked]

    # Predict button
    if st.button("Predict Survival"):
        inputs = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
        # Make a prediction
        prediction_0 = model.predict(inputs)
        probability_0 = model.predict_proba(inputs)
        prediction = prediction_0[0]
        probability = probability_0[0][1]

        # Display the results
        if prediction == 1:
            st.success(f"The passenger is predicted to SURVIVE with a probability of {probability:.2f}.")
        else:
            st.error(f"The passenger is predicted NOT to survive with a probability of {1 - probability:.2f}.")

    # Footer
    st.write("### Note:")
    st.write("This prediction is based on a machine learning model trained on the Titanic dataset and may not reflect real-world outcomes.")
