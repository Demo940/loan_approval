# Loan Approval Prediction App

This project is a simple Streamlit application that serves a pre-trained machine learning model for loan approval prediction. A user enters applicant details in the web interface, the app converts those values into the numeric representation expected by the model, applies the same scaler used during training, and then asks the saved model to predict whether the loan should be approved.

## Project Structure

- `app.py`
  The Streamlit application. It handles user input, preprocessing, scaling, prediction, and result display.
- `requirements.txt`
  The Python dependencies needed to run the app locally or on Streamlit Cloud.
- `model/loan_model.pkl`
  The serialized trained machine learning model.
- `model/scaler.pkl`
  The serialized scaler used to transform input features before prediction.

## What Happens When the App Runs

The app follows a straightforward inference pipeline:

1. It imports the required libraries:
   - `streamlit` for the UI
   - `numpy` for array formatting
   - `pickle` for loading serialized Python objects
   - `pathlib` for safe file path construction

2. It builds absolute paths to the saved model and scaler:
   - `MODEL_PATH` points to `model/loan_model.pkl`
   - `SCALER_PATH` points to `model/scaler.pkl`

3. It loads both `.pkl` files using `pickle.load(...)`:
   - the model object is restored into memory
   - the scaler object is restored into memory

4. It renders a form-like UI in Streamlit with the exact input fields needed for prediction:
   - Gender
   - Married
   - Dependents
   - Education
   - Self Employed
   - Applicant Income
   - Coapplicant Income
   - Loan Amount
   - Loan Amount Term
   - Credit History
   - Property Area

5. It converts categorical values into integers.
   Machine learning models usually do not accept raw string values like `"Male"` or `"Urban"`, so the app encodes them into numbers before prediction.

6. It combines all 11 values into a NumPy array and reshapes it into a two-dimensional structure:

```python
input_data = np.array([...]).reshape(1, -1)
```

This shape matters because most scikit-learn models expect input in the form:
- rows = number of samples
- columns = number of features

For a single prediction, the shape becomes `(1, 11)`.

7. It applies the scaler:

```python
input_data = scaler.transform(input_data)
```

This is a critical step. If the model was trained on scaled features, then new user input must be scaled in the same way. Otherwise, the model sees values in a different numeric range than it learned from during training, which can lead to poor or incorrect predictions.

8. It makes the prediction:

```python
prediction = model.predict(input_data)
```

The result is typically an array containing one value:
- `1` means approved
- `0` means not approved

9. It shows the result in the UI using Streamlit feedback messages.

## Understanding the Encoding Logic

The app currently uses the following numeric mappings:

- `Gender`
  - `Male -> 1`
  - `Female -> 0`
- `Married`
  - `Yes -> 1`
  - `No -> 0`
- `Dependents`
  - direct integer value: `0`, `1`, `2`, `3`
- `Education`
  - `Graduate -> 0`
  - `Not Graduate -> 1`
- `Self Employed`
  - `Yes -> 1`
  - `No -> 0`
- `Credit History`
  - `0` or `1` directly
- `Property Area`
  - `Rural -> 0`
  - `Semiurban -> 1`
  - `Urban -> 2`

These mappings only work correctly if they match the same encoding used during model training. If the training notebook or preprocessing script used different values, the app should be updated to match them exactly.

## Why the Scaler Is Stored Separately

The scaler is saved as its own file because preprocessing is part of the model pipeline. During training, the scaler learned statistics from the training data, such as:

- mean and standard deviation for standardization, or
- min and max values for normalization

When the app receives new user input, it must apply the same learned transformation so the model sees inputs in the same feature space it was trained on.

Without this step, even a correct model can behave badly because the numeric meaning of the features changes.

## Why `pickle.load(...)` Can Fail in Deployment

If the app runs locally but fails on Streamlit Cloud with an error like `ModuleNotFoundError` during `pickle.load(...)`, that usually means the deployment environment is missing a Python package that existed when the model was trained.

In this project, the main dependencies are:

- `streamlit`
- `numpy`
- `scikit-learn`

That is why `requirements.txt` includes those packages. When Streamlit Cloud deploys the app, it installs them before running `app.py`.

## How To Run Locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Example Test Input

You can test the UI with sample values like:

- Gender: `Male`
- Married: `Yes`
- Dependents: `1`
- Education: `Graduate`
- Self Employed: `No`
- Applicant Income: `5000`
- Coapplicant Income: `2000`
- Loan Amount: `150`
- Loan Amount Term: `360`
- Credit History: `1`
- Property Area: `Urban`

The exact output depends on the model that was trained and saved in `loan_model.pkl`.

## Important Assumptions

- The model expects exactly 11 input features in the same order used in `app.py`.
- The encoding logic in the app matches the encoding used during training.
- The scaler was fitted on the same feature order.
- The model file and scaler file are both valid pickle files.
- The deployment environment includes the libraries required to reconstruct the model object.

## Inference Flow Summary

The complete logic is:

1. Load saved model
2. Load saved scaler
3. Collect form inputs
4. Encode categorical fields into numbers
5. Build a `(1, 11)` NumPy array
6. Scale the array with the saved scaler
7. Pass the scaled array into the model
8. Convert the numeric prediction into a human-readable result

This makes the app a lightweight inference layer on top of a model that has already been trained elsewhere.
