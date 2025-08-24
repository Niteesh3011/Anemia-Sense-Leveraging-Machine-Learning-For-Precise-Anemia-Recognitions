# AnemiaSense: Leveraging Machine Learning for Anemia Recognition

AnemiaSense is a data science project that uses machine learning to predict the presence of anemia based on key blood test metrics. The project includes a full data analysis pipeline in a Jupyter Notebook and is deployed as a simple, user-friendly web application using Flask.

![AnemiaSense Web App]((https://drive.google.com/file/d/1KCSPSb7AeZyqeoeRSYcz_kJQJpEp6qUv/view?usp=drive_link)(https://drive.google.com/file/d/1MFnPnJGSm26eFxilBTcxrpxI8tFnHShh/view?usp=drive_link))

## 📋 Features

-   **Data Analysis & Visualization**: In-depth exploratory data analysis using Pandas, Matplotlib, and Seaborn.
-   **Data Preprocessing**: Handles class imbalance in the target variable ('Result') using downsampling techniques.
-   **Model Training & Evaluation**: Trains and evaluates six different classification models:
    -   Logistic Regression
    -   Random Forest
    -   Decision Tree
    -   Gaussian Naive Bayes
    -   Support Vector Machine (SVM)
    -   Gradient Boosting Classifier
-   **High-Accuracy Prediction**: The final model (Gradient Boosting Classifier) achieves 100% accuracy on the test set.
-   **Interactive Web Interface**: A simple web app built with Flask that allows users to input patient data and receive an instant prediction.

## 🛠️ Tech Stack & Dependencies

-   **Language**: Python 3.x
-   **Libraries**:
    -   Flask
    -   Pandas
    -   NumPy
    -   Scikit-learn
    -   Matplotlib
    -   Seaborn
    -   Pickle

## 📁 Project Structure

```
├── Anemia-Sense-Project/
│   ├── templates/
│   │   ├── index.html       # HTML for the input form
│   │   └── result.html      # HTML for the prediction result page
│   ├── AnemiaSense.ipynb    # Jupyter Notebook with the full analysis
│   ├── app.py               # The Flask web application script
│   ├── anemia.csv           # The dataset
│   ├── model.pkl            # Saved (pickled) trained model
│   └── README.md            # This file
```

## ⚙️ Setup and Installation

To run this project locally, please follow these steps:

**1. Clone the Repository**
```bash
git clone [https://github.com/your-username/Anemia-Sense-Leveraging-Machine-Learning.git](https://github.com/your-username/Anemia-Sense-Leveraging-Machine-Learning.git)
cd Anemia-Sense-Leveraging-Machine-Learning
```

**2. Create a Virtual Environment (Recommended)**
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
Install all the required packages using the `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*(You will need to create a `requirements.txt` file with the content provided in the next section).*

## ▶️ How to Use

### 1. Running the Web Application

The easiest way to use the predictor is via the Flask web app.

**a. Train and Save the Model (First-time setup)**
Before running the app for the first time, you need to create the `model.pkl` file. Run the `AnemiaSense.ipynb` notebook until cell 27 to train the models. Then, run the following Python code to save your trained Gradient Boosting model:

```python
# Add this to a new cell in your notebook or run as a separate script
import pickle
# 'GBC' is your trained GradientBoostingClassifier object from the notebook
with open('model.pkl', 'wb') as model_file:
    pickle.dump(GBC, model_file)
```

**b. Start the Flask Server**
From your terminal (with the virtual environment activated), run the `app.py` script:
```bash
python app.py
```

**c. Open in Browser**
Open your web browser and navigate to:
**http://127.0.0.1:5000**

You can now enter the patient's data into the form to get a prediction.

### 2. Exploring the Jupyter Notebook

To see the full data analysis, visualization, and model training process, you can run the `AnemiaSense.ipynb` notebook using Jupyter Notebook or a compatible IDE like VS Code.

## 🤖 Model Training Process

The machine learning model was developed as follows:
1.  **Data Loading**: The `anemia.csv` dataset was loaded, containing 5 features (`Gender`, `Hemoglobin`, `MCH`, `MCHC`, `MCV`) and the target variable `Result`.
2.  **Balancing**: The dataset was found to be imbalanced. To prevent model bias, the majority class was downsampled to match the number of samples in the minority class.
3.  **Model Selection**: Six different models were trained and evaluated on their accuracy.
4.  **Results**: The **Gradient Boosting Classifier**, along with Random Forest and Decision Tree, achieved **100% accuracy** on the test set and was chosen for the final application.

## 📄 requirements.txt File

Create a file named `requirements.txt` in your main project folder and add the following content:

```
Flask==3.0.3
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, please fork the repository and submit a pull request.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
