# Student Performance Prediction

Explore a student dataset, compare machine-learning classifiers, and try sample
performance predictions in a browser-based Streamlit app. The original
command-line program is also available as `Project.py`.

## Run the website locally

1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

Streamlit will print a local URL to open in your browser.

The app reads `AI-Data.csv` from the same folder as `app.py`. It compares
Decision Tree, Random Forest, Perceptron, Logistic Regression, and Neural
Network classifiers using a reproducible stratified train/test split.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. Select **Create app**, then choose this repository and the `main` branch.
4. Set the app file path to `app.py`.
5. Select **Deploy**. Streamlit Cloud installs the packages listed in
   `requirements.txt`.

Keep `AI-Data.csv`, `app.py`, and `requirements.txt` in the repository so the
deployed app can load its data and dependencies.
