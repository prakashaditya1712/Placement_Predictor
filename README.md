# LaunchLens AI — Student Placement Predictor

A beginner-friendly Streamlit project that predicts a student's placement outlook from academic, internship, communication, extra-curricular and project inputs.

## Files
- `app.py` — Streamlit user interface, prediction form, result graphics and dataset overview.
- `train_model.py` — cleans the dataset, trains/evaluates a Random Forest model, and saves `model.pkl`.
- `dataset.csv` — cleaned student placement dataset.
- `requirements.txt` — Python dependencies.
- `model.pkl` — created when you run the training script.

## Run in VS Code
1. Extract/open this folder in VS Code.
2. Open **Terminal → New Terminal**.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Train the model (this creates or replaces `model.pkl`):
   ```bash
   python train_model.py
   ```
5. Start the website:
   ```bash
   streamlit run app.py
   ```

If Streamlit prints a local URL, open that URL in your browser. Keep `app.py`, `train_model.py`, `dataset.csv`, and `model.pkl` in the same folder.

## Prediction labels
- **Positive Placement Outlook** — the model predicts the positive class.
- **More Preparation Recommended** — the model predicts the negative class.

These labels are deliberately not written as a guarantee of placement or rejection. This is an educational model and should not be used to make real hiring decisions. High performance on synthetic or rule-based labels may not generalize to real-world placement outcomes.
