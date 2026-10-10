Churn Risk Advisor

A Streamlit app that ranks telecom customers by churn risk, built from my Week 1-3 notebooks (IBM Telco Customer Churn dataset). Decision support for a retention team, not automation.

Live app: https://churn-risk-advisor-mifk87bjaqxx5xfumfjtw3.streamlit.app/


Model card: Churn Risk Advisor v1.0

Intended use: rank existing telecom customers by churn risk so a retention team can prioritize calls. Decision support, not automation. Not for: credit, pricing, or any decision that denies a service. Data: IBM Telco Customer Churn, 7,043 customers, 26.5% churn. Model: XGBClassifier (tuned with random search), chosen by 5-fold CV. Features: 30 encoded columns. Performance: CV AUC 0.850 +/- 0.012; test AUC 0.848 (test set used once). Threshold: 0.20, from a cost analysis (FN = PKR 6,000, FP = PKR 1,000) on out-of-fold training predictions; the theoretical value is 0.14 and the cost curve is flat between 0.15 and 0.20. Limitations: one US dataset from one period; no Pakistan data; associations, not causes; performance may drift as plans change. Fairness check (Week 15): compare recall across gender and seniors. Owner and version: Shoaib ur Rehman, v1.0, October 10, 2026.

Run it locally

Needs Python 3.12 or newer (the pinned xgboost 3.4.1 does not install on 3.11). Windows (Mac/Linux: source .venv/bin/activate):

cd churn-risk-advisor
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Files

app.py, churn_model.joblib, model_meta.json, requirements.txt (pins scikit-learn 1.6.1 and xgboost 3.4.1, the versions used in training), sample_customers.csv (50 customers for the batch demo), README.md.

Notebooks

Weeks 1 to 4 notebooks and write-ups: https://github.com/Shoaibrhman/customer-churn-prediction-1

Build notes

Part 1 (packaging): the Week 3 way of encoding a single customer (drop_first=True) gave P(churn) = 0.136 for a two-year-contract customer; the serving encoder gives 0.012. The parity test passes on all 7,043 customers (largest difference 0.0). Full explanation in the notebook.

Part 2 (app tests, run headlessly with Streamlit's AppTest and by hand in the browser):

Test	Result
Default inputs	74% churn probability, HIGH band, "Contact now"
Internet service = No	add-on selector disappears; probability changes to 25%
Tenure 60 + two-year contract	9%, LOW band
Move the threshold slider	probability stays 74%; band goes HIGH to WATCH at 0.95
"Why this score?"	the lab's version is Logistic Regression only; my final model is XGBoost, so I added an XGBoost version (per-customer contributions in log-odds; they add up exactly to the probability)
Batch: upload sample_customers.csv	scores 50 customers (24 above the 0.20 threshold); download works
Batch: CSV with one column deleted	clear error naming the column, no crash

The what-if table is labelled as an association learned from data, not a guaranteed effect.

Behavior that surprised me: the threshold slider changes only the band and the action, never the probability, and at the cost-based threshold of 0.20 the app flags 24 of the 50 sample customers (48%) even though only 26.5% churn, because a missed churner is assumed to cost six times a wasted offer.

Input handled badly: a batch CSV with a misspelled category (for example two year instead of Two year) raises no error; all of its dummy columns become 0 and the customer is scored as Month-to-month (P(churn) 0.029 became 0.123 in my test). A real user could easily enter this. A fix would be to validate each categorical column against the allowed values and show an error.

Part 3 (deployment): public URL: https://churn-risk-advisor-mifk87bjaqxx5xfumfjtw3.streamlit.app/. First deployment: failed. The build log said ERROR: Could not find a version that satisfies the requirement xgboost==3.4.1 (the resolver explained that xgboost 3.4.1 depends on Python>=3.12, but I had chosen Python 3.11). I fixed it by deleting the app and redeploying with Python 3.12 (the app is deployed on Python 3.12), keeping the pinned xgboost version so the saved model loads in the same library version it was trained with.

Check your understanding
Why does get_dummies(drop_first=True) break predictions when scoring a single customer? With one row, each column has only one category, which is treated as the first and dropped, so all categories become 0 and the customer is read as the reference category. No error is raised; in my test P(churn) went from 0.012 to 0.136.
What does the parity test compare, and why every training row? It compares the probabilities from the training-way features with the probabilities from the serving encoder. A mismatch can affect only some categories (for example "No phone service"), so a few rows could miss it; checking all 7,043 rows with assert np.allclose makes a single mismatch fail.
st.cache_resource or st.cache_data? st.cache_resource: it keeps one shared model object loaded once per server. st.cache_data would copy (serialize) the return value, which is slow and wrong for a model.
Why derive TotalCharges? It equals roughly tenure x monthly charge, so a user could otherwise type inconsistent values (tenure 4 with TotalCharges 5,000), and it is a strong feature. It is an approximation: it ignores past price changes.
Why pin the exact scikit-learn version? A saved model depends on the library's internal structure. Another version can fail to unpickle or silently behave differently, so the server must install exactly what trained the model (and the same for xgboost).
Would offering a two-year contract make the customer stay? Not necessarily. The drop from, say, 0.79 to 0.27 is an association: customers who choose two-year contracts differ from those who do not. A company could find out with a randomized experiment: offer the contract to a random half of at-risk customers and compare churn afterwards.
