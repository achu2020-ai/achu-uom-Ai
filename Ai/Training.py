!pip install xgboost scikit-learn pandas numpy matplotlib seaborn






import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pickle


from google.colab import files
uploaded = files.upload()   # Choose your Housing.csv file

df = pd.read_csv("ouse.csv")
print("✅ Dataset loaded successfully!")
df.head()


# Convert yes/no to 1/0
binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

# One-hot encode furnishingstatus
df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)

# Split input & target
X = df.drop(columns=['price'])
y = df['price']

# Split training/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numeric_cols = ['area','bedrooms','bathrooms','stories','parking']
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols)], remainder='passthrough')

# Define models
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb_model)
])


lr_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)
print("✅ Training completed for both models!")


def evaluate_model(name, pipeline, X_tr, X_te, y_tr, y_te):
    y_pred = pipeline.predict(X_te)
    rmse = mean_squared_error(y_te, y_pred)
    mae = mean_absolute_error(y_te, y_pred)
    r2 = r2_score(y_te, y_pred)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rmse = np.sqrt(-cross_val_score(pipeline, X_tr, y_tr, scoring='neg_mean_squared_error', cv=cv))
    print(f"\n===== {name} =====")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE : {mae:.2f}")
    print(f"R²  : {r2:.4f}")
    print(f"CV RMSE (mean ± std): {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
    return rmse

rmse_lr = evaluate_model("Linear Regression", lr_pipeline, X_train, X_test, y_train, y_test)
rmse_xgb = evaluate_model("XGBoost", xgb_pipeline, X_train, X_test, y_train, y_test)

best_pipeline = lr_pipeline if rmse_lr < rmse_xgb else xgb_pipeline
print("\n🏆 Best Model:", "Linear Regression" if rmse_lr < rmse_xgb else "XGBoost")

# Feature importance
all_features = X.columns
model = best_pipeline.named_steps['model']

if hasattr(model, 'feature_importances_'):
    fi = pd.Series(model.feature_importances_, index=all_features).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    fi.head(10).plot(kind='bar')
    plt.title("Top 10 Feature Importances (XGBoost)")
    plt.show()
elif hasattr(model, 'coef_'):
    coef = pd.Series(model.coef_, index=all_features).sort_values(key=abs, ascending=False)
    plt.figure(figsize=(8,5))
    coef.head(10).plot(kind='bar')
    plt.title("Top 10 Coefficients (Linear Regression)")
    plt.show()


with open("best_model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("💾 Model saved as best_model.pkl successfully!")
