"""
Surrogate machine learning model for predicting FCC performance:
- Primary Targets: Gasoline Octane (RON), Coke Yield (wt%), Gasoline Yield (wt%)
- Model: Multi-output Gradient Boosted Regressor (LightGBM/RandomForest) with feature importance
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def load_dataset(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    flat_records = []
    for r in records:
        flat = {
            # Categorical
            "zeolite_type": r["catalyst"]["zeolite_type"],
            "framework_type": r["catalyst"]["framework_type"],
            # Catalyst composition
            "si_al_ratio": r["catalyst"]["si_al_ratio"],
            "unit_cell_size": r["catalyst"]["unit_cell_size_angstrom"],
            "rare_earth_wt": r["catalyst"]["rare_earth_oxide_wt_pct"],
            "matrix_alumina_wt": r["catalyst"]["matrix_alumina_wt_pct"],
            "binder_clay_wt": r["catalyst"]["binder_clay_wt_pct"],
            # Textural properties
            "bet_surface_area": r["texture"]["bet_surface_area_m2_g"],
            "zeolite_surface_area": r["texture"]["zeolite_surface_area_m2_g"],
            "matrix_surface_area": r["texture"]["matrix_surface_area_m2_g"],
            "total_pore_volume": r["texture"]["total_pore_volume_cm3_g"],
            "bronsted_acid": r["texture"]["bronsted_acid_sites_umol_g"],
            "lewis_acid": r["texture"]["lewis_acid_sites_umol_g"],
            # Operating conditions
            "temp_c": r["operating_conditions"]["reaction_temp_celsius"],
            "cat_to_oil": r["operating_conditions"]["cat_to_oil_ratio"],
            "residence_time": r["operating_conditions"]["residence_time_sec"],
            "feedstock_api": r["operating_conditions"]["feedstock_api_gravity"],
            "feedstock_ccr": r["operating_conditions"]["feedstock_conradson_carbon_wt_pct"],
            # Performance Targets
            "target_ron": r["performance"]["gasoline_ron"],
            "target_coke_wt": r["performance"]["coke_yield_wt_pct"],
            "target_gasoline_yield": r["performance"]["gasoline_yield_wt_pct"],
            "target_conversion": r["performance"]["feed_conversion_wt_pct"],
        }
        flat_records.append(flat)

    df = pd.DataFrame(flat_records)
    df = pd.get_dummies(df, columns=["zeolite_type", "framework_type"], drop_first=False)
    return df


def train_surrogate_model(data_path: Path):
    df = load_dataset(data_path)

    target_cols = ["target_ron", "target_coke_wt", "target_gasoline_yield"]
    feature_cols = [c for c in df.columns if not c.startswith("target_")]

    X = df[feature_cols]
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

    print("=" * 60)
    print("      FCC CATALYST SURROGATE PREDICTOR EVALUATION")
    print("=" * 60)
    for col in target_cols:
        r2 = r2_score(y_test[col], y_pred_df[col])
        mae = mean_absolute_error(y_test[col], y_pred_df[col])
        rmse = np.sqrt(mean_squared_error(y_test[col], y_pred_df[col]))
        print(f"Target: {col:<22} | R²: {r2:6.4f} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f}")

    # Top Feature Importance
    print("-" * 60)
    print("Top Key Drivers (Feature Importances across targets):")
    importances = model.feature_importances_
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
    for feat, score in feat_series.head(7).items():
        print(f"  • {feat:<28}: {score * 100:5.2f}%")
    print("=" * 60)

    return model, feature_cols


if __name__ == "__main__":
    data_file = Path(__file__).resolve().parent.parent.parent / "data" / "catalysts_seed.json"
    train_surrogate_model(data_file)
