"""XGBoost-based predictor: predicts next-GW points using engineered features."""

from __future__ import annotations

from fpl_model.models.base import HistoricalData, PlayerPredictions, Predictor, SeasonData
from fpl_model.models.features import build_feature_matrix, build_player_features
from fpl_model.simulation.state import SquadState


class XGBoostPredictor(Predictor):
    """Predict player points using a gradient-boosted tree model.

    Features include rolling form, expected stats, fixture difficulty,
    ICT metrics, and market momentum. Falls back to form-based prediction
    if the model has not been trained.
    """

    def __init__(self, lookback_start: int = 5, recency_decay: float = 1.0) -> None:
        self.lookback_start = lookback_start
        self.recency_decay = recency_decay
        self.model = None
        self._feature_cols: list[str] | None = None

    def train(self, historical_data: HistoricalData) -> None:
        """Train XGBoost on all (player, GW) samples from historical seasons."""
        import xgboost as xgb

        rows_X = []
        rows_y = []
        rows_gw: list[int] = []

        for season_name, season_data in historical_data.seasons.items():
            gw_perf = season_data.gameweek_performances
            gameweeks = sorted(gw_perf["gameweek"].unique())

            for gw in gameweeks:
                if gw < self.lookback_start + 1:
                    continue  # need history to build features

                # Build SeasonData up to (but not including) this GW for features
                visible_data = SeasonData(
                    gameweek_performances=gw_perf[gw_perf["gameweek"] < gw],
                    fixtures=season_data.fixtures,
                    players=season_data.players,
                    teams=season_data.teams,
                    current_gameweek=gw,
                    season=season_name,
                )

                # Actual points for this GW (target)
                gw_actual = gw_perf[gw_perf["gameweek"] == gw]
                actual_map = dict(zip(gw_actual["player_code"], gw_actual["total_points"]))

                for code in season_data.players["code"]:
                    code = int(code)
                    if code not in actual_map:
                        continue
                    feats = build_player_features(visible_data, code, gw)
                    rows_X.append(feats)
                    rows_y.append(float(actual_map[code]))
                    rows_gw.append(int(gw))

        if not rows_X:
            return

        import pandas as pd
        X = pd.DataFrame(rows_X)
        self._feature_cols = list(X.columns)
        y = pd.Series(rows_y)

        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            verbosity=0,
        )
        import numpy as np

        gw_array = np.array(rows_gw)
        max_gw = gw_array.max()
        weights = self.recency_decay ** (max_gw - gw_array)

        self.model.fit(X, y, sample_weight=weights)

    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Predict next-GW points for all players."""
        matrix = build_feature_matrix(data)

        if self.model is not None and self._feature_cols is not None:
            codes = matrix["player_code"].tolist()
            X = matrix[self._feature_cols]
            y_pred = self.model.predict(X)
            predictions = {int(c): float(p) for c, p in zip(codes, y_pred)}
        else:
            # Fallback: use form_5 as prediction
            predictions = {}
            for _, row in matrix.iterrows():
                code = int(row["player_code"])
                predictions[code] = float(row.get("form_5", 2.0))

        return PlayerPredictions(predictions=predictions)
