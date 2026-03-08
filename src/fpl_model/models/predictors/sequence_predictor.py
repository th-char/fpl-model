"""LSTM-based sequence predictor for FPL player points."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from fpl_model.models.base import HistoricalData, PlayerPredictions, Predictor, SeasonData
from fpl_model.models.features import _SEQ_COLS, build_sequence_features
from fpl_model.simulation.state import SquadState


class _PointsLSTM(nn.Module):
    """LSTM that takes a sequence of per-GW features and predicts next-GW points."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_positions: int = 4):
        super().__init__()
        self.pos_embedding = nn.Embedding(num_positions + 1, 4)  # 0=unknown, 1-4=positions
        self.lstm = nn.LSTM(input_size + 4, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size), pos: (batch,) int
        pos_emb = self.pos_embedding(pos)  # (batch, 4)
        pos_emb = pos_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # (batch, seq_len, 4)
        combined = torch.cat([x, pos_emb], dim=2)
        out, _ = self.lstm(combined)
        return self.fc(out[:, -1, :]).squeeze(-1)  # (batch,)


class SequencePredictor(Predictor):
    """Predict player points using an LSTM on recent GW sequences.

    Falls back to simple mean of recent points if the model is not trained.
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        recency_decay: float = 1.0,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.recency_decay = recency_decay
        self.model: _PointsLSTM | None = None

    def train(self, historical_data: HistoricalData) -> None:
        """Train LSTM on all (player, GW) sequences from historical seasons."""
        input_size = len(_SEQ_COLS)
        self.model = _PointsLSTM(input_size, self.hidden_size, self.num_layers)

        X_list, pos_list, y_list, gw_list = [], [], [], []

        for season_data in historical_data.seasons.values():
            gw_perf = season_data.gameweek_performances
            gameweeks = sorted(gw_perf["gameweek"].unique())

            for gw in gameweeks:
                if gw < 2:
                    continue
                gw_actual = gw_perf[gw_perf["gameweek"] == gw]
                actual_map = dict(zip(gw_actual["player_code"], gw_actual["total_points"]))

                visible = SeasonData(
                    gameweek_performances=gw_perf[gw_perf["gameweek"] < gw],
                    fixtures=season_data.fixtures,
                    players=season_data.players,
                    teams=season_data.teams,
                    current_gameweek=gw,
                    season=season_data.season,
                )

                for code in season_data.players["code"]:
                    code = int(code)
                    if code not in actual_map:
                        continue
                    seq = build_sequence_features(visible, code, gw, self.seq_len)
                    player_row = season_data.players[season_data.players["code"] == code]
                    et = int(player_row.iloc[0]["element_type"]) if len(player_row) > 0 else 0
                    X_list.append(seq)
                    pos_list.append(et)
                    y_list.append(float(actual_map[code]))
                    gw_list.append(gw)

        if not X_list:
            return

        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        pos = torch.tensor(pos_list, dtype=torch.long)
        y = torch.tensor(y_list, dtype=torch.float32)

        gw_array = torch.tensor(gw_list, dtype=torch.float32)
        max_gw = gw_array.max()
        weights = self.recency_decay ** (max_gw - gw_array)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        n = len(X)
        for epoch in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                pred = self.model(X[idx], pos[idx])
                loss = (weights[idx] * (pred - y[idx]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Predict next-GW points for all players in data.players."""
        predictions: dict[int, float] = {}
        gw = data.current_gameweek

        if self.model is not None:
            self.model.eval()
            X_list, pos_list, codes = [], [], []
            for code in data.players["code"]:
                code = int(code)
                seq = build_sequence_features(data, code, gw, self.seq_len)
                player_row = data.players[data.players["code"] == code]
                et = int(player_row.iloc[0]["element_type"]) if len(player_row) > 0 else 0
                X_list.append(seq)
                pos_list.append(et)
                codes.append(code)

            with torch.no_grad():
                X = torch.tensor(np.array(X_list), dtype=torch.float32)
                pos = torch.tensor(pos_list, dtype=torch.long)
                preds = self.model(X, pos).numpy()

            for code, pred in zip(codes, preds):
                predictions[code] = float(pred)
        else:
            # Fallback: mean of recent points
            gw_perf = data.gameweek_performances
            for code in data.players["code"]:
                code = int(code)
                past = gw_perf[(gw_perf["player_code"] == code) & (gw_perf["gameweek"] < gw)]
                if len(past) > 0:
                    predictions[code] = float(past.tail(5)["total_points"].mean())
                else:
                    predictions[code] = 2.0

        return PlayerPredictions(predictions=predictions)
