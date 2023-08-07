import numpy as np
import torch
from scipy.special import expit

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3


def bresenham(img, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    while True:
        img[x0, y0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                break
            error += dy
            x0 += sx
        if e2 <= dx:
            if y0 == y1:
                break
            error += dx
            y0 += sy

def make_image(sample, ema=True, volume=True):
    height_bars = 96
    width = sample.shape[0] * 3
    img_ohlc = np.zeros((width, height_bars), dtype=np.uint8)

    max_price = max(sample[:, :4].max(), sample[:, 5].max())
    min_price = min(sample[:, :4].min(), sample[:, 5].min())
    height_scaler = (height_bars - 1) / (max_price - min_price)

    ema_y_prev = None

    for t in range(sample.shape[0]):
        open_y = round((sample[t, 1] - min_price) * height_scaler)
        img_ohlc[3*t, open_y] = 255
        close_y = round((sample[t, 0] - min_price) * height_scaler)
        img_ohlc[3*t+2, close_y] = 255

        low_y = round((sample[t, 3] - min_price) * height_scaler)
        high_y = round((sample[t, 2] - min_price) * height_scaler)
        img_ohlc[3*t+1, low_y:high_y] = 255

        if ema:
            ema_y = round((sample[t, 5] - min_price) * height_scaler)
            img_ohlc[3*t+1, ema_y] = 255
            if ema_y_prev is not None:
                bresenham(img_ohlc, 3*t-2, ema_y_prev, 3*t+1, ema_y)
            ema_y_prev = ema_y

    if not volume:
        return img_ohlc.T

    height_vol = 24
    height_whole = height_bars + height_vol if volume else 0
    img_whole = np.zeros((width, height_whole), dtype=np.uint8)
    img_whole[:, :height_bars] = img_ohlc

    max_vol = sample[:, 4].max()
    vol_scaler = (height_vol - 1)/max_vol
    for t in range(sample.shape[0]):
        vol_y = round(sample[t, 4] * vol_scaler)
        img_whole[3*t+1, height_whole-vol_y-1:height_whole-1] = 255

    return img_whole.T

class StockCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(5, 3), padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
            torch.nn.MaxPool2d((2, 1))
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(5, 3), padding=1),
            torch.nn.LeakyReLU(negative_slope=0.01, inplace=True),
            torch.nn.MaxPool2d((2, 1))
        )
        self.out_block = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(161280, 2),
            torch.nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_block(x)
        return x

class CNNStrategy:
    def __init__(self,strategy_type, retrain_every=252):
        self._retrain_every = retrain_every
        self._trained = False
        self._strategy_type = strategy_type

    def get_dollar_weights(self, backtester, adj_universe, price_by_ticker):
        if (backtester.n_day + 1) % self._retrain_every == 0:
            self.train(backtester.price_history)

        if not self._trained:
            return {ticker: 0 for ticker in adj_universe}
        else:
            return self.infer(adj_universe, backtester.price_history)

    def infer(self, adj_universe, price_history):
        Xs, fix_id_by_adj_universe = self._extract_data_for_inference(price_history, adj_universe)
        scales = Xs[:, 0, 1].reshape(-1, 1, 1).copy()
        Xs[:, :, [5]] /= scales
        Xs[:, :, :4] /= scales

        xs_pt = torch.zeros(size=(Xs.shape[0], 120, 45), dtype=torch.uint8)
        for i in range(Xs.shape[0]):
            xs_pt[i] = torch.tensor(make_image(Xs[i]))

        xs_pt = xs_pt.float() / 255.0
        xs_pt = xs_pt.unsqueeze(1)


        with torch.inference_mode():
            y_hats = self._model(xs_pt).cpu().numpy()

        if self._strategy_type == "equalpositive":

            y_argmaxes = np.argmax(y_hats, axis=-1)
            n_pos = y_argmaxes.sum()

            out = {}
            for i, (t, v) in enumerate(fix_id_by_adj_universe.items()):
                out[t] = 1/n_pos if y_argmaxes[i] > 0 else 0
            for t in adj_universe:
                if not t in out:
                    out[t] = 0
            return out

        if self._strategy_type == "equalpercent":

            n_stocks_to_invest = int(len(y_hats) * 0.10)
            sorted_stocks = sorted(zip(fix_id_by_adj_universe.keys(), y_hats[:, 1]), key=lambda x: x[1], reverse=True)

            out = {}

            for i, (ticker, _) in enumerate(sorted_stocks):
                if i < n_stocks_to_invest:
                    out[ticker] = 1/n_stocks_to_invest
                else:
                    out[ticker] = 0

            # Assign a weight of 0 to any stocks in the adjusted universe that were not included in the prediction
            for ticker in adj_universe:
                if ticker not in out:
                    out[ticker] = 0

            return out

        if self._strategy_type == "sigmoid":

            out = {}
            stocks_with_positive_y_hat = [i for i, y_hat in enumerate(y_hats) if y_hat[1] > y_hat[0]]
            y_hats_positive = y_hats[stocks_with_positive_y_hat, 1]
            mean_y_hat_positive = np.mean(y_hats_positive)
            sigma_y_hat_positive = np.std(y_hats_positive)


            for i, (ticker, _) in enumerate(fix_id_by_adj_universe.items()):
                if i in stocks_with_positive_y_hat:
                    out[ticker] = expit((y_hats[i, 1] - mean_y_hat_positive) / sigma_y_hat_positive)
                else:
                    out[ticker] = 0

            total_weight = sum(out.values())
            for ticker in out:
                out[ticker] /= total_weight

            return out

        if self._strategy_type == "marketindicator":
            positive_stocks = np.sum(y_hats[:, 1] > y_hats[:, 0])
            total_stocks = y_hats.shape[0]
            exposure_ratio = positive_stocks / total_stocks

            out = {}
            for ticker in adj_universe:
                out[ticker] = 1.0 / len(adj_universe)

            for ticker in out:
                out[ticker] *= exposure_ratio

            return out

    def _extract_data_for_inference(self, price_history, adj_universe):
        input_types =  [
                    'adj_close',
                    'adj_open',
                    'adj_high',
                    'adj_low',
                    'vol',
                ]
        price_matrix_by_type = {k: price_history.get_price_matrix(k) for k in input_types}
        input_window = 15
        ema_input_window = 40
        total_input_window = input_window + ema_input_window + 1
        ema_const = 1 - np.exp(np.log(0.5) / 7.5)

        fix_id_by_adj_universe = {t: price_history._fix_id_by_ticker[t] for t in adj_universe if t in price_history._fix_id_by_ticker}
        fix_id_by_adj_universe2 = {}
        for k, v in fix_id_by_adj_universe.items():
            is_good = True
            for t in input_types:
                price_matrix = price_matrix_by_type[t]
                if np.isnan(price_matrix[-total_input_window:, v]).any():
                    is_good = False
                    break
                if t == 'vol':
                    if price_matrix[-(input_window + 1):, v].sum() < 1:
                        is_good = False
            if is_good:
                fix_id_by_adj_universe2[k] = v
        fix_id_by_adj_universe = fix_id_by_adj_universe2

        X_pred = np.zeros((len(fix_id_by_adj_universe), input_window, len(input_types) + 1))

        for i, (t, v) in enumerate(fix_id_by_adj_universe.items()):
            adj_close = price_matrix_by_type['adj_close'][-total_input_window:, v]
            adj_close_ema = np.array(adj_close)
            for j in range(1, len(adj_close_ema)):
                adj_close_ema[j] = (1 - ema_const) * adj_close_ema[j-1] + ema_const * adj_close_ema[j]

            for j, t in enumerate(input_types):
                price_series = price_matrix_by_type[t][-input_window:, v]
                X_pred[i, :, j] = price_series
            X_pred[i, :, -1] = adj_close_ema[-15:]

        return X_pred, fix_id_by_adj_universe

    def _extract_data(self, price_history):
        input_types =  [
                    'adj_close',
                    'adj_open',
                    'adj_high',
                    'adj_low',
                    'vol',
                ]
        price_matrix_by_type = {k: price_history.get_price_matrix(k) for k in input_types}
        price_matrix = price_matrix_by_type['adj_close']

        idx_now = price_matrix.shape[0] - 1
        prediction_horizon = 5
        input_window = 15
        data_window = prediction_horizon + input_window + 1
        ema_const = 1 - np.exp(np.log(0.5) / 7.5) # decay factor for EMA

        n_datapoints = 0

        for start_date_idx in range(0, idx_now - data_window):
            for stock_idx in range(0, price_matrix.shape[1]):

                is_ok = True
                for t in input_types:
                    submatrix = price_matrix_by_type[t]
                    if np.isnan(submatrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                        is_ok = False
                        break

                    if t == 'vol':
                        if submatrix[start_date_idx:start_date_idx+input_window, stock_idx].sum() < 1:
                            is_ok = False
                            break
                if is_ok:
                    n_datapoints += 1

        X = np.zeros((n_datapoints, input_window, len(input_types) + 1))
        ema_idx = len(input_types)
        y = np.zeros((n_datapoints, 1))
        i = 0

        for stock_idx in range(0, price_matrix.shape[1]):
            adj_close_ema = np.nan

            for start_date_idx in range(0, idx_now - data_window):
                if not np.isnan(price_matrix[start_date_idx, stock_idx]):
                    if np.isnan(adj_close_ema):
                        adj_close_ema = price_matrix[start_date_idx, stock_idx]
                    else:
                        adj_close_ema = (1 - ema_const) * adj_close_ema + ema_const * price_matrix[start_date_idx, stock_idx]

                is_ok = True

                for d, t in enumerate(input_types):
                    submatrix = price_matrix_by_type[t]
                    if np.isnan(price_matrix[start_date_idx:start_date_idx+data_window, stock_idx]).any():
                        is_ok = False
                        break

                    price_series = submatrix[start_date_idx:start_date_idx+data_window, stock_idx]
                    X[i, :, d] = price_series[:input_window]

                    if t == 'vol':
                        if X[i, :, d].sum() < 1:
                            is_ok = False
                            break

                if not is_ok:
                    continue

                X[i, 0, ema_idx] = adj_close_ema
                for t in range(1, X.shape[1]):
                    X[i, t, ema_idx] = (1 - ema_const) * X[i, t-1, ema_idx] + ema_const * X[i, t, 0]

                close_matrix = price_matrix_by_type['adj_close']
                price_series = close_matrix[start_date_idx:start_date_idx+data_window, stock_idx]
                y[i] = price_series[-1]/X[i, -1, 0] - 1.0
                i += 1
        assert i == n_datapoints

        np.savez('inputs.npz', X, y)

        return X, y

    def train(self, price_history):
        Xs, ys = self._extract_data(price_history)
        scales = Xs[:, 0, 1].reshape(-1, 1, 1).copy()
        Xs[:, :, [5]] /= scales
        Xs[:, :, :4] /= scales

        ys_pt = torch.LongTensor(ys > 0)
        xs_pt = torch.zeros(size=(Xs.shape[0], 120, 45), dtype=torch.uint8)
        for i in range(Xs.shape[0]):
            xs_pt[i] = torch.tensor(make_image(Xs[i]))

        ds_train = torch.utils.data.TensorDataset(xs_pt, ys_pt)
        m = StockCNN()
        m = m.to(DEVICE)
        m = torch.nn.DataParallel(m)
        opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
        loss_fn = torch.nn.CrossEntropyLoss()

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)

        for ep in range(NUM_EPOCHS):
            train_total_loss = 0
            train_n_batches = 0
            train_n_hits = 0
            train_n_total = 0

            for xs, ys in dl_train:
                xs = xs.to(DEVICE)
                xs = xs.unsqueeze(1).float() / 255.0
                ys = ys.to(DEVICE).squeeze()
                opt.zero_grad()
                y_hat = m(xs)
                loss = loss_fn(y_hat, ys)
                loss.backward()

                train_total_loss += loss.cpu().item()
                train_n_batches += 1
                train_n_hits += (torch.argmax(y_hat, dim=-1) == ys).sum().cpu().item()
                train_n_total += y_hat.shape[0]

                opt.step()

        self._model = m
        self._trained = True


