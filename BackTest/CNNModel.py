import numpy as np
import torch
import hashlib
import os

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

    img_ohlc = np.flip(img_ohlc, 1)

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
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d((2, 1))
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=(5, 3), padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(negative_slope=0.01),
            torch.nn.MaxPool2d((2, 1))
        )
        self.out_block = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(161280, 2),
            # torch.nn.Softmax(dim=-1)
            )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.out_block(x)
        return x

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
VALID_RATIO = 0.3

def train_model(Xs, ys):
    ys_pt = torch.LongTensor(ys > 0)
    xs_pt = torch.zeros(size=(Xs.shape[0], 120, 45), dtype=torch.uint8)
    for i in range(Xs.shape[0]):
        xs_pt[i] = torch.tensor(make_image(Xs[i]))

    perm = torch.randperm(Xs.shape[0])
    xs_pt[:] = xs_pt[perm]
    ys_pt[:] = ys_pt[perm]

    split_idx = int(VALID_RATIO * Xs.shape[0])
    xs_train, xs_valid = xs_pt[split_idx:], xs_pt[:split_idx]
    ys_train, ys_valid = ys_pt[split_idx:], ys_pt[:split_idx]

    ds_train = torch.utils.data.TensorDataset(xs_train, ys_train)
    ds_valid = torch.utils.data.TensorDataset(xs_valid, ys_valid)
    m = StockCNN()
    m.train()
    m = m.to(DEVICE)
    m = torch.nn.DataParallel(m)
    opt = torch.optim.Adam(m.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss()

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=BATCH_SIZE)

    last_epoch_improved = 0
    valid_loss_last_improved = 1e6


    for ep in range(NUM_EPOCHS):
        train_total_loss = 0
        train_n_batches = 0
        train_n_hits = 0
        train_n_total = 0

        m.train()
        for i, (xs, ys) in enumerate(dl_train):
            xs = xs.to(DEVICE)
            xs = xs.unsqueeze(1).float() / 255.0
            ys = ys.to(DEVICE).squeeze()
            opt.zero_grad()
            y_hat = m(xs)
            loss = loss_fn(y_hat, ys)
            loss.backward()

            #if i == 0:
            #    print(ys[0], y_hat[0], loss.cpu().item())

            train_total_loss += loss.cpu().item()
            train_n_batches += 1
            train_n_hits += (torch.argmax(y_hat, dim=-1) == ys).sum().cpu().item()
            train_n_total += y_hat.shape[0]

            opt.step()

        # print(f'Ep[{ep}]: TLoss: {train_total_loss/train_n_batches}, TAcc: {train_n_hits/train_n_total:.2f}')

        valid_total_loss = 0
        valid_n_batches = 0
        valid_n_hits = 0
        valid_n_total = 0
        m.eval()

        with torch.inference_mode():
            for i, (xs, ys) in enumerate(dl_valid):
                xs = xs.to(DEVICE)
                xs = xs.unsqueeze(1).float() / 255.0
                ys = ys.to(DEVICE).squeeze()
                y_hat = m(xs)
                loss = loss_fn(y_hat, ys)

                #if i == 0:
                #    print(ys[0], y_hat[0])

                valid_total_loss += loss.cpu().item()
                valid_n_batches += 1
                valid_n_hits += (torch.argmax(y_hat, dim=-1) == ys).sum().cpu().item()
                valid_n_total += y_hat.shape[0]

        print(f'Ep[{ep}]: T: {train_total_loss/train_n_batches} ({train_n_hits/train_n_total:.4f}), V: {valid_total_loss/valid_n_batches} ({valid_n_hits/valid_n_total:.4f})')

        valid_loss = valid_total_loss/valid_n_batches
        if valid_loss < valid_loss_last_improved:
            valid_loss_last_improved = valid_loss
            last_epoch_improved = ep
            traincache_dir = 'traincache'
            if not os.path.exists(traincache_dir):
                os.makedirs(traincache_dir)
            torch.save(m, 'traincache/bestmodel.pt')
            print(f'Ep[{ep}]: Model perf improved is now {valid_loss}.')
        elif ep > last_epoch_improved + 2:
            print('Breaking due to earlystopping.')
            m = torch.load('traincache/bestmodel.pt')
            break

    return m

def get_hash():
    cur_path = os.path.abspath(__file__)
    return hashlib.md5(open(cur_path, 'rb').read()).hexdigest()