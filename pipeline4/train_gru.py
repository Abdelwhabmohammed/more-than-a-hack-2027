"""Stage C: a GRU sequence model trained on the same degraded DFL tables, then
ensembled with the LightGBM stacker.

    cd more-than-a-hack-2027
    python -m pipeline4.train_gru             # requires Stage A first
    python -m pipeline4.train_gru --smoke

Design (ours; PyTorch, GPU if available -- this is the step to run on Colab):

  * input: the numeric columns of the v3 feature table, standardised with
    TUNE-match statistics, as sequences (one per half, ~140 k frames);
  * model: 2-layer bidirectional GRU, hidden 96, dropout 0.1, linear head to
    H / A / D logits;
  * loss: cross-entropy on the controlled-possession-rewritten truth, weighted
    by the spell weights, over chunks of CHUNK frames with random starts
    (chunked BPTT: the hidden state is detached between chunks);
  * training runs on the degraded TUNE tables; validation on the HOLDOUT
    matches; the ensemble (GRU + stacker probabilities, averaged) is scored with
    the same decoder search as the other stages.

Nothing here reads the challenge match.
"""

import argparse
import datetime
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import pipeline4.core as core
from pipeline4 import core
from pipeline4.train_stage1 import OUT

CHUNK = 600  # frames per training chunk
HIDDEN = 96
LAYERS = 2
DROPOUT = 0.1
EPOCHS = 4
LR = 1e-3
BATCHES_PER_EPOCH = 400
MAX_FEATURES = 120  # maximum number of input features

def build_sequences(matches, scaler=None, scenarios=None, smoke=False):
    """(tables, truths, runs) as arrays, standardised; returns also the scaler."""
    tables, truths, runs_list = [], [], []
    scenarios = scenarios or (core.JUDGE_SCENARIOS if smoke else core.SCENARIOS)
    for m in matches:
        for s in scenarios:
            table, truth, runs = core.item(m, s)
            tables.append(table)
            truths.append(truth)
            runs_list.append(runs)
    names = [c for c in tables[0].columns
             if c not in ("half", "frame", "truth")][:MAX_FEATURES]
    stacked = np.concatenate([t[names].values.astype(np.float32) for t in tables])
    if scaler is None:
        mu = np.nanmean(stacked, axis=0)
        sd = np.nanstd(stacked, axis=0) + 1e-6
        scaler = (mu, sd)
    mu, sd = scaler
    seqs = [np.nan_to_num((t[names].values.astype(np.float32) - mu) / sd) for t in tables]
    return seqs, truths, runs_list, names, scaler

def train_gru(seqs, truths, weights_by_frame=None, epochs=EPOCHS, smoke=False,
              device=None):
    import torch
    import torch.nn as nn
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_in = seqs[0].shape[1]
    model = nn.GRU(n_in, HIDDEN, num_layers=LAYERS, bidirectional=True,
                   dropout=DROPOUT, batch_first=True).to(device)
    head = nn.Linear(2 * HIDDEN, 3).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=LR)
    weights_by_frame = weights_by_frame or [np.ones(len(t)) for t in truths]

    # normalise weights to mean one
    allw = np.concatenate(weights_by_frame)
    scale = allw.mean()

    rng = np.random.RandomState(core.SEED)
    criterion = nn.CrossEntropyLoss(reduction="none")
    print("   GRU on {} ({} epochs x {} chunks)".format(device, epochs, BATCHES_PER_EPOCH),
          flush=True)
    for epoch in range(1, epochs + 1):
        total, seen = 0.0, 0
        for _ in range(BATCHES_PER_EPOCH):
            si = rng.randint(0, len(seqs))
            seq, truth, w = seqs[si], truths[si], weights_by_frame[si] / scale
            n = len(seq)
            start = rng.randint(0, max(1, n - CHUNK))
            end = min(n, start + CHUNK)
            xb = torch.tensor(seq[start:end][None], dtype=torch.float32, device=device)
            yb = torch.tensor(truth[start:end], dtype=torch.long, device=device)
            wb = torch.tensor(w[start:end], dtype=torch.float32, device=device)
            mask = yb != 3  # frames outside the halves carry no loss
            if mask.sum() == 0:
                continue
            opt.zero_grad()
            out, _ = model(xb)
            logits = head(out)[0]
            losses = criterion(logits, yb.clamp(0, 2))
            loss = (losses * wb * mask.float()).sum() / (wb[mask.float().bool()].sum() + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            total += float(loss)
            seen += 1
        print("   epoch {} loss {:.4f}".format(epoch, total / max(1, seen)), flush=True)
    return model, head, device

def gru_probs(model, head, seqs, device=None, chunk=2000):
    """Per-frame H/A/D probabilities for full sequences."""
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outs = []
    for seq in seqs:
        n = len(seq)
        p = np.empty((n, 3), dtype=np.float64)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            xb = torch.tensor(seq[a:b][None], dtype=torch.float32, device=device)
            with torch.no_grad():
                out, _ = model(xb)
                logits = head(out)[0]
            pr = torch.softmax(logits, dim=1).cpu().numpy()
            p[a:b] = pr
        outs.append(p)
    return outs

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    epochs = 1 if args.smoke else args.epochs
    batches = 30 if args.smoke else BATCHES_PER_EPOCH

    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)

    # train the GRU on degraded tune tables
    print("1. standardising the feature sequences ...", flush=True)
    train_seqs, train_truths, train_runs, names, scaler = build_sequences(core.TUNE, smoke=args.smoke)
    hold_seqs, hold_truths, hold_runs, _, _ = build_sequences(core.HOLDOUT, scaler=scaler,
                                                              smoke=args.smoke)
    weights = [core.spell_weights(t) for t in train_truths]

    print("2. training the GRU ...", flush=True)
    model, head, device = _train(train_seqs, train_truths, weights, epochs, batches)

    # step 3 score the GRU and stage one models on holdout matches
    print("3. scoring the ensemble on the HOLDOUT matches ...", flush=True)
    frame_t = core.fit_frame(list(core.TUNE))
    change_t = core.fit_change(list(core.TUNE))
    gru_h = gru_probs(model, head, hold_seqs, device=device)
    cases = []
    for i, m in enumerate(core.HOLDOUT):
        for j, s in enumerate(core.JUDGE_SCENARIOS):
            table, truth, runs = core.item(m, s)
            swap_table = core.item(m, s, True)
            p1, c1 = core.predict_probs(frame_t, change_t, table, swap_table)
            k = i * len(core.JUDGE_SCENARIOS) + j
            p = np.clip((p1 + gru_h[k]) / 2, 1e-4, 1.0)
            cases.append((p, c1, table["half"].values, truth, runs))
    decoder, holdout_spell = core.decoder_search(cases)
    holdout = core.summary(cases, decoder)
    print("   ensemble HOLDOUT spellF1 {:.4f} / macroF1 {:.4f}".format(
        holdout["spellF1"], holdout["macroF1"]), flush=True)

    # step 4 save the model
    gru_path = os.path.join(OUT, "lp4_gru.pt")
    torch.save({"state": model.state_dict(), "head": head.state_dict(),
                "n_in": train_seqs[0].shape[1], "hidden": HIDDEN, "layers": LAYERS,
                "scaler_mu": scaler[0].tolist(), "scaler_sd": scaler[1].tolist(),
                "feature_names": names}, gru_path)
    core.save_params(os.path.join(OUT, "params_gru.json"),
                     variant="Let's Play pipeline4 stage 3: GRU ensemble partner",
                     model_file=os.path.relpath(gru_path, core.ROOT),
                     decoder=decoder,
                     holdout=holdout,
                     chunk=CHUNK, hidden=HIDDEN, layers=LAYERS, epochs=epochs,
                     trained_on="open DFL / IDSSE matches {} (CC BY 4.0)".format(", ".join(core.TUNE)),
                     written=datetime.datetime.now().replace(microsecond=0).isoformat())
    print("   saved {} (+ params_gru.json)  [{:.0f} s total]".format(
        os.path.relpath(gru_path, core.ROOT), time.time() - t0), flush=True)

def _train(train_seqs, train_truths, weights, epochs, batches, device=None):
    """Thin wrapper so the GRU call above stays one line; returns (model, head, device)."""
    import torch
    import torch.nn as nn
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_in = train_seqs[0].shape[1]
    model = nn.GRU(n_in, HIDDEN, num_layers=LAYERS, bidirectional=True,
                   dropout=DROPOUT, batch_first=True).to(device)
    head = nn.Linear(2 * HIDDEN, 3).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss(reduction="none")
    allw = np.concatenate(weights)
    scale = max(allw.mean(), 1e-9)
    rng = np.random.RandomState(core.SEED)
    print("   GRU on {} ({} epochs x {} chunks, chunk {})".format(
        device, epochs, batches, CHUNK), flush=True)
    for epoch in range(1, epochs + 1):
        total, seen = 0.0, 0
        for _ in range(batches):
            si = rng.randint(0, len(train_seqs))
            seq, truth, w = train_seqs[si], train_truths[si], weights[si] / scale
            n = len(seq)
            start = rng.randint(0, max(1, n - CHUNK))
            end = min(n, start + CHUNK)
            xb = torch.tensor(seq[start:end][None], dtype=torch.float32, device=device)
            yb = torch.tensor(truth[start:end], dtype=torch.long, device=device)
            wb = torch.tensor(w[start:end], dtype=torch.float32, device=device)
            mask = (yb != 3).float()
            if mask.sum() == 0:
                continue
            opt.zero_grad()
            out, _ = model(xb)
            logits = head(out)[0]
            losses = criterion(logits, yb.clamp(0, 2))
            wmask = wb * mask
            loss = (losses * wmask).sum() / (wmask.sum() + 1e-6)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            total += float(loss)
            seen += 1
        print("   epoch {} loss {:.4f}".format(epoch, total / max(1, seen)), flush=True)
    return model, head, device


if __name__ == "__main__":
    main()
