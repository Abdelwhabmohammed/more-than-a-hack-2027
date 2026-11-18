"""Pipeline 4 — the shipped possession model and its training stages.

Ships: a LightGBM frame model + change detector + time-varying Viterbi, trained
by us on the seven open DFL/IDSSE matches (CC BY 4.0) degraded by our own
simulator matched to the challenge tracking's input statistics.

    cd more-than-a-hack-2027
    python -m pipeline4.predict --out Lets_Play.csv   # regenerate the submission
    python -m pipeline4.train_dfl # retrain from external data
    python -m pipeline4.train_stage1 # Stage A: v4w + OOF stacker inputs
    python -m pipeline4.train_stack # Stage B: stacker
    python -m pipeline4.train_gru # Stage C: GRU ensemble
"""
