"""Download + parse the seven IDSSE/DFL matches into pipeline4's cache.

    cd more-than-a-hack-2027
    python -m pipeline4.prepare_dfl       # ~2.7 GB download, then parses each match

Run once before the training stages (or let train_stage1 do it implicitly --
this script just front-loads the wait).
"""

import time
from pipeline4 import core
import pipeline4.dfl_data as dfl_data

def main():
    t0 = time.time()
    for m in core.ALL_MATCHES:
        t1 = time.time()
        dfl_data.fetch(m)
        dfl_data.load(m)
        print("{} ready in {:.0f} s".format(m, time.time() - t1), flush=True)
    print("all 7 matches ready  [{:.0f} s total]".format(time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
