"""Compare two evaluation runs question by question."""

import json
import sys

from tenk_rag.config import PROJECT_ROOT


def load(name):
    return json.loads((PROJECT_ROOT / "experiments" / f"{name}.json").read_text())


def main():
    before, after = load(sys.argv[1]), load(sys.argv[2])
    rank = lambda r: 999 if r is None else r   # treat a miss as the worst rank
    better = worse = 0

    for qid, r1 in before["ranks"].items():
        r2 = after["ranks"].get(qid)
        # Marks improved questions with + and worse ones with -
        if r1 == r2:
            continue
        if rank(r2) < rank(r1):
            better += 1
            mark = "+"
        else:
            worse += 1
            mark = "-"
        print(f"{mark} {qid:<10} {str(r1):>5} -> {str(r2):>5}")

    print(f"\n{better} better, {worse} worse")


if __name__=="__main__":
    main()