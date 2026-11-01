"""CLI: python -m src.run --exp smoke --models CLATT Persistence --pairs 300015_300274 --seeds 0"""
import argparse
import logging

from src.baselines import BASELINES
from src.data import RESULTS
from src.features import pairs
from src.models import MODELS
from src.train import load_config, log, train_one

ALL_MODELS = MODELS + list(BASELINES)


def resolve(name, choices, short=lambda c: c):
    """Case-insensitive match on the full name or its short form."""
    hits = [c for c in choices if name.lower() in (c.lower(), short(c).lower())]
    if len(hits) != 1:
        raise SystemExit(f"unknown name {name!r}; choose from {choices}")
    return hits[0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--models", nargs="+", required=True)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--pairs", nargs="+")
    where.add_argument("--group", choices=["1", "2", "all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--config-override", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--final", action="store_true", help="also evaluate on the test split")
    p.add_argument("--force", action="store_true", help="rerun even if metrics.json exists")
    args = p.parse_args(argv)

    all_pairs = [f"{a}_{b}" for _, a, b in pairs()]
    if args.pairs:
        chosen = [resolve(x, all_pairs, lambda c: c.replace("sz.", "").replace("sh.", "")) for x in args.pairs]
    else:
        chosen = [f"{a}_{b}" for g, a, b in pairs() if args.group in ("all", str(g))]
    models = [resolve(m, ALL_MODELS) for m in args.models]

    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(RESULTS / "logs" / f"{args.exp}.log")])
    jobs = [(m, pr, s) for m in models for pr in chosen for s in args.seeds]
    for i, (m, pr, s) in enumerate(jobs, 1):
        log.info("[%d/%d] %s %s seed%d", i, len(jobs), m, pr, s)
        result = train_one(m, pr, s, load_config(m, args.config_override),
                           split_eval=("val", "test") if args.final else ("val",),
                           exp=args.exp, final=args.final, force=args.force)
        log.info("  val mse %.5f  skill %.4f  train %.1fs", result["val"]["mse"], result["val"]["skill"],
                 result["train_seconds"])


if __name__ == "__main__":
    main()
