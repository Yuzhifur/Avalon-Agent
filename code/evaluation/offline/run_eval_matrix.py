"""Phase-3 offline evaluation matrix (Gate 0).

Runs the full belief-quality + counterfactual comparison in one process,
loading each eval set once and reusing each backend across sets. Writes one
JSON per cell plus a consolidated summary (JSON + markdown) under
offline/eval_matrix/.

  python -m offline.run_eval_matrix [--proavalon-test-cap N] [--cells ...]

Cells (vibes off / pure model unless noted):
  detection : factor_v3 + seq_v1 (ProAvalon-trained) on self-play-90,
              avalonlogs-test, proavalon-test; factor_v2 reused from its
              committed self-play-90 report (BP is intractable on the human
              sets — ~64 s/game).
  attribution : Track A full vs no_rejected vs no_proposer (separately trained
              on ProAvalon) on self-play-90 + proavalon-test — isolates the
              rejected-history and proposer signals.
  penalty   : promoted model on self-play-90 with GoodPolicy penalties
              default / off / no_rejected (the behavior-risk double-count).
  vibes     : factor_v2 recorded-CSV vs off on self-play-90 (what LLM vibes
              added to the deployed model).

Run from code/evaluation/.
"""

import argparse
import json
import os
import time

from . import add_agent_paths
from .model_adapters import build_backend
from .replay_eval import evaluate, load_games

add_agent_paths()

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CORPORA = os.path.join(REPO, "data", "corpora")
SPLITS = os.path.join(HERE, "splits")
OUT_DIR = os.path.join(HERE, "eval_matrix")

SELFPLAY_RUNROOT = os.path.join(
    REPO, "code", "evaluation", "policy_runs_overnight", "20260608T180435Z")


def load_set(name, cap=None):
    """Return the games list for a named eval set (filtered to its test split
    for the human corpora; self-play-90 is eval-only in full)."""
    if name == "selfplay90":
        games = load_games(os.path.join(CORPORA, "selfplay90.jsonl"))
    elif name == "selfplay90_runs":          # for --vibes recorded
        games = load_games(SELFPLAY_RUNROOT)
    elif name in ("avalonlogs", "proavalon"):
        games = load_games(os.path.join(CORPORA, f"{name}.jsonl"))
        with open(os.path.join(SPLITS, f"{name}.json"), encoding="utf-8") as fh:
            wanted = set(json.load(fh)["test"])
        games = [g for g in games if g["record"].game_id in wanted]
    else:
        raise ValueError(name)
    if cap:
        games = games[:cap]
    return games


def cell(report_name, games, backend, penalties="default", vibes="off"):
    t0 = time.time()
    rep = evaluate(games, backend, penalties=penalties, vibes=vibes)
    rep["config"] = {
        "cell": report_name, "n_games": len(games),
        "backend": getattr(backend, "name", "recorded") if backend else "recorded",
        "penalties": penalties, "vibes": vibes,
        "seconds": round(time.time() - t0, 1),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{report_name}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, default=str)
    b = rep["beliefs"]
    print(f"[{report_name}] AUC={b['auc_pooled']} pair_top1={rep['pairs']['pair_top1']} "
          f"ECE={b['ece']} flip={rep['counterfactual']['evil_team_reject_flip_rate']} "
          f"({rep['config']['seconds']}s)", flush=True)
    return rep


def row(rep):
    b, p, c = rep["beliefs"], rep["pairs"], rep["counterfactual"]
    def f(x, n=3):
        return "-" if x is None else f"{x:.{n}f}"
    return (f"{rep['config']['cell']} | {rep['config']['n_games']} | "
            f"{f(b['auc_pooled'])} | {f(b['mean_p_evil_on_true_evil'])} | "
            f"{f(b['ece'])} | {f(b['nll'])} | {f(p['pair_top1'])} | {f(p['pair_top3'])} | "
            f"{f(c['evil_team_reject_flip_rate'])} | "
            f"{f(c['clean_team_new_false_reject_rate'])}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", nargs="*",
                    default=["detection", "attribution", "penalty", "vibes"])
    ap.add_argument("--proavalon-test-cap", type=int, default=None,
                    help="cap proavalon-test games for speed (AUC is stable "
                         "well below the full 14.6k)")
    ap.add_argument("--trackA-dir", default="v4_trackA_proavalon")
    ap.add_argument("--seq-dir", default="seq_v1_proavalon")
    args = ap.parse_args(argv)

    reports = []

    if "detection" in args.cells:
        a = build_backend("factor_v3", model_dir=args.trackA_dir)
        b = build_backend("seq_v1", model_dir=args.seq_dir)
        sp = load_set("selfplay90")
        av = load_set("avalonlogs")
        pa = load_set("proavalon", cap=args.proavalon_test_cap)
        reports.append(cell("detection_trackA_selfplay90", sp, a))
        reports.append(cell("detection_seq_selfplay90", sp, b))
        reports.append(cell("detection_trackA_avalonlogs", av, a))
        reports.append(cell("detection_seq_avalonlogs", av, b))
        reports.append(cell("detection_trackA_proavalon", pa, a))
        reports.append(cell("detection_seq_proavalon", pa, b))

    if "attribution" in args.cells:
        sp = load_set("selfplay90")
        pa = load_set("proavalon", cap=args.proavalon_test_cap)
        for tag, mdir, abl in (("full", args.trackA_dir, "none"),
                               ("no_rejected", f"{args.trackA_dir}_no_rejected", "no_rejected"),
                               ("no_proposer", f"{args.trackA_dir}_no_proposer", "no_proposer")):
            try:
                # score each ablated model in-distribution (matched train/eval)
                bk = build_backend("factor_v3", model_dir=mdir, ablation=abl)
            except Exception as e:
                print(f"skip attribution {tag}: {e}")
                continue
            reports.append(cell(f"attribution_{tag}_selfplay90", sp, bk))
            reports.append(cell(f"attribution_{tag}_proavalon", pa, bk))

    if "penalty" in args.cells:
        a = build_backend("factor_v3", model_dir=args.trackA_dir)
        sp = load_set("selfplay90")
        for pen in ("default", "off", "no_rejected"):
            reports.append(cell(f"penalty_{pen}_selfplay90", sp, a, penalties=pen))

    if "vibes" in args.cells:
        sp_runs = load_set("selfplay90_runs")
        reports.append(cell("vibes_recorded_v2_selfplay90", sp_runs, None,
                            vibes="recorded"))

    header = ("cell | n | AUC | meanP(evil|evil) | ECE | NLL | "
              "pair_top1 | pair_top3 | evil_flip | clean_false_reject")
    lines = [header, "|".join(["---"] * len(header.split("|")))]
    lines += [row(r) for r in reports]
    summary_md = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(summary_md + "\n")
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump([r for r in reports], fh, indent=1, default=str)
    print("\n" + summary_md)
    print(f"\nwrote {len(reports)} cells to {OUT_DIR}")


if __name__ == "__main__":
    main()
