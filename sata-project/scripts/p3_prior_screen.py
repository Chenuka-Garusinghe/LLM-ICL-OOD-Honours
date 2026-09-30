#!/usr/bin/env python3
"""P3 prior measurement (generator_spec.pdf, measuring priors).

Two stages, each scoring random profiles zero-shot and fitting the prior
surrogate (src/inference/priors.py):

  screen   every candidate name in configs/lexicons.yaml (800 profiles per
           domain) plus the abstract names f0-f9 (400 profiles); applies the
           pre-registered screening rules and proposes the final lexicon.
           Writes p3/screen_<model>.json and p3/profiles_screen_<model>.parquet.
  confirm  the final names only (400 profiles per domain), in the layout of a
           named task prompt; the abstract fit is reused from the screening.
           Writes p3/surrogate_<model>.json, which the grid's counter_prior
           strategy reads, and p3/profiles_<model>.parquet.

Profiles depend only on the stage and condition, never on the model, so every
model scores the same profiles.

  python scripts/p3_prior_screen.py --stage screen --models qwen2.5-7b-instruct
  python scripts/p3_prior_screen.py --stage confirm --models qwen2.5-7b-instruct,qwen2.5-7b-base
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config, resolve_path  # noqa: E402  (first: sets thread/MPS env vars)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.naming import load_lexicons, screening_rules  # noqa: E402
from src.inference.hf_runner import runner_from_config  # noqa: E402
from src.inference.priors import fit_surrogate, measure_margins, random_profiles, screen_lexicon  # noqa: E402
from src.inference.prompts import system_message  # noqa: E402


def profile_seed(base_seed: int, stage: str, condition: str) -> int:
    ss = np.random.SeedSequence([base_seed, zlib.crc32(stage.encode()), zlib.crc32(condition.encode())])
    return int(ss.generate_state(1)[0])


def measure(runner, stage: str, condition: str, lexicon, n: int, base_seed: int) -> pd.DataFrame:
    profiles = random_profiles(n, lexicon, np.random.default_rng(profile_seed(base_seed, stage, condition)))
    t0 = time.perf_counter()
    out = measure_margins(runner, system_message(None if condition == "abstract" else condition), profiles)
    out.insert(0, "condition", condition)
    out["stage"] = stage
    print(f"  {condition}: {n} profiles in {time.perf_counter() - t0:.0f}s, "
          f"margin mean {out['margin'].mean():+.2f} sd {out['margin'].std():.2f}")
    return out


def vocab_of(condition: str, lexicon) -> list[str]:
    return [f"f{j}" for j in range(10)] if condition == "abstract" else lexicon.names


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=("screen", "confirm"))
    parser.add_argument("--models", default="qwen2.5-7b-instruct")
    args = parser.parse_args()

    rules = screening_rules()
    out_dir = resolve_path(config.paths.results) / "p3"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {m.name: m for m in config.models}
    base_seed = config.selection.base_seed
    lexicons = load_lexicons("candidates" if args.stage == "screen" else "final")
    if args.stage == "confirm" and len(lexicons) < 2:
        parser.error("configs/lexicons.yaml has no `final` names yet; run the screen stage first")

    for name in args.models.split(","):
        runner = runner_from_config(models[name], config.inference)
        print(f"{name} on {runner.device_name} ({runner.dtype}), stage {args.stage}")
        frames, fits = [], {}
        if args.stage == "screen":
            conditions = [("abstract", None, rules["n_profiles_confirm"])] + \
                         [(d, lex, rules["n_profiles"]) for d, lex in lexicons.items()]
        else:
            screened = out_dir / f"profiles_screen_{name}.parquet"
            if screened.exists():
                prev = pd.read_parquet(screened)
                frames.append(prev[prev["condition"] == "abstract"])
                print(f"  abstract: reusing {len(frames[0])} screening profiles")
                conditions = []
            else:
                conditions = [("abstract", None, rules["n_profiles_confirm"])]
            conditions += [(d, lex, rules["n_profiles_confirm"]) for d, lex in lexicons.items()]
        for condition, lexicon, n in conditions:
            frames.append(measure(runner, args.stage, condition, lexicon, n, base_seed))
        runner.close()
        profiles = pd.concat(frames, ignore_index=True)
        for condition, g in profiles.groupby("condition", sort=False):
            lex = lexicons.get(condition)
            fits[condition] = fit_surrogate(g, vocab_of(condition, lex), alpha=rules["ridge_alpha"],
                                            n_bootstrap=rules["n_bootstrap"])
            print(f"  {condition}: out-of-fold R^2 {fits[condition]['r2_cv']:.3f}, "
                  f"label mass {fits[condition]['label_mass_median']:.3f}")

        record = {"model": name, "model_path": models[name].path, "stage": args.stage,
                  "created": time.strftime("%Y-%m-%d %H:%M"), "rules": rules, "fits": fits}
        if args.stage == "screen":
            record["screening"] = {d: screen_lexicon(fits[d], lex, rules) for d, lex in lexicons.items()}
            for d, sc in record["screening"].items():
                print(f"  {d}: pairs admitted {sum(p['admitted'] for p in sc['pairs'])}/{len(sc['pairs'])}, "
                      f"weak admitted {sum(w['admitted'] for w in sc['weak'])}/{len(sc['weak'])}, enough: {sc['enough']}")
            profiles.to_parquet(out_dir / f"profiles_screen_{name}.parquet", index=False)
            path = out_dir / f"screen_{name}.json"
        else:
            record["lexicon"] = {d: {"pairs": [list(p) for p in lex.pairs], "weak": list(lex.weak)}
                                 for d, lex in lexicons.items()}
            profiles.to_parquet(out_dir / f"profiles_{name}.parquet", index=False)
            path = out_dir / f"surrogate_{name}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=1)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
