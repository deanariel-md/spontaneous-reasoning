r"""Analysis R: the capability slope of the Cheng et al. (2025) contrast on
three scales, for reproduction Part B (the eight transported configurations), with Part A and the published values
beside it.

PowerShell (from the repository root):
    .\.venv\Scripts\python.exe scripts\run_cheng_scales.py

Inputs: data/processed/reproduction/transport_scored.parquet (Part B), numerical_scored.parquet (Part A), the
published accuracies in config/reproduction.yaml, and the scale definitions of config/instability.yaml. Benefit is
eight-shot minus zero-shot chain-of-thought accuracy and capability the zero-shot accuracy, as in
src/reproduction.capability_slope; the scales are those of analysis B (src/instability.py).

Outputs: results/cheng_scales/cheng_scales.json (with a "macros" object, prefix "cheng"), summary.md,
summary_clean.md and slopes.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for _p in (ROOT, SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src import dataset1 as d1, instability as ins, reproduction as rp  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "instability.yaml")
TRANSPORT = os.path.join(ROOT, "data", "processed", "reproduction", "transport_scored.parquet")
NUMERICAL = os.path.join(ROOT, "data", "processed", "reproduction", "numerical_scored.parquet")
OUT = os.path.join(ROOT, "results", "cheng_scales")
SC_M = {"points": "Points", "logit": "Logit", "error_reduction": "Err"}


def rel(p):
    return os.path.relpath(p, ROOT).replace("\\", "/")


def arm_arrays(scored, base, interv):
    """(configs, models, B, P, n_items): zero-shot and eight-shot accuracy (pp) per configuration and model."""
    acc = (scored.groupby(["config_id", "model_key", "arm"])["correct"].agg(["mean", "size"]).reset_index())
    n = acc["size"].unique()
    if len(n) != 1:
        raise SystemExit(f"unequal item counts across cells: {sorted(n)}")
    wide = acc.pivot_table(index=["config_id", "model_key"], columns="arm", values="mean")
    configs = sorted(scored["config_id"].unique())
    models = sorted(scored["model_key"].unique())
    idx = pd.MultiIndex.from_product([configs, models])
    wide = wide.reindex(idx)
    if wide[[base, interv]].isna().any().any():
        raise SystemExit("a (configuration, model) cell lacks an arm; nothing is imputed")
    B = 100 * wide[base].to_numpy().reshape(len(configs), len(models))
    P = 100 * wide[interv].to_numpy().reshape(len(configs), len(models))
    return configs, models, B, P, int(n[0])


def scale_rows(B, P, n_items, method, alpha):
    """Per scale: per-configuration slopes, their mean, the equal-budget slope and the cross-draw slope."""
    out = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(B, P, sc, n_items)
        per = ins.single_draw_slopes(s["x"], s["y"])
        r = {"unit": ins.SLOPE_UNITS[sc], "per_configuration": per.tolist(), "mean": float(np.mean(per)),
             "min": float(np.min(per)), "max": float(np.max(per)),
             "n_negative": int((per < 0).sum()), "n_positive": int((per > 0).sum()), "n_zero": int((per == 0).sum()),
             "n_undefined": int((~np.isfinite(per)).sum()),
             "equal_budget": d1.slope(s["y"].mean(0), s["x"].mean(0)), "replaced_cells": s["n_replaced"]}
        if B.shape[0] >= 3:
            m = ins.scale_moments(s["x"], s["y"], method, alpha)
            r["cross_draw"] = {k: m[k] for k in ("slope", "slope_set", "slope_set_sign", "R_benefit", "R_benefit_ci95",
                                                 "V_y", "W_y", "V_x", "A", "K", "M")}
        out[sc] = r
    return out


def single_rows(B, P, n_items):
    out = {}
    for sc in ins.SCALES:
        s = ins.scale_scores(B, P, sc, n_items)
        out[sc] = {"slope": float(ins.single_draw_slopes(s["x"], s["y"])[0]), "replaced_cells": s["n_replaced"]}
    return out


def _n(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}".replace("-", "−")


def render(R):
    T = R["part_b"]["all"]
    L = ["# Analysis R: the Cheng et al. (2025) capability slope on three scales", "",
         "## Setup", "",
         f"- **Data.** Reproduction Part B (`results/reproduction/summary.md`): {R['n_models']} models, the eight "
         f"transported configurations of the frozen GSM8K bank, {R['n_items']:,} test items, scorer "
         f"{R['scorer']}. Benefit is eight-shot minus zero-shot chain-of-thought accuracy; capability is the "
         "zero-shot accuracy.",
         "- **Scales** (as in analysis B, `config/instability.yaml`). Points: benefit against zero-shot accuracy. "
         "Logits: logit(eight-shot) − logit(zero-shot) against logit(zero-shot). Error reduction: 100 (P − B) / "
         "(100 − B) against zero-shot accuracy in pp. Each per-configuration score is transformed first.",
         "- **Slopes.** Per configuration, the OLS slope across models (as `src/reproduction.capability_slope`). "
         "Their mean is the plain average of the eight; the equal-budget slope regresses each model's mean benefit "
         "over the eight configurations on its mean capability (the earlier summary's 0.002 in points). The "
         "cross-draw slope treats the eight configurations as draws (Appendix A moments, jackknife t interval). "
         "The eight were drawn without replacement from the bank, so this interval is descriptive.",
         f"- **Undefined cells.** None: no per-configuration accuracy is 0% or 100% "
         f"({sum(T[sc]['replaced_cells'] for sc in ins.SCALES)} replacements).", "",
         "## Slopes per configuration", "",
         "| Configuration | Points (pp/pp) | Logits (logit/logit) | Error reduction (pp/pp) |", "|---|---|---|---|"]
    for k, c in enumerate(R["configurations"]):
        L.append(f"| {c} | " + " | ".join(_n(T[sc]["per_configuration"][k]) for sc in ins.SCALES) + " |")
    L.append("| Mean of the eight | " + " | ".join(_n(T[sc]["mean"]) for sc in ins.SCALES) + " |")
    L.append("| Range | " + " | ".join(f"{_n(T[sc]['min'])} to {_n(T[sc]['max'])}" for sc in ins.SCALES) + " |")
    L.append("| Negative / positive | " + " | ".join(f"{T[sc]['n_negative']} / {T[sc]['n_positive']}"
                                                     for sc in ins.SCALES) + " |")
    L.append("| Equal-budget slope | " + " | ".join(_n(T[sc]["equal_budget"]) for sc in ins.SCALES) + " |")
    L.append("| Cross-draw slope [95%] | " + " | ".join(
        f"{_n(T[sc]['cross_draw']['slope'])} [{_n(T[sc]['cross_draw']['slope_set']['a'])}, "
        f"{_n(T[sc]['cross_draw']['slope_set']['b'])}]" for sc in ins.SCALES) + " |")
    L.append("| R_benefit [95%] | " + " | ".join(
        f"{_n(T[sc]['cross_draw']['R_benefit'], 2)} [{_n(T[sc]['cross_draw']['R_benefit_ci95'][0], 2)}, "
        f"{_n(T[sc]['cross_draw']['R_benefit_ci95'][1], 2)}]" for sc in ins.SCALES) + " |")
    A, Pb = R["part_a"], R["published"]
    L.append("| Part A, published frame (ours) | " + " | ".join(_n(A[sc]["slope"]) for sc in ins.SCALES) + " |")
    L.append("| Published accuracies (Cheng et al.) | " + " | ".join(_n(Pb[sc]["slope"]) for sc in ins.SCALES) + " |")
    bf = R["part_b"]["bf16_only"]
    L.append(f"| BF16 models only ({R['n_models_bf16']}), mean of the eight | "
             + " | ".join(_n(bf[sc]["mean"]) for sc in ins.SCALES) + " |")
    if "part_a_bf16" in R:
        L.append(f"| BF16 models only, Part A, published frame (ours) | "
                 + " | ".join(_n(R["part_a_bf16"][sc]["slope"]) for sc in ins.SCALES) + " |")
        L.append(f"| BF16 models only, published accuracies | "
                 + " | ".join(_n(R["published_bf16"][sc]["slope"]) for sc in ins.SCALES) + " |")
    chk = R["check_points"]
    pts, lg, er = (T[sc] for sc in ins.SCALES)
    L += ["", f"**Check.** In points the per-configuration slopes equal those of `results/reproduction/summary.json` "
          f"(largest difference {chk['max_abs_difference']:.1e}); they run from {_n(pts['min'])} to {_n(pts['max'])}, "
          f"their mean is {_n(pts['mean'])}, and the equal-budget slope is {_n(pts['equal_budget'])} (reported earlier "
          f"as {_n(chk['equal_budget_earlier'])}).", "",
          f"In points the slope is negative in {pts['n_negative']} of the eight configurations and positive in "
          f"{pts['n_positive']}; in logits {lg['n_negative']} and {lg['n_positive']}; as error reduction "
          f"{er['n_negative']} and {er['n_positive']}. A negative slope means the benefit falls as zero-shot "
          f"accuracy rises. The mean slope is {_n(pts['mean'])} pp/pp in points, {_n(lg['mean'])} in logits and "
          f"{_n(er['mean'])} pp/pp as error reduction. "
          + ("Part A and the published accuracies give negative slopes on all three scales. "
             if all(A[sc]["slope"] < 0 and Pb[sc]["slope"] < 0 for sc in ins.SCALES) else "")
          + f"Without the two FP8 models the mean slope in points is {_n(bf['points']['mean'])}.", ""]
    gp = R["by_group"]
    L += [f"**Why the relative scale differs.** Averaged over the eight configurations, Cheng et al.'s strong models "
          f"({gp['strong']['n']}) have zero-shot accuracy {gp['strong']['zero_shot']:.1f}% and benefit "
          f"{_n(gp['strong']['benefit'], 2)} pp, which is {_n(gp['strong']['error_reduction'], 1)} pp of error "
          f"reduction; the weak models ({gp['weak']['n']}) have {gp['weak']['zero_shot']:.1f}%, "
          f"{_n(gp['weak']['benefit'], 2)} pp and {_n(gp['weak']['error_reduction'], 1)} pp. The strong models have "
          f"about {100 - gp['strong']['zero_shot']:.0f} pp of error left, so a change of "
          f"{abs(gp['strong']['benefit']):.1f} pp is a larger share of it than a change of "
          f"{abs(gp['weak']['benefit']):.1f} pp is of the weak models' {100 - gp['weak']['zero_shot']:.0f} pp.", ""]
    return "\n".join(L) + "\n"


def macros(R):
    t = ins.tex_num
    mac = {}

    def put(v, *parts):
        name = "cheng" + "".join(parts)
        if not name.isalpha():
            raise ValueError(f"macro name must be letters only: {name}")
        mac[name] = v

    T = R["part_b"]["all"]
    for sc in ins.SCALES:
        x, m = T[sc], SC_M[sc]
        put(t(x["mean"], 3), "Slope", m, "Mean")
        put(t(x["min"], 3), "Slope", m, "Min")
        put(t(x["max"], 3), "Slope", m, "Max")
        put(str(x["n_negative"]), "Neg", m)
        put(str(x["n_positive"]), "Pos", m)
        put(t(x["equal_budget"], 3), "EqualBudget", m)
        cd = x["cross_draw"]
        put(t(cd["slope"], 3), "Cross", m)
        put(t(cd["slope_set"]["a"], 3), "Cross", m, "Lo")
        put(t(cd["slope_set"]["b"], 3), "Cross", m, "Hi")
        put(t(cd["R_benefit"], 2), "R", m)
        put(t(R["part_a"][sc]["slope"], 3), "PartA", m)
        put(t(R["published"][sc]["slope"], 3), "Published", m)
        put(t(R["part_b"]["bf16_only"][sc]["mean"], 3), "BfSixteen", m, "Mean")
    put(str(R["n_models"]), "Models")
    put(str(len(R["configurations"])), "Configs")
    return mac


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(CONFIG, encoding="utf-8"))
    if cfg.get("status") != "fixed":
        raise SystemExit("config/instability.yaml is not fixed")
    ch = cfg["cheng"]
    base, interv = ch["baseline_arm"], ch["intervention_arm"]
    alpha = float(cfg["alpha"])
    rcfg = rp.load_config()
    tr = pd.read_parquet(TRANSPORT)
    if set(tr["scorer"]) != {rp.SCORER_VERSION}:
        raise SystemExit(f"scored with {sorted(set(tr['scorer']))}, expected {rp.SCORER_VERSION}")
    configs, models, B, P, n_items = arm_arrays(tr[tr["part"] == "transport"], base, interv)
    bf16 = [k for k, mk in enumerate(models) if rcfg["models"][mk].get("precision", "bf16") == "bf16"]
    R = {"analysis": "R, three scales",
         "config": {"path": rel(CONFIG), "sha256": hashlib.sha256(open(CONFIG, "rb").read()).hexdigest()},
         "inputs": {"transport": rel(TRANSPORT), "numerical": rel(NUMERICAL), "published": "config/reproduction.yaml"},
         "scorer": rp.SCORER_VERSION, "configurations": configs, "models": models, "n_models": len(models),
         "n_models_bf16": len(bf16), "n_items": n_items, "baseline_arm": base, "intervention_arm": interv,
         "accuracy": {"zero_shot": B.tolist(), "eight_shot": P.tolist()},
         "part_b": {"all": scale_rows(B, P, n_items, "jack_t", alpha),
                    "bf16_only": scale_rows(B[:, bf16], P[:, bf16], n_items, "jack_t", alpha)}}
    # Cheng et al.'s strong and weak groups (config/reproduction.yaml), Part B averaged over the configurations
    er = 100 * (P - B) / (100 - B)
    R["by_group"] = {}
    for grp in ("strong", "weak"):
        k = [i for i, mk in enumerate(models) if rcfg["models"][mk]["group"] == grp]
        R["by_group"][grp] = {"n": len(k), "models": [models[i] for i in k], "zero_shot": float(B[:, k].mean()),
                              "benefit": float((P - B)[:, k].mean()), "error_reduction": float(er[:, k].mean())}
    # Part A, our run of the published frame
    num = pd.read_parquet(NUMERICAL)
    _, ma, Ba, Pa, na = arm_arrays(num[num["part"] == "numerical"], base, interv)
    R["part_a"] = {**single_rows(Ba, Pa, na), "models": ma}
    # the same on the BF16 models, the main panel of every Cheng et al. comparison
    bfa = [k for k, mk in enumerate(ma) if rcfg["models"][mk].get("precision", "bf16") == "bf16"]
    R["part_a_bf16"] = {**single_rows(Ba[:, bfa], Pa[:, bfa], na), "models": [ma[k] for k in bfa]}
    # The published accuracies (Cheng et al.), same models
    pub = rp.published_table(rcfg).set_index("model_key").loc[models]
    Bp, Pp = pub["published_zero_shot"].to_numpy()[None, :], pub["published_eight_shot"].to_numpy()[None, :]
    R["published"] = single_rows(Bp, Pp, n_items)
    R["published_bf16"] = single_rows(Bp[:, bf16], Pp[:, bf16], n_items)
    # check against results/reproduction/summary.json (points)
    earlier = json.load(open(os.path.join(ROOT, "results", "reproduction", "summary.json"), encoding="utf-8"))
    ep = [earlier["transport"]["per_configuration"][c]["capability_slope"]["slope"] for c in configs]
    R["check_points"] = {"earlier_per_configuration": ep,
                         "max_abs_difference": float(np.max(np.abs(np.array(ep) -
                                                                   np.array(R["part_b"]["all"]["points"]["per_configuration"])))),
                         "equal_budget_earlier": earlier["transport"]["equal_budget_slope"]["slope"],
                         "part_a_earlier": earlier["numerical"]["capability_slope"]["slope"],
                         "published_earlier": earlier["numerical"]["published_capability_slope"]["slope"]}
    R["macros"] = macros(R)
    R["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "cheng_scales.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else
                  o.item() if isinstance(o, np.generic) else str(o))
    rows = []
    for sc in ins.SCALES:
        for c, v in zip(configs, R["part_b"]["all"][sc]["per_configuration"]):
            rows.append({"scale": sc, "set": "Part B", "configuration": c, "slope": v})
        rows += [{"scale": sc, "set": "Part B", "configuration": "mean of the eight", "slope": R["part_b"]["all"][sc]["mean"]},
                 {"scale": sc, "set": "Part B", "configuration": "equal budget", "slope": R["part_b"]["all"][sc]["equal_budget"]},
                 {"scale": sc, "set": "Part B", "configuration": "cross-draw",
                  "slope": R["part_b"]["all"][sc]["cross_draw"]["slope"]},
                 {"scale": sc, "set": "Part B, BF16 only", "configuration": "mean of the eight",
                  "slope": R["part_b"]["bf16_only"][sc]["mean"]},
                 {"scale": sc, "set": "Part A", "configuration": "cheng", "slope": R["part_a"][sc]["slope"]},
                 {"scale": sc, "set": "Part A, BF16 only", "configuration": "cheng", "slope": R["part_a_bf16"][sc]["slope"]},
                 {"scale": sc, "set": "published, BF16 only", "configuration": "Cheng et al.",
                  "slope": R["published_bf16"][sc]["slope"]},
                 {"scale": sc, "set": "published", "configuration": "Cheng et al.", "slope": R["published"][sc]["slope"]}]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "slopes.csv"), index=False)
    md = render(R)
    for name in ("summary.md", "summary_clean.md"):
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
