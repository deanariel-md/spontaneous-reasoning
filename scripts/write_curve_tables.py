r"""Curve and regret tables written from existing outputs (no estimate is
recomputed):

    results/decision/tables/figure_regret.csv          setting x rule x regret (decision.json, independent
                                                       evaluation configurations)
    results/protocol_ceiling/tables/figure_curves.csv  the protocol-ceiling curves (protocol_ceiling.json)

Each folder's summary.md and summary_clean.md gets one line, "Figure table: ...", saying what the file holds; a later run
replaces that line, so re-running this script after the analysis rewrites its summary puts the line back.

From the repository root:
    python scripts/write_curve_tables.py
"""
import argparse
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MARK = "Figure table: "

# decision.json setting keys: task and the contrast (intervention - baseline, scoring)
SETTING = {
    "gsm8k_req": ("GSM8K-29", "chain of thought - requested direct, strict"),
    "bbh_req": ("BIG-Bench Hard-28", "chain of thought - requested direct, strict"),
    "math_req": ("MATH-500", "chain of thought - requested direct, strict"),
    "gsm8k_fd": ("GSM8K-29", "chain of thought - forced direct, strict"),
    "bbh_fd": ("BIG-Bench Hard-28", "chain of thought - forced direct, strict"),
    "math_fd": ("MATH-500", "chain of thought - forced direct, strict"),
    "gsm8k_con": ("GSM8K-29", "chain of thought - requested direct, content scores"),
    "bbh_con": ("BIG-Bench Hard-28", "chain of thought - requested direct, content scores"),
    "math_con": ("MATH-500", "chain of thought - requested direct, content scores"),
    "exemplar": ("GSM8K-29", "eight-shot - zero-shot chain of thought (exemplars)"),
    "dove_mmlu": ("DOVE MMLU block, 5 models", "five-shot - zero-shot"),
    "fewshot_gsm8k": ("GSM8K-29", "five-shot - zero-shot forced direct"),
    "fewshot_bbh": ("BIG-Bench Hard-28", "five-shot - zero-shot forced direct"),
    "fewshot_math": ("MATH-500", "five-shot - zero-shot forced direct"),
    # HotpotQA, added to the decision analysis; its rows follow the fourteen above
    "hotpot_req": ("HotpotQA", "chain of thought - requested direct, strict"),
    "hotpot_fd": ("HotpotQA", "chain of thought - forced direct, strict"),
    "fewshot_hotpot": ("HotpotQA", "five-shot - zero-shot forced direct"),
}
RULE = {"capability_1": "capability slope, 1 draw", "own_1": "own gain, 1 draw", "own_2": "own gain, 2 draws",
        "shrunk_2": "shrunken own gain, 2 draws", "own_4": "own gain, 4 draws", "shrunk_4": "shrunken own gain, 4 draws",
        "own_all": "own gain, all discovery draws", "capability_all": "capability slope, all discovery draws",
        "random": "random third"}
DECISION_COLUMNS = ["setting", "task", "contrast", "rule", "rule_label", "regret_plugin", "regret_crossfit",
                    "regret_plugin_p10", "regret_plugin_p90", "share_best_third", "n_pilot_sets", "best_third_gain",
                    "panel_mean_gain", "n_configurations", "n_chosen", "M"]
CEILING_COLUMNS = ["task", "curve", "comparison", "protocols", "pairs", "kind", "K", "C", "limit_plugin",
                   "limit_plugin_lo", "limit_plugin_hi", "limit_posterior"]


def decision_rows(R):
    """One row per setting and rule: plug-in and cross-fitted regret (pp of gain per chosen model) on the independent
    evaluation configurations, as in tables/regret.csv, with the setting's best-third gain and panel mean gain."""
    rows = []
    for key, s in R["settings"].items():
        r = s["result"]
        ev = r["evaluation"]["independent"]
        task, contrast = SETTING.get(key, (key, s.get("title", "")))
        for rule, x in ev["rules"].items():
            rows.append({"setting": key, "task": task, "contrast": contrast, "rule": rule,
                         "rule_label": RULE.get(rule, rule), "regret_plugin": x["regret_plugin"],
                         "regret_crossfit": x["regret_crossfit"], "regret_plugin_p10": x.get("regret_plugin_p10"),
                         "regret_plugin_p90": x.get("regret_plugin_p90"), "share_best_third": x.get("share_best_third"),
                         "n_pilot_sets": x.get("n_sets"), "best_third_gain": ev.get("best_third_value"),
                         "panel_mean_gain": ev.get("panel_mean_gain"), "n_configurations": ev.get("n_configurations"),
                         "n_chosen": r.get("n_chosen"), "M": r.get("M")})
    return rows


def ceiling_rows(R):
    """The agreement curves per task: the headline pair (requested and forced arm, strict, points) within each
    protocol and between them, each axis's between-protocol curves, and the within-protocol curve averaged over all
    protocols; each with its limit (1 within a protocol, by construction)."""
    rows = []
    for task, T in R["tasks"].items():
        h = T["headline"]
        ci = (h.get("plugin_bootstrap") or {}).get("ci95") or [None, None]
        head_lim = (h.get("plugin"), ci[0], ci[1], h.get("posterior"))
        prot = " vs ".join(h.get("protocols", []))
        for key, comp, kind, lim in (("within_requested", "within", "empirical", (1, None, None, 1)),
                                     ("within_forced", "within", "empirical", (1, None, None, 1)),
                                     ("within_requested_model", "within", "Gaussian pair model", (1, None, None, 1)),
                                     ("between_same_draws", "between", "same draws", head_lim),
                                     ("between_disjoint", "between", "disjoint draws", head_lim),
                                     ("between_model", "between", "Gaussian pair model", head_lim)):
            for K, C in sorted(h.get(key, {}).items(), key=lambda kv: int(kv[0])):
                rows.append(dict(zip(CEILING_COLUMNS, (task, "headline_" + key, comp, prot, 1, kind, int(K), C, *lim))))
        for axis, a in T["axes"].items():
            if axis == "within":
                for key, kind in (("empirical", "empirical"), ("model", "Gaussian pair model")):
                    for K, C in sorted(a.get(key, {}).items(), key=lambda kv: int(kv[0])):
                        rows.append(dict(zip(CEILING_COLUMNS, (task, "within_all_" + key, "within",
                                                               f"{a['protocols']} protocols", a["protocols"], kind,
                                                               int(K), C, 1, None, None, 1))))
                continue
            lim = (a.get("limit_plugin"), None, None, a.get("limit_posterior"))
            for key, kind in (("same_draws", "same draws"), ("disjoint", "disjoint draws"),
                              ("model", "Gaussian pair model")):
                for K, C in sorted(a.get(key, {}).items(), key=lambda kv: int(kv[0])):
                    rows.append(dict(zip(CEILING_COLUMNS, (task, f"axis_{axis}_{key}", "between", f"axis: {axis}",
                                                           a["pairs"], kind, int(K), C, *lim))))
    return rows


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in columns})


def readme_line(summary_path, line):
    """Put `line` (which starts with MARK) at the end of a summary, replacing an earlier line with the same file name."""
    text = ""
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8", newline="") as fh:
            text = fh.read()
    eol = "\r\n" if "\r\n" in text else "\n"           # keep the file's line endings
    target = line.split("`")[1]
    kept = [x for x in text.replace("\r\n", "\n").rstrip("\n").split("\n")
            if not (x.startswith(MARK) and f"`{target}`" in x)]
    while kept and not kept[-1].strip():
        kept.pop()
    with open(summary_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(eol.join(kept + ["", line]) + eol)


DECISION_LINE = (MARK + "`tables/figure_regret.csv` holds one row per setting and rule (every setting of `decision.json`, "
                 "HotpotQA's three last; 9 rules): the "
                 "plug-in and cross-fitted regret in pp of gain per chosen model on the independent evaluation "
                 "configurations, the 10th and 90th percentiles over pilot sets, and the setting's best-third gain and "
                 "panel mean gain; read from `decision.json` by `scripts/write_curve_tables.py`.")
CEILING_LINE = (MARK + "`tables/figure_curves.csv` holds the agreement curves against K per task: the headline pair "
                "(requested and forced arm) within each protocol and between them, each axis's between-protocol "
                "curves (same draws, disjoint draws, Gaussian pair model) and the within-protocol curve over all 18 "
                "protocols, each row with its limit (plug-in with its bootstrap interval where one exists, posterior; "
                "1 within a protocol); every task of `protocol_ceiling.json`, HotpotQA's rows last; read from "
                "`protocol_ceiling.json` by `scripts/write_curve_tables.py`, which `scripts/run_protocol_ceiling.py` "
                "calls at its end.")


TABLES = {"decision": (decision_rows, DECISION_COLUMNS, "figure_regret.csv", DECISION_LINE),
          "protocol_ceiling": (ceiling_rows, CEILING_COLUMNS, "figure_curves.csv", CEILING_LINE)}


def write_one(name, root=ROOT):
    """Write one folder's figure table from its JSON and put its line into the folder's two summaries; returns
    (csv name, rows), or None when the JSON is absent."""
    rows_fn, columns, csv_name, line = TABLES[name]
    d = os.path.join(root, "results", name)
    src = os.path.join(d, f"{name}.json")
    if not os.path.exists(src):
        print(f"{src} absent; {csv_name} not written")
        return None
    rows = rows_fn(json.load(open(src, encoding="utf-8")))
    write_csv(os.path.join(d, "tables", csv_name), rows, columns)
    for s in ("summary.md", "summary_clean.md"):
        readme_line(os.path.join(d, s), line)
    print(f"wrote {os.path.join(d, 'tables', csv_name)} ({len(rows)} rows)")
    return csv_name, len(rows)


def main(argv=None, root=ROOT):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=root)
    a = ap.parse_args(argv)
    done = []
    for name in TABLES:
        r = write_one(name, a.root)
        if r:
            done.append(r)
    return done


if __name__ == "__main__":
    main()
