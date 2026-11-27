r"""Compare two paper/numbers.tex files name by name (existing macros keep their values; additions only).

From the repository root:
    python scripts/compare_numbers_tex.py BEFORE AFTER [--tsv added.tsv]

Prints the counts before and after and the numbers of changed, removed and added names, and lists every changed or
removed name. Exit status 1 when any name changed or was removed. --tsv writes every added name with its value.
"""
import argparse
import re
import sys

LINE = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")


def load(path):
    """{name: value} of the \\newcommand lines of a numbers.tex, and its other lines."""
    names, other = {}, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE.match(line)
            if m:
                if m.group(1) in names:
                    raise ValueError(f"{path}: {m.group(1)} defined twice")
                names[m.group(1)] = m.group(2)
            else:
                other.append(line)
    return names, other


def compare(a, b):
    changed = [k for k in a if k in b and a[k] != b[k]]
    removed = [k for k in a if k not in b]
    added = [k for k in b if k not in a]
    return changed, removed, added


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--tsv")
    a = ap.parse_args(argv)
    (na, oa), (nb, ob) = load(a.before), load(a.after)
    changed, removed, added = compare(na, nb)
    print(f"{len(na)} macros before, {len(nb)} after: {len(changed)} changed, {len(removed)} removed, "
          f"{len(added)} added; other lines identical: {oa == ob}")
    for k in changed:
        print(f"CHANGED {k}: {na[k]} -> {nb[k]}")
    for k in removed:
        print(f"REMOVED {k}")
    if a.tsv:
        with open(a.tsv, "w", encoding="utf-8", newline="\n") as f:
            f.write("name\tvalue\n" + "".join(f"{k}\t{nb[k]}\n" for k in added))
    return 1 if changed or removed or oa != ob else 0


if __name__ == "__main__":
    sys.exit(main())
