"""Update every reference to the GitHub and/or Hugging Face repo after renaming them.

    python scripts/rename_repos.py --gh nishsm/golf-club-detection-swingtrace \
                                   --hf nishsm/golf-club-detection-swingtrace

The Python package itself stays `swingtrace`: the import name and the CLI do not change.
Pass --dry-run to preview.
"""
import argparse
import re
from pathlib import Path

OLD = "nishsm/swingtrace"
FILES = ["pyproject.toml", "README.md", "hf/model/README.md", "hf/space/README.md",
         "hf/space/app.py", "notebooks/swingtrace_quickstart.ipynb", "assets/README.md",
         "src/swingtrace/weights.py"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gh", help="new GitHub repo, e.g. nishsm/golf-club-detection-swingtrace")
    p.add_argument("--hf", help="new Hugging Face model repo")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not (a.gh or a.hf):
        p.error("pass --gh and/or --hf")

    root = Path(__file__).resolve().parent.parent
    old_slug = OLD.split("/")[1]

    for name in FILES:
        f = root / name
        if not f.exists():
            continue
        s = orig = f.read_text()

        if a.gh:
            # github.com/<old> and colab.research.google.com/github/<old>
            s = re.sub(rf"(github\.com/){re.escape(OLD)}(?![-\w])", rf"\g<1>{a.gh}", s)
            s = re.sub(rf"(google\.com/github/){re.escape(OLD)}(?![-\w])", rf"\g<1>{a.gh}", s)
            # shields.io badge: "/" -> %2F and literal "-" must be doubled
            s = s.replace(OLD.replace("/", "%2F"),
                          a.gh.replace("-", "--").replace("/", "%2F"))
            # `cd <old-slug>` after a clone of the renamed repo
            s = s.replace(f"&& cd {old_slug}", f"&& cd {a.gh.split('/')[1]}")

        if a.hf:
            s = re.sub(rf"(huggingface\.co/){re.escape(OLD)}(?![-\w])", rf"\g<1>{a.hf}", s)
            s = s.replace(f'"{OLD}"', f'"{a.hf}"')          # weights.py, hf_hub_download
            s = re.sub(rf"(^\s*-\s*){re.escape(OLD)}$", rf"\g<1>{a.hf}", s, flags=re.M)  # Space yaml
            s = s.replace(f"[{OLD}]", f"[{a.hf}]")          # markdown link text

        if s != orig:
            print(f"{'would update' if a.dry_run else 'updated'} {name}")
            if not a.dry_run:
                f.write_text(s)

    # Report anything still pointing at the old name.
    print("\nRemaining references to the old name:")
    found = False
    for name in FILES:
        f = root / name
        if not f.exists():
            continue
        for i, line in enumerate(f.read_text().splitlines(), 1):
            for host, new in (("github.com/", a.gh), ("google.com/github/", a.gh),
                              ("huggingface.co/", a.hf)):
                if new and re.search(rf"{re.escape(host)}{re.escape(OLD)}(?![-\w])", line):
                    print(f"  {name}:{i}: {line.strip()[:100]}")
                    found = True
    if not found:
        print("  none")


if __name__ == "__main__":
    main()
