#!/usr/bin/env python3
"""
Verify that this copy of ANADEF reproduces the results shipped with it.

    python verify_reproduction.py            # main run only   (~5 min)
    python verify_reproduction.py --all      # all nine runs   (~45 min)
    python verify_reproduction.py --check    # dependencies only, runs nothing

The script snapshots the numerical outputs already in results/, re-runs the
pipeline, compares every CSV and JSON byte for byte, then rebuilds the
manuscript tables and compares those too. Anything that differs is printed.
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.abspath(__file__))
MAIN = 'Configuration_Hokkaido_FINAL.json'
VARIANTS = [
    ('Configuration_Hokkaido_FINAL_globalMc.json',      'results/output_FINAL_globalMc'),
    ('Configuration_Hokkaido_FINAL_train2004.json',     'results/output_FINAL_train2004'),
    ('Configuration_Hokkaido_FINAL_M55.json',           'results/output_FINAL_M55'),
    ('Configuration_Parameters_Hokkaido.json',          'results/output_hokkaido'),
    ('Configuration_Parameters_Hokkaido_grid04.json',   'results/output_hokkaido_grid04'),
    ('Configuration_Parameters_Hokkaido_grid04_b056.json','results/output_hokkaido_grid04_b056'),
    ('Configuration_Parameters_Hokkaido_grid03.json',   'results/output_hokkaido_grid03'),
    ('Configuration_Parameters_Hokkaido_grid03_b042.json','results/output_hokkaido_grid03_b042'),
]
DEPS = ['numpy', 'pandas', 'scipy', 'matplotlib', 'numba', 'geopandas', 'shapely', 'pyproj']
GREEN, RED, YELLOW, RESET = '\033[32m', '\033[31m', '\033[33m', '\033[0m'


def ok(msg):   print(f"  {GREEN}PASS{RESET}  {msg}")
def bad(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}WARN{RESET}  {msg}")


def check_dependencies():
    print("\n[1] Python environment")
    print(f"  python {sys.version.split()[0]}")
    missing = []
    for m in DEPS:
        try:
            mod = __import__(m)
            print(f"        {m:12s} {getattr(mod, '__version__', '?')}")
        except ImportError:
            missing.append(m)
    if missing:
        bad(f"missing packages: {', '.join(missing)}")
        print("        install with:  pip install -r requirements.txt")
        return False
    ok("all required packages importable")
    return True


def check_inputs():
    print("\n[2] Input files")
    need = ['data/input/Hokkaido_0-75km_2000-2023.csv.gz',
            'data/gis/coast_gshhs_f.geojson',
            'data/gis/faults_rgafj1991.geojson',
            MAIN]
    allgood = True
    for f in need:
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            print(f"        {f}  ({os.path.getsize(p)/1e6:.1f} MB)")
        else:
            bad(f"missing {f}"); allgood = False
    if allgood:
        ok("inputs present")
    return allgood


def numeric_files(d):
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if f.endswith(('.csv', '.json')))


def flatten(obj, prefix=''):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}/{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def compare_json(a, b):
    """Return list of substantive differences, treating NaN == NaN."""
    fa, fb = flatten(json.load(open(a))), flatten(json.load(open(b)))
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        x, y = fa.get(k), fb.get(k)
        if x == y:
            continue
        if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
            continue
        if k.endswith('/config'):        # config filename may legitimately differ
            continue
        diffs.append((k, x, y))
    return diffs


def compare_dir(ref, new, label):
    same = differ = missing = 0
    for name in numeric_files(ref):
        a, b = os.path.join(ref, name), os.path.join(new, name)
        if not os.path.exists(b):
            bad(f"{label}: {name} was not regenerated"); missing += 1; continue
        if open(a, 'rb').read() == open(b, 'rb').read():
            same += 1
        elif name.endswith('.json'):
            d = compare_json(a, b)
            if not d:
                same += 1
            else:
                differ += 1
                bad(f"{label}: {name} differs in {len(d)} value(s)")
                for k, x, y in d[:5]:
                    print(f"          {k}\n            shipped={x}\n            rerun  ={y}")
        else:
            differ += 1
            bad(f"{label}: {name} differs")
    return same, differ, missing


def run_config(cfg):
    t0 = time.time()
    r = subprocess.run([sys.executable, 'main_anadef.py', '--config', cfg],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        bad(f"{cfg} exited {r.returncode}")
        print('\n'.join(r.stdout.splitlines()[-15:]))
        print('\n'.join(r.stderr.splitlines()[-15:]))
        return False, time.time() - t0
    return True, time.time() - t0


def verify_run(cfg, outdir, snapshot_root):
    full = os.path.join(ROOT, outdir)
    ref = os.path.join(snapshot_root, os.path.basename(outdir))
    os.makedirs(ref, exist_ok=True)
    for f in numeric_files(full):
        shutil.copy(os.path.join(full, f), ref)
    n_ref = len(numeric_files(ref))
    if n_ref == 0:
        warn(f"{outdir}: no shipped outputs to compare against; will only check the run completes")
    print(f"\n  running {cfg} ...", flush=True)
    good, secs = run_config(cfg)
    if not good:
        return False
    ok(f"{cfg} completed in {secs/60:.1f} min")
    if n_ref:
        same, differ, missing = compare_dir(ref, full, os.path.basename(outdir))
        if differ == 0 and missing == 0:
            ok(f"{outdir}: all {same} numerical outputs reproduce exactly")
            return True
        return False
    return True


def verify_tables(snapshot_root):
    print("\n[4] Manuscript tables")
    shipped = os.path.join(ROOT, 'results/manuscript_tables')
    if not os.path.isdir(shipped):
        warn("results/manuscript_tables not present; skipping")
        return True
    tmp = os.path.join(snapshot_root, 'tables_rebuilt')
    os.makedirs(tmp, exist_ok=True)
    r = subprocess.run([sys.executable, 'make_tables.py',
                        '--main', 'results/output_FINAL',
                        '--globalmc', 'results/output_FINAL_globalMc',
                        '--train2004', 'results/output_FINAL_train2004',
                        '--m55', 'results/output_FINAL_M55',
                        '--out', tmp], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        bad("make_tables.py failed"); print(r.stderr[-800:]); return False
    allgood = True
    for t in ['tab2.tex', 'tab3.tex', 'tab4.tex', 'tab_paired.tex']:
        a, b = os.path.join(shipped, t), os.path.join(tmp, t)
        if not os.path.exists(b):
            bad(f"{t} not produced"); allgood = False
        elif open(a).read() == open(b).read():
            ok(f"{t} identical to the version used in the manuscript")
        else:
            bad(f"{t} differs from the manuscript version"); allgood = False
    return allgood


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', action='store_true', help='verify all nine runs, not just the main one')
    ap.add_argument('--check', action='store_true', help='check environment and inputs only')
    args = ap.parse_args()

    print("=" * 72)
    print("ANADEF v2.1 reproduction check")
    print("=" * 72)

    if not check_dependencies() or not check_inputs():
        sys.exit(1)
    if args.check:
        print("\nEnvironment looks good. Re-run without --check to verify the results.\n")
        return

    snapshot = tempfile.mkdtemp(prefix='anadef_verify_')
    print(f"\n[3] Re-running the pipeline (shipped outputs snapshotted to {snapshot})")

    results = [verify_run(MAIN, 'results/output_FINAL', snapshot)]
    if args.all:
        for cfg, outdir in VARIANTS:
            results.append(verify_run(cfg, outdir, snapshot))

    results.append(verify_tables(snapshot))

    print("\n" + "=" * 72)
    if all(results):
        print(f"{GREEN}REPRODUCTION VERIFIED{RESET} - every checked output matches what is shipped.")
        print("=" * 72 + "\n")
        sys.exit(0)
    print(f"{RED}REPRODUCTION FAILED{RESET} - see the FAIL lines above.")
    print(f"Shipped outputs were snapshotted to {snapshot} for comparison.")
    print("=" * 72 + "\n")
    sys.exit(1)


if __name__ == '__main__':
    main()
