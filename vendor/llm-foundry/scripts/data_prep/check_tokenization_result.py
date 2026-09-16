import os
import argparse
import statistics

def parse_n(name):
    parts = name.split('.', 1)
    n_str = parts[0]
    return int(n_str) if n_str.isdigit() else None

def get_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    elif os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    try:
                        total += os.path.getsize(fp)
                    except FileNotFoundError:
                        pass
        return total
    return 0

def main():
    parser = argparse.ArgumentParser(description='Compare source and target entries.')
    parser.add_argument('source', help='Source directory containing entries (files or folders)')
    parser.add_argument('target', help='Target directory containing folders')
    args = parser.parse_args()

    source_entries = {}
    for entry in os.listdir(args.source):
        path = os.path.join(args.source, entry)
        if os.path.isfile(path) or os.path.isdir(path):
            if (n := parse_n(entry)) is not None:
                source_entries[n] = get_size(path)

    target_dirs = {}
    for entry in os.listdir(args.target):
        path = os.path.join(args.target, entry)
        if os.path.isdir(path):
            if (n := parse_n(entry)) is not None:
                target_dirs[n] = get_size(path)

    source_ns = set(source_entries)
    target_ns = set(target_dirs)
    common_ns = source_ns & target_ns
    missing = {
        'source': sorted(target_ns - source_ns),
        'target': sorted(source_ns - target_ns)
    }

    if any(missing.values()):
        print("ERROR: Correspondence mismatch!")
        if missing['target']:
            print("- Missing in target:", missing['target'])
        if missing['source']:
            print("- Missing in source:", missing['source'])

    ratios = []
    for n in sorted(common_ns):
        src_size = source_entries[n]
        tgt_size = target_dirs[n]
        if src_size == 0:
            print(f"Warning: n={n} has zero source size, skipped")
            continue
        ratios.append((n, tgt_size / src_size))

    if not ratios:
        print("No valid ratios to analyze")
        return

    ratio_values = [r for _, r in ratios]
    median = statistics.median(ratio_values)

    try:
        q1, q3 = statistics.quantiles(ratio_values, n=4)[::2]
    except statistics.StatisticsError:
        q1 = q3 = median

    iqr = q3 - q1
    lower_bound = q1 - 2.0 * iqr
    upper_bound = q3 + 2.0 * iqr

    low_outliers = [(n, r) for n, r in ratios if r < lower_bound]

    print(f"\nMedian target/source ratio: {median:.2f}")
    print(f"Acceptable lower bound: {lower_bound:.2f}")

    if low_outliers:
        print("\nWARNING: Found abnormally low ratios:")
        for n, r in low_outliers:
            print(f"- n={n}: {r:.2f} (below {lower_bound:.2f}, median is {median:.2f})")
    else:
        print("\nAll ratios within acceptable range for data integrity")

if __name__ == "__main__":
    main()