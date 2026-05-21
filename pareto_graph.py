#!/usr/bin/env python3

import argparse
import csv
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from simulation_logic import System

BOLD_RED = '\033[1;31m'
RESET = '\033[0m'
X_LABEL = 'agents/user/sec'
Y_LABEL = 'agents/sec'


def params_from_config(config, num_inflight_agents):
    params = {
        'blocks_buffers': config['blocks_buffers'],
        'storage_tiers': config['storage_tiers'],
        'allow_holes_recalculation': config.get('allow_holes_recalculation', 1),
        'random_placement_on_miss': config.get('random_placement_on_miss', 0),
        'evict_on_miss': config.get('evict_on_miss', 1),
        'num_inflight_agents': num_inflight_agents,
        'steps': config['steps'],
        'iterations': config['iterations'],
        'time_between_steps': config['time_between_steps'],
        'total_gpus': config.get('total_gpus', 1),
        'step_time_in_gpu': config.get('step_time_in_gpu', 1),
        'is_linear_step_time': config['is_linear_step_time'],
        'force_hit_ratio': config.get('force_hit_ratio', 0),
        'scheduling_strategy': config.get('scheduling_strategy', 'shared_storage_least_busy'),
        'is_use_theoretical_agents': config.get('is_use_theoretical_agents', 0),
        'monitor_interval_virtual_time': 0,
    }
    for i, tier in enumerate(params['storage_tiers']):
        t = tier['type']
        if t == 'memos' and tier['num_blocks'] % tier['num_ranges'] != 0:
            raise ValueError(
                f"storage_tiers[{i}] num_blocks ({tier['num_blocks']}) must be divisible by num_ranges ({tier['num_ranges']})"
            )
        if t not in ('lru', 'memos', 'allocrelease'):
            raise ValueError(f"storage_tiers[{i}] unknown type {t!r}")
    return params


def plot_from_csv(data_file, output):
    series = {}
    with open(data_file) as f:
        for row in csv.DictReader(f):
            series.setdefault(row['config_label'], []).append(row)
    plt.figure(figsize=(10, 6))
    for label in sorted(series):
        rows = sorted(series[label], key=lambda r: int(r['num_inflight_agents']))
        xs = [float(r['agents_per_second_per_inflight']) for r in rows]
        ys = [float(r['actual_agents_per_second']) for r in rows]
        plt.plot(xs, ys, marker='o', linewidth=2, label=label)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Pareto: throughput vs per-agent throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved as {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config_file", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default="pareto_graph.png")
    parser.add_argument("-d", "--data-file", type=str, default="pareto_results.csv")
    parser.add_argument("-l", "--label", type=str, default=None)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--fresh-csv", action="store_true", help="delete -d csv before run, then write new rows")
    args = parser.parse_args()

    if args.plot_only:
        plot_from_csv(args.data_file, args.output)
        raise SystemExit

    if not args.config_file:
        parser.error("-f/--config_file is required unless --plot-only")

    with open(args.config_file) as f:
        config = json.load(f)

    label = args.label or os.path.splitext(os.path.basename(args.config_file))[0]
    fieldnames = [
        'config_label', 'num_inflight_agents', 'iterations', 'minimal_inflight',
        'actual_agents_per_second', 'agents_per_second_per_inflight', 'T',
    ]

    probe = System(params_from_config(config, 1))
    minimal = probe.minimal_inflight_agents_for_max_possible_agents_per_second
    max_inflight = 2 * minimal
    inflights = [max(1, int(round(x))) for x in np.linspace(1, max_inflight, 10)]

    xs, ys, rows = [], [], []
    print(f"{BOLD_RED}=== pareto label={label} config={args.config_file} points={len(inflights)} ==={RESET}")
    for i, n in enumerate(inflights, 1):
        params = params_from_config(config, n)
        params['iterations'] = n * 1000
        print(f"{BOLD_RED}=== sweep {i}/{len(inflights)} ==={RESET}")
        print(json.dumps(params, indent=2))
        system = System(params)
        system.simulate()
        actual = params['iterations'] / system.T
        x = actual / n
        xs.append(x)
        ys.append(actual)
        rows.append({
            'config_label': label,
            'num_inflight_agents': n,
            'iterations': params['iterations'],
            'minimal_inflight': minimal,
            'actual_agents_per_second': actual,
            'agents_per_second_per_inflight': x,
            'T': system.T,
        })
        print(f"num_inflight_agents={n} actual_agents_per_second={actual:.6f} x={x:.6f}")

    if args.fresh_csv and os.path.exists(args.data_file):
        os.remove(args.data_file)
    write_header = args.fresh_csv or not os.path.exists(args.data_file)
    with open(args.data_file, 'w' if args.fresh_csv else 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    print(f"Data written to {args.data_file} (label={label})")

    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, marker='o', linewidth=2, label=label)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Pareto: throughput vs per-agent throughput')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved as {args.output}")
