#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from simulation_logic import System

BOLD_RED = '\033[1;31m'
RESET = '\033[0m'
X_LABEL = 'agents/user/sec'
Y_LABEL = 'agents/sec'
CONCURRENCY_X_LABEL = 'concurrency'
FIELDNAMES = [
    'config_label', 'num_inflight_agents', 'iterations', 'minimal_inflight',
    'actual_agents_per_second', 'agents_per_second_per_inflight', 'T',
]
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


def load_series(data_file):
    series = {}
    with open(data_file) as f:
        for row in csv.DictReader(f):
            series.setdefault(row['config_label'], []).append(row)
    return series


def plot_pareto(series, output):
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


def plot_concurrency(series, output):
    plt.figure(figsize=(10, 6))
    for label in sorted(series):
        rows = sorted(series[label], key=lambda r: int(r['num_inflight_agents']))
        xs = [int(r['num_inflight_agents']) for r in rows]
        ys = [float(r['actual_agents_per_second']) for r in rows]
        plt.plot(xs, ys, marker='o', linewidth=2, label=label)
    plt.xlabel(CONCURRENCY_X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.title('Throughput vs concurrency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    print(f"Plot saved as {output}")


def plot_from_csv(data_file, output, concurrency_output=None):
    series = load_series(data_file)
    plot_pareto(series, output)
    if concurrency_output:
        plot_concurrency(series, concurrency_output)


def concurrency_output_path(output):
    stem, ext = os.path.splitext(output)
    return stem + '_concurrency' + ext


def _pareto_sweep_task(task):
    return pareto_sweep(*task)


def pareto_sweep(config_file, label, n_sweeps, inflight_factor):
    with open(config_file) as f:
        config = json.load(f)
    label = label or os.path.splitext(os.path.basename(config_file))[0]
    probe = System(params_from_config(config, 1))
    minimal = probe.minimal_inflight_agents_for_max_possible_agents_per_second
    max_inflight = inflight_factor * minimal
    inflights = [max(1, int(round(x))) for x in np.linspace(1, max_inflight, n_sweeps)]
    rows, xs, ys = [], [], []
    print(f"{BOLD_RED}=== pareto label={label} config={config_file} points={len(inflights)} ==={RESET}")
    for i, n in enumerate(inflights, 1):
        params = params_from_config(config, n)
        params['iterations'] = n * 1000
        print(f"{BOLD_RED}=== sweep {i}/{len(inflights)} [{label}] ==={RESET}")
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
        print(f"[{label}] num_inflight_agents={n} actual_agents_per_second={actual:.6f} x={x:.6f}")
    return label, rows, xs, ys


def write_csv(data_file, all_rows, fresh):
    if fresh and os.path.exists(data_file):
        os.remove(data_file)
    write_header = fresh or not os.path.exists(data_file)
    with open(data_file, 'w' if fresh else 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        w.writerows(all_rows)


def resolve_labels(config_files, labels):
    if not labels:
        return [None] * len(config_files)
    if len(labels) == 1:
        return labels * len(config_files)
    if len(labels) != len(config_files):
        raise ValueError(f"expected 0, 1, or {len(config_files)} labels, got {len(labels)}")
    return labels


def setup_run_dir(results_root, config_files, sweeps, inflight_factor, jobs):
    run_dir = os.path.join(results_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    seen = {}
    copied = []
    for cfg in config_files:
        base = os.path.basename(cfg)
        if base in seen:
            seen[base] += 1
            stem, ext = os.path.splitext(base)
            dest_name = f"{stem}_{seen[base]}{ext}"
        else:
            seen[base] = 0
            dest_name = base
        shutil.copy2(cfg, os.path.join(run_dir, dest_name))
        copied.append(dest_name)
    with open(os.path.join(run_dir, 'run_meta.json'), 'w') as f:
        json.dump({
            'sweeps': sweeps,
            'inflight_factor': inflight_factor,
            'jobs': jobs,
            'config_files': copied,
        }, f, indent=2)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", action="append", default=None)
    parser.add_argument("-o", "--output", type=str, default="pareto_graph.png")
    parser.add_argument("-d", "--data-file", type=str, default="pareto_results.csv")
    parser.add_argument("-l", "--label", action="append", default=None)
    parser.add_argument("-j", "--jobs", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=10, help="number of inflight points from 1 to factor*minimal")
    parser.add_argument("--inflight-factor", type=float, default=2.0)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--concurrency", action="store_true", help="also plot concurrency -> agents/sec")
    parser.add_argument("--concurrency-output", type=str, default=None)
    parser.add_argument("--fresh-csv", action="store_true")
    args = parser.parse_args()

    conc_out = args.concurrency_output or (concurrency_output_path(args.output) if args.concurrency else None)

    if args.plot_only:
        plot_from_csv(args.data_file, args.output, conc_out)
        raise SystemExit

    if not args.config_file:
        parser.error("at least one -f/--config_file is required unless --plot-only")
    labels = resolve_labels(args.config_file, args.label)
    jobs = args.jobs or min(len(args.config_file), os.cpu_count() or 1)
    run_dir = setup_run_dir(args.results_dir, args.config_file, args.sweeps, args.inflight_factor, jobs)
    data_file = os.path.join(run_dir, os.path.basename(args.data_file))
    output = os.path.join(run_dir, os.path.basename(args.output))
    print(f"Run directory: {run_dir}")
    tasks = [
        (cfg, lbl, args.sweeps, args.inflight_factor)
        for cfg, lbl in zip(args.config_file, labels)
    ]

    conc_out = os.path.join(run_dir, os.path.basename(conc_out)) if conc_out else None

    if len(tasks) == 1:
        _, rows, xs, ys = pareto_sweep(*tasks[0])
        all_rows = rows
        write_csv(data_file, all_rows, True)
        print(f"Data written to {data_file} (label={rows[0]['config_label']})")
        series = {rows[0]['config_label']: rows}
        plot_pareto(series, output)
        if conc_out:
            plot_concurrency(series, conc_out)
    else:
        print(f"Running {len(tasks)} configs with {jobs} parallel workers")
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(_pareto_sweep_task, tasks))
        all_rows = [row for _, rows, _, _ in results for row in rows]
        write_csv(data_file, all_rows, True)
        print(f"Data written to {data_file} ({len(results)} configs, {len(all_rows)} rows)")
        plot_from_csv(data_file, output, conc_out)
