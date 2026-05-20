#!/usr/bin/env python3

import argparse
import pandas as pd
import json
from datetime import datetime
from simulation_logic import System

def main(params):
    system = System(params)
    system.simulate()

    total_kv = system.hits + system.misses
    hit_rate = (system.hits / total_kv * 100) if total_kv > 0 else 0
    actual_agents_per_second = params['iterations'] / system.T

    avg_busy_ratio = system.total_busy_servers / system.sample_count / system.gpus.num_servers if system.sample_count > 0 else 0
    avg_really_busy_ratio = system.total_really_busy_servers / system.sample_count / system.gpus.num_servers if system.sample_count > 0 else 0
    
    print(f"\033[1;31mAGENTS={system.num_inflight_agents}, STEPS={params['steps']}, STEP_TIME_IN_GPU={params['step_time_in_gpu']}, TIME_BETWEEN_STEPS={params['time_between_steps']}, AVG_BUSY_RATIO={avg_busy_ratio:.4f}, AVG_REALLY_BUSY_RATIO={avg_really_busy_ratio:.4f}, HIT_RATIO={hit_rate:.4f}, THEORETICAL_MAX_AGENTS_PER_S={system.gpu_max_possible_agents_per_second:.4f}, ACTUAL_AGENTS_PER_S={actual_agents_per_second:.4f}\033[0m")

    row = {
        'sleep_between_steps' : params['time_between_steps'],
        'step_time_in_gpu' : params['step_time_in_gpu'],
        'num_inflight_agents': system.num_inflight_agents,
        'steps': params['steps'],
        'disk_usage' : 1 /(params['disk_size_in_blocks'] / (system.num_inflight_agents * params['steps'])),
        'hit_rate': hit_rate,
        'avg_busy_ratio': avg_busy_ratio,
        'avg_really_busy_ratio': avg_really_busy_ratio,
    }
    df = pd.DataFrame([row])

    output_file = params['output_file']
    if output_file.endswith('.xlsx'):
        output_file_excel = output_file
        output_file_csv = output_file.replace('.xlsx', '.csv')
    elif output_file.endswith('.csv'):
        output_file_csv = output_file
        output_file_excel = output_file.replace('.csv', '.xlsx')
    else:
        output_file_excel = output_file + '.xlsx'
        output_file_csv = output_file + '.csv'
    
    df.to_excel(output_file_excel, index=False)
    df.to_csv(output_file_csv, index=False)
    print(f"\nResults written to {output_file_excel} and {output_file_csv}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-f", "--config_file",
        type=str,
        required=True,
        help="Path to JSON config file"
    )
    
    args = parser.parse_args()
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    params = {
        'disk_size_in_blocks': config.get('disk_size_in_blocks', 1),
        'allow_holes_recalculation': config.get('allow_holes_recalculation', 1),
        'random_placement_on_miss': config.get('random_placement_on_miss', 0),
        'evict_on_miss': config.get('evict_on_miss', 1),
        'num_inflight_agents': config['num_inflight_agents'],
        'steps': config['steps'],
        'ranges': config['ranges'],
        'iterations': config['iterations'],
        'time_between_steps': config['time_between_steps'],
        'total_gpus': config.get('total_gpus', 1),
        'step_time_in_gpu': config.get('step_time_in_gpu', 1),
        'force_hit_ratio': config.get('force_hit_ratio', 0),
        'scheduling_strategy': config.get('scheduling_strategy', 'shared_storage_least_busy'),
        'is_use_theoretical_agents': config.get('is_use_theoretical_agents', 0),
        'storage_blocks_per_second': config.get('storage_blocks_per_second', 0.0),
        'output_file': config.get('output_file', f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"),
        'monitor_interval_virtual_time': config.get('monitor_interval_virtual_time', 0)
    }
    if params['disk_size_in_blocks'] % params['ranges'] != 0:
        print(
            f"Error: disk_size_in_blocks ({params['disk_size_in_blocks']}) must be divisible by ranges ({params['ranges']})"
        )
        exit(1)
    if not (0.0 <= params['force_hit_ratio'] <= 1.0):
        print(f"Error: force_hit_ratio ({params['force_hit_ratio']}) must be between 0.0 and 1.0")
        exit(1)
    if params['monitor_interval_virtual_time'] == 0:
        params['monitor_interval_virtual_time'] = (
            params['step_time_in_gpu'] + params['time_between_steps']
        ) * 100

    main(params)
