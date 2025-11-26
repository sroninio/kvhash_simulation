#!/usr/bin/env python3

import argparse
import heapq
import random
import pandas as pd
from collections import deque


class Block:
    def __init__(self, offset):
        self.owner_conv_id = -1
        self.owner_pos_in_conv = -1
        self.offset = offset
    
    def is_belongs_to(self, conv_id, pos_in_conv):
        return (self.owner_conv_id == conv_id) and (self.owner_pos_in_conv == pos_in_conv)
    
    def take_ownership(self, conv_id, pos_in_conv):
        self.owner_conv_id = conv_id
        self.owner_pos_in_conv = pos_in_conv
            


class Conversation:
    def __init__(self, conv_id, conversation_length, disk_size_in_blocks):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.kvs = []

    def is_finished(self):
        return (len(self.kvs) >= self.conversation_length)
    

class Disk:
    def __init__(self, size):
        self.disk = [Block(indx) for indx in range(size)] 

class System:
    def __init__(self, disk_size_in_blocks, steps_lower, steps_upper, allow_holes_recalculation, \
        num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id, \
        time_between_steps, total_gpus, step_time_in_gpu, to_check_cache_hit_rate):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.steps_lower = steps_lower
        self.steps_upper = steps_upper
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss
        self.range_len = self.disk_size_in_blocks // self.ranges
        self.time_between_steps = time_between_steps
        self.total_gpus = total_gpus
        self.step_time_in_gpu = step_time_in_gpu
        self.to_check_cache_hit_rate = to_check_cache_hit_rate
        

        self.prev_conv_id = first_conv_id
        self.events = []  # Min heap for events
        self.disk = disk
        self.finished_conversations = 0
        self.inflights = 0
        self.T = 0
        self.misses = 0
        self.hits = 0
        self.conversations_queue = deque()
        self.free_gpus = self.total_gpus
        
        avg_steps = (self.steps_lower + self.steps_upper) / 2
        total_time_spent_between_steps = (avg_steps + 1) * time_between_steps
        total_time_in_gpu = avg_steps * self.step_time_in_gpu
        gpu_requests_per_second = self.total_gpus * (1 / total_time_in_gpu) 
        total_agent_time = total_time_in_gpu + total_time_spent_between_steps
        minimal_agent_max_bw = gpu_requests_per_second * total_agent_time

        
        print("\033[1;33m\n=============================\033[0m")
        if self.to_check_cache_hit_rate:
            print("CACHE PERF ANALISS")        
            print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
            print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
            print(f"num_inflight_agents: {self.num_inflight_agents}")
            print(f"iterations: {self.iterations}")
            print(f"random_placement_on_miss: {self.random_placement_on_miss}")
            print(f"ranges: {self.ranges}")
            print(f"evict_on_miss: {self.evict_on_miss}")
        else:
            print("BW PERF ANALISS")        
            print(f"total_gpus: {self.total_gpus}")
            print(f"step_time_in_gpu: {self.step_time_in_gpu}")
            print(f"steps_lower: {self.steps_lower}")
            print(f"steps_upper: {self.steps_upper}")
            print(f"time_between_steps: {self.time_between_steps}") 
            print(f"\033[1;31mnum_inflight_agents: {self.num_inflight_agents}\033[0m")
            print("BW THEORETICAL NUMBERS")        
            print(f"\033[1;34mmaximal possible gpu requests per second: {gpu_requests_per_second:.8f}\033[0m")
            print(f"\033[1;31mminimal_agent_max_bw: {minimal_agent_max_bw:.2f}\033[0m")
        


    def alloc_block(self, block):
        prev_range_idx = block.offset // self.range_len if block else -1
        while True:
            range_idx = random.randrange(self.ranges)
            if range_idx != prev_range_idx or self.ranges == 1:
                break
        offset_in_range = random.randrange(self.range_len) if ((not block) or (self.random_placement_on_miss)) else (block.offset % self.range_len)
        block_offset = range_idx * self.range_len + offset_in_range 
        return self.disk.disk[block_offset] 

     

    def handle_conversation_return_event(self, conv):
        if not self.to_check_cache_hit_rate:
            return
        disable_all = False
        for indx in range(len(conv.kvs)):
            kv = conv.kvs[indx]
            valid_kv = kv.is_belongs_to(conv.conv_id, indx) 
            disable_all = disable_all or ((not self.allow_holes_recalculation) and (not valid_kv))
            if (not disable_all) and valid_kv:
                self.hits += 1
            else:
                self.misses += 1
            if not valid_kv and self.evict_on_miss:
                conv.kvs[indx] = self.alloc_block(conv.kvs[indx])
                conv.kvs[indx].take_ownership(conv.conv_id, indx)

    
    def handle_gpu_finished(self, conv):
        if not self.to_check_cache_hit_rate:
            conv.kvs.append(12)
            return
        kv = self.alloc_block(None)
        kv.take_ownership(conv.conv_id, len(conv.kvs))
        conv.kvs.append(kv)

        
    def handle_statistic_event(self):
        pass

    def get_unique_id(self):
        self.prev_conv_id += 1
        return self.prev_conv_id
    
    def enter_conv_to_gpu(self, conv):
        if self.free_gpus > 0:
            self.free_gpus -= 1
            heapq.heappush(self.events, (self.T + random.random() * 2 * self.step_time_in_gpu, {'type': 'gpu_finished', 'conv': conv}))
        else:
            self.conversations_queue.append(conv)
    
    def enter_conv_to_sleep(self, conv):
        if not conv.is_finished():
            heapq.heappush(self.events, (self.T + random.random() * 2 * self.time_between_steps, {'type': 'conv', 'conv': conv}))
        else:
            self.finished_conversations += 1
            self.inflights -= 1


    def simulate(self):
        heapq.heappush(self.events, (self.T + 0.1, {'type': 'stat'}))
        while self.finished_conversations < self.iterations:
            t, args = heapq.heappop(self.events)
            self.T = t
            if args['type'] == "stat":
                self.handle_statistic_event()
                heapq.heappush(self.events, (self.T + 0.1, {'type': 'stat'}))
            elif args['type'] == 'conv':
                conv = args['conv']
                self.handle_conversation_return_event(conv)
                self.enter_conv_to_gpu(conv)
            else: #gpu finished
                conv = args['conv'] 
                self.handle_gpu_finished(conv)
                self.enter_conv_to_sleep(conv)
                self.free_gpus += 1
                if len(self.conversations_queue) > 0:
                    conv = self.conversations_queue.popleft()
                    self.enter_conv_to_gpu(conv)
            while self.inflights < self.num_inflight_agents:
                conv = Conversation(self.get_unique_id(), random.randint(self.steps_lower, self.steps_upper), self.disk_size_in_blocks)
                self.inflights += 1
                self.enter_conv_to_sleep(conv)
                
        
        # Print cache statistics
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        if self.to_check_cache_hit_rate:
            print("CACHE RESULTS")        
            print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
            print(f"Hits: {self.hits}, Misses: {self.misses}")
        else:
            print("BW RESULTS")        
            print(f"\033[1;34mSimulation Requests Per Second: {self.iterations / self.T:.8f}\033[0m")
        print("\033[1;33m=============================\033[0m")
        return hit_rate, self.T, self.iterations
        
            

def main(disk_size_in_blocks, allow_holes_recalculation, random_placement_on_miss, evict_on_miss, agents_list, steps_list, ranges_list, sim_ratio, iterations, time_between_steps, total_gpus, step_time_in_gpu, to_check_cache_hit_rate):
    disk = Disk(disk_size_in_blocks)
    first_conv_id = 0
    results = []
    for agents in agents_list:
        for steps in steps_list:
            for ranges_val in ranges_list:
                system = System(
                    disk_size_in_blocks=disk_size_in_blocks // sim_ratio,
                    steps_lower=steps,
                    steps_upper=steps,
                    allow_holes_recalculation=allow_holes_recalculation,
                    num_inflight_agents=agents // sim_ratio,
                    iterations=iterations // (steps if to_check_cache_hit_rate else 1),
                    random_placement_on_miss=random_placement_on_miss,
                    ranges=ranges_val,
                    evict_on_miss=evict_on_miss,
                    disk=disk,
                    first_conv_id=first_conv_id,
                    time_between_steps=time_between_steps,
                    total_gpus=total_gpus,
                    step_time_in_gpu=step_time_in_gpu,
                    to_check_cache_hit_rate=to_check_cache_hit_rate
                )
                hit_rate, total_time, total_iterations = system.simulate()
                first_conv_id = system.prev_conv_id + 10
                
                # Collect results
                results.append({
                    'agents': agents,
                    'steps': steps,
                    'ranges': ranges_val,
                    'disk_size_in_blocks': disk_size_in_blocks,
                    'hit_rate': hit_rate,
                    'total_time': total_time,
                    'total_iterations': total_iterations,
                })
    
    # Write results to Excel
    df = pd.DataFrame(results)
    output_file = 'simulation_results.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\nResults written to {output_file}")
                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--disk_size_in_blocks",
        type=int,
        default=0,
        help="Disk size in blocks (default: 0)"
    )
    
    parser.add_argument(
        "--allow_holes_recalculation",
        type=int,
        default=1,
        help="Allow holes recalculation (default: 1)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=1500,
        help="Number of iterations (default: 1500)"
    )
    
    parser.add_argument(
        "--random_placement_on_miss",
        type=int,
        default=0,
        help="Random placement on miss (default: 0)"
    )
    
    parser.add_argument(
        "--evict_on_miss",
        type=int,
        default=1,
        help="Evict on miss (default: 1)"
    )
    
    parser.add_argument(
        "--agents_list",
        type=int,
        nargs='+',
        default=[1000, 2000, 4000, 8000, 16000, 32000, 64000],
        help="List of agent values to test (default: 1000 2000 4000 8000 16000 32000 64000)"
    )
    
    parser.add_argument(
        "--steps_list",
        type=int,
        nargs='+',
        default=[10, 50, 100, 150],
        help="List of step values to test (default: 10 50 100 150)"
    )
    
    parser.add_argument(
        "--ranges_list",
        type=int,
        nargs='+',
        default=[1],
        help="List of ranges values to test (default: 1 4 10)"
    )
    
    parser.add_argument(
        "--sim_ratio",
        type=int,
        default=1,
        help="Simulation ratio divider (default: 10)"
    )
    
    parser.add_argument(
        "--time_between_steps",
        type=float,
        required=True,
        help="Time between steps"
    )
    
    parser.add_argument(
        "--total_gpus",
        type=int,
        default=1,
        help="Total number of GPUs (default: 1)"
    )
    
    parser.add_argument(
        "--step_time_in_gpu",
        type=float,
        required=True,
        help="Step time in GPU"
    )
    
    parser.add_argument(
        "--to_check_cache_hit_rate",
        type=int,
        default=1,
        help="Check cache hit rate (default: 1)"
    )
    
    args = parser.parse_args()
    
    main(
        disk_size_in_blocks=args.disk_size_in_blocks,
        allow_holes_recalculation=args.allow_holes_recalculation,
        random_placement_on_miss=args.random_placement_on_miss,
        evict_on_miss=args.evict_on_miss,
        agents_list=args.agents_list,
        steps_list=args.steps_list,
        ranges_list=args.ranges_list,
        sim_ratio=args.sim_ratio,
        iterations=args.iterations,
        time_between_steps=args.time_between_steps,
        total_gpus=args.total_gpus,
        step_time_in_gpu=args.step_time_in_gpu,
        to_check_cache_hit_rate=args.to_check_cache_hit_rate
    )
