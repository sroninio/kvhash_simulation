#!/usr/bin/env python3

import argparse
import heapq
import random
import pandas as pd


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
    def __init__(self, disk_size_in_blocks, num_queries_per_agent_lower, num_queries_per_agent_upper, allow_holes_recalculation, num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.num_queries_per_agent_lower = num_queries_per_agent_lower
        self.num_queries_per_agent_upper = num_queries_per_agent_upper
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss

        self.prev_conv_id = first_conv_id
        self.events = []  # Min heap for events
        self.disk = disk
        self.finished_conversations = 0
        self.inflights = 0
        self.T = 0
        self.misses = 0
        self.hits = 0
        self.total_accesses = 0
        self.total_writes = 0

        
        print("=== Simulation Parameters ===")
        print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
        print(f"num_queries_per_agent_lower: {self.num_queries_per_agent_lower}")
        print(f"num_queries_per_agent_upper: {self.num_queries_per_agent_upper}")
        print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
        print(f"num_inflight_agents: {self.num_inflight_agents}")
        print(f"iterations: {self.iterations}")
        print(f"random_placement_on_miss: {self.random_placement_on_miss}")
        print(f"ranges: {self.ranges}")
        print(f"evict_on_miss: {self.evict_on_miss}")
        print("=============================\n")


    def alloc_block(self, block):
        range_len = self.disk_size_in_blocks // self.ranges
        consecutive_operations_before_moving_range = (self.num_inflight_agents * ((self.num_queries_per_agent_lower + self.num_queries_per_agent_upper) // 2)) // self.ranges
        range_idx = ((self.total_accesses // consecutive_operations_before_moving_range) % self.ranges)
        #print(range_idx)
        offset_in_range = random.randrange(range_len) if ((not block) or (self.random_placement_on_miss)) else (block.offset % range_len)
        block_offset = range_idx * range_len + offset_in_range 
        return self.disk.disk[block_offset] 

    def handle_conversation_return_event(self, conv):
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
        kv = self.alloc_block(None)
        kv.take_ownership(conv.conv_id, len(conv.kvs))
        conv.kvs.append(kv)
        self.total_accesses += 1
        
    def handle_statistic_event(self):
        pass

    def get_unique_id(self):
        self.prev_conv_id += 1
        return self.prev_conv_id

    def simulate(self):
        heapq.heappush(self.events, (self.T + 0.1, {'type': 'stat'}))
        while self.finished_conversations < self.iterations:
            t, args = heapq.heappop(self.events)
            self.T = t
            if args['type'] == "stat":
                self.handle_statistic_event()
                heapq.heappush(self.events, (self.T + 0.1, {'type': 'stat'}))
            else:
                conv = args['conv'] 
                self.handle_conversation_return_event(conv)
                if not conv.is_finished():
                    heapq.heappush(self.events, (self.T + random.random(), {'type': 'conv', 'conv': conv}))
                else:
                    self.finished_conversations += 1
                    self.inflights -= 1
            while self.inflights < self.num_inflight_agents:
                conv = Conversation(self.get_unique_id(), random.randint(self.num_queries_per_agent_lower, self.num_queries_per_agent_upper), self.disk_size_in_blocks)
                heapq.heappush(self.events, (self.T + random.random(), {'type': 'conv', 'conv': conv})) 
                self.inflights += 1
        
        # Print cache statistics
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
        print(f"Hits: {self.hits}, Misses: {self.misses}")
        return hit_rate
        
            

def main(disk_size_in_blocks, allow_holes_recalculation, random_placement_on_miss, evict_on_miss, agents_list, steps_list, ranges_list, sim_ratio, iterations):
    disk = Disk(disk_size_in_blocks)
    first_conv_id = 0
    results = []
    for agents in agents_list:
        for steps in steps_list:
            for ranges_val in ranges_list:
                system = System(
                    disk_size_in_blocks=disk_size_in_blocks // sim_ratio,
                    num_queries_per_agent_lower=steps,
                    num_queries_per_agent_upper=steps,
                    allow_holes_recalculation=allow_holes_recalculation,
                    num_inflight_agents=agents // sim_ratio,
                    iterations=iterations // steps,
                    random_placement_on_miss=random_placement_on_miss,
                    ranges=ranges_val,
                    evict_on_miss=evict_on_miss,
                    disk=disk,
                    first_conv_id=first_conv_id
                )
                hit_rate = system.simulate()
                first_conv_id = system.prev_conv_id + 10
                
                # Collect results
                results.append({
                    'agents': agents,
                    'steps': steps,
                    'ranges': ranges_val,
                    'disk_size_in_blocks': disk_size_in_blocks,
                    'hit_rate': hit_rate,
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
        required=True
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
        default=[1, 4, 10],
        help="List of ranges values to test (default: 1 4 10)"
    )
    
    parser.add_argument(
        "--sim_ratio",
        type=int,
        default=10,
        help="Simulation ratio divider (default: 10)"
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
        iterations=args.iterations
    )
