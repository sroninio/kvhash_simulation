#!/usr/bin/env python3

import argparse
import heapq
import random


class Block:
    def __init__(self):
        self.owner_conv_id = -1
        self.owner_pos_in_conv = -1
    
    def is_belongs_to(self, conv_id, pos_in_conv):
        return (self.owner_conv_id == conv_id) and (self.owner_pos_in_conv == pos_in_conv)
    
    def take_ownership(self, conv_id, pos_in_conv):
        self.owner_conv_id = conv_id
        self.owner_pos_in_conv = pos_in_conv
        
    

class Conversation:
    def __init__(self, conv_id, conversation_length, disk_size_in_blocks, virtual_agent_range_size_in_blocks):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.kvs = []
        self.slot_offset = -1 if virtual_agent_range_size_in_blocks == 1 else random.randrange(disk_size_in_blocks // virtual_agent_range_size_in_blocks) * virtual_agent_range_size_in_blocks 

    def is_finished(self):
        return (len(self.kvs) >= self.conversation_length)
    


class System:
    def __init__(self, disk_size_in_blocks, virtual_agent_range_size_in_blocks, num_queries_per_agent_lower, num_queries_per_agent_upper, allow_holes_recalculation, num_inflight_agents, iterations, allocate_new_on_miss):
        if disk_size_in_blocks % virtual_agent_range_size_in_blocks != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by virtual_agent_range_size_in_blocks ({virtual_agent_range_size_in_blocks})")
            exit(1)
        
        if allocate_new_on_miss and virtual_agent_range_size_in_blocks > 1:
            print(f"Error: allocate_new_on_miss ({allocate_new_on_miss}) is not supported with virtual_agent_range_size_in_blocks > 1 ({virtual_agent_range_size_in_blocks})")
            exit(1)
        
        if 1 < virtual_agent_range_size_in_blocks < num_queries_per_agent_upper:
            print(f"Error: virtual_agent_range_size_in_blocks ({virtual_agent_range_size_in_blocks}) must be at least num_queries_per_agent_upper ({num_queries_per_agent_upper})")
            exit(1)

        self.disk_size_in_blocks = disk_size_in_blocks
        self.virtual_agent_range_size_in_blocks = virtual_agent_range_size_in_blocks
        self.num_queries_per_agent_lower = num_queries_per_agent_lower
        self.num_queries_per_agent_upper = num_queries_per_agent_upper
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents
        self.iterations = iterations
        self.allocate_new_on_miss = allocate_new_on_miss

        self.prev_conv_id = 0
        self.events = []  # Min heap for events
        self.disk = [Block() for _ in range(self.disk_size_in_blocks)]
        self.finished_conversations = 0
        self.inflights = 0
        self.T = 0
        self.misses = 0
        self.hits = 0
        
        print("=== Simulation Parameters ===")
        print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
        print(f"virtual_agent_range_size_in_blocks: {self.virtual_agent_range_size_in_blocks}")
        print(f"num_queries_per_agent_lower: {self.num_queries_per_agent_lower}")
        print(f"num_queries_per_agent_upper: {self.num_queries_per_agent_upper}")
        print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
        print(f"num_inflight_agents: {self.num_inflight_agents}")
        print(f"iterations: {self.iterations}")
        print(f"allocate_new_on_miss: {self.allocate_new_on_miss}")
        print("=============================\n")

    def alloc_block(self, conv, indx_in_conv):
        block_offset = conv.slot_offset + indx_in_conv if conv.slot_offset != -1 else random.randrange(self.disk_size_in_blocks) 
        return self.disk[block_offset]

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
            if not valid_kv:
                if self.allocate_new_on_miss:
                    conv.kvs[indx] = self.alloc_block(conv, indx)
                conv.kvs[indx].take_ownership(conv.conv_id, indx)
        kv = self.alloc_block(conv, len(conv.kvs))
        kv.take_ownership(conv.conv_id, len(conv.kvs))
        conv.kvs.append(kv)
        
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
                conv = Conversation(self.get_unique_id(), random.randint(self.num_queries_per_agent_lower, self.num_queries_per_agent_upper), self.disk_size_in_blocks, self.virtual_agent_range_size_in_blocks)
                heapq.heappush(self.events, (self.T + random.random(), {'type': 'conv', 'conv': conv})) 
                self.inflights += 1
        
        # Print cache statistics
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
        print(f"Hits: {self.hits}, Misses: {self.misses}")
        
            

def main(disk_size_in_blocks, virtual_agent_range_size_in_blocks, num_queries_per_agent_lower, num_queries_per_agent_upper, allow_holes_recalculation, num_inflight_agents, iterations, allocate_new_on_miss):
    system = System(
        disk_size_in_blocks=disk_size_in_blocks,
        virtual_agent_range_size_in_blocks=virtual_agent_range_size_in_blocks,
        num_queries_per_agent_lower=num_queries_per_agent_lower,
        num_queries_per_agent_upper=num_queries_per_agent_upper,
        allow_holes_recalculation=allow_holes_recalculation,
        num_inflight_agents=num_inflight_agents,
        iterations=iterations,
        allocate_new_on_miss=allocate_new_on_miss
    )
    system.simulate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--disk_size_in_blocks",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--virtual_agent_range_size_in_blocks",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--num_queries_per_agent_lower",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--num_queries_per_agent_upper",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--allow_holes_recalculation",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--num_inflight_agents",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        required=True
    )
    
    parser.add_argument(
        "--allocate_new_on_miss",
        type=int,
        required=True
    )
    
    args = parser.parse_args()
    
    main(
        disk_size_in_blocks=args.disk_size_in_blocks,
        virtual_agent_range_size_in_blocks=args.virtual_agent_range_size_in_blocks,
        num_queries_per_agent_lower=args.num_queries_per_agent_lower,
        num_queries_per_agent_upper=args.num_queries_per_agent_upper,
        allow_holes_recalculation=args.allow_holes_recalculation,
        num_inflight_agents=args.num_inflight_agents,
        iterations=args.iterations,
        allocate_new_on_miss=args.allocate_new_on_miss
    )
