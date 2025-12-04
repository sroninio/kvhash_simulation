#!/usr/bin/env python3

import heapq
import random
import math
from collections import deque, defaultdict
from abc import ABC, abstractmethod


class Gpus(ABC):
    def __init__(self, system, step_time_in_gpu):
        self.system = system
        self.step_time_in_gpu = step_time_in_gpu
    
    @abstractmethod
    def enter_to_gpus(self, conv, blocks_to_calculate):
        """Virtual function to be implemented by subclasses"""
        pass

    @abstractmethod
    def back_from_gpu(self, conv, gpu):
        """Virtual function to be implemented by subclasses"""
        pass



class ConversationManager:
    def __init__(self, system, sleep_time_between_steps, first_conv_id, steps, context_window_size):
        self.sleep_time_between_steps = sleep_time_between_steps
        self.system = system
        self.inflights = 0
        self.finished_conversations = 0
        self.conv_id = first_conv_id
        self.steps = steps
        self.context_window_size = context_window_size
    
    def send_conversation_to_sleep(self, conv):
        if conv.is_finished():
            self.inflights -= 1
            self.finished_conversations += 1
        else:
            self.system.push_event(self.system.T + random.expovariate(1.0 / self.sleep_time_between_steps), {'type': 'back_from_between_steps_sleep', 'conv': conv}) 

    def get_unique_id(self):
        self.conv_id += 1
        return self.conv_id
    
    def create_conversation(self):
        conv = Conversation(self.get_unique_id(), self.steps, self.context_window_size)
        self.inflights += 1
        self.send_conversation_to_sleep(conv)
    

    def back_from_sleep(self,conv):
        conv.phase_in_step = 0
        conv.disable_all = False
        conv.produced_new_block_in_this_step = False


class SharedGpus(Gpus):
    """Concrete implementation of Gpus abstract class"""
    
    def __init__(self, system, num_gpus, step_time_in_gpu):
        super().__init__(system, step_time_in_gpu)
        self.free_gpus = num_gpus
        self.gpu_queue = deque()
    
    def enter_to_gpus(self, conv, blocks_to_calculate):
        if self.free_gpus > 0:
            self.free_gpus -= 1
            self.system.push_event(
                self.system.T + random.expovariate(1.0 / (self.step_time_in_gpu * blocks_to_calculate)),
                {'type': 'back_from_gpu', 'conv': conv, 'gpu' : self}
            )
        else:
            self.gpu_queue.append((conv, blocks_to_calculate))
    
    def back_from_gpu(self, conv, gpu):
        self.free_gpus += 1
        while len(self.gpu_queue) > 0 and self.free_gpus > 0:
            waiting_conv, blocks_to_calculate = self.gpu_queue.popleft()
            self.enter_to_gpus(waiting_conv, blocks_to_calculate)


class NonSharedGpus(Gpus):
    """Concrete implementation of Gpus with dedicated GPU per conversation"""
    
    def __init__(self, system, num_gpus, step_time_in_gpu):
        super().__init__(system, step_time_in_gpu)
        self.gpus = [SharedGpus(system, 1, step_time_in_gpu) for _ in range(num_gpus)]
        self.conv_to_gpu = defaultdict(lambda: random.choice(self.gpus))

    def enter_to_gpus(self, conv, blocks_to_calculate):
        gpu = self.conv_to_gpu[conv.conv_id]
        gpu.enter_to_gpus(conv, blocks_to_calculate)
    
    def back_from_gpu(self, conv, gpu):
        gpu.back_from_gpu(conv, gpu)


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
    def __init__(self, conv_id, conversation_length, context_window_len):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.finished_steps = 0
        self.context_window_len = context_window_len
        self.kvs = deque()

    def is_finished(self):
        return (self.finished_steps >= self.conversation_length)


    

class Disk:
    def __init__(self, size):
        self.disk = [Block(indx) for indx in range(size)] 


class System:
    @staticmethod
    def calculate_theoretical_bw(steps, time_between_steps, step_time_in_gpu, total_gpus):
        """Calculate theoretical bandwidth metrics"""
        total_time_spent_between_steps = (steps + 1) * time_between_steps
        total_time_in_gpu = steps * step_time_in_gpu
        gpu_requests_per_second = total_gpus * (1 / total_time_in_gpu) 
        total_agent_time = total_time_in_gpu + total_time_spent_between_steps
        minimal_agent_max_bw = gpu_requests_per_second * total_agent_time
        return gpu_requests_per_second, minimal_agent_max_bw
    
    def __init__(self, disk_size_in_blocks, steps, allow_holes_recalculation, \
        num_inflight_agents, iterations, random_placement_on_miss, ranges, evict_on_miss, disk, first_conv_id, \
        time_between_steps, total_gpus, step_time_in_gpu, context_window_size, force_hit_ratio, is_shared_storage, is_use_theoretical_agents):
        if disk_size_in_blocks % ranges != 0:
            print(f"Error: disk_size_in_blocks ({disk_size_in_blocks}) must be divisible by ranges ({ranges})")
            exit(1)
        
        if not (0.0 <= force_hit_ratio <= 1.0):
            print(f"Error: force_hit_ratio ({force_hit_ratio}) must be between 0.0 and 1.0")
            exit(1)

        gpu_requests_per_second, minimal_agent_max_bw = self.calculate_theoretical_bw(steps, time_between_steps, step_time_in_gpu, total_gpus) 
        
        self.disk_size_in_blocks = disk_size_in_blocks
        self.allow_holes_recalculation = allow_holes_recalculation
        self.num_inflight_agents = num_inflight_agents if not is_use_theoretical_agents else int(minimal_agent_max_bw)
        self.iterations = iterations
        self.random_placement_on_miss = random_placement_on_miss
        self.ranges = ranges
        self.evict_on_miss = evict_on_miss
        self.range_len = self.disk_size_in_blocks // self.ranges
        self.force_hit_ratio = force_hit_ratio
        

        self.events = []  # Min heap for events
        self.disk = disk
        self.T = 0
        self.misses = 0
        self.hits = 0
        self.event_counter = 0  # Tie-breaker for heap events

        self.conversation_manager = ConversationManager(self, time_between_steps, first_conv_id, steps, context_window_size)
        if is_shared_storage:
            self.gpus = SharedGpus(self, total_gpus, step_time_in_gpu)
        else:
            self.gpus = NonSharedGpus(self, total_gpus, step_time_in_gpu)
        

        
        print("\033[1;33m\n=============================\033[0m")
        print("CACHE PERF ANALISS")        
        print(f"disk_size_in_blocks: {self.disk_size_in_blocks}")
        print(f"allow_holes_recalculation: {self.allow_holes_recalculation}")
        print(f"num_inflight_agents: {self.num_inflight_agents}")
        print(f"iterations: {self.iterations}")
        print(f"random_placement_on_miss: {self.random_placement_on_miss}")
        print(f"ranges: {self.ranges}")
        print(f"evict_on_miss: {self.evict_on_miss}")
        print(f"context_window_size: {context_window_size}")
        print(f"disk to data set ratio: {self.disk_size_in_blocks / (self.num_inflight_agents * steps)}")
        print("BW PERF ANALISS")        
        print(f"total_gpus: {total_gpus}")
        print(f"is_shared_storage: {is_shared_storage}")
        print(f"is_use_theoretical_agents: {is_use_theoretical_agents}")
        print(f"step_time_in_gpu: {step_time_in_gpu}")
        print(f"steps: {steps}")
        print(f"time_between_steps: {time_between_steps}") 
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


    def process_conversation(self, conv):     
        if self.force_hit_ratio and conv.phase_in_step == 0: 
            conv.phase_in_step = len(conv.kvs)
            p_miss = 1 - self.force_hit_ratio
            blocks_to_recalculate = math.ceil(p_miss * len(conv.kvs))
            if blocks_to_recalculate > 0: 
                self.gpus.enter_to_gpus(conv, blocks_to_recalculate)
                return
        while conv.phase_in_step < len(conv.kvs):
            kv, indx_in_conversation = conv.kvs.popleft()
            valid_kv = kv.is_belongs_to(conv.conv_id, indx_in_conversation) 
            conv.disable_all = conv.disable_all or ((not self.allow_holes_recalculation) and (not valid_kv))
            if (not conv.disable_all) and valid_kv:
                self.hits += 1
            else:
                self.misses += 1
            if not valid_kv and self.evict_on_miss:
                kv = self.alloc_block(kv)
                kv.take_ownership(conv.conv_id, indx_in_conversation)
            conv.kvs.append((kv, indx_in_conversation)) 
            conv.phase_in_step += 1
            if not valid_kv:
                self.gpus.enter_to_gpus(conv, 1)
                return
        if not conv.produced_new_block_in_this_step:
            conv.produced_new_block_in_this_step = True
            kv = self.alloc_block(None)
            kv.take_ownership(conv.conv_id, conv.finished_steps)
            conv.kvs.append((kv, conv.finished_steps))
            if len(conv.kvs) > conv.context_window_len:
                conv.kvs.popleft()
            self.gpus.enter_to_gpus(conv, 1)
            return
        conv.finished_steps += 1
        self.conversation_manager.send_conversation_to_sleep(conv)
            
        
    def handle_statistic_event(self):
        self.push_event(self.T + 0.1, {'type': 'stat'})

    
    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))
    

    def simulate(self):
        self.push_event(self.T + 0.1, {'type': 'stat'})
        while self.conversation_manager.finished_conversations < self.iterations:
            t, counter, args = heapq.heappop(self.events)
            self.T = t
            if args['type'] == "stat":
                self.handle_statistic_event()
            elif args['type'] == 'back_from_between_steps_sleep':
                conv = args['conv']
                self.conversation_manager.back_from_sleep(conv)
                self.process_conversation(conv)
            elif args['type'] == "back_from_gpu": #gpu finished
                conv, gpu = args['conv'], args['gpu'] 
                self.gpus.back_from_gpu(conv, gpu)
                self.process_conversation(conv)
            else:
                print(f"Error: Unknown event type '{args['type']}'")
                exit(1)
            while self.conversation_manager.inflights < self.num_inflight_agents:
                self.conversation_manager.create_conversation()

                
        
        # Print cache statistics
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print("CACHE RESULTS")        
        print(f"Cache Hit Rate: {hit_rate:.2f}% ({self.hits}/{total})")
        print(f"Hits: {self.hits}, Misses: {self.misses}")
        print("BW RESULTS")        
        print(f"\033[1;34mSimulation Requests Per Second: {self.iterations / self.T:.8f}\033[0m")
        print("\033[1;33m=============================\033[0m")
        return hit_rate, self.T, self.iterations
        

