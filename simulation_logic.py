#!/usr/bin/env python3

import heapq
import random
import math
from collections import deque, defaultdict, Counter
from abc import ABC, abstractmethod
from enum import Enum
from functools import partial


class IOType(Enum):
    READ = 0
    WRITE = 1

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
            
class Disk:
    def __init__(self, size):
        self.disk = [Block(indx) for indx in range(size)] 

class Conversation:
    def __init__(self, conv_id, conversation_length, context_window_len):
        self.conv_id = conv_id
        self.conversation_length = conversation_length 
        self.finished_steps = 0
        self.context_window_len = context_window_len
        self.kvs = deque()


class ConversationManager:
    def __init__(self, system, sleep_time_between_steps, first_conv_id, steps, context_window_size):
        self.sleep_time_between_steps = sleep_time_between_steps
        self.system = system
        self.conv_id = first_conv_id
        self.steps = steps
        self.context_window_size = context_window_size

    def get_unique_id(self):
        self.conv_id += 1
        return self.conv_id
    
    def create_conversation(self):
        conv = Conversation(self.get_unique_id(), self.steps, self.context_window_size)
        return conv

class async_server(ABC):
    def __init__(self, system, num_servers, serve_time):
        self.system = system
        self.num_servers = num_servers
        self.serve_time = serve_time
        self.num_completed = 0
        self.real_works = 0
        self.works = 0
        self.total_queued = 0
        self.waiters_queue = deque()  # FIFO: append on block, popleft on wake
    
    @abstractmethod
    def _enter(self, uid, works, is_real_work):
        pass
    
    def get_busy_reallyBusy_totalQueued(self):
        return self.works, self.real_works, self.total_queued

    def enter(self, uid, works = 1, is_real_work = True):
        self.total_queued += 1
        yield from self._enter(uid, works, is_real_work)
        self.num_completed += 1

class MMC(async_server):
    def __init__(self, system, num_servers, serve_time, father = None):
        super().__init__(system, num_servers, serve_time)
        self.free_server_slots = num_servers
        self.father = father

    def _enter(self, uid, works, is_real_work):
        #import ipdb; ipdb.set_trace()
        base_queue = self if not self.father else self.father
        if (self.free_server_slots <= 0) or (len(self.waiters_queue) > 0):
            yield (0, self)
            #import ipdb; ipdb.set_trace()
        self.free_server_slots -= 1
        base_queue.total_queued -= 1
        if is_real_work:
            base_queue.real_works += 1
        base_queue.works += 1
        yield (works * (random.expovariate(1.0 / self.serve_time)), self) 
        #import ipdb; ipdb.set_trace()
        if is_real_work:
            base_queue.real_works -= 1
        base_queue.works -= 1  
        self.free_server_slots += 1



class C_MM1(async_server):    
    def __init__(self, system, num_servers, serve_time, scheduling_strategy):
        super().__init__(system, num_servers, serve_time)
        self.scheduling_strategy = scheduling_strategy
        self.servers = [MMC(self.system, 1, self.serve_time, self) for _ in range(self.num_servers)]
    
    def get_least_busy_server(self):
        def load(s):
            busy = s.num_servers - s.free_server_slots
            return busy + len(s.waiters_queue)
        server = min(self.servers, key=lambda s: (load(s), self.servers.index(s)))
        return server, self.servers.index(server)
        
    def _enter(self, uid, works, is_real_work):
        if self.scheduling_strategy == "local_storage_sticky":
            server_idx = random.randrange(len(self.servers))
            server = self.servers[server_idx]
        elif self.scheduling_strategy == "local_storage_least_busy":
            server, server_idx = self.get_least_busy_server()
        else:
            raise ValueError(f"ERROR: unexpected scheduling strategy: {self.scheduling_strategy}")
        yield from server._enter(uid, works, is_real_work)

class System:
    def calculate_theoretical_bw(self):
        """Calculate theoretical bandwidth metrics"""
        gpu_max_possible_agents_per_second = self.params['total_gpus'] / (
            self.params['steps'] * self.params['step_time_in_gpu']
        )
        minimal_inflight_agents_for_max_possible_agents_per_second = self.params['total_gpus'] * (
            1 + self.params['time_between_steps'] / self.params['step_time_in_gpu']
        )
        return gpu_max_possible_agents_per_second, minimal_inflight_agents_for_max_possible_agents_per_second

    def __init__(self, params):
        self.params = params
        self.gpu_max_possible_agents_per_second, self.minimal_inflight_agents_for_max_possible_agents_per_second = self.calculate_theoretical_bw()
        self.num_inflight_agents = self.params['num_inflight_agents'] if not self.params['is_use_theoretical_agents'] else int(self.minimal_inflight_agents_for_max_possible_agents_per_second)
        self.range_len = self.params['disk_size_in_blocks'] // self.params['ranges']

        print("System: constructing (this may take a while for large disk_size_in_blocks)...")
        self.disk = Disk(self.params['disk_size_in_blocks'])

        self.events = []  # Min heap for events
        self.T = 0
        self.event_counter = 0  # Tie-breaker for heap events


        self.agent_outside_service = MMC(self, self.num_inflight_agents, self.params['time_between_steps'])
        if self.params['scheduling_strategy'] == "shared_storage_least_busy":
            self.gpus = MMC(self, self.params['total_gpus'], self.params['step_time_in_gpu'])
        else:
            self.gpus = C_MM1(self, self.params['total_gpus'], self.params['step_time_in_gpu'], self.params['scheduling_strategy'])
        self.storage = MMC(self, 1, 1 / self.params['storage_blocks_per_second']) if self.params['storage_blocks_per_second'] else None
        self.conversation_manager = ConversationManager(self, self.params['time_between_steps'], 0, self.params['steps'], self.params['context_window_size'])

        self.completed_conversations = 0
        self.inflight_conversation_count = 0
        self.misses = 0
        self.hits = 0
        self.total_busy_servers = 0
        self.total_really_busy_servers = 0
        self.total_queue_len = 0
        self.total_sleepers = 0
        self.sample_count = 0

    def alloc_block(self, block):
        prev_range_idx = block.offset // self.range_len if block else -1
        while True:
            range_idx = random.randrange(self.params['ranges'])
            if range_idx != prev_range_idx or self.params['ranges'] == 1:
                break
        offset_in_range = random.randrange(self.range_len) if ((not block) or (self.params['random_placement_on_miss'])) else (block.offset % self.range_len)
        block_offset = range_idx * self.range_len + offset_in_range 
        return self.disk.disk[block_offset] 

    def async_handle_conversation(self, conv):
        for step in range(conv.conversation_length):
            #print(f"\033[31mconversation with id {conv.conv_id} starts step {step}\033[0m")
            disable_all, to_read, to_calc = False, 0, 0
            for _ in range(len(conv.kvs)):
                kv, indx_in_conversation = conv.kvs.popleft()
                valid_kv = kv.is_belongs_to(conv.conv_id, indx_in_conversation) if not self.params['force_hit_ratio'] else (random.random() < self.params['force_hit_ratio'])
                disable_all = disable_all or ((not self.params['allow_holes_recalculation']) and (not valid_kv))
                if (not disable_all) and (valid_kv):
                    self.hits += 1
                else:
                    self.misses += 1
                if not valid_kv and self.params['evict_on_miss']:
                    kv = self.alloc_block(kv)
                    kv.take_ownership(conv.conv_id, indx_in_conversation)
                conv.kvs.append((kv, indx_in_conversation)) 
                if not valid_kv:
                    to_calc += 1
                elif self.storage:
                    to_read += 1       
            if self.storage and to_read > 0:
                yield from self.storage.enter(conv.conv_id, to_read)
            if self.params['scheduling_strategy'] == 'local_storage_least_busy':
                _, server_idx = self.gpus.get_least_busy_server() 
                if (server_idx != (conv.conv_id % self.gpus.num_servers)) and (step > 0):
                    to_calc = len(conv.kvs)
            if self.gpus and to_calc > 0:
                yield from self.gpus.enter(conv.conv_id, works=to_calc, is_real_work=False)
            kv = self.alloc_block(None)
            kv.take_ownership(conv.conv_id, step)
            conv.kvs.append((kv, step))
            if len(conv.kvs) > conv.context_window_len:
                conv.kvs.popleft()
            yield from self.gpus.enter(conv.conv_id)    
            yield from self.agent_outside_service.enter(conv.conv_id)
        
    
    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))

    def advance_handler(self, handler):
        res = next(handler, None)
        if res is None:
            self.inflight_conversation_count -= 1
            self.completed_conversations += 1
        else:
            sleep_time, curr_server = res[0], res[1]
            if sleep_time == 0:
                curr_server.waiters_queue.append(handler)
            else:
                self.push_event(self.T + sleep_time, {'type':'handler', 'func':handler, 'async_server' : curr_server})


    def simulate(self):
        if self.params['monitor_interval_virtual_time'] > 0:
            self.push_event(self.params['monitor_interval_virtual_time'], {'type':'stat'})
        while self.completed_conversations < self.params['iterations']:
            while self.inflight_conversation_count < self.num_inflight_agents:
                conv = self.conversation_manager.create_conversation() 
                self.inflight_conversation_count += 1
                handler = self.async_handle_conversation(conv)
                self.push_event(self.T, {'type':'handler', 'func':handler, 'async_server' : None})
            t, counter, event_dict = heapq.heappop(self.events) 
            self.T = t
            if event_dict['type'] == 'stat':
                self.monitor_gpus()
                if self.params['monitor_interval_virtual_time'] > 0:
                    self.push_event(self.T + self.params['monitor_interval_virtual_time'], {'type':'stat'})
            elif event_dict['type'] == 'handler':
                srv = event_dict['async_server']
                self.advance_handler(event_dict['func'])
                if srv is not None and len(srv.waiters_queue) > 0:
                    self.advance_handler(srv.waiters_queue.popleft())
            else:
                raise ValueError(f"UNKNOWN EVENT TYPE: {event_dict['type']!r}")

    def monitor_gpus(self):
        busy, really_busy, total_queued = self.gpus.get_busy_reallyBusy_totalQueued()
        self.total_busy_servers += busy
        self.total_really_busy_servers += really_busy
        self.total_queue_len += total_queued
        sleepers = self.agent_outside_service.num_servers - self.agent_outside_service.free_server_slots
        self.total_sleepers += sleepers
        self.sample_count += 1 
        
        curr_busy_ratio = busy / self.gpus.num_servers
        curr_really_busy_ratio = really_busy / self.gpus.num_servers
        avg_busy_ratio = self.total_busy_servers / self.sample_count / self.gpus.num_servers 
        avg_really_busy_ratio = self.total_really_busy_servers / self.sample_count / self.gpus.num_servers
        avg_queue_len = self.total_queue_len / self.sample_count 
        avg_sleepers = self.total_sleepers / self.sample_count



        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        print(f"T={self.T:.2f} - Curr: Busy={curr_busy_ratio:.4f}, ReallyBusy={curr_really_busy_ratio:.4f}, Queue={total_queued:.0f}, Sleepers={sleepers:.0f}")
        print(f"T={self.T:.2f} - Avg:  Busy={avg_busy_ratio:.4f}, ReallyBusy={avg_really_busy_ratio:.4f}, Queue={avg_queue_len:.2f}, Sleepers={avg_sleepers:.2f}, HitRate={hit_rate:.8f}%, Completed={self.completed_conversations}")
        print("-" * 80)