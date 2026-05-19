#!/usr/bin/env python3

import heapq
from locale import locale_alias
import random
import math
from collections import deque, defaultdict, Counter
from abc import ABC, abstractmethod
from typing import Any
from enum import Enum
from functools import partial


class IOType(Enum):
    READ = 0
    WRITE = 1

class Waitable:
    def __init__(self):
        self.refcnt = 0

    def add_waiter(self):
        self.refcnt += 1

    def remove_waiter(self):
        self.refcnt -= 1

    def all_waited(self):
        return self.refcnt == 0

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
        self.real_works = 0
        self.works = 0
        self.total_queued = 0
        self.waiters_queue = deque()  # FIFO: append on block, popleft on wake
    

    @abstractmethod
    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        pass

    def get_busy_reallyBusy_totalQueued(self):
        return self.works, self.real_works, self.total_queued

    def async_enter(self,handler,  waitable, uid, works=1, is_real_work=True):
        return self._async_enter(handler, waitable, uid, works, is_real_work)

class MMC(async_server):
    def __init__(self, system, num_servers, serve_time, father = None):
        super().__init__(system, num_servers, serve_time)
        self.free_server_slots = num_servers
        self.father = father
    
    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        serve_time = works * (random.expovariate(1.0 / self.serve_time))
        self.system.add_work_to_system(handler, self, serve_time, is_real_work, waitable)
    
    def enter_queue(self, handler, serve_time, is_real_work, waitable):
        self.total_queued += 1
        self.waiters_queue.append((handler, serve_time, is_real_work, waitable))
    
    def exit_queue(self):
        self.total_queued -= 1
        return self.waiters_queue.popleft()
    
    def enter_service(self, is_real_work):
        base_queue = self if not self.father else self.father
        self.free_server_slots -= 1
        if is_real_work:
            base_queue.real_works += 1
        base_queue.works += 1
    
    def exit_service(self, is_real_work):
        base_queue = self if not self.father else self.father
        self.free_server_slots += 1
        if is_real_work:
            base_queue.real_works -= 1
        base_queue.works -= 1 


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

    def _async_enter(self, handler, waitable, uid, works, is_real_work):
        if self.scheduling_strategy == "local_storage_sticky":
            server_idx = random.randrange(len(self.servers))
            server = self.servers[server_idx]
        elif self.scheduling_strategy == "local_storage_least_busy":
            server, server_idx = self.get_least_busy_server()
        else:
            raise ValueError(f"ERROR: unexpected scheduling strategy: {self.scheduling_strategy}")
        server._async_enter(handler, waitable, uid, works, is_real_work) 

class Tier(C_MM1):
    def __init__(self, system, num_queues, block_serve_time, num_blocks):
        super().__init__(system, num_queues, block_serve_time, "local_storage_sticky")
        self.num_blocks = num_blocks
        self.keys_map = set()

    @abstractmethod
    def write(self, key) -> tuple[bool, Any]:
        pass

    @abstractmethod
    def remove(self, key):
        pass

    def has_key(self, key):
        return key in self.keys_map

    def read(self, key):
        yield from self.enter(key)


class LRU_Tier(Tier):
    def __init__(self, system, num_queues, block_serve_time, num_blocks):
        super().__init__(system, num_queues, block_serve_time, num_blocks)
        self._queue = deque()

    def write(self, key) -> tuple[bool, Any]:
        if key in self.keys_map:
            return True, None
        evicted = None
        if len(self.keys_map) >= self.num_blocks:
            evicted = self._queue.popleft()
            self.keys_map.discard(evicted)
        self.keys_map.add(key)
        self._queue.append(key)
        return True, evicted

    def remove(self, key):
        pass

class NON_EVICTABLE_Tier(Tier):
    def __init__(self, system, num_queues, block_serve_time, num_blocks):
        super().__init__(system, num_queues, block_serve_time, num_blocks)

    def write(self, key) -> tuple[bool, Any]:
        if key in self.keys_map:
            return True, None
        if len(self.keys_map) >= self.num_blocks:
            return False, None
        self.keys_map.add(key)
        return True, None

    def remove(self, key):
        self.keys_map.discard(key)

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
        self.storage = MMC(self, 1, 1 / self.params['storage_blocks_per_second']) if (self.params['storage_blocks_per_second'] > 0) else None
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

    def async_handle_conversation(self, d):
        conv, handler = d['conv'], d['handler']
        waitable = Waitable()
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
                self.storage.async_enter(handler, waitable, conv.conv_id, works=to_read)
                yield
            if self.params['scheduling_strategy'] == 'local_storage_least_busy':
                _, server_idx = self.gpus.get_least_busy_server() 
                if (server_idx != (conv.conv_id % self.gpus.num_servers)) and (step > 0):
                    to_calc = len(conv.kvs)
            if self.gpus and to_calc > 0:
                self.gpus.async_enter(handler, waitable, conv.conv_id, works=to_calc, is_real_work=False) 
                yield
            kv = self.alloc_block(None)
            kv.take_ownership(conv.conv_id, step)
            conv.kvs.append((kv, step))
            if len(conv.kvs) > conv.context_window_len:
                conv.kvs.popleft()
            self.gpus.async_enter(handler, waitable, conv.conv_id, works=1, is_real_work=True) 
            yield     
            self.agent_outside_service.async_enter(handler, waitable, conv.conv_id, works=1) 
            yield

    def push_event(self, time, event_dict):
        """Helper function to push events to heap with tie-breaker counter."""
        self.event_counter += 1
        heapq.heappush(self.events, (time, self.event_counter, event_dict))

    def add_work_to_system(self, handler, mmc, serve_time, is_real_work, waitable):
        waitable.add_waiter()
        if (mmc.free_server_slots <= 0) or (len(mmc.waiters_queue) > 0):
            mmc.enter_queue(handler, serve_time, is_real_work, waitable)
        else:
            mmc.enter_service(is_real_work)
            self.push_event(self.T + serve_time, {'type':'handler', 'func':handler, 'mmc' : mmc, 'is_real_work' : is_real_work, 'waitable': waitable})


    def process_completion_event(self, event_dict):
        mmc = event_dict['mmc']
        mmc.exit_service(event_dict['is_real_work'])
        if len(mmc.waiters_queue) > 0:
            (handler, serve_time, is_real_work, waitable) = mmc.exit_queue()
            mmc.enter_service(is_real_work)
            self.push_event(self.T + serve_time, {'type':'handler', 'func':handler, 'mmc' : mmc, 'is_real_work' : is_real_work, 'waitable': waitable}) 
        waitable_of_completed  = event_dict['waitable']
        waitable_of_completed.remove_waiter()
        if waitable_of_completed.all_waited():
            try:
                next(event_dict['func'])
            except StopIteration:
                self.inflight_conversation_count -= 1
                self.completed_conversations += 1
             
    def kick_off_new_conversation(self):
        conv = self.conversation_manager.create_conversation() 
        self.inflight_conversation_count += 1
        d = {'conv':conv}
        handler = self.async_handle_conversation(d)
        d['handler'] = handler
        next(handler)


    def simulate(self):
        if self.params['monitor_interval_virtual_time'] > 0:
            self.push_event(self.params['monitor_interval_virtual_time'], {'type':'stat'})
        while self.completed_conversations < self.params['iterations']:
            while self.inflight_conversation_count < self.num_inflight_agents:
                self.kick_off_new_conversation()
            t, counter, event_dict = heapq.heappop(self.events) 
            self.T = t
            if event_dict['type'] == 'stat':
                self.monitor_gpus()
            elif event_dict['type'] == 'handler':
                self.process_completion_event(event_dict)
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
        if self.params['monitor_interval_virtual_time'] > 0:
            self.push_event(self.T + self.params['monitor_interval_virtual_time'], {'type':'stat'})