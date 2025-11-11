#!/usr/bin/env python3
import argparse
import random
import math
from collections import defaultdict
from collections import deque



class Conversation:
    def __init__(self, conv_id):
        self.uses = 0
        self.conv_id = conv_id



class Simulation:
    def __init__(self, blocks_in_storage, inflight_conversations, conversation_length, total_conversations, ranges) -> None:
        if blocks_in_storage % ranges != 0:
            print(f"Error: blocks_in_storage ({blocks_in_storage}) must be divisible by ranges ({ranges})")
            #exit(1)
        
        self.blocks_in_storage = blocks_in_storage
        self.inflight_conversations = inflight_conversations
        self.conversation_length = conversation_length
        self.total_conversations = total_conversations
        self.ranges = ranges
        #import ipdb;ipdb.set_trace()

        self.inflights = deque() 
        self.conversationid_to_blocks = {}
        self.block_to_conversationid = {}
        self.hit_miss = {'hit' : 0, 'miss' :0}

        self.range_len = self.blocks_in_storage // self.ranges
        self.num_writes = 0
        self.consecutive_writes_in_range = math.ceil(self.inflight_conversations / self.ranges)
        self.next_conv_id = 0

    
    def touch_conversation(self, conversation):
        conv_id = conversation.conv_id
        #print (f"Touching {conv_id}, use = {conversation.uses} conv_to_block = {self.conversationid_to_blocks}, block_to_conv = {self.block_to_conversationid}")
        if (conversation.uses > 0):
            self.hit_miss['hit' if conv_id in self.conversationid_to_blocks else 'miss'] += 1  
        if not conv_id in self.conversationid_to_blocks:          
            curr_range = (self.num_writes // self.consecutive_writes_in_range) % self.ranges
            block = curr_range * self.range_len + random.randrange(self.range_len)
            if block in self.block_to_conversationid:
                conv_owner = self.block_to_conversationid[block] 
                del self.conversationid_to_blocks[conv_owner] 
            self.conversationid_to_blocks[conv_id] = block
            self.block_to_conversationid[block] = conv_id 
            self.num_writes += 1
        conversation.uses += 1

    def run(self):
        #import ipdb;ipdb.set_trace()
        finished_conversations = 0
        while finished_conversations < self.total_conversations:
            if len(self.inflights) < self.inflight_conversations:
                c = Conversation(self.next_conv_id) 
                self.next_conv_id += 1 
            else:
                c = self.inflights.popleft()
                self.touch_conversation(c)
            if c.uses < self.conversation_length:
                self.inflights.append(c)
            else:
                finished_conversations += 1
                #if (finished_conversations % 1000) == 0:
                    #print(finished_conversations)
        total = self.hit_miss['hit'] + self.hit_miss['miss']
        hit_rate = (self.hit_miss['hit'] / total * 100) if total > 0 else 0
        #print(f"Hit Rate: {hit_rate:.2f}% ({self.hit_miss['hit']}/{total})")
        #print(f"Hits: {self.hit_miss['hit']}, Misses: {self.hit_miss['miss']}")   
        return hit_rate
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate KV cache behavior with configurable parameters"
    )
    parser.add_argument(
        "--blocks_in_storage",
        type=int,
        required=True,
        help="Number of blocks available in storage"
    )   
    parser.add_argument(
        "--inflight_conversations",
        type=int,
        required=True,
        help="each conversation is a single block, this paramter says how many inflight conversations are"
    )
    parser.add_argument(
        "--conversation_length",
        type=int,
        required=True,
        help="how much folow up questions are in one conversation"
    ) 
    parser.add_argument(
        "--total_conversations",
        type=int,
        required=True,
        help="how many conversations in total to process"
    ) 
    parser.add_argument(
        "--ranges",
        type=int,
        required=True,
        help="num of ranges to split the block storage"
    )
    
    args = parser.parse_args()

    s = Simulation(blocks_in_storage=args.blocks_in_storage, inflight_conversations=args.inflight_conversations, conversation_length=args.conversation_length, total_conversations=args.total_conversations, ranges=args.ranges)
    print (s.run())
    '''
    for inflights in [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000, 11000, 20000]:
        res = []
        for ranges in range(1,10):
            res.append(Simulation(blocks_in_storage=10000, inflight_conversations=inflights, conversation_length=4, total_conversations=100000, ranges=ranges).run())
        print(f"inflights = {inflights} results = {res}")
    '''
    

