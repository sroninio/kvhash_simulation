#!/usr/bin/env python3




import argparse
import random
import math
from collections import defaultdict


class Conversation:
    def __init__(self):
        self.uses = 0


class Simulation:
    def __init__(self, blocks_in_storage, inflight_conversations, conversation_length, total_conversations, ranges) -> None:
        if blocks_in_storage % ranges != 0:
            print(f"Error: blocks_in_storage ({blocks_in_storage}) must be divisible by ranges ({ranges})")
            exit(1)
        
        self.blocks_in_storage = blocks_in_storage
        self.inflight_conversations = inflight_conversations
        self.conversation_length = conversation_length
        self.total_conversations = total_conversations
        self.ranges = ranges

        self.inflights = dict() #convid->conversation
        self.conversationid_to_blocks = defaultdict(lambda: set()) #convid->block
        self.block_to_conversationid = defaultdict(lambda: -1) #block->convid
        self.hit_miss = {'hit' : 0, 'miss' :0}

        self.range_len = self.blocks_in_storage // self.ranges
        self.num_writes = 0
        self.consecutive_writes_in_range = math.ceil(self.inflight_conversations / self.ranges)
        self.next_conv_id = 0

    
    def touch_conversation(self, conv_id):
        conversation = self.inflights[conv_id]
        if (conversation.uses > 0) and (len(self.inflights) == self.inflight_conversations):
           self.hit_miss['hit' if len(self.conversationid_to_blocks[conv_id]) > 0 else 'miss'] += 1  
        if len(self.conversationid_to_blocks[conv_id]) == 0: 
            curr_range = (self.num_writes // self.consecutive_writes_in_range) % self.ranges
            block = curr_range * self.range_len + random.randrange(self.range_len)
            #print(f"Writing to range {curr_range} block {block}")
            self.conversationid_to_blocks[self.block_to_conversationid[block]].discard(block)
            self.conversationid_to_blocks[conv_id].add(block) 
            self.block_to_conversationid[block] = conv_id
            self.num_writes += 1
        conversation.uses += 1
    
    def add_new_conversation(self):
        self.inflights[self.next_conv_id] = Conversation() 
        self.touch_conversation(self.next_conv_id)
        self.next_conv_id += 1
        return self.next_conv_id - 1

    def run(self):
        finished_conversations = 0
        while finished_conversations < self.total_conversations:
            if len(self.inflights) < self.inflight_conversations:
                conv_id = self.add_new_conversation()
            else:
                conv_id = random.choice(list(self.inflights.keys())) 
                self.touch_conversation(conv_id)
            if self.inflights[conv_id].uses >= self.conversation_length:
                finished_conversations += 1
                del self.inflights[conv_id]
        print (self.hit_miss)   
            

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
    s.run()

