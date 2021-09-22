'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-22 11:49:02
LastEditTime: 2021-09-22 11:51:02
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/dataloader/__init__.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''
# coding:utf-8
import sys

import time
import collections

from threading import Thread
from queue import Queue
from tqdm import tqdm



class Thread_Killer(object):
    def __init__(self):
        self.to_kill = False
    def __call__(self):
        return self.to_kill
    def set_tokill(self, to_kill):
        self.to_kill = to_kill
    
def thread_batches_feeder(to_kill, batches_queue, dataset_generator):
    while to_kill() == False:
        for step, batch_dict in dataset_generator:
            batches_queue.put((step, batch_dict), block=True)
            if to_kill() == True:
                return 

def thread_cuda_batches(to_kill, cuda_batch_queue, batch_queue):
    while to_kill() == False:
        step, batch_dict = batch_queue.get(block=True)
        batch_dict = {key : torch.from_numpy(val).cuda() for key,val in batch_dict.items()}
        cuda_batch_queue.put((step, batch_dict), block=True)
        if to_kill() == True:
            return

class DataPrefetch(collections.Iterator):
    def __init__(self, data_loader, 
                       cpuQueue_maxSize=20, 
                       cudaQueue_maxSize=10):
        self.data_loader = data_loader
        self.cpu_queue = Queue(maxsize=cpuQueue_maxSize)
        self.cuda_queue = Queue(maxsize=cudaQueue_maxSize)

        slef.prefetch_data()
        time.sleep( cudaQueue_maxSize )
    # 
    def prefetch_data(self):
        cpu_thread_killer = Thread_Killer()
        cpu_thread_killer.set_tokill(False)
        self.reader_thread = Thread(target=thread_batches_feeder, 
                            args=(cpu_thread_killer, self.cpu_queue, self.data_loader))
        self.reader_thread.daemon = True
        self.reader_thread.start()

        cuda_thread_killer = Thread_Killer()
        cuda_thread_killer.set_tokill(False)
        self.cuda_thread = Thread(target=thread_cuda_batches, 
                            args=(cuda_thread_killer, self.cuda_queue, self.cpu_queue))
        self.cuda_thread.daemon = True
        self.cuda_thread.start()

    def next(self):
        yield self.cuda_queue.get(block=True)

    def terminate(self):
        sys.exit()

def main_prefetch_criteo():
    data_path = "/Users/gaoyong/Desktop/low_bit_quantilizer_training/process_criteo/criteo_tfreocd_threshold1/"
    dataset = CriteoData(data_path)
    
    test_loader = dataset.get_data("test", 2048, 100)
    # test_loader = DataPrefetch(test_loader)

    start_time = time.time()
    for step,inputs in tqdm(test_loader, bar_format="{desc}{percentage:3.0f}{r_bar}",
                             total=100, ncols=80, desc="{}.iter".format("test")):
        if step % 10 == 0:
            print("batch: {}; inputs.keys: {} ".format(step, inputs.keys() ))
        if step >= 10:
            break
    
    end_time = time.time()
    print("read_data.time: {} ".format(end_time - start_time))

    sys.exit()


if __name__ == "__main__":
    
    main_prefetch_criteo()
