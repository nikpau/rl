"""
This heap is specialized to the use with an PER buffer. In my implementation the heap will consist of tuples
that have six entries each. The value imporant for the heap will be the key_idx (sampling priority). Therefore
this heap compares those values to order itself.
"""

class MinHeap:
    def __init__(self, max_size: int, key_idx: int):
        # Use leading zero as parent for whole tree as this will enable integer division later on
        self.heaplist = [0]
        self.max_size = max_size
        self.key_idx = key_idx # key element of the tuple to use as the comparing value by the heap
        
    def perc_up(self, i):
        while i // 2 > 0:
            if self.heaplist[i][self.key_idx] < self.heaplist[i // 2][self.key_idx]: # is item smaller than its parent?
                tmp = self.heaplist[i // 2] # save parent 
                self.heaplist[i // 2] = self.heaplist[i] # replace last item with partent
                self.heaplist[i] = tmp # replace child with former parent
            i = i // 2
            
    def insert(self,k):
        self.heaplist[self.max_size] # Replace last value in heap
        self.perc_up(self.size) # Walk up nodes until the value fits
        
    def perc_down(self,i):
        while (i * 2) <= self.size:
            mc = self.min_child(i) # Determine which child of i is the smallest
            if self.heaplist[i][self.key_idx] > self.heaplist[mc][self.key_idx]:
                tmp = self.heaplist[i]
                self.heaplist[i] = self.heaplist[mc]
                self.heaplist[mc] = tmp
            i = mc
    
    def min_child(self, i): # Determine the index of the smallest child 
        if i * 2 + 1 > self.size:
            return i * 2
        else:
            if self.heaplist[i*2][self.key_idx] < self.heaplist[i*2+1][self.key_idx]:
                return i * 2
            else:
                return i * 2 + 1
            
    def del_min(self):
       retval = self.heaplist[1]
       self.heaplist[1] = self.heaplist[self.size]
       self.size -= 1
       self.heaplist.pop()
       self.perc_down(1)
       return retval

    def build_heap(self,alist):
        i = len(alist) // 2
        self.size = len(alist)
        self.heaplist = [0] + alist[:]
        while i > 0:
            self.perc_down(i)
            i = i - 1
            
h=MinHeap()
a=[(5,88),(3,6),(6,12),(5,7),(3,8)]
h.build_heap(a)
print(h.heaplist)