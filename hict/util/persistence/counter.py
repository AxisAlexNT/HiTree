from typing import Union
from threading import RLock as ThRLock
from multiprocessing import Lock as MpLock
from readerwriterlock import rwlock
import numpy as np


class AtomicVersionCounter:
    base_lock: Union[ThRLock, MpLock]
    version: np.int64
    lock: rwlock.RWLockWrite

    def __init__(self, base_lock: Union[ThRLock, MpLock], version: np.int64 = np.int64(0)):
        def get_lock():
            return base_lock
        self.lock = rwlock.RWLockWrite(lock_factory=get_lock)
        self.version = version

    def get(self) -> np.int64:
        with self.lock.gen_rlock():
            return version

    def cas(self, expected: np.int64, target: np.int64) -> bool:
        with self.lock.gen_wlock():
            if self.version == expected:
                self.version = target
                return True
            else:
                return False
            
    def getAndIncrement(self) -> np.int64:
        with self.lock.gen_wlock():
            old_version = self.version
            self.version += 1
            return old_version
        
    def incrementAndGet(self) -> np.int64:
        with self.lock.gen_wlock():
            self.version += 1
            return self.version
