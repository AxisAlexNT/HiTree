#  MIT License
#
#  Copyright (c) 2021-2024. Aleksandr Serdiukov, Anton Zamyatin, Aleksandr Sinitsyn, Vitalii Dravgelis and Computer Technologies Laboratory ITMO University team.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
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
            return self.version

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
