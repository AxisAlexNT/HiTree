from typing import List
from common import ATUDescriptor
from readerwriterlock import rwlock
import threading


class ATUHolder(object):
    atus: List[ATUDescriptor]
    holder_lock: rwlock.RWLockWrite

    def __init__(self) -> None:
        self.atus = []
        self.holder_lock = rwlock.RWLockWrite(threading.RLock)

    def __getitem__(self, arg1):
        with self.holder_lock.gen_rlock():
            return self.atus.__getitem__(arg1)

    def __setitem__(self, arg1, arg2):
        raise Exception(
            "Direct assignment to ATU Holder is not allowed, please use provided operation-specific methods instead"
        )

    def __delitem__(self, arg1):
        raise Exception(
            "Direct deletion from ATU Holder is not allowed, please use provided operation-specific methods instead"
        )
