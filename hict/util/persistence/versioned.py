from typing import TypeVar, Generic, Dict, Optional, Tuple, List, Callable, NamedTuple
import threading
import multiprocessing
import multiprocessing.managers
from readerwriterlock import rwlock
from hict.util.persistence.exceptions import OutdatedVersionException
np.seterr(all='raise')


T = TypeVar('T')


class PersistenceBucket(Generic[T]):
    value: T
    lock: Union[threading.RLock, multiprocessing.managers.SyncManager.RLock]

    def __init__(self, value, lock):
        self.value = value
        self.lock = lock


class Versioned(Generic[T]):
    version_lock: rwlock.RWLockWriteD
    current_version: np.int64
    version_dict: Dict[np.int64, PersistenceBucket[T]]

    def __init__(self, object: T, mp_manager: Optional[multiprocessing.managers.SyncManager] = None, versions_to_keep: np.int64 = 4):
        self.mp_manager = mp_manager
        if mp_manager is not None:
            lock_factory = mp_manager.RLock
        else:
            lock_factory = threading.RLock
        self.version_lock: rwlock.RWLockWriteD = rwlock.RWLockWriteD(
            lock_factory=lock_factory
        )
        self.current_version = np.int64(0)
        self.version_dict = dict()
        self.versions_to_keep = versions_to_keep
        self.version_dict[np.int64(0)] = PersistenceBucket(
            object,
            lock_factory()
        )

    def remove_old_versions(self) -> None:
        with self.version_lock.gen_wlock():
            threshold = self.current_version - self.versions_to_keep
            for k, v in self.version_dict.items():
                if k < threshold:
                    lock: threading.RLock = v.lock
                    try:
                        lock.acquire(blocking=False)
                        del self.version_dict[k]
                        del v.value
                    finally:
                        lock.release()

    def bump_version(self, target_version: np.int64) -> None:
        with self.version_lock.gen_wlock():
            delta = target_version - self.current_version
            assert delta > 0
            for k in sorted(self.version_dict.keys(), reverse=True):
                self.version_dict[k + delta] = self.version_dict[k]
                del self.version_dict[k]

    def get_version(self, version: np.int64) -> PersistenceBucket[T]:
        lock = self.version_lock.gen_wlock()
        if lock.acquire():
            try:
                if self.current_version < version:
                    self.bump_version(version)
                lock.downgrade()
                if version in self.version_dict.keys():
                    return self.version_dict[version]
            finally:
                lock.release()
        raise VersionNotPresentException()

    def new_version(self, version: np.int64) -> PersistenceBucket[T]:
        with self.version_lock.gen_wlock():
            new_version = copy.deepcopy(
                self.version_dict[self.current_version]
            )
            self.current_version += 1
            self.version_dict[self.current_version] = new_version
            self.remove_old_versions()
