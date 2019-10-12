from sortedcontainers import SortedDict, SortedList

__all__ = ["Empty", "Queue"]


class Empty(KeyError):
    pass


class Queue:
    """Priority queue supporting update and removal at arbitrary position.

    Parameters
    ----------
    entries : iterable of (item, priority)
        The initial data in the queue. Providing this is faster than
        calling 'insert' a bunch of times.
    """

    def __init__(self, entries=()):
        self._queue = SortedDict(
            ((priority, -n), item) for n, (item, priority) in enumerate(entries)
        )
        # 'self._queue' cannot be keyed only on priority, as there may be several
        # items that have the same priority. To keep unique elements the key
        # will be '(priority, self._n)', where 'self._n' is decremented whenever
        # we add a new element. 'self._n' is negative so that elements with equal
        # priority are sorted by insertion order.
        self._n = -len(self._queue)
        # To efficiently support updating and removing items if their priority
        # is unknown we have to keep the reverse map of 'self._queue'. Because
        # items may not be hashable we cannot use a SortedDict, so we use a
        # SortedList storing '(item, key)'.
        self._items = SortedList(((v, k) for k, v in self._queue.items()))

    def __len__(self):
        return len(self._queue)

    def items(self):
        "Return an iterator over the items in the queue in arbitrary order."
        return reversed(self._queue.values())

    def peek(self):
        "Return the item and priority at the front of the queue."
        self._check_nonempty()
        ((priority, _), item) = self._queue.peekitem()
        return item, priority

    def pop(self):
        "Remove and return the item and priority at the front of the queue."
        self._check_nonempty()
        (key, item) = self._queue.popitem()
        i = self._items.index((item, key))
        del self._items[i]
        priority, _ = key
        return item, priority

    def insert(self, item, priority):
        "Insert 'item' into the queue with the given priority."
        key = (priority, self._n)
        self._items.add((item, key))
        self._queue[key] = item
        self._n -= 1

    def _check_nonempty(self):
        if not self._queue:
            raise Empty()

    def _find_first(self, item):
        self._check_nonempty()
        i = self._items.bisect_left((item, ()))
        try:
            should_be, key = self._items[i]
        except IndexError:
            raise KeyError("item is not in queue")
        if item != should_be:
            raise KeyError("item is not in queue")
        return i, key

    def remove(self, item):
        "Remove the 'item' from the queue."
        i, key = self._find_first(item)
        del self._queue[key]
        del self._items[i]

    def update(self, item, priority):
        """Update 'item' in the queue to have the given priority.

        Raises
        ------
        KeyError : if 'item' is not in the queue.
        """
        i, key = self._find_first(item)
        _, n = key
        new_key = (priority, n)

        del self._queue[key]
        self._queue[new_key] = item
        del self._items[i]
        self._items.add((item, new_key))
