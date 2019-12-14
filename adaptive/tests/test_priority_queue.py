from hypothesis import given, assume
import hypothesis.strategies as st
import pytest

from adaptive.priority_queue import Queue, Empty


item = st.floats(allow_nan=False)
priority = st.floats(allow_nan=False)
items = st.lists(st.tuples(item, priority))


@given(items, item)
def test_remove_nonexisting_item_raises(items, missing_item):
    if items:
        i, p = zip(*items)
        assume(missing_item not in i)
    q = Queue(items)
    with pytest.raises(KeyError):
        q.remove(missing_item)


@given(items, item)
def test_update_nonexisting_item_raises(items, missing_item):
    if items:
        i, p = zip(*items)
        assume(missing_item not in i)
    q = Queue(items)
    with pytest.raises(KeyError):
        q.update(missing_item, 0)


@given(items, item)
def test_remove_item_inserted_twice_removes_lowest_priority(items, missing_item):
    if items:
        i, p = zip(*items)
        assume(missing_item not in i)
    q = Queue(items)

    q.insert(missing_item, 0)
    q.insert(missing_item, 1)
    q.remove(missing_item)  # should remove priority 0 item
    # Get 'missing_item' out of the queue
    t = None
    while t != missing_item:
        t, prio = q.pop()
    assert prio == 1


@given(items)
def test_all_items_in_queue(items):
    # Items should be sorted from largest priority to smallest
    sorted_items = [item for item, _ in sorted(items, key=lambda x: -x[1])]
    assert sorted_items == list(Queue(items).items())


@given(items)
def test_pop_gives_max(items):
    q = Queue(items)
    if items:
        lq = len(q)
        should_pop = max(items, key=lambda x: x[1])
        assert should_pop == q.pop()
        assert len(q) == lq - 1
    else:
        with pytest.raises(Empty):
            q.pop()


@given(items)
def test_peek_gives_max(items):
    q = Queue(items)
    if items:
        lq = len(q)
        should_peek = max(items, key=lambda x: x[1])
        assert should_peek == q.peek()
        assert len(q) == lq
    else:
        with pytest.raises(Empty):
            q.peek()
