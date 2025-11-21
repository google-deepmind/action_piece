# Bug Fix: AssertionError in Vocabulary Construction

**Date**: 2025-11-21
**Severity**: Critical
**Status**: Fixed
**Affected Component**: `genrec/models/ActionPiece/core.py`

---

## Problem Summary

During vocabulary construction in ActionPiece tokenizer, the system encounters an `AssertionError` when trying to add a `head_id` to `pair2head_ids` that already exists in the set. This indicates a data consistency bug between two tracking data structures: `head_id2pair_cnt` and `pair2head_ids`.

## Error Message

```
File "/scratch/zl4789/action_piece_google/genrec/models/ActionPiece/core.py", line 522, in _update_pair2head_ids
    head_id not in self.pair2head_ids[pair]
AssertionError: head_id 150899 already in pair2head_ids[(1414, 1587)]
```

## Root Cause Analysis

### The Faulty Function

The bug originates from the `diff_cnt()` function at line 31 in `core.py`:

```python
def diff_cnt(cnt1, cnt2):
  """Minus the second pair2cnt from the first pair2cnt."""
  return {k: v - cnt2.get(k, 0) for k, v in cnt1.items()}  # ❌ Bug here
```

**Problem**: This implementation only iterates over keys present in `cnt1` (new counts). If a token pair exists in `cnt2` (old counts) but disappears in `cnt1` (count becomes 0), it will **not** appear in the returned difference dictionary.

### How It Breaks the System

The vocabulary construction process maintains two critical data structures:

1. **`head_id2pair_cnt[head_id][pair]`**: Maps each sequence (head_id) to its token pair counts
2. **`pair2head_ids[pair]`**: Inverted index mapping each pair to the set of sequences containing it

These must stay synchronized. The `_update_pair2head_ids()` method is responsible for this synchronization:

```python
def _update_pair2head_ids(self, diff_pair2cnt, head_id):
  for pair in diff_pair2cnt:
    if (
        diff_pair2cnt[pair] > 0
        and abs(self.head_id2pair_cnt[head_id][pair]) < self.eps
    ):
      # New pair after merging
      assert head_id not in self.pair2head_ids[pair]  # ❌ Fails here
      self.pair2head_ids[pair].add(head_id)
    elif (
        diff_pair2cnt[pair] < 0
        and abs(self.head_id2pair_cnt[head_id][pair] + diff_pair2cnt[pair]) < self.eps
    ):
      # Disappear pair after merging
      assert head_id in self.pair2head_ids[pair]
      self.pair2head_ids[pair].remove(head_id)
```

### The Bug Scenario

Let's trace through a specific scenario for `head_id 150899` and `pair (1414, 1587)`:

#### Step 1: Pair First Appears (Working Correctly)
- **Old count**: 0 (not in dictionary)
- **New count**: 2.0
- **Diff**: `{(1414, 1587): 2.0}` ✅ Included in diff
- **Action**: `head_id 150899` added to `pair2head_ids[(1414, 1587)]`
- **State**: `pair2head_ids[(1414, 1587)] = {..., 150899, ...}`

#### Step 2: Pair Disappears (BUG!)
After some merging operation, the pair completely disappears from the sequence:

- **Old count**: 2.0 (exists in `head_id2pair_cnt[150899]`)
- **New count**: 0 (does not exist in `new_pair2cnt`)
- **Diff calculation**:
  ```python
  diff_pair2cnt = diff_cnt(new_pair2cnt, head_id2pair_cnt[150899])
  # new_pair2cnt does NOT contain (1414, 1587)
  # diff_cnt only iterates over new_pair2cnt.keys()
  # Result: (1414, 1587) NOT in diff_pair2cnt ❌
  ```
- **Action**: `_update_pair2head_ids()` is **NOT** called for this pair
- **Bug**: `head_id 150899` remains in `pair2head_ids[(1414, 1587)]` even though the pair no longer exists in the sequence
- **State**:
  - `head_id2pair_cnt[150899][(1414, 1587)]` = 0 (or not in dict)
  - `pair2head_ids[(1414, 1587)]` = {..., 150899, ...} ❌ **Inconsistent!**

#### Step 3: Pair Reappears (AssertionError!)
Later, another merge operation causes the pair to appear again:

- **Old count**: 0 (not in dictionary)
- **New count**: 1.5
- **Diff**: `{(1414, 1587): 1.5}` ✅ Included in diff
- **Logic check**: `diff > 0` and `old_count ≈ 0` → Treat as "new pair"
- **Attempted action**: Try to add `head_id 150899` to `pair2head_ids[(1414, 1587)]`
- **Result**: ❌ **AssertionError!** The head_id is already in the set (never removed in Step 2)

### Visual Timeline

```
Time  →  Event                          pair2head_ids[(1414,1587)]    head_id2pair_cnt[150899][(1414,1587)]
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
  T1     Pair appears                   {150899} ✅                    2.0 ✅
  T2     Pair disappears (BUG!)         {150899} ❌ Should be {}      0 (not in dict) ✅
  T3     Pair reappears (CRASH!)        {150899} ❌                    1.5 ✅
         → AssertionError: 150899 already in set!
```

## The Fix

### Modified Code

```python
def diff_cnt(cnt1, cnt2):
  """Minus the second pair2cnt from the first pair2cnt.

  Args:
      cnt1 (dict): The first pair2cnt.
      cnt2 (dict): The second pair2cnt.

  Returns:
      dict: A duplication, not inplace.
  """
  # Include all keys from both dictionaries to detect both new and disappeared pairs
  all_keys = set(cnt1.keys()) | set(cnt2.keys())
  return {k: cnt1.get(k, 0) - cnt2.get(k, 0) for k in all_keys}
```

### What Changed

| Aspect | Before | After |
|--------|--------|-------|
| **Keys iterated** | Only `cnt1.keys()` | `cnt1.keys() ∪ cnt2.keys()` |
| **Detected changes** | New pairs, count increases/decreases | New pairs, count increases/decreases, **disappeared pairs** |
| **Missing pairs** | Ignored (count=0 not in dict) | Included with negative diff |

### Why It Works

Now when a pair disappears:

```python
# Step 2 revisited with the fix:
# Old count: 2.0 (in head_id2pair_cnt[150899])
# New count: 0 (not in new_pair2cnt)

all_keys = set(new_pair2cnt.keys()) | set(head_id2pair_cnt[150899].keys())
# (1414, 1587) is now in all_keys ✅

diff_pair2cnt[(1414, 1587)] = 0 - 2.0 = -2.0  ✅

# In _update_pair2head_ids():
# diff < 0 and new_count ≈ 0 → Removal logic triggered ✅
pair2head_ids[(1414, 1587)].remove(150899)  ✅
```

## Impact and Testing

### Impact
- **Scope**: All vocabulary construction operations
- **Frequency**: Rare but deterministic (depends on specific merge sequences)
- **Consequence**: Complete failure of vocabulary construction process

### Verification

After applying the fix, vocabulary construction should complete without assertion errors:

```bash
# Test with the original failing configuration
python build_vocab.py --category=<category> --rand_seed=42

# Expected: Vocabulary construction completes successfully
# Vocabulary file saved to: cache/AmazonReviews2014/<category>/processed/actionpiece.json
```

### Regression Risk

**Low**. The fix is a strict improvement:
- Previously ignored disappeared pairs → Now correctly handled
- No change to behavior for pairs that appear or have count changes
- More comprehensive diff computation is logically correct

## Technical Details

### Affected Code Paths

1. **Primary**: `ActionPieceCore.train()` → `_train_step()` → `_update_pair2head_ids()`
2. **Invocation**: `diff_cnt(new_pair2cnt, old_pair2cnt)` at line 637

### Data Structure Invariants

The fix restores the critical invariant:

```
∀ pair, head_id:
  (head_id in pair2head_ids[pair]) ⟺ (head_id2pair_cnt[head_id][pair] > eps)
```

**Before fix**: Violated when pairs disappeared
**After fix**: Maintained for all operations

### Floating-Point Considerations

- The code uses `eps = 1e-12` for zero-checking
- Pair counts use `float` to represent fractional weights (Set Permutation Regularization)
- The fix correctly handles floating-point comparisons via `.get(k, 0)` returning exact 0

## Related Files

- **Modified**: `genrec/models/ActionPiece/core.py` (line 31-43)
- **Affected**: All vocabulary construction workflows
  - `build_vocab.py`
  - `main.py` (all-in-one workflow)

## Lessons Learned

1. **Dictionary diff operations**: When computing differences between sparse dictionaries (where missing keys imply zero), always iterate over the union of keys, not just one side
2. **Inverted index synchronization**: When maintaining synchronized data structures (direct and inverted indices), ensure all update paths are symmetric
3. **Assertion placement**: The assertion correctly caught the inconsistency, demonstrating the value of defensive programming

## References

- **Issue**: AssertionError during vocabulary construction
- **Commit**: (To be filled after commit)
- **Original stack trace**: See "Error Message" section above
