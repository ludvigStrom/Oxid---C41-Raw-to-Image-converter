//! Session-wide undo/redo stack for keyed snapshots.
//!
//! The manager does not know about images or pipeline options. Callers `track`
//! new keys, `commit_settled` after mutations, and `undo` / `redo` to get the
//! states to restore.

use std::collections::HashMap;
use std::hash::Hash;

/// Default number of undo records kept in memory.
pub const UNDO_LIMIT: usize = 100;

struct UndoRecord<K, S> {
    /// `(key, before, after)` for each changed key in this step.
    changes: Vec<(K, S, S)>,
}

/// Chronological undo/redo history keyed by `K` with snapshot type `S`.
pub struct UndoManager<K, S> {
    undo: Vec<UndoRecord<K, S>>,
    redo: Vec<UndoRecord<K, S>>,
    last_committed: HashMap<K, S>,
    limit: usize,
}

impl<K, S> UndoManager<K, S> {
    /// Create a manager that keeps at most `limit` undo records (minimum 1).
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            last_committed: HashMap::new(),
            limit: limit.max(1),
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Drop every record and tracked baseline.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.last_committed.clear();
    }
}

impl<K: Clone + Eq + Hash, S: Clone + PartialEq> UndoManager<K, S> {
    /// Register a key at `state` without creating an undo record.
    pub fn track(&mut self, key: K, state: S) {
        self.last_committed.insert(key, state);
    }

    /// Remove a key from baselines and from every stack record.
    pub fn forget(&mut self, key: &K) {
        self.last_committed.remove(key);
        strip_key(&mut self.undo, key);
        strip_key(&mut self.redo, key);
    }

    /// Commit keys whose current state differs from the last committed baseline.
    ///
    /// `skip` is omitted (used while a gesture is in progress). Unknown keys are
    /// tracked without creating a record. Several changed keys become one step.
    pub fn commit_settled(&mut self, current: &[(K, S)], skip: Option<&K>) {
        let mut changes = Vec::new();
        for (key, state) in current {
            if skip == Some(key) {
                continue;
            }
            match self.last_committed.get(key) {
                None => {
                    self.last_committed.insert(key.clone(), state.clone());
                }
                Some(prev) if prev != state => {
                    changes.push((key.clone(), prev.clone(), state.clone()));
                    self.last_committed.insert(key.clone(), state.clone());
                }
                Some(_) => {}
            }
        }
        if changes.is_empty() {
            return;
        }
        self.redo.clear();
        self.undo.push(UndoRecord { changes });
        while self.undo.len() > self.limit {
            self.undo.remove(0);
        }
    }

    /// Pop the latest undo record and return the before-states to restore.
    pub fn undo(&mut self) -> Option<Vec<(K, S)>> {
        let record = self.undo.pop()?;
        let restore = apply_side(&record, true, &mut self.last_committed);
        self.redo.push(record);
        Some(restore)
    }

    /// Pop the latest redo record and return the after-states to restore.
    pub fn redo(&mut self) -> Option<Vec<(K, S)>> {
        let record = self.redo.pop()?;
        let restore = apply_side(&record, false, &mut self.last_committed);
        self.undo.push(record);
        Some(restore)
    }
}

fn strip_key<K: PartialEq, S>(stack: &mut Vec<UndoRecord<K, S>>, key: &K) {
    for record in stack.iter_mut() {
        record.changes.retain(|(k, _, _)| k != key);
    }
    stack.retain(|record| !record.changes.is_empty());
}

fn apply_side<K: Clone + Eq + Hash, S: Clone>(
    record: &UndoRecord<K, S>,
    before: bool,
    last_committed: &mut HashMap<K, S>,
) -> Vec<(K, S)> {
    record
        .changes
        .iter()
        .map(|(key, b, a)| {
            let state = if before { b.clone() } else { a.clone() };
            last_committed.insert(key.clone(), state.clone());
            (key.clone(), state)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mgr() -> UndoManager<u64, i32> {
        UndoManager::new(100)
    }

    fn commit(m: &mut UndoManager<u64, i32>, pairs: &[(u64, i32)]) {
        m.commit_settled(pairs, None);
    }

    #[test]
    fn track_does_not_create_undo() {
        let mut m = mgr();
        m.track(1, 0);
        assert!(!m.can_undo());
        assert!(!m.can_redo());
        commit(&mut m, &[(1, 0)]);
        assert!(!m.can_undo());
    }

    #[test]
    fn single_undo_redo() {
        let mut m = mgr();
        m.track(1, 0);
        commit(&mut m, &[(1, 5)]);
        assert!(m.can_undo());
        assert_eq!(m.undo(), Some(vec![(1, 0)]));
        assert!(!m.can_undo());
        assert!(m.can_redo());
        assert_eq!(m.redo(), Some(vec![(1, 5)]));
        assert!(m.can_undo());
        assert!(!m.can_redo());
    }

    #[test]
    fn new_edit_clears_redo() {
        let mut m = mgr();
        m.track(1, 0);
        commit(&mut m, &[(1, 5)]);
        m.undo();
        commit(&mut m, &[(1, 9)]);
        assert!(!m.can_redo());
        assert_eq!(m.undo(), Some(vec![(1, 0)]));
        assert_eq!(m.redo(), Some(vec![(1, 9)]));
    }

    #[test]
    fn batch_restore() {
        let mut m = mgr();
        m.track(1, 0);
        m.track(2, 0);
        commit(&mut m, &[(1, 3), (2, 4)]);
        assert_eq!(m.undo(), Some(vec![(1, 0), (2, 0)]));
        assert_eq!(m.redo(), Some(vec![(1, 3), (2, 4)]));
    }

    #[test]
    fn skip_while_dragging_then_commit() {
        let mut m = mgr();
        m.track(1, 0);
        m.commit_settled(&[(1, 1)], Some(&1));
        m.commit_settled(&[(1, 2)], Some(&1));
        assert!(!m.can_undo());
        m.commit_settled(&[(1, 2)], None);
        assert_eq!(m.undo(), Some(vec![(1, 0)]));
    }

    #[test]
    fn skip_does_not_block_other_keys() {
        let mut m = mgr();
        m.track(1, 0);
        m.track(2, 0);
        m.commit_settled(&[(1, 8), (2, 9)], Some(&1));
        assert_eq!(m.undo(), Some(vec![(2, 0)]));
        m.commit_settled(&[(1, 8), (2, 0)], None);
        assert_eq!(m.undo(), Some(vec![(1, 0)]));
    }

    #[test]
    fn stack_limit_drops_oldest() {
        let mut m = UndoManager::new(2);
        m.track(1, 0);
        commit(&mut m, &[(1, 1)]);
        commit(&mut m, &[(1, 2)]);
        commit(&mut m, &[(1, 3)]);
        assert_eq!(m.undo(), Some(vec![(1, 2)]));
        assert_eq!(m.undo(), Some(vec![(1, 1)]));
        assert!(!m.can_undo());
    }

    #[test]
    fn forget_strips_records_and_baseline() {
        let mut m = mgr();
        m.track(1, 0);
        m.track(2, 0);
        commit(&mut m, &[(1, 5), (2, 6)]);
        m.forget(&1);
        assert_eq!(m.undo(), Some(vec![(2, 0)]));
        m.forget(&2);
        assert!(!m.can_undo());
        assert!(!m.can_redo());
    }

    #[test]
    fn forget_only_key_drops_record() {
        let mut m = mgr();
        m.track(1, 0);
        commit(&mut m, &[(1, 5)]);
        m.forget(&1);
        assert!(!m.can_undo());
    }

    #[test]
    fn unknown_key_is_tracked_without_undo() {
        let mut m = mgr();
        commit(&mut m, &[(1, 4)]);
        assert!(!m.can_undo());
        commit(&mut m, &[(1, 7)]);
        assert_eq!(m.undo(), Some(vec![(1, 4)]));
    }

    #[test]
    fn clear_resets_everything() {
        let mut m = mgr();
        m.track(1, 0);
        commit(&mut m, &[(1, 5)]);
        m.undo();
        m.clear();
        assert!(!m.can_undo());
        assert!(!m.can_redo());
        commit(&mut m, &[(1, 5)]);
        assert!(!m.can_undo());
    }
}
