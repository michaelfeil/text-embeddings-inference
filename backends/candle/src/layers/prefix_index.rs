//! Host metadata for bounded sequence and page caches. Planning works on a clone:
//! replacement is published only after the corresponding GPU writes complete.
use std::collections::BTreeMap;

pub(super) type Token = (u32, u32); // token ID and absolute position
pub(super) const PAGE_SIZE: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheMode {
    Sequence,
    Paged,
}

#[derive(Clone)]
struct Entry {
    key: Vec<Token>,
    start: usize,
    len: usize,
    used: u64,
}

#[derive(Clone)]
pub(super) struct PrefixIndex {
    mode: CacheMode,
    capacity: usize,
    max_prefix: usize,
    entries: Vec<Entry>,
    clock: u64,
}

pub(super) struct IndexPlan {
    pub next: PrefixIndex,
    /// Physical cache row for each reused token, per sequence.
    pub hits: Vec<Vec<usize>>,
    /// Unique destination cache rows and their source rows in the original batch.
    pub writes: Vec<(usize, usize)>,
    pub evictions: usize,
}

impl PrefixIndex {
    pub fn new(mode: CacheMode, capacity: usize, max_prefix: usize) -> Self {
        let capacity = match mode {
            CacheMode::Sequence => capacity,
            CacheMode::Paged => capacity / PAGE_SIZE * PAGE_SIZE,
        };
        Self {
            mode,
            capacity,
            max_prefix: max_prefix.min(capacity),
            entries: vec![],
            clock: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn mode(&self) -> CacheMode {
        self.mode
    }
    pub fn resident_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.len).sum()
    }
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn find(&mut self, tokens: &[Token]) -> Vec<usize> {
        self.clock += 1;
        // Preserve a query token for exact hits so attention never receives an empty Q.
        let limit = tokens.len().saturating_sub(1).min(self.max_prefix);
        match self.mode {
            CacheMode::Sequence => {
                let best = self
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let n = tokens
                            .iter()
                            .zip(&e.key)
                            .take_while(|(a, b)| a == b)
                            .count()
                            .min(limit);
                        (i, n)
                    })
                    .max_by_key(|&(_, n)| n);
                if let Some((i, n)) = best.filter(|&(_, n)| n > 0) {
                    let e = &mut self.entries[i];
                    e.used = self.clock;
                    (e.start..e.start + n).collect()
                } else {
                    vec![]
                }
            }
            CacheMode::Paged => {
                let mut rows = vec![];
                for end in (PAGE_SIZE..=limit).step_by(PAGE_SIZE) {
                    let Some(e) = self.entries.iter_mut().find(|e| e.key == tokens[..end]) else {
                        break;
                    };
                    e.used = self.clock;
                    rows.extend(e.start..e.start + PAGE_SIZE);
                }
                rows
            }
        }
    }

    fn gap(&self, len: usize) -> Option<usize> {
        let mut occupied: Vec<_> = self.entries.iter().map(|e| (e.start, e.len)).collect();
        occupied.sort_unstable();
        let mut cursor = 0;
        for (start, count) in occupied {
            if start - cursor >= len {
                return Some(cursor);
            }
            cursor = start + count;
        }
        (self.capacity - cursor >= len).then_some(cursor)
    }

    fn allocate(
        &mut self,
        len: usize,
        protected: &[Token],
        evictions: &mut usize,
    ) -> Option<usize> {
        if len > self.capacity {
            return None;
        }
        loop {
            if let Some(start) = self.gap(len) {
                return Some(start);
            }
            // Paged entries retain their ancestors: reclaim cold leaves first.
            let victim = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, e)| {
                    self.mode == CacheMode::Sequence
                        || (!protected.starts_with(&e.key)
                            && !self.entries.iter().any(|child| {
                                child.key.len() > e.key.len() && child.key.starts_with(&e.key)
                            }))
                })
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i)?;
            self.entries.remove(victim);
            *evictions += 1;
        }
    }

    pub fn plan(&self, sequences: &[Vec<Token>]) -> IndexPlan {
        let mut next = self.clone();
        let hits = sequences.iter().map(|s| next.find(s)).collect();
        let mut writes = BTreeMap::new();
        let mut evictions = 0;
        let mut batch_offset = 0;
        for seq in sequences {
            let len = seq.len().min(self.max_prefix);
            let ends: Vec<usize> = match self.mode {
                CacheMode::Sequence => {
                    if len > 0 {
                        vec![len]
                    } else {
                        vec![]
                    }
                }
                CacheMode::Paged => (PAGE_SIZE..=len).step_by(PAGE_SIZE).collect(),
            };
            for end in ends {
                let key = &seq[..end];
                if next.entries.iter().any(|e| e.key == key) {
                    continue;
                }
                let count = if self.mode == CacheMode::Paged {
                    PAGE_SIZE
                } else {
                    end
                };
                let Some(start) = next.allocate(count, key, &mut evictions) else {
                    continue;
                };
                next.clock += 1;
                next.entries.push(Entry {
                    key: key.to_vec(),
                    start,
                    len: count,
                    used: next.clock,
                });
                for offset in 0..count {
                    writes.insert(start + offset, batch_offset + end - count + offset);
                }
            }
            batch_offset += seq.len();
        }
        IndexPlan {
            next,
            hits,
            writes: writes.into_iter().collect(),
            evictions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn seq(id: u32, n: usize) -> Vec<Token> {
        (0..n).map(|i| (id + i as u32, i as u32)).collect()
    }

    #[test]
    fn sequence_cache_learns_multiple_apps_and_replaces_old_entries() {
        let a = seq(10, 400);
        let b = seq(2000, 600);
        let c = seq(4000, 600);
        let cache = PrefixIndex::new(CacheMode::Sequence, 1200, 600);
        let plan = cache.plan(&[a.clone(), b.clone()]);
        assert!(plan.hits.iter().all(Vec::is_empty));
        assert_eq!(plan.next.resident_tokens(), 1000);
        let mut cache = plan.next;
        assert_eq!(cache.find(&a).len(), 399);
        let plan = cache.plan(&[c.clone()]);
        assert_eq!(plan.evictions, 1);
        cache = plan.next;
        assert_eq!(cache.find(&a).len(), 399);
        assert!(cache.find(&b).is_empty());
        assert_eq!(cache.find(&c).len(), 599);
        assert!(cache.resident_tokens() <= 1200);
    }

    #[test]
    fn paged_cache_shares_blocks_and_recognizes_positions() {
        let a = seq(10, 129);
        let mut b = a.clone();
        b[128] = (999, 128);
        let cache = PrefixIndex::new(CacheMode::Paged, 256, 256);
        let plan = cache.plan(&[a.clone(), b.clone()]);
        assert_eq!(plan.next.resident_tokens(), 128);
        assert_eq!(plan.writes.len(), 128);
        let mut cache = plan.next;
        assert_eq!(cache.find(&a).len(), 128);
        let mut moved = a.clone();
        moved[0].1 = 1;
        assert!(cache.find(&moved).is_empty());
        let c = seq(1000, 193);
        cache = cache.plan(&[c.clone()]).next;
        assert_eq!(cache.find(&c).len(), 192);
        assert!(cache.resident_tokens() <= 256);
    }

    #[test]
    fn planning_is_transactional_and_writes_have_unique_destinations() {
        for mode in [CacheMode::Sequence, CacheMode::Paged] {
            let cache = PrefixIndex::new(mode, 256, 192);
            let plan = cache.plan(&[seq(10, 192), seq(1000, 192), seq(2000, 192)]);
            assert_eq!(cache.resident_tokens(), 0);
            assert!(plan.next.resident_tokens() <= 256);
            assert!(plan.writes.windows(2).all(|w| w[0].0 < w[1].0));
            assert!(plan.writes.iter().all(|&(d, s)| d < 256 && s < 576));
        }
    }
}

/// Page tables share a transient suffix region after all layers' resident pages.
/// Only complete prefix pages are reused; suffix padding is never attended to.
#[cfg(any(test, feature = "prefix-cache-paged"))]
pub(super) struct PageLayout {
    pub tables: Vec<Vec<u32>>,
    pub suffix_writes: Vec<(usize, usize)>,
    pub transient_rows: usize,
}
#[cfg(any(test, feature = "prefix-cache-paged"))]
impl PageLayout {
    pub fn new(hits: &[Vec<usize>], lengths: &[usize], capacity: usize, layers: usize) -> Self {
        assert_eq!(hits.len(), lengths.len());
        assert_eq!(capacity % PAGE_SIZE, 0);
        let mut tables = Vec::with_capacity(hits.len());
        let mut suffix_writes = Vec::new();
        let mut transient_rows = 0;
        let mut source = 0;
        for (hit, &len) in hits.iter().zip(lengths) {
            assert_eq!(hit.len() % PAGE_SIZE, 0);
            assert!(hit.len() < len);
            let mut table: Vec<u32> = hit
                .chunks_exact(PAGE_SIZE)
                .map(|page| {
                    assert_eq!(page[0] % PAGE_SIZE, 0);
                    assert!(page
                        .iter()
                        .enumerate()
                        .all(|(i, &r)| r == page[0] + i && r < capacity));
                    (page[0] / PAGE_SIZE) as u32
                })
                .collect();
            let suffix = len - hit.len();
            let start = layers * capacity + transient_rows;
            table.extend((0..suffix.div_ceil(PAGE_SIZE)).map(|p| (start / PAGE_SIZE + p) as u32));
            suffix_writes.extend((0..suffix).map(|i| (start + i, source + i)));
            transient_rows += suffix.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            source += suffix;
            tables.push(table);
        }
        Self {
            tables,
            suffix_writes,
            transient_rows,
        }
    }

    pub fn for_layer(&self, hits: &[Vec<usize>], capacity: usize, layer: usize) -> Vec<Vec<u32>> {
        self.tables
            .iter()
            .zip(hits)
            .map(|(table, hit)| {
                table
                    .iter()
                    .enumerate()
                    .map(|(i, &page)| {
                        if i < hit.len() / PAGE_SIZE {
                            page + (layer * capacity / PAGE_SIZE) as u32
                        } else {
                            page
                        }
                    })
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod page_layout_tests {
    use super::*;
    #[test]
    fn layers_share_suffix_workspace_without_exposing_padding() {
        let hits = vec![(64..128).collect(), vec![], (0..128).collect()];
        let p = PageLayout::new(&hits, &[71, 65, 129], 256, 3);
        assert_eq!(p.transient_rows, 256);
        assert_eq!(p.tables, vec![vec![1, 12], vec![13, 14], vec![0, 1, 15]]);
        assert_eq!(
            p.for_layer(&hits, 256, 2),
            vec![vec![9, 12], vec![13, 14], vec![8, 9, 15]]
        );
        assert_eq!(p.suffix_writes.len(), 73);
        assert_eq!(p.suffix_writes[7], (832, 7));
        assert_eq!(p.suffix_writes[72], (960, 72));
    }
}

#[cfg(test)]
mod replacement_tests {
    use super::*;
    #[test]
    fn replacement_preserves_every_resident_token() {
        for mode in [CacheMode::Sequence, CacheMode::Paged] {
            let mut index = PrefixIndex::new(mode, 256, 192);
            let mut storage = vec![None; 256];
            let mut rng = 173u32;
            for _ in 0..200 {
                let mut next = || {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    rng
                };
                let sequences: Vec<Vec<Token>> = (0..1 + next() % 5)
                    .map(|_| {
                        let app = next() % 4;
                        let length = 1 + next() % 250;
                        let position = next() % 3;
                        (0..length)
                            .map(|i| (if i < 80 { i + 1 } else { app * 1000 + i }, i + position))
                            .collect()
                    })
                    .collect();
                let plan = index.plan(&sequences);
                for (sequence, hit) in sequences.iter().zip(&plan.hits) {
                    for (&token, &row) in sequence.iter().zip(hit) {
                        assert_eq!(storage[row], Some(token));
                    }
                }
                let source: Vec<_> = sequences.iter().flatten().copied().collect();
                for &(dest, row) in &plan.writes {
                    storage[dest] = Some(source[row]);
                }
                index = plan.next;
                for entry in &index.entries {
                    let expected = &entry.key[entry.key.len() - entry.len..];
                    for (i, &token) in expected.iter().enumerate() {
                        assert_eq!(storage[entry.start + i], Some(token));
                    }
                }
            }
        }
    }
}
