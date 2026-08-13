mod node;
mod token;
mod element;
mod builder;
mod node_cache;

use self::element::GreenElement;

pub(crate) use self::element::GreenElementRef;

pub use self::{
    builder::{Checkpoint, GreenNodeBuilder},
    node::{Children, GreenNode, GreenNodeData},
    node_cache::{NodeCache, SharedNodeCache},
    token::{GreenToken, GreenTokenData},
};

/// SyntaxKind is a type tag for each token or node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SyntaxKind(pub u16);

#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use super::node::GreenChild;
    use super::*;
    use crate::{TextRange, TextSize};

    #[test]
    fn assert_send_sync() {
        fn f<T: Send + Sync>() {}
        f::<GreenNode>();
        f::<GreenToken>();
        f::<GreenElement>();
    }

    #[test]
    fn test_size_of() {
        use std::mem::size_of;

        eprintln!("GreenNode          {}", size_of::<GreenNode>());
        eprintln!("GreenToken         {}", size_of::<GreenToken>());
        eprintln!("GreenElement       {}", size_of::<GreenElement>());
        assert_eq!(size_of::<GreenNode>(), size_of::<usize>());
        assert_eq!(size_of::<GreenChild>(), size_of::<usize>());
    }

    #[test]
    fn compact_children_preserve_tree_behavior() {
        let kind = SyntaxKind(0);
        let node = GreenNode::new(
            kind,
            [
                GreenToken::new(kind, "a").into(),
                GreenNode::new(kind, [GreenToken::new(kind, "bb").into()]).into(),
                GreenToken::new(kind, "c").into(),
            ],
        );

        assert_eq!(node.child_offset(0), TextSize::new(0));
        assert_eq!(node.child_offset(1), TextSize::new(1));
        assert_eq!(node.child_offset(2), TextSize::new(3));
        assert_eq!(node.child_at_range(TextRange::new(1.into(), 3.into())).unwrap().0, 1);

        let clone = node.clone();
        drop(node);
        assert_eq!(clone.to_string(), "abbc");

        let same = GreenNode::new(
            kind,
            [
                GreenToken::new(kind, "a").into(),
                GreenNode::new(kind, [GreenToken::new(kind, "bb").into()]).into(),
                GreenToken::new(kind, "c").into(),
            ],
        );
        assert_eq!(clone, same);

        let hash = |node: &GreenNode| {
            let mut hasher = DefaultHasher::new();
            node.hash(&mut hasher);
            hasher.finish()
        };
        assert_eq!(hash(&clone), hash(&same));
    }

    #[test]
    fn compact_children_clone_and_drop_across_threads() {
        let kind = SyntaxKind(0);
        let node = std::sync::Arc::new(GreenNode::new(
            kind,
            [
                GreenNode::new(kind, [GreenToken::new(kind, "node").into()]).into(),
                GreenToken::new(kind, "token").into(),
            ],
        ));

        std::thread::scope(|scope| {
            for _ in 0..8 {
                let node = std::sync::Arc::clone(&node);
                scope.spawn(move || {
                    for _ in 0..1_000 {
                        let clone = GreenNode::clone(&node);
                        assert_eq!(clone.to_string(), "nodetoken");
                    }
                });
            }
        });
    }

    #[test]
    fn checkpointed_children_keep_offsets() {
        let kind = SyntaxKind(0);
        let node = GreenNode::new(kind, (0..17).map(|_| GreenToken::new(kind, "x").into()));

        for index in [0, 7, 8, 15, 16] {
            assert_eq!(node.child_offset(index), (index as u32).into());
            assert_eq!(
                node.child_at_range(TextRange::new(
                    (index as u32).into(),
                    (index as u32 + 1).into(),
                ))
                .unwrap()
                .0,
                index,
            );
        }
        assert_eq!(node.child_at_range(TextRange::empty(8.into())).unwrap().0, 7);
        assert_eq!(node.child_at_range(TextRange::empty(16.into())).unwrap().0, 15);
        assert!(node.child_at_range(TextRange::new(7.into(), 9.into())).is_none());

        let reverse_offsets =
            node.children_with_offsets().rev().map(|child| child.rel_offset).collect::<Vec<_>>();
        assert_eq!(reverse_offsets.first().copied(), Some(16.into()));
        assert_eq!(reverse_offsets.last().copied(), Some(0.into()));

        let clone = node.clone();
        drop(node);
        assert_eq!(clone.to_string(), "xxxxxxxxxxxxxxxxx");
    }

    #[test]
    fn checkpointed_children_preserve_empty_range_ordering() {
        let kind = SyntaxKind(0);
        let empty = || GreenNode::new(kind, []).into();
        let node = GreenNode::new(
            kind,
            [
                GreenToken::new(kind, "a").into(),
                empty(),
                empty(),
                empty(),
                empty(),
                empty(),
                GreenToken::new(kind, "b").into(),
            ],
        );

        assert_eq!(node.child_at_range(TextRange::empty(1.into())).unwrap().0, 5);
        assert_eq!(node.child_at_range(TextRange::new(1.into(), 2.into())).unwrap().0, 6);
    }

    #[test]
    fn construction_cleans_up_after_iterator_panic() {
        struct PanicAfterOne {
            child: Option<GreenElement>,
        }

        impl Iterator for PanicAfterOne {
            type Item = GreenElement;

            fn next(&mut self) -> Option<Self::Item> {
                self.child.take().or_else(|| panic!("iterator panic"))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }
        }

        impl ExactSizeIterator for PanicAfterOne {
            fn len(&self) -> usize {
                if self.child.is_some() {
                    2
                } else {
                    1
                }
            }
        }

        let kind = SyntaxKind(0);
        let children = PanicAfterOne { child: Some(GreenToken::new(kind, "child").into()) };
        assert!(std::panic::catch_unwind(|| GreenNode::new(kind, children)).is_err());
    }
}
