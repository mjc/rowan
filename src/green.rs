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
    #[cfg(target_pointer_width = "64")]
    use super::node::{allocation_layout, PackedGreenChild};
    use super::token::allocation_layout as token_allocation_layout;
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
        #[cfg(target_pointer_width = "64")]
        assert_eq!(size_of::<GreenNodeData>(), size_of::<u32>());
        assert_eq!(size_of::<GreenChild>(), size_of::<usize>());
        assert_eq!(size_of::<GreenTokenData>(), size_of::<u32>());
        assert_eq!(token_allocation_layout(0, false).size(), 8);
        assert_eq!(token_allocation_layout(1, false).size(), 12);
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(size_of::<PackedGreenChild>(), size_of::<u32>());
            assert_eq!(allocation_layout(1, false).size(), 12);
            assert_eq!(allocation_layout(2, false).size(), 16);
        }
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
        assert_eq!(node.child(0).unwrap().kind(), kind);
        assert!(node.child(3).is_none());
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

    #[test]
    fn wide_child_counts_preserve_the_full_count() {
        let kind = SyntaxKind(0);
        let token = GreenToken::new(kind, "x");
        let child_count = u16::MAX as usize;
        let node = GreenNode::new(kind, (0..child_count).map(|_| token.clone().into()));
        let cloned = node.clone();
        drop(node);

        assert_eq!(cloned.children().len(), child_count);
        assert_eq!(cloned.text_len(), (child_count as u32).into());
        assert_eq!(cloned.child_offset(child_count - 1), ((child_count - 1) as u32).into());
    }

    #[test]
    fn text_only_wide_headers_preserve_full_text_lengths() {
        let kind = SyntaxKind(1);
        let text = "x".repeat(1 << 14);
        let token = GreenToken::new(SyntaxKind(u16::MAX), &text);
        let node = GreenNode::new(kind, [token.into()]);
        let cloned = node.clone();
        drop(node);

        assert_eq!(cloned.kind(), kind);
        assert_eq!(cloned.text_len(), (text.len() as u32).into());
        assert_eq!(cloned.children().next().unwrap().kind(), SyntaxKind(u16::MAX));
    }

    #[test]
    fn kind_only_wide_headers_preserve_full_kinds() {
        let kind = SyntaxKind(u16::MAX);
        let node = GreenNode::new(kind, [GreenToken::new(SyntaxKind(1), "x").into()]);

        assert_eq!(node.kind(), kind);
        assert_eq!(node.text_len(), 1.into());
    }

    #[test]
    fn wide_token_text_lengths_preserve_text_and_ownership() {
        let text = "x".repeat(u16::MAX as usize);
        let token = GreenToken::new(SyntaxKind(u16::MAX), &text);
        let cloned = token.clone();
        drop(token);

        assert_eq!(cloned.kind(), SyntaxKind(u16::MAX));
        assert_eq!(cloned.text_len(), (text.len() as u32).into());
        assert_eq!(cloned.text(), text);
    }
}
