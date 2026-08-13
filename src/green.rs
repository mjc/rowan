mod node;
mod token;
mod element;
mod builder;
mod node_cache;

use self::element::GreenElement;

pub(crate) use self::{element::GreenElementRef, node::GreenChild};

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

    use super::node::{GreenChild, GreenNodeHead};
    use super::*;
    use crate::{arc::HeaderSlice, TextRange, TextSize};

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
        #[cfg(target_pointer_width = "64")]
        assert_eq!(size_of::<GreenChild>(), 12);
        #[cfg(target_pointer_width = "64")]
        assert_eq!(size_of::<HeaderSlice<GreenNodeHead, [GreenChild; 0]>>(), 12);
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
}
