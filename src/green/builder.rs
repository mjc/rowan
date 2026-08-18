use crate::{
    green::{
        node_cache::{CacheBackend, CachedElement, NodeCache, SharedNodeCache},
        GreenNode, SyntaxKind,
    },
    NodeOrToken,
};

/// A checkpoint for maybe wrapping a node. See `GreenNodeBuilder::checkpoint` for details.
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint(usize);

/// A builder for a green tree.
#[derive(Default, Debug)]
pub struct GreenNodeBuilder<'cache> {
    cache: CacheBackend<'cache>,
    parents: Vec<(SyntaxKind, usize)>,
    children: Vec<CachedElement>,
}

impl GreenNodeBuilder<'_> {
    /// Creates new builder.
    pub fn new() -> GreenNodeBuilder<'static> {
        GreenNodeBuilder::default()
    }

    /// Reusing `NodeCache` between different `GreenNodeBuilder`s saves memory.
    /// It allows to structurally share underlying trees.
    pub fn with_cache(cache: &mut NodeCache) -> GreenNodeBuilder<'_> {
        GreenNodeBuilder {
            cache: CacheBackend::local(cache),
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Creates a builder that opts into sharing immutable green descendants with other builders
    /// using the same cache.
    ///
    /// Each finished tree keeps a distinct root allocation. Builders created with [`Self::new`] or
    /// [`Self::with_cache`] continue to use only the ordinary local cache.
    pub fn with_shared_cache(cache: &SharedNodeCache) -> GreenNodeBuilder<'_> {
        GreenNodeBuilder {
            cache: CacheBackend::shared(cache),
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Adds new token to the current branch.
    #[inline]
    pub fn token(&mut self, kind: SyntaxKind, text: &str) {
        self.children.push(self.cache.token(kind, text));
    }

    /// Start new node and make it current.
    #[inline]
    pub fn start_node(&mut self, kind: SyntaxKind) {
        let len = self.children.len();
        self.parents.push((kind, len));
    }

    /// Finish current branch and restore previous
    /// branch as current.
    #[inline]
    pub fn finish_node(&mut self) {
        let (kind, first_child) = self.parents.pop().unwrap();
        let node = self.cache.node(kind, &mut self.children, first_child);
        self.children.push(node);
    }

    /// Prepare for maybe wrapping the next node.
    /// The way wrapping works is that you first of all get a checkpoint,
    /// then you place all tokens you want to wrap, and then *maybe* call
    /// `start_node_at`.
    /// Example:
    /// ```rust
    /// # use rowan::{GreenNodeBuilder, SyntaxKind};
    /// # const PLUS: SyntaxKind = SyntaxKind(0);
    /// # const OPERATION: SyntaxKind = SyntaxKind(1);
    /// # struct Parser;
    /// # impl Parser {
    /// #     fn peek(&self) -> Option<SyntaxKind> { None }
    /// #     fn parse_expr(&mut self) {}
    /// # }
    /// # let mut builder = GreenNodeBuilder::new();
    /// # let mut parser = Parser;
    /// let checkpoint = builder.checkpoint();
    /// parser.parse_expr();
    /// if parser.peek() == Some(PLUS) {
    ///   // 1 + 2 = Add(1, 2)
    ///   builder.start_node_at(checkpoint, OPERATION);
    ///   parser.parse_expr();
    ///   builder.finish_node();
    /// }
    /// ```
    #[inline]
    pub fn checkpoint(&self) -> Checkpoint {
        Checkpoint(self.children.len())
    }

    /// Wrap the previous branch marked by `checkpoint` in a new branch and
    /// make it current.
    #[inline]
    pub fn start_node_at(&mut self, checkpoint: Checkpoint, kind: SyntaxKind) {
        let Checkpoint(checkpoint) = checkpoint;
        assert!(
            checkpoint <= self.children.len(),
            "checkpoint no longer valid, was finish_node called early?"
        );

        if let Some(&(_, first_child)) = self.parents.last() {
            assert!(
                checkpoint >= first_child,
                "checkpoint no longer valid, was an unmatched start_node_at called?"
            );
        }

        self.parents.push((kind, checkpoint));
    }

    /// Complete tree building. Make sure that
    /// `start_node_at` and `finish_node` calls
    /// are paired!
    #[inline]
    pub fn finish(mut self) -> GreenNode {
        assert_eq!(self.children.len(), 1);
        let root = match self.children.remove(0).1 {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(_) => panic!(),
        };
        self.cache.finish(root)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cursor::SyntaxNode, SharedNodeCache};

    fn build(cache: &SharedNodeCache) -> GreenNode {
        let mut builder = GreenNodeBuilder::with_shared_cache(cache);
        builder.start_node(SyntaxKind(0));
        builder.token(SyntaxKind(1), "one");
        builder.finish_node();
        builder.finish()
    }

    #[test]
    fn shared_cache_detaches_roots_but_reuses_descendants() {
        let cache = SharedNodeCache::default();
        let first = build(&cache);
        let second = build(&cache);

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq(
            first.children().next().unwrap().into_token().unwrap(),
            second.children().next().unwrap().into_token().unwrap(),
        ));

        let first_red = SyntaxNode::new_root(first.clone());
        let first_again = SyntaxNode::new_root(first);
        let second_red = SyntaxNode::new_root(second);
        assert_eq!(first_red, first_again);
        assert_eq!(first_red.first_token(), first_again.first_token());
        assert_ne!(first_red, second_red);
        assert_ne!(first_red.first_token(), second_red.first_token());
    }

    #[test]
    fn shared_cache_does_not_share_long_tokens_through_unary_nodes() {
        fn build(cache: &SharedNodeCache) -> GreenNode {
            let mut builder = GreenNodeBuilder::with_shared_cache(cache);
            builder.start_node(SyntaxKind(0));
            builder.token(SyntaxKind(1), "123456789");
            builder.finish_node();
            builder.finish()
        }

        let cache = SharedNodeCache::default();
        let first = build(&cache);
        let second = build(&cache);

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert!(!std::ptr::eq(
            first.children().next().unwrap().into_token().unwrap(),
            second.children().next().unwrap().into_token().unwrap(),
        ));
    }

    #[test]
    fn shared_cache_does_not_share_wide_nodes_through_unary_parents() {
        fn build(cache: &SharedNodeCache) -> GreenNode {
            let mut builder = GreenNodeBuilder::with_shared_cache(cache);
            builder.start_node(SyntaxKind(0));
            builder.start_node(SyntaxKind(1));
            builder.token(SyntaxKind(2), "one");
            builder.token(SyntaxKind(2), "two");
            builder.finish_node();
            builder.finish_node();
            builder.finish()
        }

        let cache = SharedNodeCache::default();
        let first = build(&cache);
        let second = build(&cache);

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert!(!std::ptr::eq::<crate::GreenNodeData>(
            first.children().next().unwrap().into_node().unwrap(),
            second.children().next().unwrap().into_node().unwrap(),
        ));
    }

    #[test]
    fn shared_cache_shares_recursively_eligible_unary_chains() {
        fn build(cache: &SharedNodeCache) -> GreenNode {
            let mut builder = GreenNodeBuilder::with_shared_cache(cache);
            builder.start_node(SyntaxKind(0));
            builder.start_node(SyntaxKind(1));
            builder.token(SyntaxKind(2), "short");
            builder.finish_node();
            builder.finish_node();
            builder.finish()
        }

        let cache = SharedNodeCache::default();
        let first = build(&cache);
        let second = build(&cache);

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert!(std::ptr::eq::<crate::GreenNodeData>(
            first.children().next().unwrap().into_node().unwrap(),
            second.children().next().unwrap().into_node().unwrap(),
        ));
    }

    #[test]
    fn clearing_shared_cache_preserves_returned_trees() {
        let cache = SharedNodeCache::default();
        let first = build(&cache);

        cache.clear();

        assert_eq!(first.to_string(), "one");
        let second = build(&cache);
        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert_eq!(first, second);
    }

    #[test]
    fn shared_cache_limits_nodes_and_tokens() {
        fn build(cache: &SharedNodeCache, text: &str) -> GreenNode {
            let mut builder = GreenNodeBuilder::with_shared_cache(cache);
            builder.start_node(SyntaxKind(0));
            builder.token(SyntaxKind(1), text);
            builder.token(SyntaxKind(1), text);
            builder.finish_node();
            builder.finish()
        }

        let cache = SharedNodeCache::default();
        let short_first = build(&cache, "short");
        let short_second = build(&cache, "short");
        let edge_first = build(&cache, "12345678");
        let edge_second = build(&cache, "12345678");
        let utf8_edge_first = build(&cache, "éééé");
        let utf8_edge_second = build(&cache, "éééé");
        let long_first = build(&cache, "long text");
        let long_second = build(&cache, "long text");

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*short_first, &*short_second));
        assert!(std::ptr::eq(
            short_first.children().next().unwrap().into_token().unwrap(),
            short_second.children().next().unwrap().into_token().unwrap(),
        ));
        assert!(std::ptr::eq(
            edge_first.children().next().unwrap().into_token().unwrap(),
            edge_second.children().next().unwrap().into_token().unwrap(),
        ));
        assert!(std::ptr::eq(
            utf8_edge_first.children().next().unwrap().into_token().unwrap(),
            utf8_edge_second.children().next().unwrap().into_token().unwrap(),
        ));
        assert!(!std::ptr::eq(
            long_first.children().next().unwrap().into_token().unwrap(),
            long_second.children().next().unwrap().into_token().unwrap(),
        ));
    }

    #[test]
    fn shared_cache_does_not_retain_wide_subtrees() {
        fn build(cache: &SharedNodeCache, middle: &str) -> GreenNode {
            let mut builder = GreenNodeBuilder::with_shared_cache(cache);
            builder.start_node(SyntaxKind(0));
            builder.start_node(SyntaxKind(1));
            for index in 0..65 {
                builder.token(SyntaxKind(2), if index == 32 { middle } else { "x" });
            }
            builder.finish_node();
            builder.finish_node();
            builder.finish()
        }

        let cache = SharedNodeCache::default();
        let first = build(&cache, "one");
        let second = build(&cache, "two");
        let repeated = build(&cache, "two");

        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
        assert!(!std::ptr::eq::<crate::GreenNodeData>(&*second, &*repeated));
        assert_eq!(second, repeated);
    }

    #[test]
    fn shared_cache_uses_local_fallback_for_ineligible_elements() {
        let cache = SharedNodeCache::default();
        let mut builder = GreenNodeBuilder::with_shared_cache(&cache);
        builder.start_node(SyntaxKind(0));
        for _ in 0..2 {
            builder.start_node(SyntaxKind(1));
            builder.token(SyntaxKind(2), "long text");
            builder.token(SyntaxKind(2), "long text");
            builder.finish_node();
        }
        builder.finish_node();
        let root = builder.finish();
        let mut children = root.children();
        let first = children.next().unwrap().into_node().unwrap();
        let second = children.next().unwrap().into_node().unwrap();

        assert!(std::ptr::eq::<crate::GreenNodeData>(first, second));
        let mut tokens = first.children();
        assert!(std::ptr::eq(
            tokens.next().unwrap().into_token().unwrap(),
            tokens.next().unwrap().into_token().unwrap(),
        ));
    }
}
