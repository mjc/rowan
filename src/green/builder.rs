use crate::{
    cow_mut::CowMut,
    green::{
        node_cache::{NodeCache, SharedNodeCache},
        GreenElement, GreenNode, GreenToken, SyntaxKind,
    },
    NodeOrToken,
};

/// A checkpoint for maybe wrapping a node. See `GreenNodeBuilder::checkpoint` for details.
#[derive(Clone, Copy, Debug)]
pub struct Checkpoint(usize);

/// A builder for a green tree.
#[derive(Default, Debug)]
pub struct GreenNodeBuilder<'cache> {
    cache: CowMut<'cache, NodeCache>,
    shared_cache: Option<&'cache SharedNodeCache>,
    parents: Vec<(SyntaxKind, usize)>,
    children: Vec<(u64, u64, GreenElement)>,
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
            cache: CowMut::Borrowed(cache),
            shared_cache: None,
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Creates a builder which structurally shares trees with other builders using the same cache.
    pub fn with_shared_cache(cache: &SharedNodeCache) -> GreenNodeBuilder<'_> {
        GreenNodeBuilder {
            cache: CowMut::Owned(NodeCache::default()),
            shared_cache: Some(cache),
            parents: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Adds new token to the current branch.
    #[inline]
    pub fn token(&mut self, kind: SyntaxKind, text: &str) {
        let (hash, token) = self.cache.token(kind, text, self.shared_cache);
        self.children.push((hash, hash, token.into()));
    }

    /// Adds an existing green token to the current branch.
    #[inline]
    pub fn token_from_green(&mut self, token: GreenToken) {
        let (hash, token) = self.cache.token_from_green(token);
        self.children.push((hash, hash, token.into()));
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
        let (hash, structural_hash, node) =
            self.cache.node(kind, &mut self.children, first_child, self.shared_cache);
        self.children.push((hash, structural_hash, node.into()));
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
        match self.children.pop().unwrap().2 {
            NodeOrToken::Node(node) => node,
            NodeOrToken::Token(_) => panic!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SharedNodeCache;

    fn build(cache: &SharedNodeCache) -> GreenNode {
        let mut builder = GreenNodeBuilder::with_shared_cache(cache);
        builder.start_node(SyntaxKind(0));
        builder.token(SyntaxKind(1), "one");
        builder.finish_node();
        builder.finish()
    }

    #[test]
    fn shared_cache_reuses_trees_across_builders() {
        let cache = SharedNodeCache::default();
        let first = build(&cache);
        let second = build(&cache);

        assert!(std::ptr::eq::<crate::GreenNodeData>(&*first, &*second));
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
    fn shared_cache_does_not_propagate_through_compound_nodes() {
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
    }
}
