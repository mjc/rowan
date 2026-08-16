//! A generic library for lossless syntax trees.
//! See `examples/s_expressions.rs` for a tutorial.
#![forbid(
    // missing_debug_implementations,
    unconditional_recursion,
    future_incompatible,
    // missing_docs,
)]
#![deny(unsafe_code)]

#[allow(unsafe_code)]
mod green;
#[allow(unsafe_code)]
pub mod cursor;

pub mod api;
mod syntax_text;
mod utility_types;

mod cow_mut;
#[cfg(feature = "serde1")]
mod serde_impls;
pub mod ast;

pub use crate::cursor::SyntaxTreeId;
pub use text_size::{TextLen, TextRange, TextSize};

/// Releases physical pages from empty internal green-tree allocation chunks.
///
/// Existing syntax trees remain valid, and reclaimed chunks are reused by later allocations.
pub fn trim_memory() {
    green::trim_memory();
}

pub use crate::{
    api::{
        Language, SyntaxElement, SyntaxElementChildren, SyntaxNode, SyntaxNodeChildren, SyntaxToken,
    },
    green::{
        Checkpoint, Children, GreenNode, GreenNodeBuilder, GreenNodeData, GreenToken,
        GreenTokenData, NodeCache, SharedNodeCache, SyntaxKind,
    },
    syntax_text::SyntaxText,
    utility_types::{Direction, NodeOrToken, TokenAtOffset, WalkEvent},
};
