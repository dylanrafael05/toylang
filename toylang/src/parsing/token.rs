use crate::core::{Span, Src};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub span: Span,
    pub kind: TokenKind
}

impl Token {
    pub fn new(span: Span, kind: TokenKind) -> Self {
        Self {
            span,
            kind
        }
    }

    pub fn text<'a>(&self, source: &'a Src) -> &'a str {
        self.span.text(source)
    }

    pub fn is_error(&self) -> bool {
        self.kind == TokenKind::Error
    }
    pub fn not_error(&self) -> bool {
        !self.is_error()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Add,
    Sub,
    Star,
    Slash,

    Ampersand,
    PtrAmpersand,
    Modulo,

    Dot,
    StaticDot,

    Equal,
    DoubleEqual,
    NotEqual,
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,

    Comma,
    Colon,

    OpenParen,
    CloseParen,

    OpenBrace,
    CloseBrace,
    
    OpenCurly,
    CloseCurly,

    Semicolon,

    If,
    Else,
    While,
    Fn,
    Let,
    Mut,
    Mod,
    Struct,
    Declare,
    Vararg,
    As,
    Import,
    Return,
    Break,
    Continue,
    Sizeof,

    And,
    Or,
    Not,

    Identifier,
    String,

    Integer,
    Decimal,

    EOI,
    Error
}