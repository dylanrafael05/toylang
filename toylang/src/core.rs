pub mod ops;
pub mod monad;
pub mod arena;

use std::{rc::Rc, fmt::Debug};

pub trait AsU64
{
    fn as_u64(&self) -> u64;
}

impl AsU64 for bool 
{
    fn as_u64(&self) -> u64 {
        if *self {1} else {0}
    }
}

#[macro_export]
macro_rules! writex {
    ($f: expr, $r: expr) => {
        write!($f, "{}", $r)
    };
}

#[macro_export]
macro_rules! formatstr {
    ($lit:literal) => {
        ($lit)
    };
    ($lit:literal, $l:expr) => {
        format!($lit, $l).as_str()
    };
    ($lit:literal, $($val:expr,)+ $l:expr) => {
        format!($lit, $($val,)+ $l).as_str()
    };
}

#[derive(Clone, PartialEq, Eq)]
pub struct Source(Rc<str>);

impl Source
{
    pub fn new(text: &str) -> Source {
        Source(text.into())
    }

    pub fn none() -> Self {
        Source("".into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub struct Ident(Rc<str>);

impl Ident
{
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl<'a, T: Into<Rc<str>>> From<T> for Ident
{
    fn from(value: T) -> Self {
        Ident(value.into())
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Position
{
    pub source: Source,
    pub index: usize
}

#[derive(Clone, PartialEq, Eq)]
pub struct Span
{
    pub source: Source,
    pub start: usize,
    pub end: usize
}

impl Debug for Span
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result 
    {
        f.write_fmt(format_args!("{{start = {}, end = {}}}", self.start, self.end))
    }
}

impl Span
{
    pub fn from(source: &Source, span: pest::Span<'_>) -> Self
    {
        Self { source: source.clone(), start: span.start(), end: span.end() }
    }

    pub fn none() -> Self 
    {
        Self { source: Source::none(), start: 0, end: 0 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity
{
    Error,
    Warning,
    Note
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic
{
    pub span: Span,
    pub severity: Severity,
    pub message: String
}

impl Diagnostic
{
    pub fn new(span: Span, severity: Severity, message: String) -> Self
    {
        Self { span, severity, message }
    }

    pub fn error(span: Span, message: String) -> Self
    {
        Self::new(span, Severity::Error, message)
    }

    pub fn warning(span: Span, message: String) -> Self
    {
        Self::new(span, Severity::Warning, message)
    }

    pub fn note(span: Span, message: String) -> Self
    {
        Self::new(span, Severity::Note, message)
    }
}