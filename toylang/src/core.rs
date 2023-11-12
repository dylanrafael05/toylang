pub mod arena;
pub mod lazy;
pub mod many;
pub mod monad;
pub mod ops;

use std::{fmt::{Debug, Display}, ops::{Add, Index, Range}, rc::Rc, sync::Arc, io::{self, Write}};

use lazy_static::lazy_static;
use termcolor::{Color, WriteColor, ColorSpec};

pub trait AsU64 {
    fn as_u64(&self) -> u64;
}

impl AsU64 for bool {
    fn as_u64(&self) -> u64 {
        if *self {
            1
        } else {
            0
        }
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Src {
    name: Arc<str>,
    content: Arc<str> 
}

lazy_static! {
    static ref NO_SRC: Src = Src { 
        name: Arc::from(""), 
        content: Arc::from("") 
    };
}

impl Index<usize> for Src {
    type Output = u8;
    fn index(&self, index: usize) -> &Self::Output {
        &self.name.as_bytes()[index]
    }
}

pub const EOI_CHAR: char = 1 as char;

impl Src {
    pub fn new(name: &str, text: &str) -> Self {
        let text = text.replace('\t', "    ");
        Src {
            name: Arc::from(name),
            content: Arc::from(text)
        }
    }

    pub fn none() -> Self {
        NO_SRC.clone()
    }

    pub fn as_str(&self) -> &str {
        self.content.as_ref()
    }

    pub fn name(&self) -> &str {
        self.name.as_ref()
    }

    pub fn at(&self, pos: &Pos) -> u8 {
        self.peek(pos, 0)
    }
    pub fn peek(&self, pos: &Pos, off: isize) -> u8 {
        if self.content.as_bytes().len() as isize <= pos.index as isize + off {
            EOI_CHAR as u8
        } else {
            self.content.as_bytes()[(pos.index as isize + off) as usize]
        }
    }
}

#[derive(Hash, Clone, Debug, PartialEq, Eq)]
pub struct Ident(Rc<str>);

impl Ident {
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn to_string(&self) -> String {
        self.0.to_string()
    }
}

impl<'a, T: Into<Rc<str>>> From<T> for Ident {
    fn from(value: T) -> Self {
        Ident(value.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pos {
    pub index: usize,
    pub col: usize,
    pub line: usize
}

impl Default for Pos {
    fn default() -> Self {
        Pos { index: 0, col: 0, line: 1 }
    }
}

impl Display for Pos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.line, self.col)
    }
}

impl PartialOrd for Pos {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.index.partial_cmp(&other.index)
    }
}
impl Ord for Pos {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Pos {
    pub fn incr(&mut self) {
        self.index += 1;
        self.col += 1;
    }
    pub fn decr(&mut self) {
        self.index -= 1;
        self.col -= 1;
    }
    pub fn incr_line(&mut self) {
        self.index += 1;
        self.col = 0;
        self.line += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: Pos,
    pub end: Pos
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Self {
            start,
            end
        }
    }
    pub fn as_range(&self) -> Range<usize> {
        self.start.index .. self.end.index
    }
    pub fn text<'a>(&self, src: &'a Src) -> &'a str {
        if self.start.index >= src.as_str().len() {"<EOI>"}
        else {&src.as_str()[self.as_range()]}
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourcePos {
    pub source: Src,
    pub index: Pos
}
impl Display for SourcePos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}:{}", self.source.name(), self.index.line, self.index.col)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct SourceSpan {
    pub source: Src,
    pub start: Pos,
    pub end: Pos,
}

impl Debug for SourceSpan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{{start = {}, end = {}}}",
            self.start, self.end
        ))
    }
}

impl SourceSpan {
    pub fn new(source: Src, span: Span) -> Self {
        Self { source, start: span.start, end: span.end }
    }

    pub fn none() -> Self {
        Self {
            source: Src::none(),
            start: Default::default(),
            end: Default::default(),
        }
    }
}

impl Add<Span> for Span {
    type Output = Span;

    fn add(self, rhs: Span) -> Self::Output {
        Self {
            start: std::cmp::min(self.start, rhs.start),
            end: std::cmp::max(self.end, rhs.end),
        }
    }
}

impl Add<Pos> for Span {
    type Output = Span;

    fn add(self, rhs: Pos) -> Self::Output {
        Self {
            start: std::cmp::min(self.start, rhs),
            end: std::cmp::max(self.end, rhs),
        }
    }
}

impl Add<Span> for SourceSpan {
    type Output = SourceSpan;

    fn add(self, rhs: Span) -> Self::Output {
        Self {
            source: self.source,
            start: std::cmp::min(self.start, rhs.start),
            end: std::cmp::max(self.end, rhs.end),
        }
    }
}

impl Add<SourceSpan> for Span {
    type Output = SourceSpan;

    fn add(self, rhs: SourceSpan) -> Self::Output {
        rhs + self
    }
}

impl Add<Pos> for SourceSpan {
    type Output = SourceSpan;

    fn add(self, rhs: Pos) -> Self::Output {
        Self {
            source: self.source,
            start: std::cmp::min(self.start, rhs),
            end: std::cmp::max(self.end, rhs),
        }
    }
}

impl Add<SourceSpan> for SourceSpan {
    type Output = SourceSpan;

    fn add(self, rhs: SourceSpan) -> Self::Output {
        if self.source != rhs.source {
            panic!("Cannot add two spans which derive from differing sources!")
        }

        Self {
            source: self.source.clone(),
            start: std::cmp::min(self.start, rhs.start),
            end: std::cmp::max(self.end, rhs.end),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    pub fn color(&self) -> Color {
        use Severity::*;
        match self {
            Error => Color::Red,
            Warning => Color::Yellow,
            Note => Color::Rgb(120u8, 120u8, 120u8)
        }
    }

    pub fn to_str(&self) -> &str {
        use Severity::*;
        match self {
            Error => "ERROR",
            Warning => "WARNING",
            Note => "NOTE"
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Diagnostic {
    pub span: SourceSpan,
    pub severity: Severity,
    pub message: String,
}

impl Diagnostic {
    pub fn new(span: SourceSpan, severity: Severity, message: String) -> Self {
        Self {
            span,
            severity,
            message,
        }
    }

    pub fn write(&self, io: &mut termcolor::StandardStream) -> io::Result<()> {
        io.set_color(ColorSpec::new()
            .set_fg(Some(self.severity.color()))
            .set_bold(true))?;

        write!(io, "{}: ", self.severity.to_str())?;

        io.set_color(ColorSpec::new()
            .set_bold(true))?;

        write!(io, "{}", self.message)?;

        io.set_color(ColorSpec::new()
            .set_dimmed(true)
            .set_italic(false))?;

        writeln!(io, " (at {}:{}:{})", 
            self.span.source.name(), 
            self.span.start.line, 
            self.span.start.col+1)?;

        let start = self.span.start;
        let end = self.span.end;

        let start_index = self.span.source.as_str()[..start.index]
            .rfind('\n')
            .map(|x| x + 1)
            .unwrap_or(0);

        let end_index = self.span.source.as_str()[start.index..]
            .find('\n')
            .map_or(self.span.source.as_str().len(), |x| x + start.index);

        let lineno = format!(" {} | ", start.line);
        let line = &self.span.source.as_str()[start_index..end_index];
        
        io.set_color(ColorSpec::new()
            .set_fg(Some(Color::Blue))
            .set_bold(true))?;

        write!(io, "{lineno}")?;
        
        io.set_color(&ColorSpec::new())?;

        writeln!(io, "{line}")?;

        let under = (start_index..end_index)
            .into_iter()
            .map(|x| if start.index <= x && x < end.index {'^'} else {' '})
            .collect::<String>();

        let lineno = lineno.chars()
            .map(|x| if x.is_digit(10) {' '} else {x})
            .collect::<String>();

        io.set_color(ColorSpec::new()
            .set_fg(Some(Color::Blue))
            .set_bold(true))?;
        write!(io, "{lineno}")?;
        
        io.set_color(ColorSpec::new()
            .set_dimmed(true))?;
        writeln!(io, "{under}")?;

        io.set_color(&ColorSpec::new())?;

        Ok(())
    }

    pub fn error(span: SourceSpan, message: String) -> Self {
        Self::new(span, Severity::Error, message)
    }

    pub fn warning(span: SourceSpan, message: String) -> Self {
        Self::new(span, Severity::Warning, message)
    }

    pub fn note(span: SourceSpan, message: String) -> Self {
        Self::new(span, Severity::Note, message)
    }
}
