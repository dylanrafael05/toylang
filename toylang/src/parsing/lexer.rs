use crate::{core::{Src, Span, Pos, SourceSpan, EOI_CHAR, Diagnostic}, binding::ProgramContext};

use super::token::{Token, TokenKind};

pub struct Lexer<'a> {
    pub(super) source: Src,
    pub(super) start: Pos,
    pos: Pos,
    pub(super) ctx: &'a mut ProgramContext
}

impl<'a> Lexer<'a> {
    fn produce(&self, kind: TokenKind) -> Token {
        Token::new(self.span(), kind)
    }
    fn produce_one(&mut self, kind: TokenKind) -> Token {
        self.pos.incr();
        self.produce(kind)
    }

    fn cur(&self) -> char {
        self.source.at(&self.pos) as char
    }
    fn peek(&self, n: isize) -> char {
        self.source.peek(&self.pos, n) as char
    }

    fn match_rep(&self, c: char, n: usize) -> bool {
        (0..n).all(|i| self.peek(i as isize) == c)
    }

    fn span(&self) -> Span {
        Span::new(self.start, self.pos)
    }

    fn srcspan(&self) -> SourceSpan {
        SourceSpan::new(self.source.clone(), self.span())
    }

    pub fn new(source: Src, ctx: &'a mut ProgramContext) -> Lexer<'a> {
        Lexer { source, start: Default::default(), pos: Default::default(), ctx }
    }
    
    pub fn next(&mut self) -> Token {
        macro_rules! match_char {
            ($($ch: pat $(, $che: expr)? => $tt: tt),* , ! => $ttl: tt) => {{
                match self.cur() {
                    $($ch $(if $che)? => {
                        // println!("Hello from {}: {}", self.cur(), stringify!($ch));
                        match_char!(@result $tt)
                    })*
                    _ => match_char!(@result $ttl)
                }
            }};

            (@result error) => {{
                self.pos.incr();
                self.ctx.diags.add_error_at(
                    self.srcspan(), 
                    format!("Invalid token {}", self.span().text(&self.source)));
                self.produce(TokenKind::Error)
            }};
            
            (@result $blk: block) => {
                $blk
            };
            
            (@result $blk: ident) => {
                self.produce_one($blk)
            };
            
            (@result [$($ch: pat => $tt: tt),* , ! => $ttl: tt]) => {{
                let pos = self.pos;
                self.pos.incr();
                match self.cur() {
                    $($ch => match_char!(@result $tt),)*
                    _ => {
                        self.pos = pos;
                        match_char!(@result $ttl)
                    }
                }
            }};
        }
        
        use TokenKind::*;
        
        // WHITESPACES //
        loop {
            match_char! [
                ' '|'\t' => {self.pos.incr();},

                '\n'|'\r' => [
                    '\r' => {self.pos.incr_line();},
                    '\n' => {self.pos.incr_line();},
                    !    => {self.pos.incr_line();}
                ],

                '/' => [
                    '/' => {
                        while !matches!(self.cur(), '\n'|'\r'|EOI_CHAR) {
                            self.pos.incr();
                        }
                    },

                    '*' => {
                        let mut count = 1;
                        while self.cur() == '*' {
                            count += 1;
                            self.pos.incr();
                        }

                        while !self.match_rep('*', count) && self.peek(count as isize) != '/' {
                            self.pos.incr();
                            if self.cur() == EOI_CHAR {
                                self.ctx.diags.add(Diagnostic::error(
                                    self.srcspan(), 
                                    format!("Unterminated long comment")));
                                break
                            }
                        }

                        for _ in 0..=count {
                            self.pos.incr();
                        }
                    },

                    ! => {break}
                ],

                ! => {break}
            ];
        }

        // ACTUAL TOKENS //
        self.start = self.pos;

        match_char! [
            EOI_CHAR => EOI,

            ',' => Comma,
            '.' => Dot,
            ':' => [
                ':' => StaticDot,
                !   => Colon
            ],

            ';' => Semicolon,

            '>' => [
                '=' => GreaterEqual,
                !   => GreaterThan
            ],

            '<' => [
                '=' => LessEqual,
                !   => LessThan
            ],

            '(' => OpenParen,
            ')' => CloseParen,
            
            '[' => OpenBrace,
            ']' => CloseBrace,

            '{' => OpenCurly,
            '}' => CloseCurly,

            '+' => Add,
            '-' => Sub,
            '*' => [
                '&' => PtrAmpersand,
                !   => Star
            ],
            '/' => Slash,

            '%' => Modulo,
            '&' => Ampersand,
            
            '=' => [
                '=' => DoubleEqual,
                !   => Equal
            ],

            '!' => [
                '=' => NotEqual,
                !   => error
            ],

            'a'..='z'|'A'..='Z'|'_' => {
                while self.cur().is_ascii_alphanumeric() || self.cur() == '_' {
                    self.pos.incr();
                }

                self.produce(match self.span().text(&self.source) {
                    
                    "if"       => If,
                    "else"     => Else,
                    "while"    => While,
                    "fn"       => Fn,
                    "let"      => Let,
                    "mut"      => Mut,
                    "mod"      => Mod,
                    "struct"   => Struct,
                    "declare"  => Declare,
                    "as"       => As,
                    "vararg"   => Vararg,
                    "sizeof"   => Sizeof,

                    "import"   => Import,

                    "return"   => Return,
                    "break"    => Break,
                    "continue" => Continue,

                    "and"      => And,
                    "or"       => Or,
                    "not"      => Not,

                    _          => Identifier
                })
            },

            '0'..='9' => {
                while self.cur().is_digit(10) {
                    self.pos.incr();
                }

                if self.cur() == '.' {
                    while self.cur().is_digit(10) {
                        self.pos.incr();
                    }

                    self.produce(Decimal)
                } else {
                    self.produce(Integer)
                }
            },

            '"' => {
                self.pos.incr();

                while (self.cur() != '"' 
                    || self.cur() == '"' && self.peek(-1) == '\\')
                    && self.cur() != EOI_CHAR {
                    self.pos.incr();
                }

                self.pos.incr();

                if self.cur() == EOI_CHAR {
                    self.ctx.diags.add_error_at(self.srcspan(), format!("Unterminated string literal"));
                    self.produce(Error)
                } else {
                    self.produce(String)
                }
            },

            ! => error
        ]
    }

    pub fn collect(&mut self) -> Vec<Token> {
        let mut body = vec![];
        loop {
            let next = self.next();
            body.push(next.clone());

            if next.kind == TokenKind::EOI {break}
        }

        body
    }
}