use std::cell::OnceCell;

use crate::{core::{Diagnostic, SourceSpan, Pos, ops::{BinOp, UnOp, ArithOp, CompOp}, Severity, Ident, Src, monad::Wrapper, utils::FlagCell}, binding::ProgramContext};

use super::{ast::{ExprAST, AST, Expr, StmtAST, Block, Stmt, IfNode, ElseKind, TypeSpecAST, TypeSpec, ArgSpec, StructDefSpec, File}, lexer::Lexer, token::{TokenKind, Token}};


pub struct Parser<'a> {
    src: Src,
    tokens: Vec<Token>,
    index: usize,
    ctx: &'a mut ProgramContext
}

type Result<T> = std::result::Result<T, Diagnostic>;
trait Parse<'a> = Fn(&mut Parser<'a>) -> Result<ExprAST>;

impl<'a> Parser<'a> {
    fn cur(&self) -> &Token {
        &self.peek(0)
    }
    fn cur_kind(&self) -> &TokenKind {
        &self.cur().kind
    }

    fn peek(&self, off: isize) -> &Token {
        let sum = (self.index as isize + off) as usize;
        if sum < self.tokens.len() {
            &self.tokens[sum]
        } else {
            self.tokens.last().unwrap()
        }
    }
    fn peek_kind(&self, off: isize) -> &TokenKind {
        &self.peek(off).kind
    }

    fn advance(&mut self) -> Token {
        let tok = self.cur().clone();
        self.index += 1;
        tok
    }
    fn advance_with<T>(&mut self, v: T) -> T {
        self.advance();
        v
    }
    
    fn advance_to(&mut self, tts: &[TokenKind]) {
        while !tts.contains(self.cur_kind()) && self.cur_kind() != &TokenKind::EOI {
            self.advance();
        }
    }

    fn advance_past(&mut self, tts: &[TokenKind]) {
        self.advance_to(tts);

        while tts.contains(self.cur_kind()) && self.cur_kind() != &TokenKind::EOI {
            self.advance();
        }
    }

    fn text(&self) -> &str {
        self.cur().text(&self.src)
    }
    fn srcspan(&self) -> SourceSpan {
        SourceSpan::new(self.src.clone(), self.cur().span.clone())
    }
    fn prev_end(&self) -> Pos {
        if self.index == 0 {self.cur().span.start}
        else {self.peek(-1).span.end}
    }

    fn error<T>(&mut self, msg: &str) -> Result<T> {
        self.advance_with(
            Err(Diagnostic::error(self.srcspan(), msg.to_owned())))
    }

    fn expect(&mut self, tt: &[TokenKind], msg: &str) -> Result<Token> {
        let span = self.srcspan();
        let tok = self.advance();
        if !tt.contains(&tok.kind) && tok.not_error() {
            Err(Diagnostic::new(span, Severity::Error, msg.to_owned()))
        } else {
            Ok(tok)
        }
    }

    fn left_op(&mut self, tts: &[(TokenKind, BinOp)], next: impl Parse<'a>) -> Result<ExprAST> {
        let lhs = next(self)?;
        if let Some((_, op)) = tts.iter().find(|(tt, _)| tt == self.cur_kind()) {
            self.advance();
            let rhs = self.left_op(tts, next)?;
            Ok(AST::new(
                lhs.span.clone() + rhs.span.clone(),
                Expr::Binary { op: op.clone(), lhs: Box::new(lhs), rhs: Box::new(rhs) }
            ))
        } else {
            Ok(lhs)
        }
    }

    pub fn new(mut lexer: Lexer<'a>) -> Parser<'a> {
        let tokens = lexer.collect();

        // for tok in &tokens {
        //     println!("{:?}: {:?}", tok.kind, tok.text(&lexer.source))
        // }

        Parser {
            src: lexer.source.clone(),
            ctx: lexer.ctx,
            index: 0,
            
            tokens
        }
    }

    fn parse_str(span: &SourceSpan, s: &str) -> Result<String> {
        let mut out = String::new();
        out.reserve_exact(s.len());

        let mut chars = s[1..s.len()-1].chars();
        let mut cm = chars.next();

        while let Some(c) = cm {
            if c == '\\' {
                cm = chars.next();
                out.push(match cm.unwrap() {
                    'n'  => '\n',
                    'r'  => '\r',
                    't'  => '\t',
                    '0'  => '\0',
                    '\\' => '\\',
                    '"'  => '"',

                    n => return Err(
                        Diagnostic::error(
                            span.clone(), 
                            format!("Unknown escape character '\\{}'", n))),
                });
            } else {out.push(c);}

            cm = chars.next();
        }

        Ok(out)
    }

    pub fn term(&mut self) -> Result<ExprAST> {
        match self.cur_kind() {
            TokenKind::Integer => {
                self.advance_with(Ok(AST::new(
                    self.srcspan(), 
                    Expr::Integer(self.text().parse().unwrap(), None))))
            }

            TokenKind::Decimal => {
                self.advance_with(Ok(AST::new(
                    self.srcspan(), 
                    Expr::Decimal(self.text().parse().unwrap(), None))))
            }
            
            TokenKind::String => {
                self.advance_with(Ok(AST::new(
                    self.srcspan(), 
                    Expr::String(Self::parse_str(&self.srcspan(), self.text())?))))
            }

            TokenKind::Identifier => {
                self.advance_with(Ok(AST::new(
                    self.srcspan(), 
                    Expr::Identifier(self.text().into()))))
            }

            TokenKind::OpenParen => {
                self.advance();
                let inner = self.expr()?;
                self.expect(&[TokenKind::CloseParen], "Expected ')' to match '('")?;

                Ok(inner)
            }

            TokenKind::Sizeof => {
                let start = self.srcspan();

                self.advance();
                let ty = self.typespec()?;
                
                Ok(AST::new(
                    start + self.prev_end(),
                    Expr::Sizeof(ty.content)
                ))
            }

            TokenKind::OpenCurly => Ok(self.block().map(Expr::Block)),

            _ => self.error("Expected an expression")
        }
    }

    pub fn comma_sep_full<T>(&mut self, item: impl Fn(&mut Parser<'a>) -> Result<T>, enders: &[TokenKind]) -> Result<Vec<T>> {
        let mut out = vec![];
        while !enders.contains(self.cur_kind()) {
            if !out.is_empty() {
                self.expect(&[TokenKind::Comma], "Expected a comma")?;
            }
            out.push(item(self)?);
        }
        Ok(out)
    }
    pub fn comma_sep<T>(&mut self, item: impl Fn(&mut Parser<'a>) -> Result<AST<T>>, enders: &[TokenKind]) -> Result<Vec<T>> {
        self.comma_sep_full(|x| item(x).map(|x| x.content), enders)
    }
    
    pub fn static_dot(&mut self) -> Result<ExprAST> {
        let mut lhs = self.term()?;
        
        while self.cur_kind() == &TokenKind::StaticDot && self.peek_kind(1) != &TokenKind::OpenCurly {
            self.advance();
            match self.cur_kind() {
                TokenKind::Identifier => {
                    lhs = AST::new(
                        self.srcspan() + lhs.span.clone(),
                        Expr::StaticDot { op: Box::new(lhs), id: self.ident()?.content }
                    );
                }

                TokenKind::OpenBrace => {
                    self.advance();

                    let args = self.comma_sep_full(Self::typespec, &[TokenKind::CloseBrace])?;

                    self.expect(&[TokenKind::CloseBrace], "Expected a closing '[' for generic argument specification")?;

                    lhs = AST::new(
                        self.srcspan() + lhs.span.clone(),
                        Expr::Gen { op: Box::new(lhs), args }
                    );
                }

                _ => {
                    self.ctx.diags.add_error_at(self.srcspan(), format!("Expected a valid identifier as the right-hand-side of a '::' expression"))
                }
            }   
        }

        Ok(lhs)
    }

    pub fn dot(&mut self) -> Result<ExprAST> {
        let mut lhs = self.static_dot()?;

        loop {
            if self.cur_kind() == &TokenKind::Dot {
                self.advance();
                let ident = self.expect(
                    &[TokenKind::Identifier], 
                    "Expected a valid identifier as the right-hand-side of a '.' expression")?;

                lhs = AST::new(
                    self.srcspan() + lhs.span.clone(),
                    Expr::Dot { op: Box::new(lhs), id: ident.text(&self.src).into() }
                );
            } else if self.cur_kind() == &TokenKind::OpenParen {
                self.advance();
                let args = self.comma_sep_full(Self::expr, &[TokenKind::CloseParen])?;
                let end = self.expect(&[TokenKind::CloseParen], "Expected closing parenthesis for function call")?;
    
                lhs = AST::new(
                    lhs.span.clone() + end.span,
                    Expr::Call { func: Box::new(lhs), args }
                );
            } else {break}
        }

        Ok(lhs)
    }

    pub fn unary(&mut self) -> Result<ExprAST> {
        match self.cur_kind() {
            TokenKind::PtrAmpersand => {
                let start = self.advance();
                let inner = self.unary()?;
                Ok(AST::new(
                    start.span + inner.span.clone(),
                    Expr::Addressof(Box::new(inner))
                ))
            }
            
            TokenKind::Ampersand => {
                let start = self.advance();
                let inner = self.unary()?;
                Ok(AST::new(
                    start.span + inner.span.clone(),
                    Expr::Ref(Box::new(inner))
                ))
            }

            TokenKind::Star => {
                let start = self.advance();
                let inner = self.unary()?;
                Ok(AST::new(
                    start.span + inner.span.clone(),
                    Expr::Deref(Box::new(inner))
                ))
            }
            
            TokenKind::Not => {
                let start = self.advance();
                let inner = self.unary()?;
                Ok(AST::new(
                    start.span + inner.span.clone(),
                    Expr::Unary { op: UnOp::Not, target: Box::new(inner) }
                ))
            }
            
            TokenKind::Sub => {
                let start = self.advance();
                let inner = self.unary()?;
                Ok(AST::new(
                    start.span + inner.span.clone(),
                    Expr::Unary { op: UnOp::Negate, target: Box::new(inner) }
                ))
            }

            _ => self.dot()
        }
    }

    pub fn r#as(&mut self) -> Result<ExprAST> {
        let start = self.srcspan();
        let op = self.unary()?;

        let expr = if self.cur_kind() == &TokenKind::As {
            self.advance();
            let typ = self.typespec()?.content;

            Expr::Cast { op: Box::new(op), typ }
        } else {op.content};

        Ok(AST::new(
            start + self.prev_end(),
            expr
        ))
    }

    pub fn mul(&mut self) -> Result<ExprAST> {
        self.left_op(&[
            (TokenKind::Star,  BinOp::Arith(ArithOp::Multiply)), 
            (TokenKind::Slash, BinOp::Arith(ArithOp::Divide)),
            (TokenKind::Modulo, BinOp::Arith(ArithOp::Modulo))
        ], Self::r#as)
    }
    pub fn add(&mut self) -> Result<ExprAST> {
        self.left_op(&[
            (TokenKind::Add, BinOp::Arith(ArithOp::Add)), 
            (TokenKind::Sub, BinOp::Arith(ArithOp::Subtract))
        ], Self::mul)
    }
    pub fn comp(&mut self) -> Result<ExprAST> {
        self.left_op(&[
            (TokenKind::DoubleEqual,  BinOp::Equals),
            (TokenKind::NotEqual,     BinOp::NotEquals),
            (TokenKind::GreaterThan,  BinOp::Comparison(CompOp::GreaterThan)),
            (TokenKind::GreaterEqual, BinOp::Comparison(CompOp::GreaterEqual)),
            (TokenKind::LessThan,     BinOp::Comparison(CompOp::LessThan)),
            (TokenKind::LessEqual,    BinOp::Comparison(CompOp::LessEqual)),
        ], Self::add)
    }
    pub fn or(&mut self) -> Result<ExprAST> {
        self.left_op(&[(TokenKind::Or, BinOp::Or)], Self::comp)
    }
    pub fn and(&mut self) -> Result<ExprAST> {
        self.left_op(&[(TokenKind::And, BinOp::And)], Self::or)
    }

    /// Always puts diagnostics into the context.
    pub fn any_block(&mut self, next: impl Fn(&mut Parser<'a>) -> Result<StmtAST>) -> AST<Block> {
        let start = self.srcspan();

        let _ = self.expect(&[TokenKind::OpenCurly], "Expected an open curly brace!").map_err(|d| {
            self.ctx.diags.add(d);
            self.advance_past(&[TokenKind::OpenCurly]);
        });

        let mut body = vec![];
        let mut tail = None;

        while self.cur_kind() != &TokenKind::CloseCurly {
            let stmt = next(self).unwrap_or_else(|d| {
                self.ctx.diags.add(d);
                self.advance_to(&[TokenKind::Semicolon, TokenKind::CloseCurly]);
                AST::new(SourceSpan::none(), Stmt::Error)
            });

            // Handle inner statements and final statement
            if self.cur_kind() == &TokenKind::Semicolon {
                self.advance();
                body.push(stmt);
            } else if let Stmt::Expr(expr) = stmt.content {
                tail = Some(Box::new(expr));
                break;
            } else {
                break;
            }
        }

        let _ = self.expect(&[TokenKind::CloseCurly], "Expected a closing curly brace or semicolon!").map_err(|d| {
            self.ctx.diags.add(d);
            self.advance_past(&[TokenKind::CloseCurly]);
        });

        AST::new(
            start + self.prev_end(),
            Block { body, tail }
        )
    }

    pub fn block(&mut self) -> AST<Block> {
        self.any_block(Self::stmt)
    }

    pub fn r#if(&mut self) -> Result<AST<IfNode>> {
        let start = self.srcspan();
        self.expect(&[TokenKind::If], "Expected an 'if' keyword")?;

        let cond = Box::new(self.expr().unwrap_or_else(|d| {
            self.ctx.diags.add(d);
            self.advance_to(&[TokenKind::OpenCurly]);

            AST::new(SourceSpan::none(), Expr::Error)
        }));
        
        let block = self.block().content;

        let tail = match self.cur_kind() {
            TokenKind::Else => {
                self.advance();
                match self.cur_kind() {
                    TokenKind::If => {
                        let inner = self.r#if()?;
                        ElseKind::ElseIf(Box::new(inner))
                    }

                    _ => {
                        let block = self.block();
                        ElseKind::Else(block)
                    }
                }
            }

            _ => ElseKind::None
        };

        Ok(AST::new(
            start + self.prev_end(),
            IfNode { cond, block, tail }
        ))
    }

    pub fn expr(&mut self) -> Result<ExprAST> {
        match self.cur_kind() {
            TokenKind::If => Ok(self.r#if()?.map(Expr::If)),

            _ => self.and()
        }
    }

    // TODO: make more like static_dot?
    pub fn typespec_ident(&mut self) -> Result<TypeSpecAST> {
        let mut span = self.srcspan();
        let mut idents: Vec<Ident> = vec![
            self.expect(
                &[TokenKind::Identifier], 
                "Unexpected non-identifier type name")?
            .text(&self.src)
            .into()
        ];
        
        while self.cur_kind() == &TokenKind::StaticDot {
            self.advance();
            let ident = self.expect(
                &[TokenKind::Identifier], 
                "Expected a valid identifier as the right-hand-side of a '::' expression")?;
            span = span + ident.span;

            idents.push(ident.text(&self.src).into())
        }

        match idents.as_slice()
        {
            [x] => Ok(AST::new(
                span,
                TypeSpec::Named(x.clone())
            )),

            [y @ .., x] => Ok(AST::new(
                span,
                TypeSpec::Scoped(Vec::from(y), x.clone())
            )),

            [] => panic!("Should never be like this!")
        }
    }

    pub fn typespec(&mut self) -> Result<TypeSpecAST> {
        match self.cur_kind() {
            TokenKind::Ampersand => {
                let amp = self.advance();
                let inner = self.typespec()?;
                Ok(AST::new(
                    amp.span + inner.span.clone(),
                    TypeSpec::Ref(Box::new(inner))
                ))
            }

            TokenKind::Star => {
                let amp = self.advance();
                let inner = self.typespec()?;
                Ok(AST::new(
                    amp.span + inner.span.clone(),
                    TypeSpec::Ptr(Box::new(inner))
                ))
            }

            _ => {
                let base = self.typespec_ident()?;

                if self.cur_kind() == &TokenKind::OpenBrace {
                    self.advance();

                    let args = self.comma_sep(Self::typespec, &[TokenKind::CloseBrace])?;

                    let cls = self.expect(&[TokenKind::CloseBrace], "Expected a closing '[' for generic argument specification")?;

                    Ok(AST::new(
                        base.span + cls.span,
                        TypeSpec::Gen(Box::new(base.content), args)
                    ))
                } else {Ok(base)}
            }
        }
    }

    pub fn r#while(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::While], "Expected a 'while' keyword")?;

        let cond = self.expr().unwrap_or_else(|d| {
            self.ctx.diags.add(d);
            self.advance_to(&[TokenKind::OpenCurly]);

            AST::new(SourceSpan::none(), Expr::Error)
        });

        let block = self.block().content;

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::While {cond, block}
        ))
    }

    pub fn ident(&mut self) -> Result<AST<Ident>> {
        let span = self.srcspan();
        let i = self.expect(&[TokenKind::Identifier], "Expected an identifier")?;
        Ok(AST::new(
            span,
            i.text(&self.src).into()
        ))
    }

    pub fn r#let(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Let], "Expected a 'let' keyword")?;

        let name = self.ident()?.content;

        let ty = if self.cur_kind() != &TokenKind::Equal {
            Some(self.typespec()?)
        } else {None};

        self.expect(&[TokenKind::Equal], "Expected a '=' during 'let' statement")?;

        let value = self.expr()?;

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::Let { name, ty, value }
        ))
    }

    pub fn arg(&mut self) -> Result<AST<ArgSpec>> {
        let start = self.srcspan();

        let name = self.ident()?.content;
        let type_spec = self.typespec()?;

        Ok(AST::new(
            start + self.prev_end(),
            ArgSpec {name, type_spec}
        ))
    }

    pub fn func(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Fn], "Expected 'fn' keyword")?;

        let name = self.ident()?.content;

        let gen_args = if self.cur_kind() == &TokenKind::OpenBrace {
            self.advance();

            let args = self.comma_sep(Self::ident, &[TokenKind::CloseBrace])?;

            self.expect(&[TokenKind::CloseBrace], "Expected a closing '[' for generic argument specification")?;

            args
        } else {vec![]};

        self.expect(&[TokenKind::OpenParen], "Expected open parentheses")?;
        
        let args = self.comma_sep_full(Self::arg, &[TokenKind::CloseParen])?;

        self.expect(&[TokenKind::CloseParen], "Expected closing parentheses")?;

        let ty = if self.can_start_ty() {
            Some(self.typespec()?)
        } else {None};

        self.expect(&[TokenKind::Colon], "Expected colon")?;

        let body = self.expr()?;

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::Fn { name, gen_args, args, ty, body, error: FlagCell::new() }
        ))
    }

    const START_TY: [TokenKind; 3] = [TokenKind::Ampersand, TokenKind::Star, TokenKind::Identifier];

    fn can_start_ty(&self) -> bool {
        Self::START_TY.contains(self.cur_kind())
    }

    pub fn declare_func(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Declare], "Expected 'declare' keyword")?;

        let vararg = if self.cur_kind() == &TokenKind::Vararg {
            self.advance();
            true
        } else {false};

        self.expect(&[TokenKind::Fn], "Expected 'fn' keyword")?;

        let name = self.ident()?.content;

        self.expect(&[TokenKind::OpenParen], "Expected open parentheses")?;
        
        let args = self.comma_sep(Self::arg, &[TokenKind::CloseParen])?;

        self.expect(&[TokenKind::CloseParen], "Expected closing parentheses")?;

        let ty = if self.can_start_ty() {Some(self.typespec()?)} else {None};

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::DeclareFn { name, args, ty, vararg }
        ))
    }

    pub fn r#struct(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Struct], "Expected 'struct' keyword")?;

        let name = self.ident()?.content;

        let args = if self.cur_kind() == &TokenKind::OpenBrace {
            self.advance();

            let args = self.comma_sep(Self::ident, &[TokenKind::CloseBrace])?;

            self.expect(&[TokenKind::CloseBrace], "Expected a closing brace for generic args")?;

            args
        } else {vec![]};

        self.expect(&[TokenKind::OpenCurly], "Expected opening to struct body")?;

        let fields = self.comma_sep(Self::arg, &[TokenKind::CloseCurly])?;

        self.expect(&[TokenKind::CloseCurly], "Expected close to struct body")?;

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::Struct(StructDefSpec{name, fields, args, error: FlagCell::new()})
        ))
    }

    pub fn expr_stmt(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        let lhs = self.expr()?;

        let stmt = if self.cur_kind() == &TokenKind::Equal {
            self.advance();

            let rhs = self.expr()?;
            Stmt::Assign { lhs, rhs }
        } else {Stmt::Expr(lhs)};

        Ok(AST::new(start + self.prev_end(), stmt))
    }

    pub fn import(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Import], "Expected an 'import' keyword")?;
        let str = self.expect(&[TokenKind::String], "Expected a 'string' import")?;

        Ok(AST::new(
            start + self.prev_end(),
            Stmt::Import(Self::parse_str(&self.srcspan(), str.text(&self.src))?)
        ))
    }

    pub fn r#continue(&mut self) -> Result<StmtAST> {
        let span = self.srcspan();

        self.expect(&[TokenKind::Continue], "Expected a 'continue' keyword")?;

        Ok(AST::new(
            span, Stmt::Continue
        ))
    }
    pub fn r#break(&mut self) -> Result<StmtAST> {
        let span = self.srcspan();

        self.expect(&[TokenKind::Break], "Expected a 'continue' keyword")?;

        Ok(AST::new(
            span, Stmt::Break
        ))
    }

    const NOT_EXPR_START: &'static [TokenKind] = &[TokenKind::Semicolon, TokenKind::CloseCurly];
    
    pub fn r#return(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Return], "Expected a 'return' keyword")?;
        let retval = if Self::NOT_EXPR_START.contains(self.cur_kind()) {None} else {
            Some(self.expr()?)
        };

        Ok(AST::new(
            start + self.prev_end(), 
            Stmt::Return(retval)
        ))
    }

    pub fn stmt(&mut self) -> Result<StmtAST> {
        match self.cur_kind() {
            TokenKind::Struct   => self.r#struct(),
            TokenKind::While    => self.r#while(),
            TokenKind::Fn       => self.func(),
            TokenKind::Let      => self.r#let(),

            TokenKind::Break    => self.r#break(),
            TokenKind::Continue => self.r#continue(),
            TokenKind::Return   => self.r#return(),

            _ => self.expr_stmt()
        }
    }

    pub fn many_until<T>(&mut self, f: impl Fn(&mut Parser<'a>) -> Result<T>, starters: &[TokenKind], enders: &[TokenKind]) -> Vec<T> {
        let mut out = vec![];
        while !enders.contains(self.cur_kind()) {
            match f(self) {
                Ok(x) => out.push(x),
                Err(n) => {
                    self.ctx.diags.add(n);
                    self.advance_to(starters);
                }
            }
        }
        out
    }

    pub fn r#mod(&mut self) -> Result<StmtAST> {
        let start = self.srcspan();

        self.expect(&[TokenKind::Mod], "Expected 'mod' keyword")?;

        let name = self.ident()?.content;

        let stmt = match self.cur_kind() {
            TokenKind::OpenCurly => {
                self.advance();
                let body = self.many_until(Self::top, &Self::TOP_START, &[TokenKind::CloseCurly]);
                self.expect(&[TokenKind::CloseCurly], "Expected a closing curly brace for 'mod' declaration")?;
                Stmt::Mod { name, body, scope_id: OnceCell::new(), error: FlagCell::new() }
            },

            _ => {
                Stmt::ModHead { name, scope_id: OnceCell::new(), error: FlagCell::new() }
            }
        };

        Ok(AST::new(
            start + self.prev_end(),
            stmt
        ))
    }

    pub fn top(&mut self) -> Result<StmtAST> {
        match self.cur_kind() {
            TokenKind::Mod => self.r#mod(),
            TokenKind::Declare => self.declare_func(),
            TokenKind::Fn => self.func(),
            TokenKind::Struct => self.r#struct(),
            TokenKind::Import => self.import(),

            _ => Err(Diagnostic::error(self.srcspan(), format!("Invalid start of top-level statement")))
        }
    }

    const TOP_START: &'static [TokenKind] = &[TokenKind::Mod, TokenKind::Struct, TokenKind::Declare, TokenKind::Fn, TokenKind::Import];

    pub fn file(&mut self) -> File {
        let mut body = vec![];

        while self.cur_kind() != &TokenKind::EOI {
            match self.top() {
                Ok(x) => body.push(x),
                Err(d) => self.ctx.diags.add(d)
            }

            self.advance_to(Self::TOP_START)
        }

        File(body)
    }

}
