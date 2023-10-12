use crate::binding::ProgramContext;
use crate::binding::Type;
use crate::core::Ident;
use crate::core::Span;
use crate::core::monad::Wrapper;
use crate::core::ops::ArithOp;
use crate::core::ops::CompOp;
use crate::core::ops::ConstFloat;
use crate::core::ops::ConstInt;
use crate::core::ops::{UnOp, BinOp};

pub trait Symbol
    where Self: Sized
{
    type Kind;

    fn kind(&self) -> &Self::Kind;
}

pub trait OwnedSymbol : Symbol
    where Self: Sized
{
    fn into_kind(self) -> Self::Kind;
}

impl<T : Symbol> Symbol for &T
{
    type Kind = T::Kind;

    fn kind(&self) -> &Self::Kind {
        (*self).kind()
    }
}

pub trait UnboundSymbol
{
    type Bound : BoundSymbol;
    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound;
}
pub trait BoundSymbol
{
    fn none() -> Self;
    fn error() -> Self;
}

pub trait TypedSymbol
{
    fn get_type(&self) -> &Type;
}

#[derive(Debug, PartialEq, Clone)]
pub struct AST<Content>
{
    pub content: Content,
    pub span: Span
}

impl<Content : Symbol> Symbol for AST<Content>
{
    type Kind = Content::Kind;

    fn kind(&self) -> &Self::Kind {
        self.content.kind()
    }
}
impl<Content : OwnedSymbol> OwnedSymbol for AST<Content>
{
    fn into_kind(self) -> Self::Kind {
        self.content.into_kind()
    }
}

impl<Content : BoundSymbol> BoundSymbol for AST<Content> {
    fn error() -> Self {
        AST { content: Content::error(), span: Span::none() }
    }
    fn none() -> Self {
        AST { content: Content::none(), span: Span::none() }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Typed<Content>
{
    content: Content,
    typ: Type
}

impl<C> Typed<C>
{
    pub fn new(typ: Type, content: C) -> Self
    {
        Self { content, typ }
    }
}

impl<C> TypedSymbol for Typed<C>
{
    fn get_type(&self) -> &Type {
        &self.typ
    }
}

impl<Sym> Wrapper for Typed<Sym>
{
    type Value = Sym;
    type Of<T> = Typed<T>;
    
    fn map<T, F: FnOnce(Self::Value) -> T>(self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            typ: self.typ.clone(),
            content: func(self.content)
        }
    }

    fn map_ref<'a, T, F: FnOnce(&'a Sym) -> T>(&'a self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            typ: self.typ.clone(),
            content: func(&self.content)
        }
    }
}

impl<Sym : Symbol> Symbol for Typed<Sym>
{
    type Kind = Sym::Kind;

    fn kind(&self) -> &Self::Kind {
        self.content.kind()
    }
}
impl<Sym : OwnedSymbol> OwnedSymbol for Typed<Sym>
{
    fn into_kind(self) -> Self::Kind {
        self.content.into_kind()
    }
}

impl<T: TypedSymbol> TypedSymbol for AST<T>
{
    fn get_type(&self) -> &Type {
        self.content.get_type()
    }
}

impl<Sym> Wrapper for AST<Sym>
{
    type Value = Sym;
    type Of<T> = AST<T>;

    fn map<T, F: FnOnce(Self::Value) -> T>(self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            span: self.span.clone(),
            content: func(self.content)
        }
    }

    fn map_ref<'a, T, F: FnOnce(&'a Sym) -> T>(&'a self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            span: self.span.clone(),
            content: func(&self.content)
        }
    }
}

impl<Sym : UnboundSymbol> AST<Sym>
{
    pub fn bind(&self, ctx: &mut ProgramContext) -> AST<Sym::Bound> {
        AST::<Sym::Bound> {
            content: self.bind_content(ctx),
            span: self.span.clone()
        }
    }
    pub fn bind_content(&self, ctx: &mut ProgramContext) -> Sym::Bound
    {
        ctx.diags.push_span(self.span.clone());
        let out = self.content.bind(ctx);
        ctx.diags.pop_span();

        out
    }
}

impl<Sym : BoundSymbol> BoundSymbol for Typed<Sym>
{
    fn error() -> Self {
        Self {
            typ: Type::error(),
            content: Sym::error()
        }
    }

    fn none() -> Self {
        Self {
            typ: Type::none(),
            content: Sym::none()
        }
    }
}

pub type ExprAST = AST<Expr>;
pub type StmtAST = AST<Stmt>;
pub type TypeSpecAST = AST<TypeSpec>;

#[derive(Debug, PartialEq, Symbol)]
pub enum Stmt
{
    Fn{ name: Ident, args: Vec<ArgSpec>, ty: Option<TypeSpecAST>, body: ExprAST },
    Let{ name: Ident, ty: Option<TypeSpecAST>, value: ExprAST },
    
    While{ cond: ExprAST, block: Block },

    Assign{ lhs: Ident, rhs: ExprAST },

    Expr(ExprAST)
}

#[derive(Debug, PartialEq, Symbol)]
pub enum Expr
{
    Identifier(Ident),

    Integer(ConstInt, Option<Type>),
    Decimal(ConstFloat, Option<Type>),
    Bool(bool),

    String(String),

    Unary{ op: UnOp, target: Box<ExprAST> },
    Binary{ op: BinOp, lhs: Box<ExprAST>, rhs: Box<ExprAST> },

    Call{ func: Box<ExprAST>, args: Vec<ExprAST> },

    Block(Block),
    If(IfNode),

    EOI // TODO: necessary?
}

#[derive(Debug, PartialEq)]
pub struct ArgSpec
{
    pub name: Ident,
    pub type_spec: TypeSpecAST
}

#[derive(Debug, PartialEq, Symbol)]
pub enum TypeSpec
{
    Named(Ident),
    Ptr(Box<TypeSpecAST>),
    Ref(Box<TypeSpecAST>)
}

#[derive(Debug, PartialEq, Symbol)]
pub struct Block
{
    pub body: Vec<StmtAST>,
    pub tail: Option<Box<ExprAST>>
}

#[derive(Debug, PartialEq, Symbol)]
pub struct IfNode
{
    pub cond: Box<ExprAST>,
    pub block: Block,
    pub tail: ElseKind
}

#[derive(Debug, PartialEq)]
pub enum ElseKind
{
    None,
    Else(AST<Block>),
    ElseIf(Box<AST<IfNode>>)
}

#[derive(Debug, Symbol)]
pub struct Program(pub Vec<StmtAST>);


  ////////////////////////
 //// Implementation ////
////////////////////////

use crate::core::Source;
use pest::iterators::Pair;
use pest::Parser;
use toylang_derive::Symbol;

use crate::parsing::Parseable;
use crate::parsing::grammar::{Grammar, Rule};

impl Parseable<Rule> for ArgSpec
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self 
    {
        let mut pairs = pair.into_inner();

        let name = Ident::parse_next(source, &mut pairs);
        let type_spec = TypeSpecAST::parse_next(source, &mut pairs);

        Self { name, type_spec }
    }
}

impl Parseable<Rule> for Block
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self 
    {
        let mut pairs = pair.into_inner(); 
        
        let body = StmtAST::vec_parse_next(source, &mut pairs);
        let tail = ExprAST::box_parse_next_maybe(source, &mut pairs);

        Self { body, tail }
    }
}

impl Parseable<Rule> for IfNode
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self
    {
        let mut pairs = pair.into_inner();

        let cond = ExprAST::box_parse_next(source, &mut pairs);
        let block = Block::parse_next(source, &mut pairs);
        
        let tail = match pairs.next()
        {
            None => ElseKind::None,
            Some(pair) => match pair.as_rule()
            {
                Rule::else_clause => ElseKind::Else(AST::parse(source, pair)),
                Rule::elif_clause => ElseKind::ElseIf(AST::box_parse(source, pair)),

                unknown => panic!("Unknown else clause {unknown:?}")
            }
        };

        Self { cond, block, tail }
    }
}

macro_rules! bin_op {
    ($source: ident, $pair: ident; $($ma: ident => $mb: expr),+) => {{
        let mut pairs = $pair.into_inner();

        let lhs = ExprAST::box_parse_next($source, &mut pairs);
        let op = match pairs.next().unwrap().as_rule()
        {
            $(Rule::$ma => $mb,)+

            unknown => panic!("Unexpected operator {unknown:?}")
        };
        let rhs = ExprAST::box_parse_next($source, &mut pairs);

        Expr::Binary {op, lhs, rhs}
    }};
}

impl<Kind : Parseable<Rule>> Parseable<Rule> for AST<Kind>
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self 
    {
        Self
        {
            span: Span::from(source, pair.as_span()),
            content: Kind::parse(source, pair)
        }
    }
}

impl Parseable<Rule> for Stmt
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self
    {
        match pair.as_rule()
        {
            Rule::let_stmt =>
            {
                let mut pairs = pair.into_inner();

                let name = Ident::parse_next(source, &mut pairs);
                let ty = Option::<TypeSpecAST>::parse_next(source, &mut pairs);
                let value = ExprAST::parse_next(source, &mut pairs);

                Stmt::Let { name, ty, value }
            },

            Rule::fn_stmt =>
            {
                let mut pairs = pair.into_inner();

                let name = Ident::parse_next(source, &mut pairs);
                let args = ArgSpec::vec_parse_next(source, &mut pairs);
                let ty = Option::<TypeSpecAST>::parse_next(source, &mut pairs);
                let body = ExprAST::parse_next(source, &mut pairs);

                Stmt::Fn { name, args, ty, body }
            },
            
            Rule::while_stmt => 
            {
                let mut pairs = pair.into_inner();

                let cond = ExprAST::parse_next(source, &mut pairs);
                let block = Block::parse_next(source, &mut pairs);

                Stmt::While { cond, block }
            },

            Rule::assgn_stmt => 
            {
                let mut pairs = pair.into_inner();

                let lhs = Ident::parse_next(source, &mut pairs);
                let rhs = ExprAST::parse_next(source, &mut pairs);

                Stmt::Assign { lhs, rhs }
            },

            _ => Stmt::Expr(ExprAST::parse(source, pair))
        }
    }
}

impl Parseable<Rule> for Expr
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self
    {
        match pair.as_rule()
        {
            Rule::ident => Expr::Identifier(Ident::parse(source, pair)),

            Rule::integer => Expr::Integer(String::parse(source, pair).parse().unwrap(), None),
            Rule::decimal => Expr::Decimal(String::parse(source, pair).parse().unwrap(), None),

            Rule::typed_integer => 
            {
                // TODO: should this be hard-coded?
                use crate::binding::IntType::*;
                use crate::binding::UIntType::*;

                let mut pairs = pair.into_inner();

                let int = String::parse_next(source, &mut pairs).parse().unwrap();
                let typ = match pairs.next().unwrap().as_str()
                {
                    "i8" => Type::Int(I8),
                    "i32" => Type::Int(I32),
                    "i64" => Type::Int(I64),
                    "isize" => Type::Int(ISize),

                    "u8" => Type::UInt(U8),
                    "u32" => Type::UInt(U32),
                    "u64" => Type::UInt(U64),
                    "usize" => Type::UInt(USize),

                    _ => unreachable!()
                };
                
                Expr::Integer(int, Some(typ))
            },

            Rule::typed_decimal =>
            {
                // TODO: should this be hard-coded?
                use crate::binding::FloatType::*;

                let mut pairs = pair.into_inner();

                let int = String::parse_next(source, &mut pairs).parse().unwrap();
                let typ = match pairs.next().unwrap().as_str()
                {
                    "f32" => Type::Float(F32),
                    "f64" => Type::Float(F64),

                    _ => unreachable!()
                };
                
                Expr::Decimal(int, Some(typ))
            }

            Rule::call => 
            {
                let mut pairs = pair.into_inner();

                let func = ExprAST::box_parse_next(source, &mut pairs);
                let args = ExprAST::parse_rest(source, &mut pairs);

                Expr::Call {func, args}
            },

            Rule::unary_expr => 
            {
                let mut pairs = pair.into_inner();

                let op = match pairs.next().unwrap().as_rule()
                {
                    Rule::minus => UnOp::Negate,
                    Rule::not   => UnOp::Not,

                    unknown => panic!("Invalid unary operator rule {unknown:?}")
                };
                let target = ExprAST::box_parse_next(source, &mut pairs);
                
                Expr::Unary { op, target }
            }

            Rule::muldiv_expr => bin_op!(source, pair; star => BinOp::Arith(ArithOp::Multiply), slash => BinOp::Arith(ArithOp::Divide)),
            Rule::addsub_expr => bin_op!(source, pair; plus => BinOp::Arith(ArithOp::Add), minus => BinOp::Arith(ArithOp::Subtract)),
            Rule::comp_expr => bin_op!(source, pair; 
                dequal  => BinOp::Equals, 
                nequal  => BinOp::NotEquals, 
                greater => BinOp::Comparison(CompOp::GreaterThan), 
                grequal => BinOp::Comparison(CompOp::GreaterEqual),
                less    => BinOp::Comparison(CompOp::LessThan),
                lequal  => BinOp::Comparison(CompOp::LessEqual)
            ),
            Rule::or_expr => bin_op!(source, pair; or => BinOp::Or),
            Rule::and_expr => bin_op!(source, pair; and => BinOp::And),

            Rule::block => Expr::Block(Block::parse(source, pair)),

            Rule::if_stmt => Expr::If(IfNode::parse(source, pair)),

            Rule::EOI => Expr::EOI,
            
            unknown => todo!("Unfinished: {unknown:?}")
        }
    }
}

impl Parseable<Rule> for Option<TypeSpecAST>
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self
    {
        pair.into_inner().next().map(|p| TypeSpecAST::parse(source, p))
    }
}

impl Parseable<Rule> for TypeSpec
{
    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self
    {
        if pair.as_rule() != Rule::r#type 
        {
            panic!("Non type rule entered to parse type!");
        }

        let pair = pair.into_inner().next().unwrap();

        match pair.as_rule()
        {
            Rule::ident => TypeSpec::Named(pair.as_str().into()),
            Rule::ptr_type => TypeSpec::Ptr(TypeSpecAST::box_parse(source, pair.into_inner().nth(1).unwrap())),
            Rule::ref_type => TypeSpec::Ref(TypeSpecAST::box_parse(source, pair.into_inner().nth(1).unwrap())),

            unknown => panic!("Unknown rule {unknown:?}")
        }
    }
}

pub fn parse(ipt: &str) -> Result<Program, pest::error::Error<Rule>>
{
    let source = Source::new(ipt);

    Ok(Program(
        Grammar::parse(Rule::prog, ipt)?
            .into_iter()
            .map(|p| StmtAST::parse(&source, p))
                .collect()
    ))
}