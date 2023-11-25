use std::cell::OnceCell;
use std::collections::HashMap;
use std::collections::HashSet;
use toylang_derive::Symbol;

use crate::binding::ProgramContext;
use crate::binding::Scope;
use crate::core::arena::ArenaID;
use crate::core::monad::Wrapper;
use crate::core::ops::ConstFloat;
use crate::core::ops::ConstInt;
use crate::core::ops::{BinOp, UnOp};
use crate::core::Ident;
use crate::core::SourceSpan;
use crate::core::utils::FlagCell;
use crate::types::Type;

pub trait Symbol
where
    Self: Sized,
{
    type Kind;

    fn kind(&self) -> &Self::Kind;
}

pub trait OwnedSymbol: Symbol
where
    Self: Sized,
{
    fn into_kind(self) -> Self::Kind;
}

impl<T: Symbol> Symbol for &T {
    type Kind = T::Kind;

    fn kind(&self) -> &Self::Kind {
        (*self).kind()
    }
}

pub trait UnboundSymbol {
    type Bound: BoundSymbol;
    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound;
}
pub trait BoundSymbol {
    fn none() -> Self;
    fn error() -> Self;

    fn is_error(&self) -> bool;
    fn is_none(&self) -> bool;

    fn not_error(&self) -> bool {
        !self.is_error()
    }
    fn not_none(&self) -> bool {
        !self.is_none()
    }
}

pub trait TypedSymbol {
    fn get_type(&self) -> &Type;
}

#[derive(Debug, PartialEq, Clone)]
pub struct AST<Content> {
    pub content: Content,
    pub span: SourceSpan,
}

impl<T> AST<T> {
    pub fn new(span: SourceSpan, content: T) -> Self {
        Self {
            span,
            content
        }
    }
}

impl<Content: Symbol> Symbol for AST<Content> {
    type Kind = Content::Kind;

    fn kind(&self) -> &Self::Kind {
        self.content.kind()
    }
}
impl<Content: OwnedSymbol> OwnedSymbol for AST<Content> {
    fn into_kind(self) -> Self::Kind {
        self.content.into_kind()
    }
}

impl<Content: BoundSymbol> BoundSymbol for AST<Content> {
    fn error() -> Self {
        AST {
            content: Content::error(),
            span: SourceSpan::none(),
        }
    }
    fn none() -> Self {
        AST {
            content: Content::none(),
            span: SourceSpan::none(),
        }
    }

    fn is_error(&self) -> bool {
        self.content.is_error()
    }
    fn is_none(&self) -> bool {
        self.content.is_none()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Typed<Content> {
    content: Content,
    typ: Type,
}

impl<C> Typed<C> {
    pub fn new(typ: Type, content: C) -> Self {
        Self { content, typ }
    }
}

impl<C> TypedSymbol for Typed<C> {
    fn get_type(&self) -> &Type {
        &self.typ
    }
}

impl<Sym> Wrapper for Typed<Sym> {
    type Value = Sym;
    type Of<T> = Typed<T>;

    fn map<T, F: FnOnce(Self::Value) -> T>(self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            typ: self.typ.clone(),
            content: func(self.content),
        }
    }

    fn map_ref<'a, T, F: FnOnce(&'a Sym) -> T>(&'a self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            typ: self.typ.clone(),
            content: func(&self.content),
        }
    }
}

impl<Sym: Symbol> Symbol for Typed<Sym> {
    type Kind = Sym::Kind;

    fn kind(&self) -> &Self::Kind {
        self.content.kind()
    }
}
impl<Sym: OwnedSymbol> OwnedSymbol for Typed<Sym> {
    fn into_kind(self) -> Self::Kind {
        self.content.into_kind()
    }
}

impl<T: TypedSymbol> TypedSymbol for AST<T> {
    fn get_type(&self) -> &Type {
        self.content.get_type()
    }
}

impl<Sym> Wrapper for AST<Sym> {
    type Value = Sym;
    type Of<T> = AST<T>;

    fn map<T, F: FnOnce(Self::Value) -> T>(self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            span: self.span.clone(),
            content: func(self.content),
        }
    }

    fn map_ref<'a, T, F: FnOnce(&'a Sym) -> T>(&'a self, func: F) -> Self::Of<T> {
        Self::Of::<T> {
            span: self.span.clone(),
            content: func(&self.content),
        }
    }
}

impl<Sym: UnboundSymbol> AST<Sym> {
    pub fn bind(&self, ctx: &mut ProgramContext) -> AST<Sym::Bound> {
        AST::<Sym::Bound> {
            content: self.bind_content(ctx),
            span: self.span.clone(),
        }
    }
    pub fn bind_content(&self, ctx: &mut ProgramContext) -> Sym::Bound {
        ctx.diags.push_span(self.span.clone());
        let out = self.content.bind(ctx);
        ctx.diags.pop_span();

        out
    }
}

impl AST<Stmt> {
    pub fn general_pass<T, F: FnMut(&Stmt, &mut ProgramContext) -> T>(
        &self,
        func: &mut F,
        ctx: &mut ProgramContext,
    ) -> Vec<T>
    {
        ctx.diags.push_span(self.span.clone());
        let out = self.content.general_pass(func, ctx);
        ctx.diags.pop_span();

        out
    }

    // todo: this better.
    // maybe a trait?
    pub fn load_mod_definitions(&self, ctx: &mut ProgramContext) {
        ctx.diags.push_span(self.span.clone());
        self.content.load_mod_definitions(ctx);
        ctx.diags.pop_span();
    }

    pub fn load_fn_definitions(&self, ctx: &mut ProgramContext) {
        ctx.diags.push_span(self.span.clone());
        self.content.load_fn_definitions(ctx);
        ctx.diags.pop_span();
    }

    pub fn define_structs(&self, ctx: &mut ProgramContext) {
        ctx.diags.push_span(self.span.clone());
        self.content.define_structs(ctx);
        ctx.diags.pop_span();
    }

    pub fn load_struct_definitions(&self, ctx: &mut ProgramContext) {
        ctx.diags.push_span(self.span.clone());
        self.content.load_struct_definitions(ctx);
        ctx.diags.pop_span();
    }
}

impl<Sym: BoundSymbol> BoundSymbol for Typed<Sym> {
    fn error() -> Self {
        Self {
            typ: Type::error(),
            content: Sym::error(),
        }
    }

    fn none() -> Self {
        Self {
            typ: Type::none(),
            content: Sym::none(),
        }
    }

    fn is_error(&self) -> bool {
        self.content.is_error()
    }
    fn is_none(&self) -> bool {
        self.content.is_none()
    }
}

pub type ExprAST = AST<Expr>;
pub type StmtAST = AST<Stmt>;
pub type TypeSpecAST = AST<TypeSpec>;

#[derive(Debug, PartialEq)]
pub struct StructDef {
    pub fields: Vec<(Ident, Type)>,
}

#[derive(Debug, PartialEq)]
pub struct StructDefSpec {
    pub name: Ident,
    pub fields: Vec<ArgSpec>,
    pub args: Vec<Ident>,
    pub error: FlagCell
}

#[derive(Debug, PartialEq, Symbol)]
pub enum Stmt {
    DeclareFn {
        name: Ident,
        args: Vec<ArgSpec>,
        ty: Option<TypeSpecAST>,
        vararg: bool
    },
    Fn {
        name: Ident,
        gen_args: Vec<Ident>,
        args: Vec<AST<ArgSpec>>,
        ty: Option<TypeSpecAST>,
        body: ExprAST,
        error: FlagCell
    },
    Let {
        name: Ident,
        ty: Option<TypeSpecAST>,
        value: ExprAST,
    },

    Mod {
        name: Ident,
        body: Vec<StmtAST>,
        scope_id: OnceCell<ArenaID<Scope>>,
        error: FlagCell
    },
    ModHead {
        name: Ident,
        scope_id: OnceCell<ArenaID<Scope>>,
        error: FlagCell
    },

    While {
        cond: ExprAST,
        block: Block,
    },

    Assign {
        lhs: ExprAST,
        rhs: ExprAST,
    },

    Import(String),

    Struct(StructDefSpec),

    Return(Option<ExprAST>),
    Break,
    Continue,

    Expr(ExprAST),
    Error
}

#[derive(Debug, PartialEq, Symbol)]
pub enum Expr {
    Identifier(Ident),

    Integer(ConstInt, Option<Type>),
    Decimal(ConstFloat, Option<Type>),
    Bool(bool),

    String(String),

    Unary {
        op: UnOp,
        target: Box<ExprAST>,
    },
    Binary {
        op: BinOp,
        lhs: Box<ExprAST>,
        rhs: Box<ExprAST>,
    },

    Call {
        func: Box<ExprAST>,
        args: Vec<ExprAST>,
    },

    Deref(Box<ExprAST>),
    Ref(Box<ExprAST>),
    
    Addressof(Box<ExprAST>),

    Block(Block),
    If(IfNode),

    Dot {
        op: Box<ExprAST>,
        id: Ident,
    },
    StaticDot {
        op: Box<ExprAST>,
        id: Ident,
    },
    Gen {
        op: Box<ExprAST>,
        args: Vec<TypeSpecAST>,
    },

    Cast {
        op: Box<ExprAST>,
        typ: TypeSpec,
    },

    Sizeof(TypeSpec),

    EOI, // TODO: necessary?
    Error
}

#[derive(Debug, PartialEq)]
pub struct ArgSpec {
    pub name: Ident,
    pub type_spec: TypeSpecAST,
}

#[derive(Debug, PartialEq, Symbol)]
pub enum TypeSpec {
    Named(Ident),
    Scoped(Vec<Ident>, Ident),
    Ptr(Box<TypeSpecAST>),
    Ref(Box<TypeSpecAST>),
    RefMut(Box<TypeSpecAST>),
    RefMove(Box<TypeSpecAST>),
    Gen(Box<TypeSpec>, Vec<TypeSpec>)
}

#[derive(Debug, PartialEq, Symbol)]
pub struct Block {
    pub body: Vec<StmtAST>,
    pub tail: Option<Box<ExprAST>>,
}

#[derive(Debug, PartialEq, Symbol)]
pub struct IfNode {
    pub cond: Box<ExprAST>,
    pub block: Block,
    pub tail: ElseKind,
}

#[derive(Debug, PartialEq)]
pub enum ElseKind {
    None,
    Else(AST<Block>),
    ElseIf(Box<AST<IfNode>>),
}

#[derive(Debug, Symbol)]
pub struct File(pub Vec<StmtAST>);

#[derive(Debug, Symbol)]
pub struct Program(pub HashMap<String, File>, pub HashSet<String>);