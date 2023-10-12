use std::collections::HashMap;
use std::fmt::{Display, Formatter, Debug};
use std::ops::Deref;
use toylang_derive::Symbol;

use crate::core::arena::{Arena, ArenaID, ArenaRef, ArenaMut};
use crate::core::monad::Wrapper;
use crate::core::ops::{CompOp, ArithOp, UnOp, BinOp, Constant, ConstInt, ConstFloat};
use crate::core::{Span, Ident, Diagnostic};
use crate::parsing::ast::{StmtAST, Stmt, TypeSpec, Expr, Symbol, AST, BoundSymbol, OwnedSymbol, Typed, UnboundSymbol, TypedSymbol, ElseKind, Block, IfNode, Program};
use crate::writex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComptimeType
{
    IntLit,
    FloatLit
}

impl Display for ComptimeType
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self 
        {
            Self::IntLit   => "integer literal",
            Self::FloatLit => "float literal",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntType
{
    I8,
    I32,
    I64,
    ISize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UIntType
{
    U8,
    U32,
    U64,
    USize
}
/*
impl IntType 
{
    pub fn bit_size(&self, ctx: CodegenContext) -> u32
    {
        match self 
        {
            IntType::I8    => 8,
            IntType::I32   => 32,
            IntType::I64   => 64,
            IntType::ISize => ctx.target.get_target_data().get_pointer_byte_size(None) * 8,
        }
    }
}

impl UIntType 
{
    pub fn bit_size(&self, ctx: CodegenContext) -> u32
    {
        match self 
        {
            UIntType::U8    => 8,
            UIntType::U32   => 32,
            UIntType::U64   => 64,
            UIntType::USize => ctx.target.get_target_data().get_pointer_byte_size(None) * 8,
        }
    }
} */

impl Display for IntType
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self
        {
            Self::I8 => "i8",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::ISize => "isize",
        })
    }
}

impl Display for UIntType
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self
        {
            Self::U8 => "u8",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::USize => "usize"
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatType
{
    F32,
    F64
}

impl Display for FloatType
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self 
        {
            Self::F32 => "f32",
            Self::F64 => "f64"
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Symbol)]
pub enum Type
{
    Error,
    // Comptime(ComptimeType),

    Unit,

    Bool,

    Int(IntType),
    UInt(UIntType),
    Float(FloatType),

    Char,
    Str,

    Ptr(Box<Type>),
    Ref(Box<Type>),

    Named(Ident),

    Function(FunctionType),
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conversion
{
    Ptr(Type, Type),
    IntToInt(IntType, IntType),
    FloatToFloat(FloatType, FloatType),
    IntToFloat(IntType, FloatType),
    FloatToInt(FloatType, IntType)
}

impl Type 
{
    fn compatible(&self, other: &Type) -> bool
    {
        if *self == Type::Error || *other == Type::Error {
            true
        }
        else {
            self == other
        }
    }

    fn concrete(self, ctx: &mut ProgramContext) -> Type
    {
        if let Type::Function(f) = self 
        {
            ctx.diags.add_error(format!("Cannot have a variable of type {f}"));
            Type::error()
        }
        else {self}
        /*if let Type::Comptime(ct) = self
        {
            ctx.add_error(format!("Cannot have a value of type '{ct}'"));
            Type::Error
        }
        else 
        {
            self
        }*/
    }

    fn as_fn(&self) -> Option<&FunctionType>
    {
        match self 
        {
            Type::Function(f) => Some(f),
            _ => None
        }
    }

    /*
    pub fn int_bit_size(&self, ctx: CodegenContext) -> Option<u32>
    {
        match self 
        {
            Type::Int(i) => Some(i.bit_size(ctx)),
            Type::UInt(u) => Some(u.bit_size(ctx)),

            _ => None
        }
    }
    */
}

impl Display for Type
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Bool => f.write_str("bool"),
            Self::Char => f.write_str("char"),
            Self::Str => f.write_str("str"),

            Self::Unit => f.write_str("unit"),

            Self::Error => f.write_str("?"),

            Self::Int(int) => write!(f, "{int}"),
            Self::UInt(int) => write!(f, "{int}"),
            Self::Float(float) => write!(f, "{float}"),

            Self::Named(x) => f.write_str(x.as_str()),
            Self::Ptr(t) => f.write_fmt(format_args!("*{t}")),
            Self::Ref(t) => f.write_fmt(format_args!("&{t}")),

            Self::Function(x) => writex!(f, x)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType
{
    pub ret: Box<Type>,
    pub args: Vec<Box<Type>>
}

impl Display for FunctionType
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for (i, arg) in self.args.iter().enumerate() {
            writex!(f, arg)?;
            if i != self.args.len() {write!(f, ", ")?;}
        }
        write!(f, ") -> ")?;
        writex!(f, self.ret)
    }
}

#[derive(Debug)]
pub struct FunctionDefinition
{
    pub name: Ident,
    pub typ: Type, 
    pub argnames: Vec<Ident>, 
    pub scope: ArenaID<Scope>,

    pub body: BoundExprAST,
    pub argdefs: Vec<ArenaID<Definition>>,
}

impl TypedSymbol for FunctionDefinition
{
    fn get_type(&self) -> &Type {
        &self.typ
    }
}

#[derive(Debug)]
pub enum Definition
{
    Variable{ name: Ident, typ: Type },
    Parameter{ name: Ident, typ: Type },
    Function (FunctionDefinition)
}

impl Definition
{
    pub fn name(&self) -> &Ident
    {
        match self
        {
            Self::Variable { name, .. } => name,
            Self::Parameter { name, .. } => name,
            Self::Function(FunctionDefinition { name, .. }) => name,
        }
    }

    pub fn function(name: Ident, typ: Type, argnames: Vec<Ident>, scope: ArenaID<Scope>) -> Definition
    {
        Definition::Function(
            FunctionDefinition { name, typ, argnames, scope, body: AST::none(), argdefs: Default::default() }
        )
    }

    pub fn into_fn(self) -> Option<FunctionDefinition>
    {
        match self 
        {
            Self::Function(f) => Some(f),
            _ => None
        }
    }

    pub fn as_fn(&self) -> Option<&FunctionDefinition>
    {
        match self 
        {
            Self::Function(f) => Some(f),
            _ => None
        }
    }

    pub fn as_fn_mut(&mut self) -> Option<&mut FunctionDefinition>
    {
        match self 
        {
            Self::Function(f) => Some(f),
            _ => None
        }
    }
}

impl TypedSymbol for Definition
{
    fn get_type(&self) -> &Type {
        match self 
        {
            Self::Variable { name: _, typ, .. } => typ,
            Self::Parameter { name: _, typ } => typ,
            Self::Function(FunctionDefinition { name: _, typ, .. }) => typ
        }
    }
}

pub struct Scopes
{
    pub arena: Arena<Scope>,
    current: ArenaID<Scope>
}

impl Scopes
{
    pub fn new() -> Scopes
    {
        let mut x = Scopes { arena: Arena::new(), current: ArenaID::default() };
        x.arena.add(Scope::global());

        x
    }

    pub fn get<'a>(&'a self, id: ArenaID<Scope>) -> ArenaRef<'a, Scope>
    {
        self.arena.get(id)
    }
    pub fn get_mut<'a>(&'a mut self, id: ArenaID<Scope>) -> ArenaMut<'a, Scope>
    {
        self.arena.get_mut(id)
    }

    pub fn cur<'a>(&'a self) -> ArenaRef<'a, Scope>
    {
        self.get(self.current)
    }
    pub fn cur_mut<'a>(&'a mut self) -> ArenaMut<'a, Scope>
    {
        self.get_mut(self.current)
    }

    pub fn count(&self) -> usize {self.arena.len()}

    pub fn current(&self) -> ArenaID<Scope>
    {
        self.current
    }

    pub fn create(&mut self) -> ArenaID<Scope>
    {
        self.arena.add(Scope::new(self.current))
    }

    pub fn enter(&mut self, scope: ArenaID<Scope>)
    {
        self.current = scope;
    }
    
    pub fn exit(&mut self)
    {
        self.current = self.arena.get(self.current).parent
            .unwrap_or(ArenaID::default());
    }

    pub fn lookup_at(&self, id: ArenaID<Scope>, ident: &Ident) -> Option<ArenaID<Definition>>
    {
        let scope = id.get(&self.arena);
        let def = scope.definitions.get(ident);

        if def.is_none() && id.get(&self.arena).parent.is_some()
        {
            let parent = id.get(&self.arena).parent.unwrap();
            self.lookup_at(parent, ident)
        }
        else 
        {
            def.cloned()
        }
    }

    pub fn lookup(&self, ident: &Ident) -> Option<ArenaID<Definition>>
    {
        self.lookup_at(self.current, ident)
    }
}

pub struct Scope
{
    definitions: HashMap<Ident, ArenaID<Definition>>,
    parent: Option<ArenaID<Scope>>
}

impl Scope
{
    pub fn global() -> Self
    {
        Scope { definitions: HashMap::new(), parent: None }
    }

    pub fn new(parent: ArenaID<Scope>) -> Self
    {
        Scope { definitions: HashMap::new(), parent: Some(parent) }
    }

    pub fn define(&mut self, definition: ArenaID<Definition>, defs: &mut Arena<Definition>)
    {
        self.definitions.insert(definition.get(&defs).name().clone(), definition);
    }

    pub fn define_lit(&mut self, definition: Definition, defs: &mut Arena<Definition>) -> ArenaID<Definition>
    {
        let def = defs.add(definition);
        self.define(def, defs);
        def
    }

    pub fn definitions(&self) -> &HashMap<Ident, ArenaID<Definition>>
    {
        &self.definitions
    }
}

pub struct DiagnosticHandler
{
    diagnostics: Vec<Diagnostic>,
    spans: Vec<Span>,
}

impl DiagnosticHandler
{
    pub fn new() -> Self 
    {
        Self {
            diagnostics: vec![],
            spans: vec![]
        }
    }

    pub fn current_span(&self) -> &Span {self.spans.last().unwrap()}
    pub fn pop_span(&mut self) -> Span {self.spans.pop().unwrap()}
    pub fn push_span(&mut self, span: Span) {self.spans.push(span)}

    pub fn in_span<T, F: FnMut() -> T>(&mut self, span: Span, mut f: F) -> T
    {
        self.push_span(span);
        let t = f();
        self.pop_span();
        t
    }

    pub fn diagnostics(&self) -> &Vec<Diagnostic>
    {&self.diagnostics}

    pub fn add_diagnostic(&mut self, diag: Diagnostic)
    {
        self.diagnostics.push(diag);
    }

    pub fn add_error(&mut self, message: String)
    {
        self.add_error_at(self.current_span().clone(), message);
    }
    pub fn add_error_at(&mut self, span: Span, message: String)
    {
        self.diagnostics.push(Diagnostic::error(span, message));
    }

    pub fn add_warning(&mut self, message: String)
    {
        self.add_warning_at(self.current_span().clone(), message);
    }
    pub fn add_warning_at(&mut self, span: Span, message: String)
    {
        self.diagnostics.push(Diagnostic::warning(span, message));
    }
    
    pub fn add_note(&mut self, message: String)
    {
        self.add_note_at(self.current_span().clone(), message);
    }
    pub fn add_note_at(&mut self, span: Span, message: String)
    {
        self.diagnostics.push(Diagnostic::note(span, message));
    }
}

pub struct ProgramContext
{
    pub diags: DiagnosticHandler,
    pub scopes: Scopes,
    pub defs: Arena<Definition>
}

impl ProgramContext
{
    pub fn new() -> Self
    {
        ProgramContext 
        { 
            diags: DiagnosticHandler::new(),
            scopes: Scopes::new(),
            defs: Arena::new()
        }
    }
}

impl BoundSymbol for BoundExpr
{
    fn none() -> Self {Self::None}
    fn error() -> Self {Self::Error}
}
impl BoundSymbol for BoundStmt
{
    fn none() -> Self {Self::None}
    fn error() -> Self {Self::Error}
}
impl BoundSymbol for Type
{
    fn none() -> Self {Self::Unit}
    fn error() -> Self {Self::Error}
}

pub type BoundExprAST = AST<Typed<BoundExpr>>;
pub type BoundStmtAST = AST<Typed<BoundStmt>>;

impl BoundStmtAST
{
    pub fn as_expr(self) -> Option<BoundExprAST> 
    {
        if matches!(self.kind(), BoundStmt::Expr(..) | BoundStmt::Error | BoundStmt::None)
        {
            None
        }
        else 
        {
            Some(self.map_deep(|x| match x {
                BoundStmt::Expr(e) => e,
                BoundStmt::Error => BoundExpr::Error,
                BoundStmt::None => BoundExpr::None,
                _ => unreachable!()
            }))
        }
    }
}

#[derive(Debug)]
pub struct BoundBinary
{
    pub lhs: Box<BoundExprAST>,
    pub rhs: Box<BoundExprAST>
}

#[derive(Debug, Symbol)]
pub enum BoundExpr
{
    Error,
    None,

    Negate(Box<BoundExprAST>),
    Not(Box<BoundExprAST>),

    Arithmetic(BoundBinary, ArithOp),

    And(BoundBinary),
    Or(BoundBinary),

    Equals(BoundBinary),
    NotEquals(BoundBinary),

    Comparison(BoundBinary, CompOp),
    
    Block(BoundBlock),

    If(BoundIf),

    Integer(ConstInt),
    Decimal(ConstFloat),
    Bool(bool),

    Identifier(ArenaID<Definition>),

    Call(Box<BoundExprAST>, Vec<BoundExprAST>)
}

#[derive(Debug)]
pub struct BoundIf
{
    pub cond: Box<BoundExprAST>,
    pub block: BoundBlock,
    pub else_block: Option<Box<BoundExprAST>>
}

impl BoundIf 
{
    fn calculate_type(&self, ctx: &mut ProgramContext) -> &Type 
    {
        let blk_type = self.block.get_type();

        if *blk_type == Type::Unit 
        {
            return &Type::Unit
        }

        if Some(blk_type) != self.else_block.as_ref().map(|e| e.get_type())
        {
            ctx.diags.add_warning("If block does not return the same type as its else.".to_owned());
            &Type::Error
        }
        else 
        {
            blk_type
        }
    }
}

#[derive(Debug, Symbol)]
pub struct BoundBlock
{
    pub scope: ArenaID<Scope>,
    pub body: Vec<BoundStmtAST>, 
    pub tail: Option<Box<BoundExprAST>>
}

impl BoundSymbol for BoundBlock
{
    fn error() -> Self {
        BoundBlock { scope: ArenaID::default(), body: vec![], tail: None }
    }
    fn none() -> Self {
        Self::error()
    }
}

impl BoundBlock
{
    fn get_type(&self) -> &Type {
        self.tail.as_ref().map_or(&Type::Unit, |b| b.get_type())
    }
}

#[derive(Debug, Symbol)]
pub enum BoundStmt
{
    Error,

    None,

    Expr(BoundExpr),

    While{ cond: BoundExprAST, block: BoundBlock },
    
    Assign{ first: bool, target: ArenaID<Definition>, value: Box<BoundExprAST> },
}

#[derive(Debug, Symbol)]
pub struct BoundProgram(pub Vec<BoundStmtAST>);

impl BoundSymbol for BoundProgram
{
    fn none() -> Self {
        BoundProgram(vec![])
    }

    fn error() -> Self {
        BoundProgram(vec![])
    }
}

impl StmtAST
{
    pub fn load_fn_definitions(&self, ctx: &mut ProgramContext)
    {
        match &self.content
        {
            Stmt::Fn { name, args, ty, .. } => {

                let scope = ctx.scopes.create();

                let func = Definition::function( 
                    name.clone(), 
                    Type::Function(FunctionType { 
                        ret: Box::new(ty.as_ref().map(|ts| ts.bind_content(ctx)).unwrap_or(Type::Unit)), 
                        args: args.iter().map(|arg| arg.type_spec.bind_content(ctx)).map(Box::new).collect()
                    }),
                    args.iter().map(|arg| arg.name.clone()).collect(),
                    scope
                );

                let func = ctx.defs.add(func);

                ctx.scopes.cur_mut().define(func, &mut ctx.defs);

            },

            _ => {}
        };
    }
}

impl UnboundSymbol for Stmt
{
    type Bound = Typed<BoundStmt>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound
    {
        match &self
        {
            Stmt::Let { name, ty, value } =>
            {
                let value = value.bind(ctx);
                let valty = value.get_type().clone();

                let ty = if ty.is_some()
                {
                    let ty = ty.as_ref().unwrap().bind_content(ctx);

                    if !ty.compatible(&valty)
                    {
                        ctx.diags.add_error(format!("Incompatible type between assignee ({ty}) and value ({valty})"));
                        return Self::Bound::error();
                    }
                    else 
                    {
                        ty
                    }
                }
                else
                {
                    valty.clone()
                };

                let ty = ty.concrete(ctx);

                let var = ctx.defs.add(Definition::Variable { name: name.clone(), typ: ty });
                ctx.scopes.cur_mut().define(var, &mut ctx.defs);

                Typed::new(
                    valty,
                    BoundStmt::Assign { first: true, target: var, value: Box::new(value) }
                )
            }

            Stmt::Assign { lhs, rhs } =>
            {
                let def_id = ctx.scopes.lookup(&lhs);

                if def_id.is_none()
                {
                    return Self::Bound::error();
                }
                
                let def_id = def_id.unwrap();
                let def = ctx.defs.get(def_id);

                let ty = def.get_type().clone();
                
                let value = rhs.bind(ctx);
                let valty = value.get_type().clone();

                if !ty.compatible(&valty)
                {
                    ctx.diags.add_error(format!("Incompatible type between assignee ({ty}) and value ({valty})"));
                    return Self::Bound::error();
                }

                Typed::new(
                    valty,
                    BoundStmt::Assign { first: false, target: def_id, value: Box::new(value) }
                )
            }

            Stmt::Expr(e) => 
            {
                let e = e.bind(ctx);

                Typed::new(
                    e.get_type().clone(),
                    BoundStmt::Expr(e.into_kind())
                )
            },

            Stmt::Fn { name, args: _, ty: _, body } => 
            {
                let def = ctx.scopes.lookup(name).expect("Should have already defined this!");

                let (ty, argnames, scope) = match ArenaRef::into_ref(def.get(&ctx.defs))
                {
                    Definition::Function(f) => (
                        f.get_type().as_fn().unwrap().clone(), 
                        f.argnames.clone(), 
                        f.scope
                    ),
                    _ => unreachable!()
                };

                ctx.scopes.enter(scope);

                for (typ, name) in ty.args.iter().zip(argnames.clone().iter())
                {
                    let param = ctx.scopes.cur_mut()
                        .define_lit(Definition::Parameter { name: name.clone(), typ: typ.deref().clone() }, &mut ctx.defs);

                    def.get_mut(&mut ctx.defs)
                        .as_fn_mut()
                        .unwrap()
                        .argdefs
                        .push(param);

                    println!("{:?}", param);
                }

                let body = body.bind(ctx);
                let bodyty = body.get_type();

                ctx.scopes.exit();

                if !ty.ret.compatible(bodyty)
                {
                    ctx.diags.add_error(format!("Function expects return of type {ty} but got {bodyty}"));
                    Typed::error()
                }
                else 
                {
                    let mut def = def.get_mut(&mut ctx.defs);
                    let def = def
                        .as_fn_mut()
                        .unwrap();

                    def.body = body;
                    Typed::none()
                }
            },

            _ => todo!("Bruh")
        }
    }
}

impl UnboundSymbol for TypeSpec
{
    type Bound = Type;
    fn bind(&self, ctx: &mut ProgramContext) -> Type
    {
        match &self
        {
            TypeSpec::Named(name) => 
            {
                match name.as_str()
                {
                    "i8" => Type::Int(IntType::I8),
                    "i32" => Type::Int(IntType::I32),
                    "i64" => Type::Int(IntType::I64),
                    "isize" => Type::Int(IntType::ISize),
                    
                    "u8" => Type::UInt(UIntType::U8),
                    "u32" => Type::UInt(UIntType::U32),
                    "u64" => Type::UInt(UIntType::U64),
                    "usize" => Type::UInt(UIntType::USize),

                    "f32" => Type::Float(FloatType::F32),
                    "f64" => Type::Float(FloatType::F64),

                    "char" => Type::Char,
                    "str" => Type::Str,

                    "bool" => Type::Bool,
                    "unit" => Type::Unit,

                    _ => Type::Named(name.clone())
                }
            },

            TypeSpec::Ptr(inner) => Type::Ptr(Box::new(inner.bind_content(ctx))),
            TypeSpec::Ref(inner) => Type::Ref(Box::new(inner.bind_content(ctx)))
        }
    }
}

impl UnboundSymbol for Block
{
    type Bound = Typed<BoundBlock>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound 
    {
        let scope = ctx.scopes.create();
        ctx.scopes.enter(scope);

        let block = BoundBlock {
            scope,
            body: self.body.iter().map(|e| e.bind(ctx)).collect(),
            tail: self.tail.as_ref().map(|e| e.bind(ctx)).map(Box::new)
        };

        ctx.scopes.exit();
        
        Typed::new(
            block.get_type().clone(),
            block,
        )
    }
}

impl UnboundSymbol for IfNode
{
    type Bound = Typed<BoundExpr>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound
    {
        match &self.tail
        {
            ElseKind::None => {
                Typed::new(
                    Type::Unit,
                    BoundExpr::If(
                        BoundIf { 
                            cond: Box::new(self.cond.bind(ctx)), 
                            block: self.block.bind(ctx).into_kind(), 
                            else_block: None
                        }
                    )
                )
            },

            ElseKind::Else(block) => {
                let ifn = BoundIf { 
                    cond: Box::new(self.cond.bind(ctx)), 
                    block: self.block.bind(ctx).into_kind(), 
                    else_block: Some(Box::new(block.bind(ctx).map_deep(BoundExpr::Block)))
                };

                Typed::new(
                    ifn.calculate_type(ctx).clone(),
                    BoundExpr::If(ifn)
                )
            },

            ElseKind::ElseIf(elif) => {
                let ifn = BoundIf { 
                    cond: Box::new(self.cond.bind(ctx)), 
                    block: self.block.bind(ctx).into_kind(), 
                    else_block: Some(Box::new(elif.bind(ctx)))
                };

                Typed::new(
                    ifn.calculate_type(ctx).clone(),
                    BoundExpr::If(ifn)
                )
            }
        }
    }
}

impl UnboundSymbol for Expr
{
    type Bound = Typed<BoundExpr>;
    
    fn bind(&self, ctx: &mut ProgramContext) -> Typed<BoundExpr>
    {
        match &self
        {
            Expr::Block(block) => 
            {
                let block = block.bind(ctx);
                
                Typed::new(
                    block.get_type().clone(),
                    BoundExpr::Block(block.into_kind()),
                )
            },

            Expr::Integer(integer, typ) => 
            {
                Typed::new(
                    typ.clone().unwrap_or(Type::Int(IntType::I32)),
                    BoundExpr::Integer(*integer)
                )
            },

            Expr::Decimal(decimal, typ) =>
            {
                Typed::new(
                    typ.clone().unwrap_or(Type::Float(FloatType::F32)),
                    BoundExpr::Decimal(*decimal)
                )
            },

            Expr::Bool(b) => Constant::Bool(*b).into(),

            Expr::Identifier(name) =>
            {
                let def = ctx.scopes.lookup(name);

                if def.is_none()
                {
                    ctx.diags.add_error(format!("Unknown identifier {}", name.as_str()));
                    return Typed::error();
                }
                
                let def_id = def.unwrap();
                let def = def_id.get(&ctx.defs);
                
                Typed::new(
                    def.get_type().clone(),
                    BoundExpr::Identifier(def_id)
                )

            },

            Expr::Unary { op, target } =>
            {
                let target = target.bind(ctx);
                let ty = target.get_type();

                match op
                {
                    UnOp::Negate => {
                        if matches!(ty, Type::Int(..) | Type::Float(..))
                        {
                            Typed::new(
                                ty.clone(),
                                BoundExpr::Negate(Box::new(target))
                            )
                        }
                        else 
                        {
                            ctx.diags.add_error(format!("Cannot negate value of type {ty}"));
                            Typed::error()
                        }
                    },

                    UnOp::Not => {
                        if matches!(ty, Type::Bool)
                        {
                            Typed::new(
                                Type::Bool,
                                BoundExpr::Not(Box::new(target))
                            )
                        }
                        else 
                        {
                            ctx.diags.add_error(format!("Cannot apply not to value of non-boolean type {ty}"));
                            Typed::error()
                        }
                    }
                }
            },

            Expr::Binary { op, lhs, rhs } => 
            {
                let lhs = lhs.bind(ctx);
                let rhs = rhs.bind(ctx);
                
                let lty = lhs.get_type();
                let rty = rhs.get_type();

                match op 
                {
                    BinOp::Arith(arith) =>
                    {
                        if lty != rty 
                        {
                            ctx.diags.add_error(format!("Cannot perform operation {arith} on distinct types {lty} and {rty}"));
                            Typed::error()
                        }
                        else 
                        {
                            Typed::new(
                                lty.clone(),
                                BoundExpr::Arithmetic(BoundBinary { lhs: Box::new(lhs), rhs: Box::new(rhs) }, arith.clone())
                            )
                        }
                    },

                    BinOp::Equals | BinOp::NotEquals =>
                    {
                        if !lty.compatible(rty)
                        {
                            ctx.diags.add_error(format!("Cannot compare two values of differing types {lty} {rty}"));
                            Typed::error()
                        }
                        // else if lcnst.is_some() && rcnst.is_some()
                        // {
                        //     Constant::Bool(lcnst == rcnst).into()
                        // }
                        else 
                        {
                            Typed::new(
                                Type::Bool,
                                if *op == BinOp::Equals {BoundExpr::Equals} else {BoundExpr::NotEquals} (BoundBinary { lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            )
                        }
                    },

                    BinOp::And | BinOp::Or => 
                    {
                        if lty != rty 
                        {
                            ctx.diags.add_error(format!("Cannot compare two values of differing types {lty} {rty}"));
                            Typed::error()
                        }
                        else if *lty != Type::Bool
                        {
                            ctx.diags.add_error(format!("Cannot perform logic on non-boolean type {lty}"));
                            Typed::error()
                        }
                        else 
                        {
                            Typed::new(
                                Type::Bool,
                                if *op == BinOp::And {BoundExpr::And} else {BoundExpr::Or} (BoundBinary { lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            )
                        }
                    },

                    BinOp::Comparison(comp) =>
                    {
                        if !lty.compatible(rty)
                        {
                            ctx.diags.add_error(format!("Cannot compare two values of differing types {lty} {rty}"));
                            Typed::error()
                        }
                        else 
                        {
                            Typed::new(
                                Type::Bool,
                                BoundExpr::Comparison(BoundBinary { lhs: Box::new(lhs), rhs: Box::new(rhs) }, comp.clone())
                            )
                        }
                    }
                }
            },

            Expr::If(ifn) => ifn.bind(ctx),

            Expr::Call { func, args } =>
            {
                let func = func.bind(ctx);
                let args = args.iter().map(|x| x.bind(ctx)).collect::<Vec<_>>();

                let fty = func.get_type();

                if let Type::Function(fty) = fty 
                {
                    if args.len() != fty.args.len() {
                        ctx.diags.add_error(format!("Incorrect number of arguments for call to function of type {fty}"));
                        Typed::error()
                    }
                    else {
                        for (arg, argty) in args.iter().zip(fty.args.iter()) {
                            if !arg.get_type().compatible(&argty) {
                                ctx.diags.add_error_at(arg.span.clone(), format!("Expected argument of type {argty}, got {}", arg.get_type()));
                                return Typed::error();
                            }
                        }

                        Typed::new(
                            (*fty.ret).clone(),
                            BoundExpr::Call(Box::new(func), args)
                        )
                    }
                }
                else 
                {
                    if *fty != Type::Error
                    {
                        ctx.diags.add_error(format!("Cannot call non-function value of type {fty}"));
                    }
                    Typed::error()
                }
            },

            Expr::EOI => Typed::none(),

            _ => todo!()
        }
    }
}

impl UnboundSymbol for Program
{
    type Bound = BoundProgram;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound
    {
        self.0.iter().for_each(|x| x.load_fn_definitions(ctx));
        BoundProgram(self.0.iter().map(|x| x.bind(ctx)).collect::<Vec<_>>())
    }
}