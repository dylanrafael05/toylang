use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::io;
use std::ops::Deref;
use std::path::Path;
use toylang_derive::Symbol;

use crate::core::arena::{Arena, ArenaID, ArenaMut, ArenaRef, ArenaValue};
use crate::core::lazy::OnceBuildable;
use crate::core::monad::Wrapper;
use crate::core::ops::{ArithOp, BinOp, CompOp, ConstFloat, ConstInt, Constant, UnOp};
use crate::core::{Diagnostic, Ident, SourceSpan, Src};
use crate::parsing;
use crate::parsing::ast::{
    Block, BoundSymbol, ElseKind, Expr, IfNode, OwnedSymbol, Stmt, Symbol, TypeSpec,
    Typed, TypedSymbol, UnboundSymbol, AST, ExprAST, Program,
};
use crate::types::{
    FloatType, FunctionType, IntType, LitType, RefKind, Type, TypeDefinition, UIntType,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComptimeType {
    IntLit,
    FloatLit,
}

impl Display for ComptimeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::IntLit => "integer literal",
                Self::FloatLit => "float literal",
            }
        )
    }
}

#[derive(Debug)]
pub struct FunctionDefinition {
    pub typ: Type,
    pub argnames: Vec<Ident>,

    pub body_info: Option<FunctionBodyInfo>,
}

#[derive(Debug)]
pub struct FunctionBodyInfo {
    pub scope: ArenaID<Scope>,
    pub body_ast: BoundExprAST,
    pub argdefs: Vec<ArenaID<Definition>>,
}

impl FunctionBodyInfo {
    pub fn from_scope(scope: ArenaID<Scope>) -> Self {
        let mut val = Self::default();

        val.scope = scope;

        val
    }
}

impl Default for FunctionBodyInfo {
    fn default() -> Self {
        Self {
            scope: Default::default(),
            body_ast: BoundExprAST::none(),
            argdefs: Vec::new(),
        }
    }
}

impl TypedSymbol for FunctionDefinition {
    fn get_type(&self) -> &Type {
        &self.typ
    }
}

#[derive(Debug)]
pub enum DefinitionKind {
    Variable(Type),
    Parameter(Type),
    Function(FunctionDefinition),
    Type(TypeDefinition),
    Scope(ArenaID<Scope>),
}

#[derive(Debug)]
pub struct Definition {
    name: Ident,
    scope: ArenaID<Scope>,
    pub kind: DefinitionKind,
}

impl Definition {
    pub fn name(&self) -> &Ident {
        &self.name
    }
    pub fn full_name(&self, scopes: &Arena<Scope>) -> String {
        self.scope.get(&scopes).full_name(&scopes) + "::" + self.name.as_str()
    }

    pub fn compiled_name(&self, scopes: &Arena<Scope>) -> String {
        match &self.kind {
            DefinitionKind::Function(f) => match &f.body_info {
                Some(..) => {
                    if self.name.as_str() == "main" && self.scope == ArenaID::default() {
                        "main".into()
                    } else {
                        self.full_name(scopes)
                    }
                }
                None => self.name.to_string(),
            },

            _ => self.full_name(scopes),
        }
    }

    pub fn scope(&self) -> ArenaID<Scope> {
        self.scope
    }

    pub fn set_scope(&mut self, scope: ArenaID<Scope>) {
        self.scope = scope;
    }

    pub fn new(name: Ident, kind: DefinitionKind) -> Definition {
        Definition {
            name,
            scope: Default::default(),
            kind,
        }
    }

    pub fn function(
        name: Ident,
        typ: Type,
        argnames: Vec<Ident>,
        scope: ArenaID<Scope>,
    ) -> Definition {
        Self::new(
            name,
            DefinitionKind::Function(FunctionDefinition {
                typ,
                argnames,
                body_info: Some(FunctionBodyInfo::from_scope(scope)),
            }),
        )
    }
    pub fn function_decl(name: Ident, typ: Type, argnames: Vec<Ident>) -> Definition {
        Self::new(
            name,
            DefinitionKind::Function(FunctionDefinition {
                typ,
                argnames,
                body_info: None,
            }),
        )
    }

    pub fn into_fn(self) -> Option<FunctionDefinition> {
        match self.kind {
            DefinitionKind::Function(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_fn(&self) -> Option<&FunctionDefinition> {
        match self.kind {
            DefinitionKind::Function(ref f) => Some(f),
            _ => None,
        }
    }

    pub fn as_fn_mut(&mut self) -> Option<&mut FunctionDefinition> {
        match self.kind {
            DefinitionKind::Function(ref mut f) => Some(f),
            _ => None,
        }
    }

    pub fn into_ty(self) -> Option<TypeDefinition> {
        match self.kind {
            DefinitionKind::Type(f) => Some(f),
            _ => None,
        }
    }

    pub fn as_ty(&self) -> Option<&TypeDefinition> {
        match self.kind {
            DefinitionKind::Type(ref f) => Some(f),
            _ => None,
        }
    }

    pub fn as_ty_mut(&mut self) -> Option<&mut TypeDefinition> {
        match self.kind {
            DefinitionKind::Type(ref mut f) => Some(f),
            _ => None,
        }
    }

    pub fn try_get_type(&self) -> Option<&Type> {
        match self.kind {
            DefinitionKind::Variable(ref typ) => Some(typ),
            DefinitionKind::Parameter(ref typ) => Some(typ),
            DefinitionKind::Function(FunctionDefinition { ref typ, .. }) => Some(typ),

            DefinitionKind::Type(..) | DefinitionKind::Scope(..) => None,
        }
    }
}

impl TypedSymbol for Definition {
    fn get_type(&self) -> &Type {
        self.try_get_type()
            .expect("Calls to Definition::get_type should not fail. Use try_get_type if this is the expected behaviour.")
    }
}

pub struct Scopes {
    pub arena: Arena<Scope>,
    current: ArenaID<Scope>,
}

impl Scopes {
    pub fn new() -> Scopes {
        let mut x = Scopes {
            arena: Arena::new(),
            current: ArenaID::default(),
        };
        x.arena.add(Scope::global());

        x
    }

    pub fn get<'a>(&'a self, id: ArenaID<Scope>) -> ArenaRef<'a, Scope> {
        self.arena.get(id)
    }
    pub fn get_mut<'a>(&'a mut self, id: ArenaID<Scope>) -> ArenaMut<'a, Scope> {
        self.arena.get_mut(id)
    }

    pub fn cur<'a>(&'a self) -> ArenaRef<'a, Scope> {
        self.get(self.current)
    }
    pub fn cur_mut<'a>(&'a mut self) -> ArenaMut<'a, Scope> {
        self.get_mut(self.current)
    }

    pub fn count(&self) -> usize {
        self.arena.len()
    }

    pub fn cur_id(&self) -> ArenaID<Scope> {
        self.current
    }

    pub fn create(&mut self) -> ArenaID<Scope> {
        self.arena.add(Scope::new(self.current))
    }
    pub fn create_named(&mut self, name: Ident, defs: &mut Definitions) -> ArenaID<Scope> {
        let sc = self.arena.add(Scope::new_named(self.current, name.clone()));
        self.cur_mut()
            .define_lit(Definition::new(name, DefinitionKind::Scope(sc)), defs);
        sc
    }

    pub fn enter(&mut self, scope: ArenaID<Scope>) {
        self.current = scope;
    }

    pub fn exit(&mut self) {
        self.current = self
            .arena
            .get(self.current)
            .parent
            .unwrap_or(ArenaID::default());
    }
    pub fn exit_to(&mut self, id: ArenaID<Scope>) {
        while self.current != id {
            self.exit();

            if self.current == ArenaID::default() && id != ArenaID::default() {
                panic!("{:?} was not found in the heirarchy of current scopes!", id)
            }
        }
    }

    pub fn lookup_at(&self, id: ArenaID<Scope>, ident: &Ident) -> Option<ArenaID<Definition>> {
        let scope = id.get(&self.arena);
        let def = scope.definitions.get(ident);

        if def.is_none() && id.get(&self.arena).parent.is_some() {
            let parent = id.get(&self.arena).parent.unwrap();
            self.lookup_at(parent, ident)
        } else {
            def.cloned()
        }
    }

    pub fn lookup(&self, ident: &Ident) -> Option<ArenaID<Definition>> {
        self.lookup_at(self.current, ident)
    }
}

pub struct Scope {
    definitions: HashMap<Ident, ArenaID<Definition>>,
    parent: Option<ArenaID<Scope>>,
    name: Option<Ident>,
    id: Option<ArenaID<Scope>>,
}

impl ArenaValue for Scope {
    fn assign_id(&mut self, id: ArenaID<Self>) {
        self.id = Some(id);
    }
}

impl Scope {
    pub fn global() -> Self {
        Scope {
            definitions: HashMap::new(),
            parent: None,
            name: None,
            id: None,
        }
    }

    pub fn new(parent: ArenaID<Scope>) -> Self {
        Scope {
            definitions: HashMap::new(),
            parent: Some(parent),
            name: None,
            id: None,
        }
    }
    pub fn new_named(parent: ArenaID<Scope>, name: Ident) -> Self {
        Scope {
            definitions: HashMap::new(),
            parent: Some(parent),
            name: Some(name),
            id: None,
        }
    }

    pub fn name(&self) -> Option<&Ident> {
        self.name.as_ref()
    }
    pub fn full_name(&self, arena: &Arena<Scope>) -> String {
        match self.parent {
            None => "<global>".into(),
            Some(p) => match &self.name {
                None => p.get(arena).full_name(arena),
                Some(n) => if p == Default::default() {n.as_str().to_owned()} else {
                    p.get(arena).full_name(arena) + "::" + n.as_str()
                },
            },
        }
    }

    pub fn id(&self) -> ArenaID<Scope> {
        self.id
            .expect("ID should never be called before a scope is added to a scope arena!")
    }

    pub fn define(&mut self, definition: ArenaID<Definition>, defs: &mut Arena<Definition>) {
        let mut def_mut = definition.get_mut(defs);
        def_mut.set_scope(self.id());

        self.definitions
            .insert(definition.get(&defs).name().clone(), definition);
    }

    pub fn define_lit(
        &mut self,
        definition: Definition,
        defs: &mut Arena<Definition>,
    ) -> ArenaID<Definition> {
        let def = defs.add(definition);
        self.define(def, defs);
        def
    }

    pub fn definitions(&self) -> &HashMap<Ident, ArenaID<Definition>> {
        &self.definitions
    }
    pub fn take_definitions(&mut self) -> HashMap<Ident, ArenaID<Definition>> {
        std::mem::replace(&mut self.definitions, HashMap::new())
    }
}

pub struct DiagnosticHandler {
    diagnostics: Vec<Diagnostic>,
    spans: Vec<SourceSpan>,
}

impl DiagnosticHandler {
    pub fn new() -> Self {
        Self {
            diagnostics: vec![],
            spans: vec![],
        }
    }

    pub fn current_span(&self) -> &SourceSpan {
        self.spans.last().unwrap()
    }
    pub fn pop_span(&mut self) -> SourceSpan {
        self.spans.pop().unwrap()
    }
    pub fn push_span(&mut self, span: SourceSpan) {
        self.spans.push(span)
    }

    pub fn in_span<T, F: FnMut() -> T>(&mut self, span: SourceSpan, mut f: F) -> T {
        self.push_span(span);
        let t = f();
        self.pop_span();
        t
    }

    pub fn diagnostics(&self) -> &Vec<Diagnostic> {
        &self.diagnostics
    }

    pub fn add(&mut self, diag: Diagnostic) {
        match self.diagnostics.binary_search_by(|x| (x.span.start.index, x.span.source.name()).cmp(&(diag.span.start.index, diag.span.source.name()))) {
            Ok(x)|Err(x) => self.diagnostics.insert(x, diag)
        }
    }

    pub fn add_error(&mut self, message: String) {
        self.add_error_at(self.current_span().clone(), message);
    }
    pub fn add_error_at(&mut self, span: SourceSpan, message: String) {
        self.add(Diagnostic::error(span, message));
    }

    pub fn add_warning(&mut self, message: String) {
        self.add_warning_at(self.current_span().clone(), message);
    }
    pub fn add_warning_at(&mut self, span: SourceSpan, message: String) {
        self.add(Diagnostic::warning(span, message));
    }

    pub fn add_note(&mut self, message: String) {
        self.add_note_at(self.current_span().clone(), message);
    }
    pub fn add_note_at(&mut self, span: SourceSpan, message: String) {
        self.add(Diagnostic::note(span, message));
    }
}

pub type Definitions = Arena<Definition>;

pub struct ProgramContext {
    pub diags: DiagnosticHandler,
    pub scopes: Scopes,
    pub defs: Definitions
}

impl ProgramContext {
    pub fn new() -> Self {
        ProgramContext {
            diags: DiagnosticHandler::new(),
            scopes: Scopes::new(),
            defs: Arena::new()
        }
    }
}

impl BoundSymbol for BoundExpr {
    fn none() -> Self {
        Self::None
    }
    fn error() -> Self {
        Self::Error
    }

    fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }
    fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}
impl BoundSymbol for BoundStmt {
    fn none() -> Self {
        Self::None
    }
    fn error() -> Self {
        Self::Error
    }

    fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }
    fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }
}
impl BoundSymbol for Type {
    fn none() -> Self {
        LitType::None.into()
    }
    fn error() -> Self {
        Self::Error
    }

    fn is_error(&self) -> bool {
        match self {
            Type::Error => true,
            Type::Literal(LitType::Ref(x, ..)) => x.is_error(),
            Type::Literal(LitType::Ptr(x)) => x.is_error(),

            _ => false,
        }
    }
    fn is_none(&self) -> bool {
        matches!(self, Self::Literal(LitType::Unit))
    }
}

pub type BoundExprAST = AST<Typed<BoundExpr>>;
pub type BoundStmtAST = AST<Typed<BoundStmt>>;

impl BoundStmtAST {
    pub fn as_expr(self) -> Option<BoundExprAST> {
        if matches!(
            self.kind(),
            BoundStmt::Expr(..) | BoundStmt::Error | BoundStmt::None
        ) {
            None
        } else {
            Some(self.map_deep(|x| match x {
                BoundStmt::Expr(e) => e,
                BoundStmt::Error => BoundExpr::Error,
                BoundStmt::None => BoundExpr::None,
                _ => unreachable!(),
            }))
        }
    }
}

#[derive(Debug)]
pub struct BoundBinary {
    pub lhs: Box<BoundExprAST>,
    pub rhs: Box<BoundExprAST>,
}

#[derive(Debug, Symbol)]
pub enum BoundExpr {
    Error,
    None,
    
    Zeroinit(Type),

    Negate(Box<BoundExprAST>),
    Not(Box<BoundExprAST>),

    Arithmetic(BoundBinary, ArithOp),
    PtrArithmetic(BoundBinary, ArithOp),

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
    String(String),

    Symbol(ArenaID<Definition>),

    Dot(Box<BoundExprAST>, usize),

    DirectCall(ArenaID<Definition>, Vec<BoundExprAST>),
    IndirectCall(Box<BoundExprAST>, Vec<BoundExprAST>),

    Cast(Box<BoundExprAST>, Type),

    Deref(Box<BoundExprAST>),

    VariableRef(ArenaID<Definition>),
    DotRef(Box<BoundExprAST>, usize),
    DerefRef(Box<BoundExprAST>)
}

#[derive(Debug)]
pub struct BoundIf {
    pub cond: Box<BoundExprAST>,
    pub block: BoundBlock,
    pub else_block: Option<Box<BoundExprAST>>,
}

impl BoundIf {
    fn calculate_type(&self, ctx: &mut ProgramContext) -> &Type {
        let blk_type = self.block.get_type();

        if *blk_type == LitType::Unit.into() {
            Type::unit_ref()
        } else if Some(blk_type) != self.else_block.as_ref().map(|e| e.get_type()) {
            ctx.diags
                .add_warning("If block does not return the same type as its else.".to_owned());
            &Type::Error
        } else {
            blk_type
        }
    }
}

#[derive(Debug, Symbol)]
pub struct BoundBlock {
    pub scope: ArenaID<Scope>,
    pub body: Vec<BoundStmtAST>,
    pub tail: Option<Box<BoundExprAST>>,
}

impl BoundSymbol for BoundBlock {
    fn error() -> Self {
        BoundBlock {
            scope: ArenaID::default(),
            body: vec![],
            tail: None,
        }
    }
    fn none() -> Self {
        Self::error()
    }

    fn is_error(&self) -> bool {
        self.scope == ArenaID::default()
    }
    fn is_none(&self) -> bool {
        self.scope == ArenaID::default()
    }
}

impl BoundBlock {
    fn get_type(&self) -> &Type {
        self.tail
            .as_ref()
            .map_or(Type::unit_ref(), |b| b.get_type())
    }
}

#[derive(Debug, Symbol)]
pub enum BoundStmt {
    Error,

    None,

    Expr(BoundExpr),

    While {
        cond: BoundExprAST,
        block: BoundBlock,
    },

    Assign {
        target: Box<BoundExprAST>,
        value: Box<BoundExprAST>,
    },

    Let {
        target: ArenaID<Definition>,
        value: Box<BoundExprAST>
    }
}

#[derive(Debug, Symbol)]
pub struct BoundProgram(pub Vec<Vec<BoundStmtAST>>);

impl BoundSymbol for BoundProgram {
    fn none() -> Self {
        BoundProgram(vec![])
    }

    fn error() -> Self {
        BoundProgram(vec![])
    }

    fn is_error(&self) -> bool {
        self.0.is_empty()
    }
    fn is_none(&self) -> bool {
        self.0.is_empty()
    }
}

impl Stmt {
    pub fn general_pass<T, F: FnMut(&Stmt, &mut ProgramContext) -> T>(
        &self,
        func: &mut F,
        ctx: &mut ProgramContext,
    ) -> Vec<T> {
        match self {
            Stmt::Mod {
                name: _,
                body,
                scope_id,
            } => {
                let exit = ctx.scopes.cur_id();
                ctx.scopes.enter(*scope_id.get().unwrap());

                let body = body
                    .iter()
                    .flat_map(|x| x.kind().general_pass(func, ctx))
                    .collect();

                ctx.scopes.exit_to(exit);

                body
            }

            Stmt::ModHead { name: _, scope_id } => {
                ctx.scopes.enter(*scope_id.get().unwrap());

                vec![]
            }

            _ => vec![func(self, ctx)],
        }
    }

    pub fn load_mod_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Mod {
                name,
                body,
                scope_id,
            } => {
                let c = ctx.scopes.cur_id();

                scope_id
                    .set(ctx.scopes.create_named(name.clone(), &mut ctx.defs))
                    .expect("This should be the first time this occurs!");

                ctx.scopes.enter(*scope_id.get().unwrap());

                body.iter().for_each(|x| x.kind().load_mod_definitions(ctx));

                ctx.scopes.exit_to(c);
            }

            Stmt::ModHead { name, scope_id } => {
                scope_id
                    .set(ctx.scopes.create_named(name.clone(), &mut ctx.defs))
                    .expect("This should be the first time this occurs!");
                ctx.scopes.enter(*scope_id.get().unwrap());
            }

            _ => {}
        }
    }

    pub fn load_struct_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Struct(s) => {
                ctx.scopes.cur_mut().define_lit(
                    Definition::new(
                        s.name.clone(),
                        DefinitionKind::Type(TypeDefinition {
                            typ: OnceBuildable::new(),
                        }),
                    ),
                    &mut ctx.defs,
                );
            }

            Stmt::Mod { .. } | Self::ModHead { .. } => {
                self.general_pass(&mut Self::load_struct_definitions, ctx);
            }

            _ => {}
        };
    }

    pub fn define_structs(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Struct(s) => {
                let def_id = ctx.scopes.lookup(&s.name).unwrap();
                {
                    let def = def_id.get(&ctx.defs);
                    let def = def.as_ty().unwrap();

                    def.typ.begin_build();
                }

                let mut map = HashMap::new();

                for arg in s.fields.iter() {
                    let argty = arg.type_spec.bind(ctx).into_kind();

                    let argty = match argty {
                        Type::Ref(r) if r.get(&ctx.defs).as_ty().unwrap().typ.is_building() => {
                            ctx.diags.add_error_at(arg.type_spec.span.clone(), 
                                format!("Storing value of type {} in struct {} causes recursion", 
                                    r.get(&ctx.defs).full_name(&ctx.scopes.arena),
                                    def_id.get(&ctx.defs).full_name(&ctx.scopes.arena)
                                ));

                            Type::Error
                        }

                        _ => argty
                    };

                    if map.insert(arg.name.clone(), (map.len(), argty)).is_some() {
                        // TODO: add IdentAST for spanning of identifiers
                        ctx.diags.add_error_at(
                            arg.type_spec.span.clone(),
                            format!("Duplicate definition of field {}", arg.name.as_str()),
                        )
                    }
                }

                let def = def_id.get(&ctx.defs);
                let def = def.as_ty().unwrap();

                def.typ
                    .set(LitType::Struct(map))
                    .expect("This should not have been previously run.");
            }

            Stmt::Mod { .. } | Self::ModHead { .. } => {
                self.general_pass(&mut Self::define_structs, ctx);
            }

            _ => {}
        };
    }

    pub fn load_fn_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Fn { name, args, ty, .. } => {
                let scope = ctx.scopes.create();

                let func = Definition::function(
                    name.clone(),
                    LitType::Function(FunctionType {
                        ret: Box::new(
                            ty.as_ref()
                                .map(|ts| ts.bind_content(ctx))
                                .unwrap_or(LitType::Unit.into()),
                        ),
                        args: args
                            .iter()
                            .map(|arg| arg.type_spec.bind_content(ctx))
                            .map(Box::new)
                            .collect(),
                    })
                    .into(),
                    args.iter().map(|arg| arg.name.clone()).collect(),
                    scope,
                );

                let func = ctx.defs.add(func);

                ctx.scopes.cur_mut().define(func, &mut ctx.defs);
            }

            Stmt::DeclareFn { name, args, ty } => {
                let func = Definition::function_decl(
                    name.clone(),
                    FunctionType {
                        ret: Box::new(
                            ty.as_ref()
                                .map(|ts| ts.bind_content(ctx))
                                .unwrap_or(LitType::Unit.into()),
                        ),
                        args: args
                            .iter()
                            .map(|arg| arg.type_spec.bind_content(ctx))
                            .map(Box::new)
                            .collect(),
                    }
                    .into(),
                    args.iter().map(|arg| arg.name.clone()).collect(),
                );

                let func = ctx.defs.add(func);

                ctx.scopes.cur_mut().define(func, &mut ctx.defs);
            }

            Self::Mod { .. } | Self::ModHead { .. } => {
                self.general_pass(&mut Self::load_fn_definitions, ctx);
            }

            _ => {}
        };
    }
}

impl UnboundSymbol for Stmt {
    type Bound = Typed<BoundStmt>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound {
        match self {
            Stmt::Error => Typed::error(),

            Stmt::Import(..) => Typed::none(),

            Stmt::Mod { .. } | Stmt::ModHead { .. } => {
                self.general_pass(&mut Self::bind, ctx);
                Typed::none()
            }

            Stmt::Let { name, ty, value } => {
                let value = value.bind(ctx);
                let valty = value.get_type().clone();

                let ty = if ty.is_some() {
                    let ty = ty.as_ref().unwrap().bind_content(ctx);

                    if !ty.compatible(&valty) {
                        ctx.diags.add_error(format!(
                            "Incompatible type between assignee ({}) and initializer ({})",
                            ty.to_string(&ctx.defs, &ctx.scopes.arena),
                            valty.to_string(&ctx.defs, &ctx.scopes.arena)
                        ));
                        return Self::Bound::error();
                    } else {
                        ty
                    }
                } else {
                    valty.clone()
                };

                let ty = ty.concrete(ctx);

                let var = ctx
                    .defs
                    .add(Definition::new(name.clone(), DefinitionKind::Variable(ty)));
                ctx.scopes.cur_mut().define(var, &mut ctx.defs);

                Typed::new(
                    valty,
                    BoundStmt::Let {
                        target: var,
                        value: Box::new(value),
                    },
                )
            }

            Stmt::Assign { lhs, rhs } => {
                let lhs = lhs.bind_as_lvalue(ctx);
                let ty = lhs.get_type().clone();

                let value = rhs.bind(ctx);
                let valty = value.get_type().clone();

                if !ty.ptr_compatible(&valty) {
                    ctx.diags.add_error(format!(
                        "Incompatible type between assignee ({}) and value ({})",
                        ty.to_string(&ctx.defs, &ctx.scopes.arena),
                        valty.to_string(&ctx.defs, &ctx.scopes.arena)
                    ));
                    return Self::Bound::error();
                }

                Typed::new(
                    valty,
                    BoundStmt::Assign {
                        target: Box::new(lhs),
                        value: Box::new(value),
                    },
                )
            }

            Stmt::Expr(e) => {
                let e = e.bind(ctx);

                Typed::new(e.get_type().clone(), BoundStmt::Expr(e.into_kind()))
            }

            Stmt::Fn {
                name,
                args: _,
                ty: _,
                body,
            } => {
                let def = ctx
                    .scopes
                    .lookup(name)
                    .expect("Should have already defined this!");

                let (ty, argnames, scope) = match &ArenaRef::into_ref(def.get(&ctx.defs)).kind {
                    DefinitionKind::Function(f) => (
                        f.get_type().as_fn().unwrap().clone(),
                        f.argnames.clone(),
                        f.body_info.as_ref().unwrap().scope,
                    ),
                    _ => unreachable!(),
                };

                ctx.scopes.enter(scope);

                for (typ, name) in ty.args.iter().zip(argnames.clone().iter()) {
                    let param = ctx.scopes.cur_mut().define_lit(
                        Definition::new(
                            name.clone(),
                            DefinitionKind::Parameter(typ.deref().clone()),
                        ),
                        &mut ctx.defs,
                    );

                    def.get_mut(&mut ctx.defs)
                        .as_fn_mut()
                        .unwrap()
                        .body_info
                        .as_mut()
                        .unwrap()
                        .argdefs
                        .push(param);
                }

                let body = body.bind(ctx);
                let bodyty = body.get_type();

                ctx.scopes.exit();

                if !ty.ret.compatible(bodyty) {
                    ctx.diags.add_error(format!(
                        "Function expects return of type {} but got {}",
                        ty.ret.to_string(&ctx.defs, &ctx.scopes.arena),
                        bodyty.to_string(&ctx.defs, &ctx.scopes.arena)
                    ));
                    Typed::error()
                } else {
                    let mut def = def.get_mut(&mut ctx.defs);
                    let def = def.as_fn_mut().unwrap();

                    def.body_info.as_mut().unwrap().body_ast = body;
                    Typed::none()
                }
            }

            Stmt::While { cond, block } => Typed::new(
                Type::none(),
                BoundStmt::While {
                    cond: cond.bind(ctx),
                    block: block.bind(ctx).into_kind(),
                },
            ),

            Stmt::DeclareFn { .. } => Typed::none(),

            Stmt::Struct(..) => Typed::none(),
        }
    }
}

impl UnboundSymbol for TypeSpec {
    type Bound = Type;
    fn bind(&self, ctx: &mut ProgramContext) -> Type {
        match &self {
            TypeSpec::Named(name) => match name.as_str() {
                "i8" => IntType::I8.into(),
                "i32" => IntType::I32.into(),
                "i64" => IntType::I64.into(),
                "isize" => IntType::ISize.into(),

                "u8" => UIntType::U8.into(),
                "u32" => UIntType::U32.into(),
                "u64" => UIntType::U64.into(),
                "usize" => UIntType::USize.into(),

                "f32" => FloatType::F32.into(),
                "f64" => FloatType::F64.into(),

                "char" => LitType::Char.into(),
                "str" => LitType::Str.into(),

                "bool" => LitType::Bool.into(),
                "unit" => LitType::Unit.into(),

                _ => {
                    if let Some(lookup) = ctx.scopes.lookup(name) {
                        Type::Ref(lookup)
                    } else {
                        ctx.diags
                            .add_error(format!("Unknown type {}", name.as_str()));
                        Type::error()
                    }
                }
            },

            TypeSpec::Scoped(scopes, n) => {
                let mut scope = ctx.scopes.current;

                for s in scopes {
                    let new_scope = ctx.scopes.lookup_at(scope, s);
                    if let Some(new_scope) = new_scope {
                        let new_scope = new_scope.get(&ctx.defs);
                        scope = match &new_scope.kind {
                            DefinitionKind::Scope(x) => *x,
                            _ => {
                                ctx.diags.add_error(format!(
                                    "Unexpected non-scope item {}",
                                    new_scope.full_name(&ctx.scopes.arena)
                                ));
                                return Type::error();
                            }
                        }
                    } else {
                        ctx.diags.add_error(format!(
                            "Could not find {} in scope {}",
                            s.as_str(),
                            scope.get(&ctx.scopes.arena).full_name(&ctx.scopes.arena)
                        ));
                        return Type::error();
                    }
                }

                if let Some(lookup) = ctx.scopes.lookup_at(scope, n) {
                    Type::Ref(lookup)
                } else {
                    ctx.diags.add_error(format!(
                        "Unknown type {}::{}",
                        scope.get(&ctx.scopes.arena).full_name(&ctx.scopes.arena),
                        n.as_str()
                    ));
                    Type::error()
                }
            }

            TypeSpec::Ptr(inner) => LitType::Ptr(Box::new(inner.bind_content(ctx))).into(),
            TypeSpec::Ref(inner) => {
                LitType::Ref(Box::new(inner.bind_content(ctx)), RefKind::Plain).into()
            }
            TypeSpec::RefMut(inner) => {
                LitType::Ref(Box::new(inner.bind_content(ctx)), RefKind::Mut).into()
            }
            TypeSpec::RefMove(inner) => {
                LitType::Ref(Box::new(inner.bind_content(ctx)), RefKind::Move).into()
            }
        }
    }
}

impl UnboundSymbol for Block {
    type Bound = Typed<BoundBlock>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound {
        let scope = ctx.scopes.create();
        ctx.scopes.enter(scope);

        let block = BoundBlock {
            scope,
            body: self.body.iter().map(|e| e.bind(ctx)).collect(),
            tail: self.tail.as_ref().map(|e| e.bind(ctx)).map(Box::new),
        };

        ctx.scopes.exit();

        Typed::new(block.get_type().clone(), block)
    }
}

impl UnboundSymbol for IfNode {
    type Bound = Typed<BoundExpr>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound {
        match &self.tail {
            ElseKind::None => Typed::new(
                LitType::Unit.into(),
                BoundExpr::If(BoundIf {
                    cond: Box::new(self.cond.bind(ctx)),
                    block: self.block.bind(ctx).into_kind(),
                    else_block: None,
                }),
            ),

            ElseKind::Else(block) => {
                let ifn = BoundIf {
                    cond: Box::new(self.cond.bind(ctx)),
                    block: self.block.bind(ctx).into_kind(),
                    else_block: Some(Box::new(block.bind(ctx).map_deep(BoundExpr::Block))),
                };

                Typed::new(ifn.calculate_type(ctx).clone(), BoundExpr::If(ifn))
            }

            ElseKind::ElseIf(elif) => {
                let ifn = BoundIf {
                    cond: Box::new(self.cond.bind(ctx)),
                    block: self.block.bind(ctx).into_kind(),
                    else_block: Some(Box::new(elif.bind(ctx))),
                };

                Typed::new(ifn.calculate_type(ctx).clone(), BoundExpr::If(ifn))
            }
        }
    }
}

impl Expr {
    fn bind_as_lvalue(&self, ctx: &mut ProgramContext) -> Typed<BoundExpr> {
        match self {
            Expr::Identifier(x) => {
                let def_id = ctx.scopes.lookup(&x);

                if def_id.is_none() {
                    ctx.diags
                        .add_error(format!("Unknown identifier {}", x.as_str()));
                    Typed::error()
                } else if let DefinitionKind::Variable(ty) = &def_id.unwrap().get(&ctx.defs).kind {
                    Typed::new(
                        Type::Literal(LitType::Ptr(Box::new(ty.clone()))),
                        BoundExpr::VariableRef(def_id.unwrap())
                    )
                } else {
                    ctx.diags
                        .add_error(format!("Cannot treat r-value {} as an l-value", def_id.unwrap().get(&ctx.defs).full_name(&ctx.scopes.arena)));
                    Typed::error()
                }
            }

            Expr::Deref(op) => {
                let op = op.bind(ctx);

                match op.get_type() {
                    Type::Literal(LitType::Ptr(..)) => {
                        Typed::new(
                            op.get_type().clone(),
                            BoundExpr::DerefRef(Box::new(op))
                        )
                    }

                    x => {
                        if x.not_error() {
                            ctx.diags
                                .add_error(format!("Cannot deref a non-pointer type"));
                        }
                        Typed::error()
                    }
                }
            }

            Expr::Error => Typed::error(),

            _ => {
                ctx.diags
                    .add_error(format!("Expected an l-value"));
                Typed::error()
            }
        }
    }
}

impl ExprAST {
    fn bind_as_lvalue(&self, ctx: &mut ProgramContext) -> AST<Typed<BoundExpr>> {
        ctx.diags.push_span(self.span.clone());
        let out = self.content.bind_as_lvalue(ctx);
        ctx.diags.pop_span();

        AST {
            content: out,
            span: self.span.clone()
        }
    }
}

impl UnboundSymbol for Expr {
    type Bound = Typed<BoundExpr>;

    fn bind(&self, ctx: &mut ProgramContext) -> Typed<BoundExpr> {
        match self {
            Expr::Error => Typed::error(),
            
            Expr::StaticDot { op, id } => {
                let lhs = op.bind(ctx);

                if let BoundExpr::Symbol(lhs) = lhs.kind() {
                    if let DefinitionKind::Scope(s) = lhs.get(&ctx.defs).kind {
                        let l = ctx.scopes.lookup_at(s, id);

                        if let Some(l) = l {
                            Typed::new(
                                l.get(&ctx.defs)
                                    .try_get_type()
                                    .map_or(Type::unit(), Type::clone),
                                BoundExpr::Symbol(l),
                            )
                        } else {
                            ctx.diags.add_error(format!(
                                "Could not find item {} in scope {}",
                                id.as_str(),
                                s.get(&ctx.scopes.arena).full_name(&ctx.scopes.arena)
                            ));
                            Typed::error()
                        }
                    } else {
                        ctx.diags.add_error(format!(
                            "Could not find item {}",
                            id.as_str()
                        ));
                        Typed::error()
                    }
                } else {
                    if lhs.not_error() {
                        ctx.diags
                            .add_error(format!("Left hand side of :: must be a symbol!"));
                    }

                    Typed::error()
                }
            }

            Expr::Dot { op, id } => {
                let lhs = op.bind(ctx);

                match lhs.get_type() {
                    Type::Ref(x) => {
                        let x = x.get(&ctx.defs);
                        let typ = x.as_ty().unwrap().typ.get().unwrap();

                        match typ {
                            LitType::Struct(hm) => {
                                if let Some((i, ty)) = hm.get(id) {
                                    Typed::new(ty.clone(), BoundExpr::Dot(Box::new(lhs), *i))
                                } else {
                                    // TODO: implement full name for all definitions
                                    ctx.diags.add_error(format!(
                                        "No member {} in type {}",
                                        id.as_str(),
                                        x.name.as_str()
                                    ));
                                    Typed::error()
                                }
                            }

                            // TODO: code duplication! ew!
                            _ => {
                                ctx.diags.add_error(format!(
                                    "Left side of '.' must be a structure type"
                                ));
                                Typed::error()
                            }
                        }
                    }

                    ty => {
                        if ty.not_error() {
                            ctx.diags
                                .add_error(format!("Left side of '.' must be a structure type"));
                        }
                        Typed::error()
                    }
                }
            },

            Expr::Deref(op) => {
                let op = op.bind(ctx);

                match op.get_type() {
                    Type::Literal(LitType::Ptr(ty)) => {
                        Typed::new(
                            (**ty).clone(),
                            BoundExpr::Deref(Box::new(op))
                        )
                    }

                    x => {
                        if x.not_error() {
                            ctx.diags
                                .add_error(format!("Cannot deref a non-pointer type"));
                        }
                        Typed::error()
                    }
                }
            }
            
            Expr::Addressof(op) => {
                op.kind().bind_as_lvalue(ctx)
            }

            Expr::Block(block) => {
                let block = block.bind(ctx);

                Typed::new(
                    block.get_type().clone(),
                    BoundExpr::Block(block.into_kind()),
                )
            }

            Expr::Integer(integer, typ) => Typed::new(
                typ.clone().unwrap_or(IntType::I32.into()),
                BoundExpr::Integer(*integer),
            ),

            Expr::Decimal(decimal, typ) => Typed::new(
                typ.clone().unwrap_or(FloatType::F32.into()),
                BoundExpr::Decimal(*decimal),
            ),

            Expr::String(x) => {
                // todo: is this the desired type?
                Typed::new(
                    LitType::Ptr(Box::new(UIntType::U8.into())).into(),
                    BoundExpr::String(x.clone()),
                )
            }

            Expr::Bool(b) => Constant::Bool(*b).into(),

            Expr::Identifier(name) => {
                let def = ctx.scopes.lookup(name);

                if def.is_none() {
                    ctx.diags
                        .add_error(format!("Unknown identifier {}", name.as_str()));
                    return Typed::error();
                }

                let def_id = def.unwrap();
                let def = def_id.get(&ctx.defs);

                Typed::new(
                    def.try_get_type().map_or(Type::unit(), Type::clone),
                    BoundExpr::Symbol(def_id),
                )
            }

            Expr::Unary { op, target } => {
                let target = target.bind(ctx);
                let ty = target.get_type();

                match op {
                    UnOp::Negate => {
                        if matches!(ty.as_lit(), Some(LitType::Int(..) | LitType::Float(..))) {
                            Typed::new(ty.clone(), BoundExpr::Negate(Box::new(target)))
                        } else {
                            ctx.diags.add_error(format!(
                                "Cannot negate value of type {}",
                                ty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        }
                    }

                    UnOp::Not => {
                        if matches!(ty.as_lit(), Some(LitType::Bool)) {
                            Typed::new(LitType::Bool.into(), BoundExpr::Not(Box::new(target)))
                        } else {
                            ctx.diags.add_error(format!(
                                "Cannot apply not to value of non-boolean type {}",
                                ty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        }
                    }
                }
            }

            Expr::Binary { op, lhs, rhs } => {
                let lhs = lhs.bind(ctx);
                let rhs = rhs.bind(ctx);

                let lty = lhs.get_type();
                let rty = rhs.get_type();

                match op {
                    BinOp::Arith(arith) => {
                        if matches!(lty.as_lit(), Some(LitType::Ptr(..))) {
                            if !matches!(arith, ArithOp::Add | ArithOp::Subtract) {
                                ctx.diags.add_error(format!(
                                    "Cannot perform operation {arith} on pointer type {}",
                                    lty.to_string(&ctx.defs, &ctx.scopes.arena)
                                ));
                                Typed::error()
                            } else if !matches!(
                                rty.as_lit(),
                                Some(LitType::Int(..) | LitType::UInt(..))
                            ) {
                                ctx.diags.add_error(format!(
                                    "Cannot perform operation {arith} on pointer type {} with type {}", 
                                    lty.to_string(&ctx.defs, &ctx.scopes.arena),
                                    rty.to_string(&ctx.defs, &ctx.scopes.arena)
                                ));
                                Typed::error()
                            } else {
                                Typed::new(
                                    lty.clone(),
                                    BoundExpr::PtrArithmetic(
                                        BoundBinary {
                                            lhs: Box::new(lhs),
                                            rhs: Box::new(rhs),
                                        },
                                        arith.clone(),
                                    ),
                                )
                            }
                        } else if lty != rty && lty.not_error() {
                            ctx.diags.add_error(format!(
                                "Cannot perform operation {arith} on distinct types {} and {}",
                                lty.to_string(&ctx.defs, &ctx.scopes.arena),
                                rty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        } else {
                            Typed::new(
                                lty.clone(),
                                BoundExpr::Arithmetic(
                                    BoundBinary {
                                        lhs: Box::new(lhs),
                                        rhs: Box::new(rhs),
                                    },
                                    arith.clone(),
                                ),
                            )
                        }
                    }

                    BinOp::Equals | BinOp::NotEquals => {
                        if !lty.compatible(rty) {
                            ctx.diags.add_error(format!(
                                "Cannot compare two values of differing types {} {}",
                                lty.to_string(&ctx.defs, &ctx.scopes.arena),
                                rty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        }
                        // else if lcnst.is_some() && rcnst.is_some()
                        // {
                        //     Constant::Bool(lcnst == rcnst).into()
                        // }
                        else {
                            Typed::new(
                                LitType::Bool.into(),
                                if *op == BinOp::Equals {
                                    BoundExpr::Equals
                                } else {
                                    BoundExpr::NotEquals
                                }(BoundBinary {
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                }),
                            )
                        }
                    }

                    BinOp::And | BinOp::Or => {
                        if lty != rty {
                            ctx.diags.add_error(format!(
                                "Cannot compare two values of differing types {} {}",
                                lty.to_string(&ctx.defs, &ctx.scopes.arena),
                                rty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        } else if *lty != LitType::Bool.into() {
                            ctx.diags.add_error(format!(
                                "Cannot perform logic on non-boolean type {}",
                                lty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        } else {
                            Typed::new(
                                LitType::Bool.into(),
                                if *op == BinOp::And {
                                    BoundExpr::And
                                } else {
                                    BoundExpr::Or
                                }(BoundBinary {
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                }),
                            )
                        }
                    }

                    BinOp::Comparison(comp) => {
                        if !lty.compatible(rty) {
                            ctx.diags.add_error(format!(
                                "Cannot compare two values of differing types {} {}",
                                lty.to_string(&ctx.defs, &ctx.scopes.arena),
                                rty.to_string(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        } else {
                            Typed::new(
                                LitType::Bool.into(),
                                BoundExpr::Comparison(
                                    BoundBinary {
                                        lhs: Box::new(lhs),
                                        rhs: Box::new(rhs),
                                    },
                                    comp.clone(),
                                ),
                            )
                        }
                    }
                }
            }

            Expr::If(ifn) => ifn.bind(ctx),

            Expr::Call { func, args } => {
                let func = func.bind(ctx);
                let args = args.iter().map(|x| x.bind(ctx)).collect::<Vec<_>>();

                // Handle zeroinits early
                match func.kind() {
                    BoundExpr::Symbol(xid) => {
                        let x = xid.get(&ctx.defs);
                        match &x.kind {
                            DefinitionKind::Type(_) =>
                                return Typed::new(
                                    // TODO: why this so wacky?
                                    Type::Ref(*xid),
                                    BoundExpr::Zeroinit(Type::Ref(*xid))
                                ),

                            _ => {}
                        }
                    }

                    _ => {}
                }

                // Normal flow
                let fty = func.get_type();

                if let Some(fty) = fty.as_fn() {
                    if args.len() != fty.args.len() {
                        ctx.diags.add_error(format!(
                            "Incorrect number of arguments for call to function of type {}",
                            fty.to_string(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    } else {
                        for (arg, argty) in args.iter().zip(fty.args.iter()) {
                            if !arg.get_type().compatible(&argty) {
                                ctx.diags.add_error_at(
                                    arg.span.clone(),
                                    format!(
                                        "Expected argument of type {}, got {}",
                                        argty.to_string(&ctx.defs, &ctx.scopes.arena),
                                        arg.get_type().to_string(&ctx.defs, &ctx.scopes.arena)
                                    ),
                                );
                                return Typed::error();
                            }
                        }

                        if let BoundExpr::Symbol(def) = func.kind() {
                            Typed::new((*fty.ret).clone(), BoundExpr::DirectCall(*def, args))
                        } else {
                            panic!("Unknown function value {:?}", func.kind())
                        }
                    }
                } else {
                    if fty.not_error() {
                        ctx.diags.add_error(format!(
                            "Cannot call non-function value of type {}",
                            fty.to_string(&ctx.defs, &ctx.scopes.arena)
                        ));
                    }
                    Typed::error()
                }
            }

            Expr::Cast { op, typ } => {
                let op = op.bind(ctx);
                let optyp = op.get_type();

                let typ = typ.bind(ctx);

                use LitType::*;

                if typ.is_error() || op.is_error() {
                    Typed::error()
                } else if &typ == optyp {
                    ctx.diags
                        .add_error(format!("Cannot cast a value to its own type"));
                    Typed::error()
                } else if !matches!(
                    (optyp.as_lit(), typ.as_lit()),
                    (
                        Some(Int(..) | UInt(..) | Float(..)),
                        Some(Int(..) | UInt(..) | Float(..))
                    ) | (
                        Some(Ptr(..) | Int(..) | UInt(..)),
                        Some(Ptr(..) | Int(..) | UInt(..))
                    )
                ) {
                    ctx.diags.add_error(format!(
                        "Cannot cast a value of type {} to a value of {}",
                        optyp.to_string(&ctx.defs, &ctx.scopes.arena),
                        typ.to_string(&ctx.defs, &ctx.scopes.arena)
                    ));
                    Typed::error()
                } else {
                    Typed::new(typ.clone(), BoundExpr::Cast(Box::new(op), typ))
                }
            }

            Expr::EOI => Typed::none(),
        }
    }
}

impl Program {

    pub fn new(root_name: String, ctx: &mut ProgramContext) -> Program {
        let mut prog = Program(HashMap::new(), HashSet::new());

        fn make_file(root: &str, file: &str, prog: &mut Program, ctx: &mut ProgramContext) -> io::Result<()> {
            const STD_MARKER: &'static str = "std:";

            let is_std = file.starts_with(STD_MARKER);

            let full_file;

            if is_std {
                let exe = std::env::current_exe().unwrap();
                let exe = exe.canonicalize().unwrap();
                let exe = exe.parent().unwrap();

                let std = exe.join("std");

                full_file = std.join(&file[STD_MARKER.len() ..])
                    .as_os_str()
                    .to_str().unwrap()
                    .to_owned();
            } else {
                full_file = Path::new(&root).join(file)
                    .canonicalize().unwrap()
                    .as_os_str()
                    .to_str().unwrap()
                    .to_owned();
            }

            let ipt = std::fs::read_to_string(full_file)?;
            let ipt = ipt.as_str();

            let src = Src::new(file, ipt);

            let lexer = parsing::lexer::Lexer::new(src, ctx);
            let mut parser = parsing::lexer::Parser::new(lexer);

            let ast = parser.file();
            
            prog.1.insert(file.to_owned());
            
            for x in &ast.0 {
                match x.kind() {
                    Stmt::Import(imported) => {
                        if !prog.1.contains(imported) {
                            let _ = make_file(root, imported.as_str(), prog, ctx).map_err(|_| {
                                ctx.diags.add_error_at(
                                    x.span.clone(), 
                                    format!("Could not load file '{}'", imported));
                            });
                        }
                    }

                    _ => {}
                }
            }

            prog.0.insert(file.to_owned(), ast);

            Ok(())
        }

        let filebuf = Path::new(&root_name)
            .canonicalize().unwrap();

        let file = filebuf.file_name().unwrap()
            .to_str().unwrap();
        let root = filebuf.parent().unwrap()
            .to_str().unwrap();

        make_file(
            root,
            file,
            &mut prog, 
            ctx
        ).expect("Invalid input file!");

        prog
    }


    pub fn bind(&mut self, ctx: &mut ProgramContext) -> BoundProgram {

        self.0.iter().for_each(|(_, x)| {
            x.0.iter().for_each(|x| x.load_mod_definitions(ctx));
            ctx.scopes.exit_to(ArenaID::default());
        });

        self.0.iter().for_each(|(_, x)| {
            x.0.iter().for_each(|x| x.load_struct_definitions(ctx));
            ctx.scopes.exit_to(ArenaID::default());
        });

        self.0.iter().for_each(|(_, x)| {
            x.0.iter().for_each(|x| x.define_structs(ctx));
            ctx.scopes.exit_to(ArenaID::default());
        });

        self.0.iter().for_each(|(_, x)| {
            x.0.iter().for_each(|x| x.load_fn_definitions(ctx));
            ctx.scopes.exit_to(ArenaID::default());
        });

        BoundProgram(self.0.iter().map(|(_, x)| x.0.iter().map(|x| x.bind(ctx)).collect()).collect::<Vec<_>>())
    }
}
