use std::borrow::Cow;
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::rc::Rc;
use hashlink::LinkedHashMap;
use toylang_derive::Symbol;

use crate::core::arena::{Arena, ArenaID, ArenaMut, ArenaRef, ArenaValue};
use crate::core::lazy::OnceBuildable;
use crate::core::monad::Wrapper;
use crate::core::ops::{ArithOp, BinOp, CompOp, ConstFloat, ConstInt, Constant, UnOp};
use crate::core::utils::{Itertools, FlagCell};
use crate::core::{Diagnostic, Ident, SourceSpan, Src};
use crate::parsing;
use crate::parsing::ast::{
    Block, BoundSymbol, ElseKind, Expr, IfNode, OwnedSymbol, Stmt, Symbol, TypeSpec,
    Typed, TypedSymbol, UnboundSymbol, AST, ExprAST, Program,
};
use crate::types::{
    FloatType, FunctionType, IntType, LitType, RefKind, Type, TypeDefinition, UIntType, DotError,
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

#[derive(Debug, Clone)]
pub struct GenericInfo {
    pub args: RcList<Type>
}

impl GenericInfo {
    pub fn new() -> Self {
        Self {
            args: Rc::new([])
        }
    }

    pub fn from_args(args: RcList<Type>) -> Self {
        Self {
            args
        }
    }
}

#[derive(Debug)]
pub struct FunctionDefinition {
    pub typ: Type,
    pub argnames: Vec<Ident>,
    pub body_info: Option<FunctionBodyInfo>,
    pub gen_info: Option<GenericInfo>
}

impl FunctionDefinition {
    pub fn is_declare(&self) -> bool {
        self.body_info.is_none()
    }
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
    PlaceholderType,
    Scope(ArenaID<Scope>),
}

pub type RcList<T> = Rc<[T]>;

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub enum SymbolRef {
    Basic(ArenaID<Definition>),
    Generic(ArenaID<Definition>, RcList<Type>)
}

impl From<ArenaID<Definition>> for SymbolRef {
    fn from(value: ArenaID<Definition>) -> Self {
        Self::Basic(value)
    }
}
impl<'a> From<&'a ArenaID<Definition>> for SymbolRef {
    fn from(value: &'a ArenaID<Definition>) -> Self {
        Self::Basic(*value)
    }
}


impl SymbolRef {
    pub fn id(&self) -> ArenaID<Definition> {
        match self {
            Self::Basic(x) => *x,
            Self::Generic(x, ..) => *x
        }
    }
    pub fn gen_args_src(&self, defs: &Definitions) -> Option<RcList<Type>> {
        match self {
            Self::Basic(..) => None,
            Self::Generic(x, ..) => {
                let x = x.get(defs);
                x.gen_args().cloned()
            }
        }
    }
    pub fn gen_args_this(&self) -> Option<RcList<Type>> {
        match self {
            Self::Basic(..) => None,
            Self::Generic(_, g) => Some(g.clone()),
        }
    }
    pub fn gen_args(&self, defs: &Definitions) -> Option<(RcList<Type>, RcList<Type>)> {
        match self {
            Self::Basic(..) => None,
            Self::Generic(x, g) => Some((x.get(defs).gen_args().unwrap().clone(), g.clone())),
        }
    }
    pub fn replace_all<'a>(&'a self, old: &[Type], new: &[Type]) -> Cow<'a, Self> {
        match self {
            Self::Basic(..) => Cow::Borrowed(self),
            Self::Generic(x, args) => {
                Cow::Owned(Self::Generic(
                    *x,
                    args.to_vec()
                        .iter_mut()
                        .map(|t| {t.replace_all(old, new); t.clone()})
                        .collect()
                ))
            }
        }
    }

    // TODO: needing to copy around defs and scopes! is this necessary, or can these be static?
    pub fn get_type(&self, defs: &Definitions, _scopes: &Arena<Scope>) -> Type {
        match self {
            Self::Basic(x) => x.get(defs).get_type().clone(),
            Self::Generic(x, args) => {
                let def = x.get(defs);
                
                // TODO: necessary to have this logic here?
                match &def.kind {
                    DefinitionKind::Type(..) => Type::Gen(*x, args.to_vec()),
                    DefinitionKind::Function(func) => {
                        let ty = func.get_type();
                        ty.replaced_all(&func.gen_info.as_ref().unwrap().args, args)
                    }

                    k => panic!("Symbol references cannot contain {k:?}")
                }
            }
        }
    }
    pub fn compiled_name(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        if let Some(gargs) = self.gen_args_this() {
            self.id().get(defs).compiled_name_with_args(&gargs, defs, scopes)
        } else {
            self.id().get(defs).compiled_name(scopes)
        }
    }
}

#[derive(Debug)]
pub struct Definition {
    name: Ident,
    scope: ArenaID<Scope>,
    pub kind: DefinitionKind,
}

impl ArenaValue for Definition {
    fn assign_id(&mut self, id: ArenaID<Self>) {
        if let DefinitionKind::Type(td) = &mut self.kind {
            td.typ = Type::Ref(id);
        }
    }
}

impl Definition {
    pub fn name(&self) -> &Ident {
        &self.name
    }
    pub fn full_name(&self, scopes: &Arena<Scope>) -> String {
        self.scope.get(&scopes).full_name_of(self.name.as_str(), scopes)
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
    
    pub fn compiled_name_with_args(&self, args: &[Type], defs: &Arena<Definition>, scopes: &Arena<Scope>) -> String {
        self.compiled_name(scopes) + "::<>gen[" + args.iter().map(|x| x.full_name(defs, scopes)).collect::<Vec<_>>().join(";").as_str() + "]"
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
        gen_info: Option<GenericInfo>
    ) -> Definition {
        Self::new(
            name,
            DefinitionKind::Function(FunctionDefinition {
                typ,
                argnames,
                body_info: Some(FunctionBodyInfo::from_scope(scope)),
                gen_info
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
                gen_info: None
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
        match &self.kind {
            DefinitionKind::Variable(typ) => Some(typ),
            DefinitionKind::Parameter(typ) => Some(typ),
            DefinitionKind::Function(FunctionDefinition { typ, .. }) => Some(typ),
            DefinitionKind::Type(td) => Some(&td.typ),

            DefinitionKind::Scope(..) | DefinitionKind::PlaceholderType => None,
        }
    }

    pub fn gen_args(&self) -> Option<&RcList<Type>> {
        match &self.kind {
            DefinitionKind::Function(f) => f.gen_info.as_ref().map(|x| &x.args),
            DefinitionKind::Type(t) => Some(&t.gen_args),

            _ => None
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
    pub fn create_named(&mut self, name: Ident, defs: &mut Definitions) -> Result<ArenaID<Scope>, ArenaID<Definition>> {
        let sc = self.arena.add(Scope::new_named(self.current, name.clone()));
        self.cur_mut()
            .define_lit(Definition::new(name, DefinitionKind::Scope(sc)), defs)?;
        Ok(sc)
    }

    pub fn enter(&mut self, scope: ArenaID<Scope>) -> ArenaID<Scope> {
        let c = self.current;
        self.current = scope;
        c
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
            None => "".into(),
            Some(p) => match &self.name {
                None => p.get(arena).full_name(arena),
                Some(n) => if p == Default::default() {n.as_str().to_owned()} else {
                    p.get(arena).full_name(arena) + "::" + n.as_str()
                },
            },
        }
    }
    pub fn full_name_of(&self, child: &str, arena: &Arena<Scope>) -> String {
        let full = self.full_name(arena);
        if full.is_empty() { child.to_owned() } else { full + "::" + child }
    }

    pub fn id(&self) -> ArenaID<Scope> {
        self.id
            .expect("ID should never be called before a scope is added to a scope arena!")
    }

    pub fn define(&mut self, definition: ArenaID<Definition>, defs: &mut Arena<Definition>) -> Result<(), ArenaID<Definition>> {
        let mut def_mut = definition.get_mut(defs);
        def_mut.set_scope(self.id());

        match self.definitions.entry(definition.get(&defs).name().clone()) {
            Entry::Occupied(o) => Err(*o.get()),
            Entry::Vacant(v) => {
                v.insert(definition);
                Ok(())
            }
        }
    }

    pub fn define_lit(
        &mut self,
        definition: Definition,
        defs: &mut Arena<Definition>,
    ) -> Result<ArenaID<Definition>, ArenaID<Definition>> {
        let def = defs.add(definition);
        self.define(def, defs)?;
        Ok(def)
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
    pub defs: Definitions,

    pub declared_fns: HashSet<String>,

    pub cur_func_ty: Option<FunctionType>,
    pub inside_loop: bool,
}

impl ProgramContext {
    pub fn new() -> Self {
        ProgramContext {
            diags: DiagnosticHandler::new(),
            scopes: Scopes::new(),
            defs: Arena::new(),

            declared_fns: HashSet::new(),

            cur_func_ty: None,
            inside_loop: false
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

#[derive(Debug, Clone)]
pub struct BoundBinary {
    pub lhs: Box<BoundExprAST>,
    pub rhs: Box<BoundExprAST>,
}

#[derive(Debug, Clone, Symbol)]
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

    Type(SymbolRef),
    Fn(SymbolRef),

    Dot(Box<BoundExprAST>, usize),

    DirectCall(Box<BoundExprAST>, Vec<BoundExprAST>),
    IndirectCall(Box<BoundExprAST>, Vec<BoundExprAST>),

    Cast(Box<BoundExprAST>, Type),

    Deref(Box<BoundExprAST>),

    VariableRef(ArenaID<Definition>),

    DotRef(Box<BoundExprAST>, usize),
    DerefRef(Box<BoundExprAST>),

    Sizeof(Type)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ValueKind {
    L, R
}

impl BoundExprAST {
    pub fn auto_deref(&self, kind: ValueKind) -> BoundExprAST {
        let mut out = self.clone();
        while let Some(df) = out.get_type().deref().cloned() {
            if df.is_error() {break}
            if kind == ValueKind::L && df.deref().is_none() {break}

            out = AST::new(
                out.span.clone(),
                Typed::new(
                    df, BoundExpr::Deref(Box::new(out))
                )
            );
        }
        out
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Symbol, Clone)]
pub struct BoundBlock {
    pub scope: ArenaID<Scope>,
    pub body: Vec<BoundStmtAST>,
    pub tail: Option<Box<BoundExprAST>>,
    pub is_terminator: bool
}

impl BoundSymbol for BoundBlock {
    fn error() -> Self {
        BoundBlock {
            scope: ArenaID::default(),
            body: vec![],
            tail: None,
            is_terminator: false
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

#[derive(Debug, Symbol, Clone)]
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
    },

    Return(Option<BoundExpr>),
    Break,
    Continue,
}

impl BoundExpr {
    pub fn is_terminator(&self) -> bool {
        match self {
            BoundExpr::If(BoundIf {cond, block, else_block}) => {
                cond.kind().is_terminator() || (block.kind().is_terminator && else_block.as_ref().is_some_and(|x| x.kind().is_terminator()))
            }

            BoundExpr::Block(block) => block.kind().is_terminator,

            _ => false
        }
    }
}

impl BoundStmt {
    pub fn is_terminator(&self) -> bool {
        match self {
            BoundStmt::Return(..) | BoundStmt::Break | BoundStmt::Continue => true,
            BoundStmt::Expr(x) => x.is_terminator(),

            _ => false
        }
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
                error
            } => {
                if error.get() {return vec![]}

                let exit = ctx.scopes.cur_id();
                ctx.scopes.enter(*scope_id.get().unwrap());

                let body = body
                    .iter()
                    .flat_map(|x| x.general_pass(func, ctx))
                    .collect();

                ctx.scopes.exit_to(exit);

                body
            }

            Stmt::ModHead { name: _, scope_id, error } => {
                if error.get() {return vec![]}

                ctx.scopes.enter(*scope_id.get().unwrap());

                vec![]
            }

            _ => vec![func(self, ctx)],
        }
    }

    pub fn get_or_make_mod(err: &FlagCell, name: &Ident, ctx: &mut ProgramContext) -> ArenaID<Scope> {
        let found = ctx.scopes.lookup(name);

        let scope = if let Some(found) = found {
            let found = found.get(&ctx.defs);
            if let DefinitionKind::Scope(x) = found.kind {
                Some(x)
            } else {
                // short circuit with dummy value on error
                ctx.diags.add_error(format!("Attempt to redefine {}", ctx.scopes.cur().full_name_of(name.as_str(), &ctx.scopes.arena)));
                err.mark();
                return ctx.scopes.cur_id();
            }
        } else {None};

        scope.unwrap_or_else(|| ctx.scopes.create_named(name.clone(), &mut ctx.defs).unwrap())
    }

    pub fn load_mod_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Mod {
                name,
                body,
                scope_id,
                error
            } => {
                if error.get() {return}

                let c = ctx.scopes.cur_id();

                scope_id
                    .set(Self::get_or_make_mod(error, name, ctx))
                    .expect("This should be the first time this occurs!");

                ctx.scopes.enter(*scope_id.get().unwrap());

                body.iter().for_each(|x| x.load_mod_definitions(ctx));

                ctx.scopes.exit_to(c);
            }

            Stmt::ModHead { name, scope_id, error } => {
                if error.get() {return}
                scope_id
                    .set(Self::get_or_make_mod(error, name, ctx))
                    .expect("This should be the first time this occurs!");
                ctx.scopes.enter(*scope_id.get().unwrap());
            }

            _ => {}
        }
    }

    pub fn load_struct_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Struct(s) => {
                let scope = match ctx.scopes.create_named(Ident::from("<type>".to_owned() + s.name.as_str()), &mut ctx.defs) {
                    Ok(x) => x,
                    Err(..) => {
                        ctx.diags.add_error(format!("Attempt to redefine type {}", ctx.scopes.cur().full_name_of(s.name.as_str(), &ctx.scopes.arena)));
                        s.error.mark();
                        return
                    }
                };

                let mut gen_args = vec![];

                for arg in &s.args {
                    let arg = match scope.get_mut(&mut ctx.scopes.arena).define_lit(
                        Definition::new(arg.clone(), DefinitionKind::PlaceholderType),
                        &mut ctx.defs
                    ) {
                        Ok(x) => x,
                        Err(..) => {
                            ctx.diags.add_error(format!("Attempt to redefine generic argument {}", arg.as_str()));
                            s.error.mark();
                            return
                        }
                    };

                    gen_args.push(Type::Ref(arg));
                }

                match ctx.scopes.cur_mut().define_lit(
                    Definition::new(
                        s.name.clone(),
                        DefinitionKind::Type(TypeDefinition {
                            inner_type: OnceBuildable::new(),
                            gen_args: gen_args.into(),
                            scope,
                            typ: Type::Error
                        }),
                    ),
                    &mut ctx.defs,
                ) {
                    Ok(..) => {},
                    Err(..) => {
                        ctx.diags.add_error(format!("Attempt to redefine type {}", ctx.scopes.cur().full_name_of(s.name.as_str(), &ctx.scopes.arena)));
                        s.error.mark()
                    },
                };
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
                if s.error.get() {return}

                let def_id = ctx.scopes.lookup(&s.name).unwrap();
                {
                    let def = def_id.get(&ctx.defs);
                    let def = def.as_ty().unwrap();

                    def.inner_type.begin_build();

                    ctx.scopes.enter(def.scope);
                }

                let mut map = LinkedHashMap::new();

                for arg in s.fields.iter() {
                    let argty = arg.type_spec.bind(ctx).into_kind();

                    // dbg!(&argty);
                    let argty = match argty {
                        Type::Ref(r) if matches!(r.get(&ctx.defs).kind, DefinitionKind::PlaceholderType) => argty,

                        Type::Ref(r) if r.get(&ctx.defs).as_ty().unwrap().inner_type.is_building() => {
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

                def.inner_type
                    .set(LitType::Struct(map))
                    .expect("This should not have been previously run.");

                ctx.scopes.exit();
            }

            Stmt::Mod { .. } | Self::ModHead { .. } => {
                self.general_pass(&mut Self::define_structs, ctx);
            }

            _ => {}
        };
    }

    pub fn load_fn_definitions(&self, ctx: &mut ProgramContext) {
        match self {
            Stmt::Fn { name, gen_args, args, ty, body: _, error } => {
                let scope = ctx.scopes.create();

                // TODO: code duplication with struct definition!
                let mut gen_args_ty = vec![];

                for arg in gen_args {
                    let arg = match scope.get_mut(&mut ctx.scopes.arena).define_lit(
                        Definition::new(arg.clone(), DefinitionKind::PlaceholderType),
                        &mut ctx.defs
                    ) {
                        Ok(x) => x,
                        Err(..) => {
                            ctx.diags.add_error(format!("Attempt to redefine generic argument {}", arg.as_str()));
                            error.mark();
                            return
                        }
                    };

                    gen_args_ty.push(Type::Ref(arg));
                }

                let cur = ctx.scopes.enter(scope);

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
                            .map(|arg| arg.content.type_spec.bind_content(ctx))
                            .map(Box::new)
                            .collect(),
                        vararg: false
                    })
                    .into(),
                    args.iter().map(|arg| arg.content.name.clone()).collect(),
                    scope,
                    if gen_args_ty.is_empty() {None} else {Some(GenericInfo::from_args(gen_args_ty.into()))},
                );

                ctx.scopes.exit_to(cur);

                let func = ctx.defs.add(func);

                if let Err(..) = ctx.scopes.cur_mut().define(func, &mut ctx.defs) {
                    ctx.diags.add_error(format!("Attempt to redefine function {}", func.get(&ctx.defs).full_name(&ctx.scopes.arena)));
                    error.mark();
                }
            }

            Stmt::DeclareFn { name, args, ty, vararg } => {
                
                if ctx.declared_fns.contains(name.as_str()) {
                    ctx.diags.add_error(format!("Attempt to redefine external function {}", name.as_str()));
                }

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

                        vararg: *vararg
                    }
                    .into(),
                    args.iter().map(|arg| arg.name.clone()).collect(),
                );

                let func = ctx.defs.add(func);

                if let Err(..) = ctx.scopes.cur_mut().define(func, &mut ctx.defs) {
                    ctx.diags.add_error(format!("Attempt to redefine function {}", func.get(&ctx.defs).full_name(&ctx.scopes.arena)));
                }

                ctx.declared_fns.insert(name.to_string());
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
                            ty.full_name(&ctx.defs, &ctx.scopes.arena),
                            valty.full_name(&ctx.defs, &ctx.scopes.arena)
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

                if let Err(..) = ctx.scopes.cur_mut().define(var, &mut ctx.defs) {
                    ctx.diags.add_error(format!("Attempt to redefine {}", var.get(&ctx.defs).name().as_str()))
                }

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

                if !ty.deref_compatible(&valty) {
                    ctx.diags.add_error(format!(
                        "Incompatible type between assignee ({}) and value ({})",
                        ty.full_name(&ctx.defs, &ctx.scopes.arena),
                        valty.full_name(&ctx.defs, &ctx.scopes.arena)
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

            Stmt::Break => {
                if !ctx.inside_loop {
                    ctx.diags.add_error(format!("Cannot 'break' outside of a loop"));
                    Typed::error()
                } else {
                    Typed::new(
                        Type::unit(),
                        BoundStmt::Break
                    )
                }
            }
            
            Stmt::Continue => {
                if !ctx.inside_loop {
                    ctx.diags.add_error(format!("Cannot 'continue' outside of a loop"));
                    Typed::error()
                } else {
                    Typed::new(
                        Type::unit(),
                        BoundStmt::Continue
                    )
                }
            }
            
            Stmt::Return(expr) => {
                let expr = expr.as_ref().map(|x| x.bind(ctx));

                match expr {
                    None => if !ctx.cur_func_ty.as_ref().unwrap().ret.compatible(Type::unit_ref()) {
                        ctx.diags.add_error(format!(
                            "Cannot return without a value from a function of type {}",
                            ctx.cur_func_ty.as_ref().unwrap().full_name(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    } else {
                        Typed::new(
                            Type::unit(),
                            BoundStmt::Return(None)
                        )
                    },
                    
                    Some(expr) => if !ctx.cur_func_ty.as_ref().unwrap().ret.compatible(expr.get_type()) {
                        ctx.diags.add_error(format!(
                            "Cannot return a value of type {} from a function of type {}",
                            expr.get_type().full_name(&ctx.defs, &ctx.scopes.arena),
                            ctx.cur_func_ty.as_ref().unwrap().full_name(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    } else {
                        Typed::new(
                            Type::unit(),
                            BoundStmt::Return(Some(expr.into_kind()))
                        )
                    }
                }
            }

            Stmt::Fn {
                name,
                gen_args: _,
                args,
                ty: _,
                body,
                error
            } => {
                if error.get() {return Typed::error()}

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
                ctx.cur_func_ty = Some(ty.clone());

                for (typ, (name, spec)) in ty.args.iter().zip(argnames.clone().iter().zip(args)) {
                    let param = ctx.scopes.cur_mut().define_lit(
                        Definition::new(
                            name.clone(),
                            DefinitionKind::Parameter(typ.deref().clone()),
                        ),
                        &mut ctx.defs,
                    );

                    let param = match param {
                        Ok(x) => x,
                        Err(x) => {
                            ctx.diags.add_error_at(spec.span.clone(), format!("Duplicate argument {}", name.as_str()));
                            x
                        }
                    };

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

                ctx.cur_func_ty = None;
                ctx.scopes.exit();

                if !ty.ret.compatible(bodyty) {
                    ctx.diags.add_error(format!(
                        "Function expects return of type {} but got {}",
                        ty.ret.full_name(&ctx.defs, &ctx.scopes.arena),
                        bodyty.full_name(&ctx.defs, &ctx.scopes.arena)
                    ));
                    Typed::error()
                } else {
                    let mut def = def.get_mut(&mut ctx.defs);
                    let def = def.as_fn_mut().unwrap();

                    def.body_info.as_mut().unwrap().body_ast = body;
                    Typed::none()
                }
            }

            Stmt::While { cond, block } => {
                ctx.inside_loop = true;
                let block = block.bind(ctx).into_kind();
                ctx.inside_loop = false;

                Typed::new(
                    Type::none(),
                    BoundStmt::While {
                        cond: cond.bind(ctx),
                        block
                    },
                )
            }

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
                        "Unknown type {}",
                        scope.get(&ctx.scopes.arena).full_name_of(n.as_str(), &ctx.scopes.arena)
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

            TypeSpec::Gen(base, args) => {
                let base = base.bind(ctx);
                match &base {
                    Type::Ref(x) => Type::Gen(*x, args.iter().map(|x| x.bind(ctx)).collect()),

                    _ => {
                        ctx.diags.add_error(format!("Cannot parameterize non-generic type"));
                        Type::Error
                    }
                }
            }
        }
    }
}

impl UnboundSymbol for Block {
    type Bound = Typed<BoundBlock>;

    fn bind(&self, ctx: &mut ProgramContext) -> Self::Bound {
        let scope = ctx.scopes.create();
        ctx.scopes.enter(scope);

        let mut body = vec![];
        let mut unreachable = vec![];

        let mut terminated = false;

        for stmt in &self.body {
            let stmt = stmt.bind(ctx);
            let is_term = stmt.kind().is_terminator();

            if terminated {&mut unreachable} else {&mut body}.push(stmt);

            if is_term {
                terminated = true;
            }
        }

        if !unreachable.is_empty() {
            let span = unreachable[0].span.clone() + unreachable[unreachable.len() - 1].span.clone();
            ctx.diags.add_warning_at(span, format!("Unreachable code"));
        }

        let tail = self.tail.as_ref().map(|e| e.bind(ctx)).map(Box::new);

        let block = BoundBlock {
            scope,
            body,
            tail,
            is_terminator: terminated
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
                    Type::Literal(LitType::Ptr(..)) | Type::Literal(LitType::Ref(..)) => {
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

            Expr::Dot { op, id } => {
                let lhs = op.bind_as_lvalue(ctx);
                let lhs = lhs.auto_deref(ValueKind::L);

                // dbg!(&lhs);

                let ty = lhs.get_type().deref().unwrap();

                match ty.dot(id, &ctx.defs) {
                    Ok((i, ty)) => Typed::new(
                        Type::Literal(LitType::Ptr(Box::new(ty))), 
                        BoundExpr::DotRef(Box::new(lhs), i)
                    ),

                    Err(DotError::NoFieldFound) => {
                        ctx.diags.add_error(format!(
                            "No member {} in type {}",
                            id.as_str(),
                            ty.full_name(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    }

                    Err(DotError::NotAStructure) => {
                        if ty.not_error() {
                            ctx.diags
                                .add_error(format!("Left side of '.' must be a structure type"));
                        }
                        Typed::error()
                    }
                }
            }

            Expr::Error => Typed::error(),

            _ => {
                let op = self.bind(ctx);

                match op.get_type() {
                    Type::Literal(LitType::Ref(..)) => op,
                    Type::Error => Typed::error(),

                    _ => {
                        ctx.diags
                            .add_error(format!("Expected an l-value"));
                        Typed::error()
                    }
                }
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

            Expr::Gen { op, args } => {
                let lhs = op.bind(ctx);

                match lhs.kind() {
                    BoundExpr::Fn(func) => {
                        let args: RcList<_> = args.iter().map(|x| x.bind_content(ctx)).collect();
                        
                        let func_id = func.id();

                        let mut func = func_id.get_mut(&mut ctx.defs);
                        let func = func
                            .as_fn_mut()
                            .unwrap();

                        let gen_info = func.gen_info.as_mut().unwrap();
                        
                        if gen_info.args.len() != args.len() {
                            ctx.diags.add_error(format!("Incorrect number of generic arguments"));
                        }

                        let mut typ = lhs.get_type().clone();
                        typ.replace_all(&gen_info.args, &args);

                        Typed::new(
                            typ,
                            BoundExpr::Fn(SymbolRef::Generic(func_id, args))
                        )
                    },
                    BoundExpr::Type(ty) => {
                        let args: RcList<_> = args.iter().map(|x| x.bind_content(ctx)).collect();

                        let ty_id = ty.id();

                        let ty = ty_id.get(&ctx.defs);
                        let ty = ty.as_ty().unwrap();

                        if ty.gen_args.len() != args.len() {
                            ctx.diags.add_error(format!("Incorrect number of generic arguments"));
                        }

                        dbg!(ty_id);

                        Typed::new(
                            Type::unit(),
                            BoundExpr::Type(SymbolRef::Generic(ty_id, args))
                        )
                    },
                    
                    _ => {
                        if lhs.not_error() {
                            ctx.diags
                                .add_error(format!("Left hand side of ::[...] must be a type or function"));
                        }
    
                        Typed::error()
                    }
                }
            }
            
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
                                match &l.get(&ctx.defs).kind {
                                    DefinitionKind::Function(..) => BoundExpr::Fn(SymbolRef::Basic(l)),
                                    DefinitionKind::Type(..) => BoundExpr::Type(SymbolRef::Basic(l)),
                                    _ => BoundExpr::Symbol(l)
                                },
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
                            .add_error(format!("Left hand side of :: must be a symbol"));
                    }

                    Typed::error()
                }
            }

            Expr::Dot { op, id } => {
                let lhs = op.bind(ctx);
                let lhs = lhs.auto_deref(ValueKind::R);


                let ty = lhs.get_type();

                match ty.dot(id, &ctx.defs) {
                    Ok((i, ty)) => Typed::new(
                        ty, 
                        BoundExpr::Dot(Box::new(lhs), i)
                    ),

                    Err(DotError::NoFieldFound) => {
                        ctx.diags.add_error(format!(
                            "No member {} in type {}",
                            id.as_str(),
                            ty.full_name(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    }

                    Err(DotError::NotAStructure) => {
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
                    Type::Literal(LitType::Ptr(ty)) | Type::Literal(LitType::Ref(ty, ..)) => {
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
            
            Expr::Ref(op) => {
                let op = op.kind().bind_as_lvalue(ctx);
                let deref = match op.get_type().deref() {
                    Some(x) => x,
                    None => {
                        if op.get_type().not_error() {
                            ctx.diags.add_error(format!(
                                "Could not take a reference to type {}", 
                                op.get_type().full_name(&ctx.defs, &ctx.scopes.arena)
                            ));
                        }

                        return Typed::error();
                    }
                };

                Typed::new(
                    Type::Literal(LitType::Ref(Box::new(deref.clone()), RefKind::Plain)),
                    op.into_kind()
                )
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

            Expr::Sizeof(ty) => Typed::new(
                UIntType::USize.into(),
                BoundExpr::Sizeof(ty.bind(ctx))
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

                let ty = def.try_get_type().map_or(Type::unit(), Type::clone);

                Typed::new(
                    ty,
                    match &def.kind {
                        DefinitionKind::Function(..) => BoundExpr::Fn(SymbolRef::Basic(def_id)),
                        DefinitionKind::Type(..) => BoundExpr::Type(SymbolRef::Basic(def_id)),
                        _ => BoundExpr::Symbol(def_id)
                    }
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
                                ty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                ty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                    lty.full_name(&ctx.defs, &ctx.scopes.arena)
                                ));
                                Typed::error()
                            } else if !matches!(
                                rty.as_lit(),
                                Some(LitType::Int(..) | LitType::UInt(..))
                            ) {
                                ctx.diags.add_error(format!(
                                    "Cannot perform operation {arith} on pointer type {} with type {}", 
                                    lty.full_name(&ctx.defs, &ctx.scopes.arena),
                                    rty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                lty.full_name(&ctx.defs, &ctx.scopes.arena),
                                rty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                lty.full_name(&ctx.defs, &ctx.scopes.arena),
                                rty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                lty.full_name(&ctx.defs, &ctx.scopes.arena),
                                rty.full_name(&ctx.defs, &ctx.scopes.arena)
                            ));
                            Typed::error()
                        } else if *lty != LitType::Bool.into() {
                            ctx.diags.add_error(format!(
                                "Cannot perform logic on non-boolean type {}",
                                lty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                                lty.full_name(&ctx.defs, &ctx.scopes.arena),
                                rty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                    BoundExpr::Type(ty) => {
                        let ty = ty.get_type(&ctx.defs, &ctx.scopes.arena);
                        return Typed::new(
                            ty.clone(),
                            BoundExpr::Zeroinit(ty)
                        );
                    }

                    _ => {}
                }

                // Normal flow
                let fty = func.get_type();

                if let Some(fty) = fty.as_fn() {
                    if args.len() != fty.args.len() && !fty.vararg || (args.len() < fty.args.len()) {
                        ctx.diags.add_error(format!(
                            "Incorrect number of arguments for call to function of type {}",
                            fty.full_name(&ctx.defs, &ctx.scopes.arena)
                        ));
                        Typed::error()
                    } else {
                        for (arg, argty) in args.iter().zip_or_none(fty.args.iter()) {
                            let argty = if let Some(argty) = argty {argty} else {continue};
                            if !arg.get_type().compatible(&argty) {
                                ctx.diags.add_error_at(
                                    arg.span.clone(),
                                    format!(
                                        "Expected argument of type {}, got {}",
                                        argty.full_name(&ctx.defs, &ctx.scopes.arena),
                                        arg.get_type().full_name(&ctx.defs, &ctx.scopes.arena)
                                    ),
                                );
                                return Typed::error();
                            }
                        }

                        Typed::new(
                            (*fty.ret).clone(),
                            BoundExpr::DirectCall(Box::new(func), args)
                        )
                    }
                } else {
                    if fty.not_error() {
                        ctx.diags.add_error(format!(
                            "Cannot call non-function value of type {}",
                            fty.full_name(&ctx.defs, &ctx.scopes.arena)
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
                        optyp.full_name(&ctx.defs, &ctx.scopes.arena),
                        typ.full_name(&ctx.defs, &ctx.scopes.arena)
                    ));
                    Typed::error()
                } else {
                    println!("CONVERSION {:?} -> {:?}", optyp, typ);
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
            let mut parser = parsing::parser::Parser::new(lexer);

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


    pub fn bind(&self, ctx: &mut ProgramContext) -> BoundProgram {

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
