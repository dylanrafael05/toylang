use std::{
    collections::HashMap,
    fmt::{Display, Formatter},
};

use toylang_derive::Symbol;

use crate::{
    binding::{Definition, Definitions, ProgramContext, Scope},
    core::{arena::{ArenaID, Arena}, lazy::OnceBuildable, Ident},
    llvm::CodegenContext,
    parsing::ast::BoundSymbol,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntType {
    I8,
    I32,
    I64,
    ISize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UIntType {
    U8,
    U32,
    U64,
    USize,
}

impl Into<LitType> for IntType {
    fn into(self) -> LitType {
        LitType::Int(self)
    }
}
impl Into<Type> for IntType {
    fn into(self) -> Type {
        LitType::Int(self).into()
    }
}

impl IntType {
    pub fn bit_size(&self, ctx: &CodegenContext) -> u32 {
        match self {
            IntType::I8 => 8,
            IntType::I32 => 32,
            IntType::I64 => 64,
            IntType::ISize => ctx.target.get_target_data().get_pointer_byte_size(None) * 8,
        }
    }
}

impl Into<LitType> for UIntType {
    fn into(self) -> LitType {
        LitType::UInt(self)
    }
}
impl Into<Type> for UIntType {
    fn into(self) -> Type {
        LitType::UInt(self).into()
    }
}

impl UIntType {
    pub fn bit_size(&self, ctx: &CodegenContext) -> u32 {
        match self {
            UIntType::U8 => 8,
            UIntType::U32 => 32,
            UIntType::U64 => 64,
            UIntType::USize => ctx.target.get_target_data().get_pointer_byte_size(None) * 8,
        }
    }
}

impl Display for IntType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::I8 => "i8",
                Self::I32 => "i32",
                Self::I64 => "i64",
                Self::ISize => "isize",
            }
        )
    }
}

impl Display for UIntType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::U8 => "u8",
                Self::U32 => "u32",
                Self::U64 => "u64",
                Self::USize => "usize",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatType {
    F32,
    F64,
}

impl Into<LitType> for FloatType {
    fn into(self) -> LitType {
        LitType::Float(self)
    }
}
impl Into<Type> for FloatType {
    fn into(self) -> Type {
        LitType::Float(self).into()
    }
}

impl Display for FloatType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::F32 => "f32",
                Self::F64 => "f64",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefKind {
    Plain,
    Mut,
    Move,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LitType {
    None,

    Unit,

    Bool,

    Int(IntType),
    UInt(UIntType),
    Float(FloatType),

    Char,
    Str,

    Ptr(Box<Type>),
    Ref(Box<Type>, RefKind),

    Struct(HashMap<Ident, (usize, Type)>),

    Function(FunctionType),
}

impl LitType {
    pub fn to_string(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        match self {
            Self::Bool => "bool".to_owned(),
            Self::Char => "char".to_owned(),
            Self::Str => "str".to_owned(),

            Self::None => "!".to_owned(),
            Self::Unit => "unit".to_owned(),

            Self::Int(int) => format!("{int}"),
            Self::UInt(int) => format!("{int}"),
            Self::Float(float) => format!("{float}"),

            Self::Ptr(t) => format!("*{}", t.to_string(defs, scopes)),

            Self::Ref(t, RefKind::Plain) => format!("&{}", t.to_string(defs, scopes)),
            Self::Ref(t, RefKind::Mut) => format!("&mut {}", t.to_string(defs, scopes)),
            Self::Ref(t, RefKind::Move) => format!("&move {}", t.to_string(defs, scopes)),

            Self::Function(x) => x.to_string(defs, scopes),

            Self::Struct(x) => {
                format!(
                    "{{ {} }}",
                    x.iter()
                        .map(|(name, ty)| format!("{}: {}", name.as_str(), ty.1.to_string(defs, scopes)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
        }
    }
}

impl Into<Type> for LitType {
    fn into(self) -> Type {
        Type::Literal(self)
    }
}

#[derive(Debug, Clone, Symbol)]
pub struct TypeDefinition {
    pub typ: OnceBuildable<LitType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Symbol)]
pub enum Type {
    Error,
    Ref(ArenaID<Definition>),
    Literal(LitType),
}

impl Type {
    pub fn compatible(&self, other: &Type) -> bool {
        if *self == Type::Error || *other == Type::Error {
            true
        } else {
            self == other
        }
    }
    pub fn ptr_compatible(&self, other: &Type) -> bool {
        if *self == Type::Error || *other == Type::Error {
            true
        } else {
            self.ptr_val().unwrap() == other
        }
    }

    pub fn unit_ref<'a>() -> &'a Type {
        &Self::Literal(LitType::Unit)
    }
    pub fn unit() -> Type {
        Self::Literal(LitType::Unit)
    }

    pub fn concrete(self, ctx: &mut ProgramContext) -> Type {
        if let Some(f) = self.as_fn() {
            ctx.diags.add_error(format!(
                "Cannot have a variable of type {}",
                f.to_string(&ctx.defs, &ctx.scopes.arena)
            ));
            Type::error()
        } else {
            self
        }
    }

    pub fn as_fn(&self) -> Option<&FunctionType> {
        match self {
            Type::Literal(LitType::Function(f)) => Some(f),
            _ => None,
        }
    }

    pub fn as_lit(&self) -> Option<&LitType> {
        match self {
            Type::Literal(x) => Some(x),
            _ => None,
        }
    }
    pub fn ptr_val(&self) -> Option<&Type> {
        match self {
            Type::Literal(LitType::Ptr(x)) => Some(x.as_ref()),
            _ => None
        }
    }

    pub fn must_lit(&self) -> &LitType {
        self.as_lit().unwrap()
    }
    pub fn into_lit(self) -> Option<LitType> {
        match self {
            Type::Literal(x) => Some(x),
            _ => None,
        }
    }

    pub fn to_string(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        match self {
            Type::Error => "?".to_owned(),
            Type::Literal(x) => x.to_string(defs, scopes),
            Type::Ref(x) => x.get(defs).full_name(scopes).to_string(),
        }
    }
}

/*
impl Display for Type
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self
        {
            Self::Error => f.write_str("?"),
            Self::Ref(x) => f.write_str(x.as_str()),
            Self::Literal(x) => writex!(f, x)
        }
    }
}
*/

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionType {
    pub ret: Box<Type>,
    pub args: Vec<Box<Type>>,
}

impl FunctionType {
    pub fn to_string(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        format!(
            "({}) -> {}",
            self.args
                .iter()
                .map(|a| a.to_string(defs, scopes))
                .collect::<Vec<_>>()
                .join(", "),
            self.ret.to_string(defs, scopes)
        )
    }
}

impl Into<LitType> for FunctionType {
    fn into(self) -> LitType {
        LitType::Function(self)
    }
}

impl Into<Type> for FunctionType {
    fn into(self) -> Type {
        LitType::Function(self).into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Conversion {
    Ptr(Type, Type),
    IntToInt(IntType, IntType),
    FloatToFloat(FloatType, FloatType),
    IntToFloat(IntType, FloatType),
    FloatToInt(FloatType, IntType),
}
