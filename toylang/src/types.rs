use std::{
    fmt::{Display, Formatter},
    hash::Hash
};

use hashlink::LinkedHashMap;
use toylang_derive::Symbol;

use crate::{
    binding::{Definition, Definitions, ProgramContext, Scope, RcList},
    core::{arena::{ArenaID, Arena}, lazy::OnceBuildable, Ident},
    llvm::CodegenContext,
    parsing::ast::BoundSymbol,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntType {
    I8,
    I32,
    I64,
    ISize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefKind {
    Plain,
    Mut,
    Move,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LitType {
    None,
    Placehold,

    Unit,

    Bool,

    Int(IntType),
    UInt(UIntType),
    Float(FloatType),

    Char,
    Str,

    Ptr(Box<Type>),
    Ref(Box<Type>, RefKind),

    Struct(LinkedHashMap<Ident, (usize, Type)>),

    Function(FunctionType),
}

impl LitType {
    pub fn full_name(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        match self {
            Self::Placehold => "<placeholder>".to_owned(),

            Self::Bool => "bool".to_owned(),
            Self::Char => "char".to_owned(),
            Self::Str => "str".to_owned(),

            Self::None => "!".to_owned(),
            Self::Unit => "unit".to_owned(),

            Self::Int(int) => format!("{int}"),
            Self::UInt(int) => format!("{int}"),
            Self::Float(float) => format!("{float}"),

            Self::Ptr(t) => format!("*{}", t.full_name(defs, scopes)),

            Self::Ref(t, RefKind::Plain) => format!("&{}", t.full_name(defs, scopes)),
            Self::Ref(t, RefKind::Mut) => format!("&mut {}", t.full_name(defs, scopes)),
            Self::Ref(t, RefKind::Move) => format!("&move {}", t.full_name(defs, scopes)),

            Self::Function(x) => x.full_name(defs, scopes),

            Self::Struct(x) => {
                format!(
                    "{{ {} }}",
                    x.iter()
                        .map(|(name, ty)| format!("{}: {}", name.as_str(), ty.1.full_name(defs, scopes)))
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

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DotError {
    NotAStructure,
    NoFieldFound
}

#[derive(Debug, Clone, Symbol)]
pub struct TypeDefinition {
    pub inner_type: OnceBuildable<LitType>,
    pub scope: ArenaID<Scope>,
    pub gen_args: RcList<Type>,
    pub typ: Type
}

#[derive(Debug, Clone, PartialEq, Eq, Symbol, Hash)]
pub enum Type {
    Error,
    Ref(ArenaID<Definition>),

    // TODO: how to make this better: needing to go into and out of vec to
    // allow for mutability
    Gen(ArenaID<Definition>, Vec<Type>),
    Literal(LitType)
}

impl Type {
    pub fn compatible(&self, other: &Type) -> bool {
        if *self == Type::Error || *other == Type::Error {
            true
        } else {
            self == other
        }
    }
    pub fn deref_compatible(&self, other: &Type) -> bool {
        if *self == Type::Error || *other == Type::Error {
            true
        } else {
            self.deref().unwrap() == other
        }
    }

    pub fn is_generic(&self, defs: &Arena<Definition>) -> bool {
        match self {
            Type::Ref(x) => !x.get(defs).as_ty().unwrap().gen_args.is_empty(),
            _ => false
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
                f.full_name(&ctx.defs, &ctx.scopes.arena)
            ));
            Type::error()
        } else if self.is_generic(&ctx.defs) {
            ctx.diags.add_error(format!(
                "Cannot have a variable of generic type {}",
                self.full_name(&ctx.defs, &ctx.scopes.arena)
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
    pub fn deref(&self) -> Option<&Type> {
        match self {
            Type::Literal(LitType::Ptr(x)) => Some(x.as_ref()),
            Type::Literal(LitType::Ref(x, ..)) => Some(x.as_ref()),
            Type::Error => Some(&Type::Error),
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

    pub fn full_name(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        match self {
            Type::Error => "?".to_owned(),
            Type::Literal(x) => x.full_name(defs, scopes),
            Type::Ref(x) => x.get(defs).full_name(scopes).to_string(),
            Type::Gen(x, args) => x.get(defs).full_name(scopes).to_string() + "[" + &args.iter().map(|x| x.full_name(defs, scopes)).collect::<Vec<_>>().join(", ") + "]"
        }
    }

    pub fn dot(&self, ident: &Ident, defs: &Arena<Definition>) -> Result<(usize, Type), DotError> {
        use LitType::*;

        match self {
            Type::Literal(Struct(x)) => {
                x.get(ident).ok_or(DotError::NoFieldFound).cloned()
            }

            Type::Gen(x, args) => {
                let ty = x.get(defs);
                let ty = ty.as_ty().unwrap();

                let gens = &ty.gen_args;

                // TODO: any way to remove this call to 'clone'?
                let ty = Type::Literal(ty.inner_type.get().unwrap().clone());
                let mut ty = ty.dot(ident, defs)?;

                ty.1.replace_all(gens, args);

                Ok(ty)
            }

            Type::Ref(x) => {
                let ty = x.get(defs);
                let ty = ty.as_ty().unwrap();
                let ty = ty.inner_type.get().unwrap();
                
                // TODO: any way to remove this call to 'clone'?
                let ty = Type::Literal(ty.clone());

                ty.dot(ident, defs)
            }

            _ => Err(DotError::NotAStructure)
        }
    }
    
    pub fn has_inners(&self) -> bool {
        use LitType::*;

        match self {
            Type::Literal(x) if matches!(x, Ptr(..) | Ref(..) | Struct(..) | Function(..)) => true,
            Type::Gen(..) => true,
            _ => false
        }
    }

    /// Get all instances of the provided type in this type as mutable references
    pub fn get_references(&mut self, typ: &Type) -> Vec<&mut Type> {
        use LitType::*;

        if self.has_inners() {
            match self {
                Type::Literal(lt) => match lt {
                    Ptr(x) => x.get_references(typ),
                    Ref(x, ..) => x.get_references(typ),
                    Struct(x) => x.values_mut().flat_map(|(_, x)| x.get_references(typ)).collect(),
                    Function(FunctionType {ret, args, ..}) => args.iter_mut().flat_map(|x| x.get_references(typ)).chain(ret.get_references(typ).into_iter()).collect(),

                    _ => unreachable!()
                }

                Type::Gen(.., y) => y.iter_mut().flat_map(|x| x.get_references(typ)).collect(),

                _ => unreachable!()
            }
        }
        else if self == typ { vec![self] } 
        else { vec![] }
    }

    /// Replace all instances of the given type in this type
    pub fn replace(&mut self, typ: &Type, new: &Type) {
        for ty in self.get_references(typ) {
            *ty = new.clone();
        }
    }
    
    /// Replace all instances of the given type in this type
    pub fn replace_all(&mut self, typ: &[Type], new: &[Type]) {
        for (old, new) in typ.iter().zip(new) {
            self.replace(old, new);
        }
    }
    
    /// Replace all instances of the given type in this type
    pub fn replaced_all(&self, typ: &[Type], new: &[Type]) -> Type {
        let mut out = self.clone();
        out.replace_all(typ, new);
        out
    }

    // TODO: swap argument order so that this is called on the type to be inferred
    // over.infer(self, template)
    // -- AKA --
    // <T>.infer(<**Vec[i32]>, <**Vec[T]>)
    // -- OR --
    // self.infer(from, based_on)

    /// Implements type inference by finding the first instance of 'over' 
    /// within the provided 'template' type and returning the corresponding
    /// value within the structure of this type, if it exists. 
    /// 
    /// ```Ok(None)``` indicates that while no inference was found, there was 
    /// no error, while ```Err(())``` indicates that no inference could possibly
    /// be found since the structures do not match
    pub fn infer(&self, over: &Type, template: &Type) -> Result<Option<&Type>, ()> {
        // Match over this and template
        match (self, template) {
            // If we have found over, return this
            (_, tmp) if tmp == over => Ok(Some(self)),

            // If this is an error, infer as error
            (Self::Error, _) => Ok(Some(&Self::Error)),

            // If both types are literal, check them further
            (Self::Literal(ty), Self::Literal(oty)) => match (ty, oty) {

                // If both types are pointers or references, check the inner values
                (LitType::Ptr(ty), LitType::Ptr(oty)) 
                    => ty.infer(over, oty),
                (LitType::Ref(ty, rtx), LitType::Ref(oty, rty)) if rtx == rty 
                    => ty.infer(over, oty),
                
                // If both types are structs, return the first valid inference for the fields,
                // erroring out early if a difference is found
                (LitType::Struct(tys), LitType::Struct(otys)) => {
                    for ((_, (_, ty)), (_, (_, oty))) in tys.iter().zip(otys.iter()) {
                        match ty.infer(over, oty)? {
                            Some(x) => return Ok(Some(x)),
                            None => continue
                        }
                    }

                    Ok(None)
                }
                // If both types are functions, return the first valid inference for the arguments,
                // erroring out early if a difference is found, falling back to the inference for the
                // return type
                (LitType::Function(FunctionType { ret, args, .. }), LitType::Function(FunctionType { ret: oret, args: oargs, .. })) => {
                    for (ty, oty) in args.iter().zip(oargs.iter()) {
                        match ty.infer(over, oty)? {
                            Some(x) => return Ok(Some(x)),
                            None => continue
                        }
                    }

                    ret.infer(over, oret)
                }

                // If both types are otherwise identical, report valid
                (x, y) if (x == y) => Ok(None),

                // Otherwise, error out
                _ => Err(())
            },

            // Otherwise, error out
            _ => Err(())
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionType {
    pub ret: Box<Type>,
    pub args: Vec<Box<Type>>,
    pub vararg: bool
}

impl FunctionType {
    pub fn full_name(&self, defs: &Definitions, scopes: &Arena<Scope>) -> String {
        format!(
            "({}) -> {}",
            self.args
                .iter()
                .map(|a| a.full_name(defs, scopes))
                .collect::<Vec<_>>()
                .join(", ") + if self.vararg {"..."} else {""},
            self.ret.full_name(defs, scopes)
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
