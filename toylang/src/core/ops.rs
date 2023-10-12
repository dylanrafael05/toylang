use std::fmt::Display;

use crate::{binding::{ProgramContext, BoundExpr, Type, IntType, FloatType}, parsing::ast::Typed, writex};

pub type ConstInt = i128;
pub type ConstFloat = f64;

#[derive(Debug, Clone, PartialEq)]
pub enum Constant 
{
    Int(ConstInt),
    Float(ConstFloat),
    Bool(bool)
}

impl Display for Constant
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Constant::*;
        match self {
            Int(x) => writex!(f, x),
            Float(x) => writex!(f, x),
            Bool(x) => writex!(f, x)
        }
    }
}

impl Into<Typed<BoundExpr>> for Constant 
{
    fn into(self) -> Typed<BoundExpr> {
        use Constant::*;
        match self 
        {
            Int(x) => Typed::new(Type::Int(IntType::I32), BoundExpr::Integer(x)),
            Float(x) => Typed::new(Type::Float(FloatType::F32), BoundExpr::Decimal(x)),
            Bool(x) => Typed::new(Type::Bool, BoundExpr::Bool(x))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnOp
{
    Negate,
    Not
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp
{
    Arith(ArithOp),

    Equals,
    NotEquals,

    Comparison(CompOp),

    And,
    Or
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithOp
{
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl Display for ArithOp
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", match self 
        {
            ArithOp::Add => "+",
            ArithOp::Subtract => "-",
            ArithOp::Multiply => "*",
            ArithOp::Divide => "/"
        })
    }
}

#[macro_export]
macro_rules! checked {
    ($l: expr => $r: expr, +) => {
        $l.checked_add($r)
    };
    ($l: expr => $r: expr, -) => {
        $l.checked_sub($r)
    };
    ($l: expr => $r: expr, *) => {
        $l.checked_mul($r)
    };
    ($l: expr => $r: expr, /) => {
        $l.checked_div($r)
    };
    ($l: expr => $r: expr, %) => {
        $l.checked_rem($r)
    };
}

macro_rules! implement_op {
    ($ty: literal, $ctx: ident, $lhs: ident $op: tt $rhs: ident) => {{
        let out = checked!($lhs => $rhs, $op);
        if out.is_none()
        {
            $ctx.diags.add_error(format!("Invalid {} constant!", $ty));
        }
        out
    }};
}

macro_rules! implement_int {
    ($ctx: ident, $lhs: ident $op: tt $rhs: ident) => {
       implement_op!("integer", $ctx, $lhs $op $rhs) 
    };
}

impl ArithOp
{
    pub fn constant(&self, ctx: &mut ProgramContext, lhs: Constant, rhs: Constant) -> Option<Constant>
    {
        use Constant::*;

        match (lhs, rhs) 
        {
            (Int(lhs),  Int(rhs))  => self.const_int(ctx, lhs, rhs).map(Int),
            (Float(lhs), Float(rhs)) => self.const_float(ctx, lhs, rhs).map(Float),

            (Int(lhs), Float(rhs)) => self.const_float(ctx, lhs as ConstFloat, rhs).map(Float),
            (Float(lhs), Int(rhs)) => self.const_float(ctx, lhs, rhs as ConstFloat).map(Float),

            (l, r) => {
                ctx.diags.add_error(format!("Invalid operation {l} {self} {r}"));
                None
            }
        }
    }

    pub fn const_int(&self, ctx: &mut ProgramContext, lhs: ConstInt, rhs: ConstInt) -> Option<ConstInt>
    {
        match self
        {
            ArithOp::Add      => implement_int!(ctx, lhs + rhs),
            ArithOp::Subtract => implement_int!(ctx, lhs - rhs),
            ArithOp::Multiply => implement_int!(ctx, lhs * rhs),
            ArithOp::Divide   => implement_int!(ctx, lhs / rhs)
        }
    }

    pub fn const_float(&self, _ctx: &mut ProgramContext, lhs: ConstFloat, rhs: ConstFloat) -> Option<ConstFloat>
    {
        match self
        {
            ArithOp::Add      => Some(lhs + rhs),
            ArithOp::Subtract => Some(lhs - rhs),
            ArithOp::Multiply => Some(lhs * rhs),
            ArithOp::Divide   => Some(lhs / rhs)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompOp
{
    GreaterThan,
    GreaterEqual,
    LessThan,
    LessEqual,
}