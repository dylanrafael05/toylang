use std::collections::HashMap;

use inkwell::{context::Context, targets::TargetMachine, types::{AnyTypeEnum, BasicType, BasicTypeEnum}, values::{FunctionValue, BasicValueEnum, BasicValue, AnyValueEnum, AnyValue}, builder::Builder, module::Module};

use crate::{binding::{BoundProgram, ProgramContext, BoundExpr, BoundExprAST, Type, IntType, BoundStmtAST, BoundStmt, BoundBlock, Definition}, parsing::ast::{OwnedSymbol, TypedSymbol, Typed}, core::{AsU64, ops::ArithOp, arena::{ArenaID, ArenaRef}}, formatstr};

pub struct CodegenContext<'a>
{
    pub ctx: &'a Context,
    pub module: Module<'a>,
    pub builder: Builder<'a>,
    pub target: TargetMachine,
    pub cur_func: Option<FunctionValue<'a>>,
    pub definition_values: HashMap<ArenaID<Definition>, AnyValueEnum<'a>>
}

#[derive(Debug, Clone)]
pub enum TLValue<'a>
{
    Unit,
    Value(BasicValueEnum<'a>)
}

impl<'a> TLValue<'a>
{
    pub fn as_inner(&self) -> Option<BasicValueEnum<'a>>
    {
        use TLValue::*;
        match self
        {
            Unit => None,
            Value(x) => Some(x.clone())
        }
    }
}

impl<'a, T : BasicValue<'a>> From<T> for TLValue<'a>
{
    fn from(value: T) -> Self 
    {
        Self::Value(value.as_basic_value_enum())
    }
}

pub struct BasicTypeHelper;

impl BasicTypeHelper
{
    pub fn from<'a>(any: AnyTypeEnum<'a>) -> BasicTypeEnum<'a>
    {
        use AnyTypeEnum::*;
        match any 
        {
            ArrayType(x) => BasicTypeEnum::ArrayType(x),
            FloatType(x) => BasicTypeEnum::FloatType(x),
            IntType(x) => BasicTypeEnum::IntType(x),
            PointerType(x) => BasicTypeEnum::PointerType(x),
            StructType(x) => BasicTypeEnum::StructType(x),
            VectorType(x) => BasicTypeEnum::VectorType(x),

            _ => panic!("Could not convert {any:?} to a basic type.")
        }
    }
}

pub struct BasicValueHelper;

impl BasicValueHelper
{
    pub fn from<'a>(any: AnyValueEnum<'a>) -> BasicValueEnum<'a>
    {
        use AnyValueEnum::*;
        match any 
        {
            ArrayValue(x) => BasicValueEnum::ArrayValue(x),
            FloatValue(x) => BasicValueEnum::FloatValue(x),
            IntValue(x) => BasicValueEnum::IntValue(x),
            PointerValue(x) => BasicValueEnum::PointerValue(x),
            StructValue(x) => BasicValueEnum::StructValue(x),
            VectorValue(x) => BasicValueEnum::VectorValue(x),

            _ => panic!("Could not convert {any:?} to a basic type.")
        }
    }
}

impl<'a> CodegenContext<'a>
{
    pub fn new(ctx: &'a Context, target: TargetMachine) -> Self
    {
        // let unit = module.struct_of(vec![], false);
        // let def = NamedStructDef::Defined(unit);

        // module.types.add_named_struct_def("unit".into(), def);

        let module = ctx.create_module("output");
        let builder = ctx.create_builder();

        Self {
            ctx,
            module,
            builder,
            target,

            cur_func: None,
            definition_values: HashMap::new()
        }
    }

    pub fn build(&mut self, _prog: BoundProgram, mut ctx: ProgramContext)
    {
        for (name, def) in ctx.scopes.cur().definitions()
        {
            let id = *def;
            let def = def.get(&ctx.defs);
            let def = def.as_fn().unwrap();
    
            let ty = self.build_type(def.get_type());
            let func = self.module.add_function(name.as_str(), ty.into_function_type(), None);

            self.definition_values.insert(id, func.into());
        }

        for (name, def) in ctx.scopes.cur_mut().definitions().clone()
        {
            let def = def.take(&mut ctx.defs);
            let def = def.into_value().into_fn().unwrap();
            
            let cur_func = self.module.get_function(name.as_str()).unwrap();
            self.cur_func = Some(cur_func);
            let entry = self.ctx.append_basic_block(cur_func, "entry");

            self.builder.position_at_end(entry);

            for (i, arg) in 
                def.argdefs.iter().enumerate()
            {
                self.definition_values.insert(
                    *arg, 
                    cur_func.get_nth_param(i as u32)
                        .unwrap()
                        .as_any_value_enum()
                );
            }

            for (_, def) in 
                def.scope.get(&ctx.scopes.arena).definitions()
            {
                println!("DEF = {:?}", def);

                let defs = def.get(&ctx.defs);

                if !matches!(ArenaRef::as_ref(&defs), Definition::Variable {..})
                {
                    continue;
                }

                let ty = self.build_type(defs.get_type());
                self.definition_values.insert(
                    *def,
                    self.builder.build_alloca(
                        BasicTypeHelper::from(ty), defs.name().as_str()
                    ).unwrap().as_any_value_enum()
                );
            }

            let eval = self.build_expr(def.body, &mut ctx).as_inner();
            
            self.builder.build_return(eval.as_ref().map(|x| x as &dyn BasicValue)).unwrap();
        }
    }

    pub fn build_type(&mut self, typ: &Type) -> AnyTypeEnum<'a>
    {
        match typ {
            Type::Bool => self.ctx.bool_type().into(),
            Type::Unit => self.ctx.void_type().into(),

            Type::Int(it) => match it {
                IntType::I8    => self.ctx.i8_type(),
                IntType::I32   => self.ctx.i32_type(),
                IntType::I64   => self.ctx.i64_type(),
                IntType::ISize => self.ctx.ptr_sized_int_type(
                    &self.target.get_target_data(), 
                    None
                )
            }.into(),

            Type::Function(fty) => 
            {
                BasicTypeEnum::try_from(self.build_type(fty.ret.as_ref())).unwrap().fn_type(
                    fty.args.iter().map(|x| 
                        BasicTypeEnum::try_from(self.build_type(x.as_ref())).unwrap().into()
                    ).collect::<Vec<_>>().as_slice(), 
                    false
                ).into()
            },

            _ => todo!()
        }
    }

    pub fn build_stmt(&mut self, stmt: BoundStmtAST, ctx: &mut ProgramContext)
    {
        let span = stmt.span.clone();
        let typ = stmt.get_type().clone();

        match stmt.into_kind()
        {
            BoundStmt::Expr(e) => {
                self.build_expr(BoundExprAST {
                    span,
                    content: Typed::<BoundExpr>::new(typ, e)
                }, ctx);
            },

            BoundStmt::Assign { first, target, value } => {
                println!("{:?}", self.definition_values);

                if first
                {
                    let target_v = target.get(&ctx.defs);
                    let ty = self.build_type(target_v.get_type());
                    self.definition_values.insert(
                        target,
                        self.builder.build_alloca(
                            BasicTypeHelper::from(ty), target_v.name().as_str()
                        ).unwrap().as_any_value_enum()
                    );
                }

                let dval = self.definition_values[&target].into_pointer_value();
                let vval = self.build_expr(*value, ctx);

                self.builder.build_store(
                    dval,
                    vval.as_inner().unwrap()
                ).expect("building store should work");
            }

            x => todo!("{x:?}")
        }
    }

    fn build_block(&mut self, block: BoundBlock, ctx: &mut ProgramContext) -> TLValue<'a>
    {
        for stmt in block.body 
        {
            self.build_stmt(stmt, ctx);
        }

        if let Some(expr) = block.tail
        {
            self.build_expr(*expr, ctx)
        }
        else
        {
            TLValue::Unit
        }
    }

    fn build_expr(&mut self, expr: BoundExprAST, ctx: &mut ProgramContext) -> TLValue<'a>
    {
        let ty = expr.get_type().clone();
        let span = expr.span.clone();

        match expr.into_kind()
        {
            BoundExpr::Bool(x) => self.ctx.bool_type().const_int(x.as_u64(), false).into(),
            BoundExpr::Integer(x) => self.build_type(&ty).into_int_type().const_int(x as u64, false).into(),
            BoundExpr::Decimal(x) => self.build_type(&ty).into_float_type().const_float(x).into(),

            BoundExpr::Block(x) => self.build_block(x, ctx),

            BoundExpr::Identifier(x) => match ArenaRef::as_ref(&x.get(&ctx.defs))
            {
                Definition::Parameter { .. } 
                    => BasicValueHelper::from(self.definition_values[&x]).into(),
                Definition::Variable { name: _, typ} => 
                {
                    let typ = self.build_type(typ);

                    self.builder.build_load(
                        BasicTypeHelper::from(typ),
                        self.definition_values[&x].into_pointer_value(),
                        ""
                    ).expect("").into()
                },
                
                _ => todo!("what")
            },

            BoundExpr::Arithmetic(bin, op) => 
            {
                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                match ty
                {
                    Type::Int(..) =>
                    {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();

                        match op
                        {
                            ArithOp::Add 
                                => self.builder.build_int_add(lhs, rhs, ""),
                            ArithOp::Subtract 
                                => self.builder.build_int_sub(lhs, rhs, ""),
                            ArithOp::Multiply 
                                => self.builder.build_int_mul(lhs, rhs, ""),
                            ArithOp::Divide
                                => self.builder.build_int_signed_div(lhs, rhs, "")
                        }
                    }.unwrap().into(),

                    Type::UInt(..) =>
                    {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();
                        
                        match op
                        {
                            ArithOp::Add 
                                => self.builder.build_int_add(lhs, rhs, ""),
                            ArithOp::Subtract 
                                => self.builder.build_int_sub(lhs, rhs, ""),
                            ArithOp::Multiply 
                                => self.builder.build_int_mul(lhs, rhs, ""),
                            ArithOp::Divide
                                => self.builder.build_int_unsigned_div(lhs, rhs, "")
                        }
                    }.unwrap().into(),

                    Type::Float(..) =>
                    {
                        let lhs = lhs.into_float_value();
                        let rhs = rhs.into_float_value();
                        
                        match op
                        {
                            ArithOp::Add 
                                => self.builder.build_float_add(lhs, rhs, ""),
                            ArithOp::Subtract 
                                => self.builder.build_float_sub(lhs, rhs, ""),
                            ArithOp::Multiply 
                                => self.builder.build_float_mul(lhs, rhs, ""),
                            ArithOp::Divide
                                => self.builder.build_float_div(lhs, rhs, "")
                        }
                    }.unwrap().into(),

                    x => todo!("Unimplemented arithmetic type {x}")
                }
            }

            BoundExpr::If(bound_if) => {
                let comp = self.build_expr(*bound_if.cond, ctx);

                let on_true = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), 
                    formatstr!("on_true_{}", span.start)
                );

                let end_if = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), 
                    formatstr!("endif_{}", span.start)
                );
                
                let on_false = if bound_if.else_block.is_some() 
                {
                    end_if.clone()
                }
                else 
                {
                    self.ctx.append_basic_block(
                        self.cur_func.unwrap(), 
                        formatstr!("on_false_{}", span.start)
                    )
                };

                self.builder.build_conditional_branch(
                    comp.as_inner().unwrap().into_int_value(), 
                    on_true, 
                    on_false
                ).unwrap();

                self.builder.position_at_end(on_true);
                let block_value = self.build_block(bound_if.block, ctx);
                self.builder.build_unconditional_branch(end_if).unwrap();

                let else_value = if let Some(else_block) = bound_if.else_block
                {
                    self.builder.position_at_end(on_false);
                    let val = self.build_expr(*else_block, ctx);
                    self.builder.build_unconditional_branch(end_if).unwrap();

                    val
                }
                else 
                {
                    TLValue::Unit
                };

                self.builder.position_at_end(end_if);

                if ty != Type::Unit
                {
                    let ty = self.build_type(&ty);
                    let phi = self.builder.build_phi(
                        BasicTypeHelper::from(ty),
                        formatstr!("phi_{}", span.start)
                    ).expect("phi construction should not fail");

                    phi.add_incoming(&[
                        (&block_value.as_inner().unwrap(), on_true),
                        (&else_value.as_inner().unwrap(),  on_false)
                    ]);

                    phi.as_basic_value().into()
                }
                else 
                {   
                    TLValue::Unit
                }
            }

            _ => todo!()
        }
    }
}