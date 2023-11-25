use std::{collections::HashMap, borrow::Cow};

use inkwell::{
    builder::Builder,
    context::Context,
    module::Module,
    targets::TargetMachine,
    types::{
        AnyType, AnyTypeEnum, BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FunctionType,
        IntType as InkwellIntType,
    },
    values::{
        AnyValue, AnyValueEnum, BasicMetadataValueEnum, BasicValue, BasicValueEnum, FunctionValue,
        IntValue,
    },
    AddressSpace, FloatPredicate, IntPredicate, basic_block::BasicBlock,
};

use crate::{
    binding::{
        BoundBlock, BoundExpr, BoundExprAST, BoundProgram, BoundStmt, BoundStmtAST, Definition,
        DefinitionKind, Definitions, ProgramContext, Scope, SymbolRef, RcList,
    },
    core::{
        arena::{ArenaID, ArenaRef, Arena},
        ops::ArithOp,
        AsU64, collections::{stack::Stack, OneSidedCollection}, Ident,
    },
    parsing::ast::{OwnedSymbol, Typed, TypedSymbol, Symbol},
    types::{IntType, LitType, Type, UIntType},
};

pub struct CodegenTypes<'a> {
    pub ctx: &'a Context,
    pub target: &'a TargetMachine,
    pub module: Module<'a>,
    pub type_values: HashMap<ArenaID<Definition>, AnyTypeEnum<'a>>,
    pub gen_ty_impls: HashMap<ArenaID<Definition>, HashMap<RcList<Type>, AnyTypeEnum<'a>>>,
    pub gen_fn_impls: HashMap<ArenaID<Definition>, HashMap<RcList<Type>, AnyTypeEnum<'a>>>,
    pub cur_gen_vals: (RcList<Type>, RcList<Type>)
}

pub struct CodegenLitTypes<'a> {
    pub ctx: &'a Context,
    pub target: &'a TargetMachine,
    pub module: Module<'a>,
    pub type_values: HashMap<ArenaID<Definition>, AnyTypeEnum<'a>>,
}

#[derive(Debug)]
pub struct Loop<'a> {
    pub cond: BasicBlock<'a>,
    pub end:  BasicBlock<'a>
}

#[derive(Debug)]
pub struct FnVal<'a> {
    pub func: FunctionValue<'a>,
    pub sym:  SymbolRef
}

pub struct CodegenContext<'a> {
    pub ctx: &'a Context,
    pub module: Module<'a>,
    pub builder: Builder<'a>,
    pub target: &'a TargetMachine,

    pub cur_func: Option<FunctionValue<'a>>,
    pub cur_loop: Option<Loop<'a>>,

    pub definition_values: HashMap<SymbolRef, AnyValueEnum<'a>>,
    pub unfinished_fns: Stack<FnVal<'a>>,

    pub types: CodegenTypes<'a>,
}

#[derive(Debug, Clone)]
pub enum TLValue<'a> {
    Unit,
    Value(BasicValueEnum<'a>),
}

impl<'a> TLValue<'a> {
    pub fn as_inner(&self) -> Option<BasicValueEnum<'a>> {
        use TLValue::*;
        match self {
            Unit => None,
            Value(x) => Some(x.clone()),
        }
    }

    pub fn block(&self) -> Option<BasicBlock<'a>> {
        use TLValue::*;
        match self {
            Unit => None,
            Value(x) => x.as_instruction_value().and_then(|x| x.get_parent())
        }
    }
}

impl<'a, T: BasicValue<'a>> From<T> for TLValue<'a> {
    fn from(value: T) -> Self {
        Self::Value(value.as_basic_value_enum())
    }
}

pub struct BasicTypeHelper;

impl BasicTypeHelper {
    pub fn from<'a>(any: AnyTypeEnum<'a>) -> BasicTypeEnum<'a> {
        use AnyTypeEnum::*;
        match any {
            ArrayType(x) => BasicTypeEnum::ArrayType(x),
            FloatType(x) => BasicTypeEnum::FloatType(x),
            IntType(x) => BasicTypeEnum::IntType(x),
            PointerType(x) => BasicTypeEnum::PointerType(x),
            StructType(x) => BasicTypeEnum::StructType(x),
            VectorType(x) => BasicTypeEnum::VectorType(x),

            _ => panic!("Could not convert {any:?} to a basic type."),
        }
    }

    pub fn func<'a>(
        ret: AnyTypeEnum<'a>,
        param_types: &[BasicMetadataTypeEnum<'a>],
        is_var_args: bool,
    ) -> FunctionType<'a> {
        match ret {
            AnyTypeEnum::VoidType(v) => v.fn_type(param_types, is_var_args),

            other => BasicTypeEnum::try_from(other)
                .unwrap()
                .fn_type(param_types, is_var_args),
        }
    }
}

pub struct BasicValueHelper;

impl BasicValueHelper {
    pub fn from<'a>(any: AnyValueEnum<'a>) -> BasicValueEnum<'a> {
        use AnyValueEnum::*;
        match any {
            ArrayValue(x) => BasicValueEnum::ArrayValue(x),
            FloatValue(x) => BasicValueEnum::FloatValue(x),
            IntValue(x) => BasicValueEnum::IntValue(x),
            PointerValue(x) => BasicValueEnum::PointerValue(x),
            StructValue(x) => BasicValueEnum::StructValue(x),
            VectorValue(x) => BasicValueEnum::VectorValue(x),

            _ => panic!("Could not convert {any:?} to a basic type."),
        }
    }
}

impl<'a> CodegenTypes<'a> {
    pub fn new(ctx: &'a Context, target: &'a TargetMachine, module: Module<'a>) -> Self {
        Self {
            ctx,
            target,
            module,
            type_values:  HashMap::new(),
            gen_ty_impls: HashMap::new(),
            gen_fn_impls: HashMap::new(),
            cur_gen_vals: (RcList::from([]), RcList::from([]))
        }
    }

    pub fn transform(&self, typ: &mut Type) {
        typ.replace_all(&self.cur_gen_vals.0, &self.cur_gen_vals.1)
    }

    // TODO: rename to differentiate between transforming in-place and transforming
    // with clone?
    pub fn transform_ref<'b>(&self, sym: &'b SymbolRef) -> Cow<'b, SymbolRef> {
        sym.replace_all(&self.cur_gen_vals.0, &self.cur_gen_vals.1)
    }

    pub fn get(&mut self, typ: &Type, defs: &Definitions, scopes: &Arena<Scope>) -> AnyTypeEnum<'a> {
        
        // dbg!(typ.full_name(defs, scopes));

        // TODO: remove this clone?
        let mut typ = typ.clone();
        self.transform(&mut typ);

        // dbg!(typ.full_name(defs, scopes));
        // dbg!(&self.cur_gen_vals);

        match &typ {
            Type::Gen(b, args) => {
                self.gen_ty_impls
                    .entry(*b)
                    .or_insert(Default::default());

                let args: RcList<_> = args.iter().map(Clone::clone).collect();

                if !self.gen_ty_impls[b].contains_key(&args) {
                    let mono = self.build_monomorphization(*b, &args, defs, scopes);

                    self.gen_ty_impls.get_mut(b).unwrap().insert(args.clone(), mono);
                }

                self.gen_ty_impls[b][&args]
            }

            Type::Literal(x) => self.build(x, defs, scopes),

            Type::Ref(r) => {
                if !self.type_values.contains_key(r) {
                    let def = r.get(defs);
                    let name = def.name();
                    dbg!(&*def);
                    let ty = self.build_named(
                        &def.as_ty()
                            .unwrap()
                            .inner_type
                            .get()
                            .expect("This should have already bene defined"),
                        defs,
                        scopes,
                        name.as_str(),
                    );

                    self.type_values.insert(*r, ty);
                }

                *self.type_values.get(r).unwrap()
            }

            Type::Error => panic!("Code with errors should never be compiled!"),
        }
    }

    pub fn build_monomorphization(
        &mut self,
        typ: ArenaID<Definition>,
        args: &RcList<Type>,
        defs: &Arena<Definition>,
        scopes: &Arena<Scope>
    ) -> AnyTypeEnum<'a> {
        let typdef = typ.get(defs);
        let typdef = typdef.as_ty().unwrap();

        // should Type::Literal be nuked?
        let mut typ = Type::Literal(typdef.inner_type.get().unwrap().clone());

        for (arg, gen) in args.iter().zip(typdef.gen_args.iter()) {
            typ.replace(gen, arg);
        }

        self.get(&typ, defs, scopes)
    }

    pub fn build_named(
        &mut self,
        typ: &LitType,
        defs: &Definitions,
        scopes: &Arena<Scope>,
        name: &str,
    ) -> AnyTypeEnum<'a> {
        match typ {
            LitType::Struct(s) => {
                let ls = self.ctx.opaque_struct_type(name);
                ls.set_body(
                    s.values()
                        .map(|(_, x)| BasicTypeHelper::from(self.build(x.as_lit().unwrap(), defs, scopes)))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    false,
                );
                ls.as_any_type_enum()
            }

            _ => panic!(
                "Cannot build a named type from literal type {}",
                typ.full_name(defs, scopes)
            ),
        }
    }

    pub fn build(&mut self, typ: &LitType, defs: &Definitions, scopes: &Arena<Scope>) -> AnyTypeEnum<'a> {
        match typ {
            LitType::Bool => self.ctx.bool_type().into(),
            LitType::Unit => self.ctx.void_type().into(),

            LitType::Int(it) => match it {
                IntType::I8 => self.ctx.i8_type(),
                IntType::I32 => self.ctx.i32_type(),
                IntType::I64 => self.ctx.i64_type(),
                IntType::ISize => self.ptr_sized_int_type(),
            }
            .into(),
            LitType::UInt(it) => match it {
                UIntType::U8 => self.ctx.i8_type(),
                UIntType::U32 => self.ctx.i32_type(),
                UIntType::U64 => self.ctx.i64_type(),
                UIntType::USize => self.ptr_sized_int_type(),
            }
            .into(),

            LitType::Ptr(..) => self.ctx.i8_type().ptr_type(AddressSpace::from(0)).into(),
            LitType::Ref(..) => self.ctx.i8_type().ptr_type(AddressSpace::from(0)).into(),

            LitType::Function(fty) => BasicTypeHelper::func(
                self.get(fty.ret.as_ref(), &defs, &scopes),
                fty.args
                    .iter()
                    .map(|x| BasicTypeEnum::try_from(self.get(x, &defs, &scopes)).unwrap().into())
                    .collect::<Vec<_>>()
                    .as_slice(),
                fty.vararg,
            )
            .into(),

            LitType::Struct(s) => self
                .ctx
                .struct_type(
                    s.values()
                        .map(|(_, x)| BasicTypeHelper::from(self.build(x.as_lit().unwrap(), defs, scopes)))
                        .collect::<Vec<_>>()
                        .as_slice(),
                    false,
                )
                .as_any_type_enum(),

            x => todo!("{} cannot yet be built", x.full_name(&defs, &scopes)),
        }
    }

    pub fn ptr_sized_int_type(&self) -> InkwellIntType<'a> {
        self.ctx
            .ptr_sized_int_type(&self.target.get_target_data(), None)
    }
}

impl<'a> CodegenContext<'a> {
    pub fn new(ctx: &'a Context, target: &'a TargetMachine) -> Self {
        // let unit = module.struct_of(vec![], false);
        // let def = NamedStructDef::Defined(unit);

        // module.types.add_named_struct_def("unit".into(), def);

        let module = ctx.create_module("output");
        let builder = ctx.create_builder();

        Self {
            ctx,
            builder,
            target,

            cur_func: None,
            cur_loop: None,

            definition_values: HashMap::new(),

            types: CodegenTypes::new(ctx, &target, module.clone()),
            unfinished_fns: Stack::new(),

            module,
        }
    }

    pub fn get_fn(&mut self, ctx: &mut ProgramContext, symbol: &SymbolRef) -> FunctionValue<'a> {
        if self.definition_values.contains_key(symbol) {
            self.definition_values[symbol].into_function_value()
        } else {
            let ty = symbol.get_type(&ctx.defs, &ctx.scopes.arena);
            let name = symbol.compiled_name(&ctx.defs, &ctx.scopes.arena);

            dbg!(symbol);
            dbg!(&name);
            
            let ty = self.types.build(ty.must_lit(), &ctx.defs, &ctx.scopes.arena);

            let fnval = self
                .module
                .add_function(&name, ty.into_function_type(), None);

            dbg!(&fnval);

            if !symbol.id().get(&ctx.defs).as_fn().unwrap().is_declare() {
                self.unfinished_fns.push(FnVal { func: fnval, sym: symbol.clone() });
            }

            self.definition_values.insert(symbol.clone(), fnval.as_any_value_enum());

            fnval
        }
    }

    pub fn build(&mut self, _prog: BoundProgram, mut ctx: ProgramContext) {

        let main = ctx.scopes.lookup(&Ident::new("main"))
            .unwrap();

        let main = SymbolRef::Basic(main);
        self.get_fn(&mut ctx, &main);

        while !self.unfinished_fns.is_empty() {
            let fns = self.unfinished_fns.drain().collect::<Vec<_>>();
            dbg!(&self.unfinished_fns);
            
            for fnval in fns {
                if let Some(args) = fnval.sym.gen_args(&ctx.defs) {
                    self.types.cur_gen_vals = (args.0.clone(), args.1.clone());
                }

                let entry = self.ctx.append_basic_block(fnval.func, "entry");
                self.cur_func = Some(fnval.func);

                self.builder.position_at_end(entry);

                let def = fnval.sym.id();
                let def = def.take(&mut ctx.defs);
                let def = def.into_value().into_fn().unwrap();

                let body_info = def.body_info.unwrap();

                for (i, arg) in body_info.argdefs.iter().enumerate() {
                    self.definition_values.insert(
                        arg.into(),
                        fnval.func
                            .get_nth_param(i as u32)
                            .unwrap()
                            .as_any_value_enum(),
                    );
                }

                for (_, def) in body_info.scope.get(&ctx.scopes.arena).definitions() {
                    let def = def.get(&ctx.defs);

                    if !matches!(ArenaRef::as_ref(&def).kind, DefinitionKind::Variable(..)) {
                        continue;
                    }

                    let ty = self.types.build(def.get_type().must_lit(), &ctx.defs, &ctx.scopes.arena);
                    self.definition_values.insert(
                        ArenaRef::id(&def).into(),
                        self.builder
                            .build_alloca(BasicTypeHelper::from(ty), def.name().as_str())
                            .unwrap()
                            .as_any_value_enum(),
                    );
                }

                let is_terminal = body_info.body_ast.kind().is_terminator();
                let eval = self.build_expr(body_info.body_ast, &mut ctx).as_inner();

                if !is_terminal {
                    if fnval.func.get_name().to_str().unwrap() == "main" {
                        let exit = self.module.add_function(
                            "exit", 
                            self.ctx.void_type().fn_type(&[self.ctx.i32_type().into()], false), 
                            None
                        );
                        self.builder.build_call(exit, &[self.ctx.i32_type().const_zero().into()], "").unwrap();
                        self.builder.build_unreachable().unwrap();
                    } else {
                        self.builder
                            .build_return(eval.as_ref().map(|x| x as &dyn BasicValue))
                            .unwrap();
                    }
                }
                
                self.types.cur_gen_vals = (RcList::from([]), RcList::from([]));
                self.cur_func = None;
            }
        }

        // TODO: propogate associated generics: switch to finding monomorphisms at the llvm build step
        // to reduce the complexity of finding all monomorphisms. This can be a part of the following:

        // TODO: implement a breadth-first-traversal of calls to prevent unnecessary functions from being compiled.
        // use a stack of to-be-defined function declarations to prevent builder positioning from being spoiled,
        // as well as to prevent recursion.
        
        // TODO: create a universal identifier type so that { func } and { generic_func::[i32] } can be represented
        // as one in the same, reducing code bloat.

        // TODO: consider ways to reduce cloning of types

        // TODO: potentially switch to `Rc<[Type]>` for generic arguments, since they are shared

        // TODO: actually implement type inference!
    }

    pub fn build_stmt(&mut self, stmt: BoundStmtAST, ctx: &mut ProgramContext) {
        let span = stmt.span.clone();
        let typ = stmt.get_type().clone();

        match stmt.into_kind() {
            BoundStmt::Expr(e) => {
                self.build_expr(
                    BoundExprAST {
                        span,
                        content: Typed::<BoundExpr>::new(typ, e),
                    },
                    ctx,
                );
            }

            BoundStmt::Let {
                target,
                value
            } => {
                let target_v = target.get(&ctx.defs);
                let ty = self.types.get(target_v.get_type(), &ctx.defs, &ctx.scopes.arena);
                self.definition_values.insert(
                    target.into(),
                    self.builder
                        .build_alloca(BasicTypeHelper::from(ty), target_v.name().as_str())
                        .unwrap()
                        .as_any_value_enum(),
                );

                let dval = self.definition_values[&target.into()].into_pointer_value();
                let vval = self.build_expr(*value, ctx);

                self.builder
                    .build_store(dval, vval.as_inner().unwrap())
                    .expect("building store should work");
            }

            BoundStmt::Assign {
                target,
                value,
            } => {
                let dval = self.build_expr(*target, ctx).as_inner().unwrap().into_pointer_value();
                let vval = self.build_expr(*value, ctx);

                self.builder
                    .build_store(dval, vval.as_inner().unwrap())
                    .expect("building store should work");
            }

            BoundStmt::While { cond, block } => {
                let cond_bb = self
                    .ctx
                    .append_basic_block(self.cur_func.unwrap(), "");

                let loop_bb = self
                    .ctx
                    .append_basic_block(self.cur_func.unwrap(), "");

                let exit_bb = self
                    .ctx
                    .append_basic_block(self.cur_func.unwrap(), "");

                self.cur_loop = Some(Loop{ cond: cond_bb, end: exit_bb });

                self.builder.build_unconditional_branch(cond_bb).unwrap();

                self.builder.position_at_end(cond_bb);
                let cond = self
                    .build_expr(cond, ctx)
                    .as_inner()
                    .unwrap()
                    .into_int_value();

                if self.block_unterminated() {
                    self.builder
                        .build_conditional_branch(cond, loop_bb, exit_bb)
                        .unwrap();
                }

                self.builder.position_at_end(loop_bb);

                self.build_block(block, ctx);

                if self.block_unterminated() {
                    self.builder.build_unconditional_branch(cond_bb).unwrap();
                }
                
                self.cur_loop = None;

                self.builder.position_at_end(exit_bb);
            }
            
            BoundStmt::Return(expr) => {
                let expr = expr.map(|expr| BoundExprAST {
                    span,
                    content: Typed::<BoundExpr>::new(typ, expr),
                });

                let expr = match expr {
                    Some(x) => {
                        Some(self.build_expr(x, ctx).as_inner().unwrap())
                    }

                    None => None
                };

                self.builder.build_return(expr.as_ref().map(|x| x as &dyn BasicValue<'a>))
                    .unwrap();
            }

            BoundStmt::Continue => {
                self.builder.build_unconditional_branch(self.cur_loop.as_ref().unwrap().cond)
                    .unwrap();
            }

            BoundStmt::Break => {
                self.builder.build_unconditional_branch(self.cur_loop.as_ref().unwrap().end)
                    .unwrap();
            }
            
            x => todo!("{x:?}"),
        }
    }

    pub fn build_fn_val(&mut self, expr: BoundExpr, ctx: &mut ProgramContext) -> AnyValueEnum<'a> {
        match expr {
            BoundExpr::Fn(mut sym) => {
                self.get_fn(ctx, &self.types.transform_ref(&mut sym)).as_any_value_enum()
            }

            _ => panic!("Cannot build {expr:?} as a function value")
        }
    }

    fn build_block(&mut self, block: BoundBlock, ctx: &mut ProgramContext) -> TLValue<'a> {
        for stmt in block.body {
            self.build_stmt(stmt, ctx);
        }

        if let Some(expr) = block.tail {
            if self.block_unterminated() {
                self.build_expr(*expr, ctx)
            } else {
                TLValue::Unit
            }
        } else {
            TLValue::Unit
        }
    }

    fn build_eq(
        &mut self,
        ty: &Type,
        lhs: BasicValueEnum<'a>,
        rhs: BasicValueEnum<'a>,
        ctx: &mut ProgramContext,
    ) -> IntValue<'a> {
        match ty.as_lit() {
            Some(LitType::Bool | LitType::Char | LitType::Int(..) | LitType::UInt(..)) => {

                let lhs = lhs.into_int_value();
                let rhs = rhs.into_int_value();

                self.builder
                    .build_int_compare(IntPredicate::EQ, lhs, rhs, "")
                    .unwrap()
            }

            Some(LitType::Float(..)) => {
                let lhs = lhs.into_float_value();
                let rhs = rhs.into_float_value();

                self.builder
                    .build_float_compare(FloatPredicate::OEQ, lhs, rhs, "")
                    .unwrap()
            }

            Some(LitType::Ptr(..)) => {
                let lhs = lhs.into_pointer_value();
                let rhs = rhs.into_pointer_value();

                self.builder
                    .build_int_compare(IntPredicate::EQ, lhs, rhs, "")
                    .unwrap()
            }

            _ => panic!(
                "Unimplemented equality-comparable type {}",
                ty.full_name(&ctx.defs, &ctx.scopes.arena)
            ),
        }
    }

    fn cur_block(&self) -> BasicBlock<'a> {
        self.builder.get_insert_block().unwrap()
    }
    fn block_unterminated(&self) -> bool {
        self.cur_block().get_terminator().is_none()
    }

    fn build_expr(&mut self, expr: BoundExprAST, ctx: &mut ProgramContext) -> TLValue<'a> {
        let ty = expr.get_type().clone();

        let kind = expr.into_kind();

        match kind {
            BoundExpr::Bool(x) => self.ctx.bool_type().const_int(x.as_u64(), false).into(),
            BoundExpr::Integer(x) => self
                .types
                .get(&ty, &ctx.defs, &ctx.scopes.arena)
                .into_int_type()
                .const_int(x as u64, false)
                .into(),
            BoundExpr::Decimal(x) => self
                .types
                .get(&ty, &ctx.defs, &ctx.scopes.arena)
                .into_float_type()
                .const_float(x)
                .into(),
            
            BoundExpr::Sizeof(ty) => {
                let usize = self.types.build(&LitType::UInt(UIntType::USize), &ctx.defs, &ctx.scopes.arena);
                let ty = self.types.get(&ty, &ctx.defs, &ctx.scopes.arena);

                let sz = ty.size_of().unwrap();
                sz.const_cast(usize.into_int_type(), false).into()
            }

            BoundExpr::Block(x) => self.build_block(x, ctx),

            BoundExpr::Symbol(x) => match ArenaRef::as_ref(&x.get(&ctx.defs)).kind {
                DefinitionKind::Parameter(..) => {
                    BasicValueHelper::from(self.definition_values[&x.into()]).into()
                }
                DefinitionKind::Variable(ref typ) => {
                    let typ = self.types.get(typ, &ctx.defs, &ctx.scopes.arena);

                    self.builder
                        .build_load(
                            BasicTypeHelper::from(typ),
                            self.definition_values[&x.into()].into_pointer_value(),
                            "",
                        )
                        .expect("")
                        .into()
                }

                _ => todo!("what"),
            },

            BoundExpr::Negate(op) => {
                let ty = op.get_type().clone();
                let op = self.build_expr(*op, ctx).as_inner().unwrap();

                match ty.as_lit() {
                    Some(ty @ LitType::Int(..)) => {
                        let bty = self.types.build(ty, &ctx.defs, &ctx.scopes.arena);

                        self.builder.build_int_sub(
                            bty.into_int_type().const_zero(),
                            op.into_int_value(),
                            "",
                        )
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::Float(..)) => self
                        .builder
                        .build_float_neg(op.into_float_value(), "")
                        .unwrap()
                        .into(),

                    _ => todo!(
                        "Unimplemented signed arithmetic type {}",
                        ty.full_name(&ctx.defs, &ctx.scopes.arena)
                    ),
                }
            }

            BoundExpr::String(x) => self
                .builder
                .build_global_string_ptr(x.as_str(), "")
                .unwrap()
                .into(),

            BoundExpr::Not(x) => {
                let x = self.build_expr(*x, ctx);
                self.builder
                    .build_not(x.as_inner().unwrap().into_int_value(), "")
                    .unwrap()
                    .into()
            }

            BoundExpr::Comparison(bin, op) => {
                let ty = bin.lhs.get_type().clone();
                
                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                match ty.as_lit() {
                    Some(LitType::Int(..)) => {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();

                        self.builder
                            .build_int_compare(op.as_signed_int_predicate(), lhs, rhs, "")
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::UInt(..)) => {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();

                        self.builder
                            .build_int_compare(op.as_unsigned_int_predicate(), lhs, rhs, "")
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::Float(..)) => {
                        let lhs = lhs.into_float_value();
                        let rhs = rhs.into_float_value();

                        self.builder
                            .build_float_compare(op.as_float_predicate(), lhs, rhs, "")
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::Ptr(..)) => {
                        let lhs = lhs.into_pointer_value();
                        let rhs = rhs.into_pointer_value();

                        self.builder
                            .build_int_compare(op.as_unsigned_int_predicate(), lhs, rhs, "")
                            .unwrap()
                            .into()
                    }

                    _ => todo!("Unimplemented arithmetic type {}", ty.full_name(&ctx.defs, &ctx.scopes.arena)),
                }
            }

            BoundExpr::Equals(bin) => {
                let ty = bin.lhs.get_type().clone();
                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                self.build_eq(&ty, lhs, rhs, ctx).into()
            }

            BoundExpr::NotEquals(bin) => {
                let ty = bin.lhs.get_type().clone();
                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                let eq = self.build_eq(&ty, lhs, rhs, ctx);

                self.builder.build_not(eq, "").unwrap().into()
            }

            BoundExpr::DirectCall(ex, args) => {
                let args = args
                    .into_iter()
                    .map(|x| {
                        BasicMetadataValueEnum::from(self.build_expr(x, ctx).as_inner().unwrap())
                    })
                    .collect::<Vec<_>>();

                let func = self.build_fn_val(ex.into_kind(), ctx);

                self.builder
                    .build_call(
                        func.into_function_value(),
                        args.as_slice(),
                        "",
                    )
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .map_or(TLValue::Unit, Into::into)
            }

            BoundExpr::PtrArithmetic(bin, op) => {
                let pty = match bin.lhs.get_type().as_lit()
                {
                    Some(LitType::Ptr(x)) => *x.clone(),
                    _ => panic!("Non pointers should not be passed into the lhs of BoundExpr::PtrArithmetic.")
                };

                let pty = BasicTypeHelper::from(self.types.get(&pty, &ctx.defs, &ctx.scopes.arena));

                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                let rhs = match &op {
                    ArithOp::Add => rhs.into_int_value(),
                    ArithOp::Subtract => self
                        .builder
                        .build_int_neg(rhs.into_int_value(), "")
                        .unwrap(),

                    _ => panic!("Impossible"),
                };

                unsafe {
                    self.builder
                        .build_gep(pty, lhs.into_pointer_value(), &[rhs], "")
                        .unwrap()
                        .into()
                }
            }

            BoundExpr::Arithmetic(bin, op) => {
                let lhs = self.build_expr(*bin.lhs, ctx).as_inner().unwrap();
                let rhs = self.build_expr(*bin.rhs, ctx).as_inner().unwrap();

                match ty.as_lit() {
                    Some(LitType::Int(..)) => {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();

                        match op {
                            ArithOp::Add => self.builder.build_int_add(lhs, rhs, ""),
                            ArithOp::Subtract => self.builder.build_int_sub(lhs, rhs, ""),
                            ArithOp::Multiply => self.builder.build_int_mul(lhs, rhs, ""),
                            ArithOp::Divide => self.builder.build_int_signed_div(lhs, rhs, ""),
                            ArithOp::Modulo => self.builder.build_int_signed_rem(lhs, rhs, ""),
                        }
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::UInt(..)) => {
                        let lhs = lhs.into_int_value();
                        let rhs = rhs.into_int_value();

                        match op {
                            ArithOp::Add => self.builder.build_int_add(lhs, rhs, ""),
                            ArithOp::Subtract => self.builder.build_int_sub(lhs, rhs, ""),
                            ArithOp::Multiply => self.builder.build_int_mul(lhs, rhs, ""),
                            ArithOp::Divide => self.builder.build_int_unsigned_div(lhs, rhs, ""),
                            ArithOp::Modulo => self.builder.build_int_unsigned_rem(lhs, rhs, ""),
                        }
                    }
                    .unwrap()
                    .into(),

                    Some(LitType::Float(..)) => {
                        let lhs = lhs.into_float_value();
                        let rhs = rhs.into_float_value();

                        match op {
                            ArithOp::Add => self.builder.build_float_add(lhs, rhs, ""),
                            ArithOp::Subtract => self.builder.build_float_sub(lhs, rhs, ""),
                            ArithOp::Multiply => self.builder.build_float_mul(lhs, rhs, ""),
                            ArithOp::Divide => self.builder.build_float_div(lhs, rhs, ""),
                            ArithOp::Modulo => self.builder.build_float_rem(lhs, rhs, ""),
                        }
                    }
                    .unwrap()
                    .into(),

                    _ => todo!("Unimplemented arithmetic type {}", ty.full_name(&ctx.defs, &ctx.scopes.arena)),
                }
            }

            BoundExpr::And(bin) => {
                let bb = self.builder.get_insert_block().unwrap();
                let lhs = self
                    .build_expr(*bin.lhs, ctx)
                    .as_inner()
                    .unwrap()
                    .into_int_value();

                let fallthrough = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), ""
                );
                let end = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), ""
                );

                self.builder
                    .build_conditional_branch(lhs, fallthrough, end)
                    .unwrap();

                self.builder.position_at_end(fallthrough);

                let rhs = self
                    .build_expr(*bin.rhs, ctx)
                    .as_inner()
                    .unwrap()
                    .into_int_value();

                self.builder.build_unconditional_branch(end).unwrap();

                self.builder.position_at_end(end);

                let phi = self.builder.build_phi(self.ctx.bool_type(), "").unwrap();

                phi.add_incoming(&[
                    (&self.ctx.bool_type().const_zero(), bb),
                    (&rhs, fallthrough),
                ]);

                phi.as_basic_value().into()
            }

            BoundExpr::Or(bin) => {
                let bb = self.builder.get_insert_block().unwrap();
                let lhs = self
                    .build_expr(*bin.lhs, ctx)
                    .as_inner()
                    .unwrap()
                    .into_int_value();

                let fallthrough = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), ""
                );
                let end = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), ""
                );

                self.builder
                    .build_conditional_branch(lhs, end, fallthrough)
                    .unwrap();

                self.builder.position_at_end(fallthrough);

                let rhs = self
                    .build_expr(*bin.rhs, ctx)
                    .as_inner()
                    .unwrap()
                    .into_int_value();

                self.builder.build_unconditional_branch(end).unwrap();

                self.builder.position_at_end(end);

                let phi = self.builder.build_phi(self.ctx.bool_type(), "").unwrap();

                phi.add_incoming(&[
                    (&self.ctx.bool_type().const_all_ones(), bb),
                    (&rhs, fallthrough),
                ]);

                phi.as_basic_value().into()
            }

            BoundExpr::If(bound_if) => {
                let comp = self.build_expr(*bound_if.cond, ctx);

                if !self.block_unterminated() {
                    return TLValue::Unit;
                }

                let on_true = self.ctx.append_basic_block(
                    self.cur_func.unwrap(), "",
                );

                let end_if = self
                    .ctx
                    .append_basic_block(self.cur_func.unwrap(), "");

                let on_false = if bound_if.else_block.is_none() {
                    end_if.clone()
                } else {
                    self.ctx.append_basic_block(
                        self.cur_func.unwrap(), "",
                    )
                };

                self.builder
                    .build_conditional_branch(
                        comp.as_inner().unwrap().into_int_value(),
                        on_true,
                        on_false,
                    )
                    .unwrap();

                self.builder.position_at_end(on_true);
                let block_value = self.build_block(bound_if.block, ctx);
                if self.block_unterminated() {
                    self.builder.build_unconditional_branch(end_if).unwrap();
                }

                let else_value = if let Some(else_block) = bound_if.else_block {
                    self.builder.position_at_end(on_false);
                    let val = self.build_expr(*else_block, ctx);

                    if self.block_unterminated() {
                        self.builder.build_unconditional_branch(end_if).unwrap();
                    }

                    val
                } else {
                    TLValue::Unit
                };

                self.builder.position_at_end(end_if);

                if ty != LitType::Unit.into() {
                    let ty = self.types.get(&ty, &ctx.defs, &ctx.scopes.arena);
                    let phi = self
                        .builder
                        .build_phi(BasicTypeHelper::from(ty), "")
                        .expect("phi construction should not fail");

                    phi.add_incoming(&[
                        (&block_value.as_inner().unwrap(), block_value.block().unwrap_or(on_true)),
                        (&else_value.as_inner().unwrap(), else_value.block().unwrap_or(on_false)),
                    ]);

                    phi.as_basic_value().into()
                } else {
                    TLValue::Unit
                }
            }

            BoundExpr::Cast(op, mut castty) => {
                self.types.transform(&mut castty);
                let opty = op.get_type().clone();
                self.build_cast(*op, castty.into_lit().unwrap(), opty.into_lit().unwrap(), ctx)
            }

            BoundExpr::Dot(op, i) => {
                let op = self.build_expr(*op, ctx);

                self.builder
                    .build_extract_value(
                        op.as_inner().unwrap().into_struct_value(),
                        i as u32,
                        ""
                    )
                    .unwrap()
                    .into()
            }
            
            BoundExpr::DotRef(op, i) => {
                let ty = op.get_type().deref().unwrap().clone();
                let op = self.build_expr(*op, ctx);

                self.builder
                    .build_struct_gep(
                        BasicTypeHelper::from(self.types.get(&ty, &ctx.defs, &ctx.scopes.arena).as_any_type_enum()),
                        op.as_inner().unwrap().into_pointer_value(),
                        i as u32,
                        ""
                    )
                    .unwrap()
                    .into()
            }

            BoundExpr::Deref(op) => {
                let ty = self.types.get(op.get_type().deref().unwrap(), &ctx.defs, &ctx.scopes.arena);
                let op = self.build_expr(*op, ctx);

                self.builder
                    .build_load(BasicTypeHelper::from(ty), op.as_inner().unwrap().into_pointer_value(), "")
                    .expect("Building load should work")
                    .into()
            }

            BoundExpr::VariableRef(id) => {
                BasicValueHelper::from(self.definition_values[&id.into()]).into()
            }

            BoundExpr::DerefRef(x) 
                => self.build_expr(*x, ctx),

            BoundExpr::Zeroinit(ty) => {
                let ty = self.types.get(&ty, &ctx.defs, &ctx.scopes.arena);
                BasicTypeHelper::from(ty).const_zero().into()
            }

            BoundExpr::Error | BoundExpr::None | BoundExpr::Type(..) => TLValue::Unit,

            BoundExpr::IndirectCall(..) => todo!(),

            BoundExpr::Fn(..) => panic!("Cannot build {kind:?} as a value")
        }
    }

    fn build_cast(
        &mut self,
        op: BoundExprAST,
        ty: LitType,
        opty: LitType,
        ctx: &mut ProgramContext,
    ) -> TLValue<'a> {
        let op = self.build_expr(op, ctx).as_inner().unwrap();

        let oty = self.types.build(&ty, &ctx.defs, &ctx.scopes.arena);

        use LitType::*;

        match dbg!((opty, ty)) {
            (Int(..) | UInt(..), Int(..)) => self
                .builder
                .build_int_cast_sign_flag(op.into_int_value(), oty.into_int_type(), true, "")
                .unwrap()
                .into(),

            (Int(..) | UInt(..), UInt(..)) => self
                .builder
                .build_int_cast_sign_flag(op.into_int_value(), oty.into_int_type(), false, "")
                .unwrap()
                .into(),

            (Int(..), Float(..)) => self
                .builder
                .build_signed_int_to_float(op.into_int_value(), oty.into_float_type(), "")
                .unwrap()
                .into(),

            (UInt(..), Float(..)) => self
                .builder
                .build_unsigned_int_to_float(op.into_int_value(), oty.into_float_type(), "")
                .unwrap()
                .into(),

            (Float(..), Int(..)) => self
                .builder
                .build_float_to_signed_int(op.into_float_value(), oty.into_int_type(), "")
                .unwrap()
                .into(),

            (Float(..), UInt(..)) => self
                .builder
                .build_float_to_unsigned_int(op.into_float_value(), oty.into_int_type(), "")
                .unwrap()
                .into(),

            (Int(..) | UInt(..), Ptr(..)) => {
                println!("CONVERSIONNNNNN!");
                self
                .builder
                .build_int_to_ptr(op.into_int_value(), oty.into_pointer_type(), "")
                .unwrap()
                .into()
            }

            (Ptr(..), Ptr(..)) => op.into(),

            (i, o) => panic!(
                "Unknown cast of {} -> {}",
                i.full_name(&ctx.defs, &ctx.scopes.arena),
                o.full_name(&ctx.defs, &ctx.scopes.arena)
            ),
        }
    }
}
