#![feature(min_specialization)]
#![feature(trait_alias)]
#![feature(try_trait_v2)]
#![feature(generic_arg_infer)]

pub mod binding;
pub mod core;
pub mod llvm;
pub mod parsing;
pub mod types;

use std::ffi::OsString;

use inkwell::{
    context::Context,
    targets::{Target, TargetMachine},
};
use llvm::CodegenContext;
use parsing::ast::Program;
use termcolor::ColorChoice;
pub use toylang_derive::*;

use clap::Parser as CParser;

#[derive(CParser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to compile
    file: OsString,
}

use crate::binding::ProgramContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let file = args.file;

    let mut stdout = termcolor::StandardStream::stdout(ColorChoice::Always);
    let mut ctx = ProgramContext::new();

    let mut prog = Program::new(
        file.to_str().unwrap().to_owned(), 
        &mut ctx);
    let prog = prog.bind(&mut ctx);

    let diags = ctx.diags.diagnostics();
    for diag in diags {
        diag.write(&mut stdout)?;
    }
    // exit(0);
    // for def in ctx.defs.iter() {
    //     println!("{:#?}", *def);
    // }
    // exit(0);

    if diags.is_empty() {
        Target::initialize_native(&Default::default()).expect("Native target works");
        let target = Target::get_first()
            .unwrap()
            .create_target_machine(
                &TargetMachine::get_default_triple(),
                TargetMachine::get_host_cpu_name().to_str().unwrap(),
                TargetMachine::get_host_cpu_features().to_str().unwrap(),
                inkwell::OptimizationLevel::Aggressive,
                inkwell::targets::RelocMode::Default,
                inkwell::targets::CodeModel::Default,
            )
            .expect("Creating the target should work!");

        let llvm_ctx = Context::create();
        let mut codegen = CodegenContext::new(&llvm_ctx, &target);

        codegen.build(prog, ctx);

        println!("{}", codegen.module.print_to_string().to_str().unwrap());

        let ex = codegen
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::Default)
            .unwrap();

        let f = unsafe { ex.get_function::<unsafe extern "C" fn()>("main") }.unwrap();

        unsafe {
            f.call();
        }
    }

    Ok(())
}
