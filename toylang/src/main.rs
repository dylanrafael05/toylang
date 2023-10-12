pub mod core;
pub mod parsing;
pub mod binding;
pub mod llvm;

use std::ffi::OsString;

use inkwell::{context::Context, targets::{TargetMachine, Target}};
use llvm::CodegenContext;
pub use toylang_derive::*;

use clap::Parser as CParser;

#[derive(CParser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args 
{
    /// File to compile
    #[arg(short, long)]
    file: OsString
}

use crate::{binding::ProgramContext, parsing::ast::UnboundSymbol};

fn main() -> Result<(), Box<dyn std::error::Error>> 
{
    let args = Args::parse();
    println!("{}", args.file.to_str().unwrap());
    let ipt = std::fs::read_to_string(
        std::fs::canonicalize(args.file).unwrap())
        .expect("User should provide valid file");
    let ipt = ipt.as_str();
    
    match parsing::ast::parse(ipt)
    {
        Ok(ast) =>
        {
            let mut ctx = ProgramContext::new();

            let ast = ast.bind(&mut ctx);

            let diags = ctx.diags.diagnostics();
            if diags.is_empty()
            {
                Target::initialize_native(&Default::default()).expect("Native target works");
                let target = Target::get_first().unwrap().create_target_machine(
                    &TargetMachine::get_default_triple(), 
                    TargetMachine::get_host_cpu_name().to_str().unwrap(), 
                    TargetMachine::get_host_cpu_features().to_str().unwrap(), 
                    inkwell::OptimizationLevel::Aggressive, 
                    inkwell::targets::RelocMode::Default, 
                    inkwell::targets::CodeModel::Default
                ).expect("Creating the target should work!");

                let llvm_ctx = Context::create();
                let mut codegen = CodegenContext::new(
                    &llvm_ctx, 
                    target
                );
                
                codegen.build(ast, ctx);

                println!(
                    "{}",
                    codegen.module.print_to_string().to_str().unwrap()
                );
            }
            else 
            {
                for diag in diags
                {
                    println!("{diag:?}")
                }
            }

            Ok(())
        },

        Err(e) => 
        {
            println!("{e}");

            Err(Box::new(e))
        }
    }
}