#![feature(min_specialization)]
#![feature(trait_alias)]
#![feature(try_trait_v2)]
#![feature(generic_arg_infer)]
#![feature(panic_update_hook)]

pub mod binding;
pub mod core;
pub mod llvm;
pub mod parsing;
pub mod types;

use core::{Severity, utils::{Itertools, PathExtensions}};
use std::{ffi::OsString, path::Path, process::{Command, Stdio, exit}, io::{stderr, Write}};

use inkwell::{
    context::Context,
    targets::{Target, TargetMachine, FileType},
};
use llvm::CodegenContext;
use parsing::ast::Program;
use termcolor::{ColorChoice, WriteColor, ColorSpec, Color};
pub use toylang_derive::*;

use clap::Parser as CParser;

#[derive(CParser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to compile
    file: OsString,

    /// Output name
    #[arg(short, long, value_name="FILE")]
    out: Option<OsString>
}

use crate::binding::ProgramContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::panic::update_hook(|prev, info| {
        let mut stdout = termcolor::StandardStream::stdout(ColorChoice::Always);

        let mut spec = ColorSpec::new();
        let spec = spec
            .set_fg(Some(Color::Red))
            .set_bold(true)
            .set_dimmed(true);

        stdout.set_color(spec)
            .unwrap();

        writeln!(&mut stdout, "[[ INTERNAL COMPILER ERROR ]]").unwrap();

        stdout.set_color(spec.set_dimmed(false))
            .unwrap();

        stdout.flush().unwrap();

        prev(info);
    });

    let mut stdout = termcolor::StandardStream::stdout(ColorChoice::Always);

    let args = Args::parse();
    let file = args.file;

    if !Path::new(&file).exists() {
        println!("Could not find file '{}'", file.to_str().unwrap());
        exit(-1)
    }

    let out = match args.out {
        None => Path::new(&file)
            .canonicalize()?
            .join(".out"),

        Some(x) => x.into()
    };

    let mut ctx = ProgramContext::new();

    let prog = Program::new(
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

    if diags.iter().none(|x| x.severity == Severity::Error) {
        Target::initialize_native(&Default::default()).expect("Native target works");
        let target = Target::get_first()
            .unwrap()
            .create_target_machine(
                &TargetMachine::get_default_triple(),
                TargetMachine::get_host_cpu_name().to_str().unwrap(),
                TargetMachine::get_host_cpu_features().to_str().unwrap(),
                inkwell::OptimizationLevel::Aggressive,
                inkwell::targets::RelocMode::PIC,
                inkwell::targets::CodeModel::Default,
            )
            .expect("Creating the target should work!");

        let llvm_ctx = Context::create();
        let mut codegen = CodegenContext::new(&llvm_ctx, &target);

        codegen.build(prog, ctx);

        println!("{}", codegen.module.print_to_string().to_str().unwrap());
        
        match codegen.module.verify() {
            Ok(()) => println!("Verified! Now outputting . . ."),
            Err(x) => return Err(Box::new(x))
        };

        let tmp = out.replace_filename(|x| OsString::from(x.to_str().unwrap().to_owned() + ".o"));

        target.write_to_file(&codegen.module, FileType::Object, tmp.as_path())?;

        let clang = Command::new("clang")
            .arg(tmp.as_os_str())
            .arg("-o")
            .arg(out.as_os_str())
            .stderr(Stdio::inherit())
            .output();
    
        std::fs::remove_file(tmp).expect("Could not remove temporary file!");

        if let Err(x) = clang {
            writeln!(stderr(), "No clang found! Please install clang").unwrap();
            return Err(Box::new(x))
        }

        /*
        let ex = codegen
            .module
            .create_jit_execution_engine(inkwell::OptimizationLevel::Default)
            .unwrap();

        let f = unsafe { ex.get_function::<unsafe extern "C" fn()>("main") }.unwrap();

        unsafe {
            f.call();
        }
        */
    } else {
        std::process::exit(1);
    }

    Ok(())
}
