use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "parsing/grammar.pest"]
pub struct Grammar;