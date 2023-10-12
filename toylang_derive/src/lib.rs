use quote::quote;
use syn::{parse_macro_input, DeriveInput};


#[proc_macro_derive(Symbol)]
pub fn derive_symbol(input: proc_macro::TokenStream) -> proc_macro::TokenStream
{
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    proc_macro::TokenStream::from(quote!{
        impl crate::parsing::ast::Symbol for #name
        {
            type Kind = Self;
            fn kind(&self) -> &Self {&self}
        }

        impl crate::parsing::ast::OwnedSymbol for #name 
        {
            fn into_kind(self) -> Self {self}
        }
    })
}