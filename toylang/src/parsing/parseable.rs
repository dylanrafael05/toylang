/*
use pest::iterators::{Pair, Pairs};
use pest::RuleType;

use crate::core::Ident;
use crate::core::Source;

// TRAITS //

pub trait Parseable<Rule>
where
    Self: Sized,
    Rule: RuleType,
{
    // standard series

    fn parse(source: &Source, pair: Pair<'_, Rule>) -> Self;

    fn parse_next(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Self {
        Self::parse(source, pairs.next().unwrap())
    }

    fn parse_next_maybe(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Option<Self> {
        pairs.next().map(|p| Self::parse(source, p))
    }

    fn parse_rest(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Vec<Self> {
        pairs.map(|p| Self::parse(source, p)).collect()
    }

    // box series

    fn box_parse(source: &Source, pair: Pair<'_, Rule>) -> Box<Self> {
        Box::new(Self::parse(source, pair))
    }

    fn box_parse_next(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Box<Self> {
        Self::box_parse(source, pairs.next().unwrap())
    }

    fn box_parse_next_maybe(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Option<Box<Self>> {
        pairs.next().map(|p| Self::box_parse(source, p))
    }

    fn box_parse_rest(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Vec<Box<Self>> {
        pairs.map(|p| Self::box_parse(source, p)).collect()
    }

    // vec series

    fn vec_parse(source: &Source, pair: Pair<'_, Rule>) -> Vec<Self> {
        pair.into_inner().map(|p| Self::parse(source, p)).collect()
    }

    fn vec_parse_next(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Vec<Self> {
        Self::vec_parse(source, pairs.next().unwrap())
    }

    fn vec_parse_next_maybe(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Option<Vec<Self>> {
        pairs.next().map(|p| Self::vec_parse(source, p))
    }

    fn vec_parse_rest(source: &Source, pairs: &mut Pairs<'_, Rule>) -> Vec<Vec<Self>> {
        pairs.map(|p| Self::vec_parse(source, p)).collect()
    }
}

// BASE IMPLEMENTATIONS //

impl<Rule: RuleType> Parseable<Rule> for String {
    fn parse(_: &Source, pair: Pair<'_, Rule>) -> Self {
        Self::from(pair.as_str().trim())
    }
}

impl<Rule: RuleType> Parseable<Rule> for Ident {
    fn parse(_: &Source, pair: Pair<'_, Rule>) -> Self {
        pair.as_str().into()
    }
}
*/