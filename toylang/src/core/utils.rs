use std::{path::{Path, PathBuf}, ffi::{OsStr, OsString}, cell::Cell};

pub struct ZipOrNone<I1: Iterator, I2: Iterator>(I1, I2);

impl<I1: Iterator, I2: Iterator> Iterator for ZipOrNone<I1, I2> {
    type Item = (I1::Item, Option<I2::Item>);
    fn next(&mut self) -> Option<Self::Item> {
        Some((self.0.next()?, self.1.next()))
    }
}

pub trait Itertools: Iterator + Sized {
    fn none(&mut self, mut f: impl FnMut(Self::Item) -> bool) -> bool {
        !self.any(|x| f(x))
    }
    fn zip_or_none<Iter: Iterator>(self, iter: Iter) -> ZipOrNone<Self, Iter> {
        ZipOrNone(self, iter)
    }
}

impl<Iter: Iterator> Itertools for Iter {}

pub trait PathExtensions {
    fn replace_filename(&self, f: impl FnOnce(&OsStr) -> OsString) -> PathBuf;
}

impl PathExtensions for PathBuf {
    fn replace_filename(&self, f: impl FnOnce(&OsStr) -> OsString) -> PathBuf {
        match self.parent() {
            Some(x) => x.join(f(self.file_name().unwrap())),
            None => Self::from(f(self.file_name().unwrap()))
        }
    }
}
impl PathExtensions for Path {
    fn replace_filename(&self, f: impl FnOnce(&OsStr) -> OsString) -> PathBuf {
        self.to_path_buf().replace_filename(f)
    }
}


#[derive(Debug, PartialEq, Eq)]
pub struct FlagCell(Cell<bool>);

impl FlagCell {
    pub fn new() -> Self {Self(Cell::new(false))}
    pub fn get(&self) -> bool {self.0.get()}
    pub fn mark(&self) {self.0.set(true)}
}

impl Default for FlagCell {
    fn default() -> Self {
        Self::new()
    }
}


trait InnerClone {
    type Output;
    fn cloned(&self) -> Self::Output;
}

macro_rules! inner_clone_impl {
    ($lfirst: lifetime $tyfirst: ident $valfirst: ident, $($l: lifetime $ty: ident $val: ident),*) => {
        inner_clone_impl!{@inner $lfirst $tyfirst $valfirst, $($l $ty $val),*}
        inner_clone_impl!{$($l $ty $val),*}
    };
    
    ($lfirst: lifetime $tyfirst: ident $valfirst: ident) => {};
    
    () => {};
    
    (@inner $($l: lifetime $ty: ident $val: ident),+) => {
        impl<$($l),+, $($ty),+> InnerClone for ($(&$l $ty),+) where $($ty: Clone),+ {
            type Output = ($($ty),+);
            fn cloned(&self) -> Self::Output {
                let ($($val),+) = *self;
                ($($val.clone()),+)
            }
        }
    };
}

inner_clone_impl!{'a A a, 'b B b, 'c C c, 'd D d, 'e E e, 'f F f, 'g G g, 'h H h}