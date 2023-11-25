pub trait OneSidedCollection<T> {
    fn new() -> Self;

    fn top(&self) -> Option<&T>;
    fn pop(&mut self) -> Option<T>;
    fn push(&mut self, val: T);

    fn len(&self) -> usize;
    
    fn clear(&mut self) {while !self.is_empty() {self.pop();}}
    fn is_empty(&self) -> bool {self.len() == 0}
}

pub trait TwoSidedCollection<T> {
    fn new() -> Self;

    fn front(&self) -> Option<&T>;
    fn pop_front(&mut self) -> Option<T>;
    fn push_front(&mut self, val: T);

    fn back(&self) -> Option<&T>;
    fn pop_back(&mut self) -> Option<T>;
    fn push_back(&mut self, val: T);
    
    fn len(&self) -> usize;
    
    fn clear(&mut self) {while !self.is_empty() {self.pop_back();}}
    fn is_empty(&self) -> bool {self.len() == 0}
}

pub trait OneSidedBuilder {
    type Type<T>: OneSidedCollection<T>;
}
pub trait TwoSidedBuilder {
    type Type<T>: TwoSidedCollection<T>;
}

macro_rules! impl_collection {
    (one-sided [$($scope: ident)::*] :: $ty: ident @ $builder: ident => top: $top: ident, pop: $pop: ident, push: $push: ident; $($rest: tt)*) => {
        impl_collection!(@def $builder);
        impl_collection!(@one [$($scope)::*] :: $ty @ $builder => top: $top, pop: $pop, push: $push);
        impl_collection!($($rest)*);
    };
    (two-sided [$($scope: ident)::*] :: $ty: ident @ $builder: ident => front: $front: ident, back: $back: ident, pop_front: $pop_front: ident, pop_back: $pop_back: ident, push_front: $push_front: ident, push_back: $push_back: ident; $($rest: tt)*) => {
        impl_collection!(@def $builder);
        impl_collection!(@two [$($scope)::*] :: $ty @ $builder => front: $front, back: $back, pop_front: $pop_front, pop_back: $pop_back, push_front: $push_front, push_back: $push_back);
        impl_collection!($($rest)*);
    };

    () => {};
    
    (@def $ty: ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $ty;
    };
    
    (@two [$($scope: ident)::*] :: $ty: ident @ $builder: ident => front: $front: ident, back: $back: ident, pop_front: $pop_front: ident, pop_back: $pop_back: ident, push_front: $push_front: ident, push_back: $push_back: ident) => {
        impl<T> $crate::core::collections::TwoSidedCollection<T> for $($scope::)* $ty <T> {
            fn new() -> Self {Self::new()}
            
            fn front(&self) -> Option<&T> {self.$front()}
            fn pop_front(&mut self) -> Option<T> {self.$pop_front()}
            fn push_front(&mut self, val: T) {self.$push_front(val)}

            fn back(&self) -> Option<&T> {self.$back()}
            fn pop_back(&mut self) -> Option<T> {self.$pop_back()}
            fn push_back(&mut self, val: T) {self.$push_back(val)}

            fn len(&self) -> usize {self.len()}
        }
        
        impl $crate::core::collections::TwoSidedBuilder for $builder {
            type Type<T> = $($scope::)* $ty<T>;
        }

        impl_collection!(@one [$($scope)::*] :: $ty @ $builder => top: $back, pop: $pop_back, push: $push_back);
    };
    (@one [$($scope: ident)::*] :: $ty: ident @ $builder: ident => top: $top: ident, pop: $pop: ident, push: $push: ident) => {
        impl<T> $crate::core::collections::OneSidedCollection<T> for $($scope::)* $ty <T> {
            fn new() -> Self {Self::new()}

            fn top(&self) -> Option<&T> {self.$top()}
            fn pop(&mut self) -> Option<T> {self.$pop()}
            fn push(&mut self, val: T) {self.$push(val)}
            
            fn len(&self) -> usize {self.len()}
        }

        impl $crate::core::collections::OneSidedBuilder for $builder {
            type Type<T> = $($scope::)* $ty<T>;
        }
    };
}

impl_collection! {
    one-sided [std::vec]::Vec @ VecBuilder => top: last, pop: pop, push: push;
    two-sided [std::collections]::LinkedList @ LinkedListBuilder => front: front, back: back, pop_front: pop_front, pop_back: pop_back, push_front: push_front, push_back: push_back;
    two-sided [std::collections]::VecDeque @ VecDequeBuilder => front: front, back: back, pop_front: pop_front, pop_back: pop_back, push_front: push_front, push_back: push_back;
}

pub mod stack {
    use super::{OneSidedBuilder, VecBuilder, OneSidedCollection};

    #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
    pub struct Stack<T, Builder: OneSidedBuilder = VecBuilder>(Builder::Type<T>);

    pub struct Drain<'a, T, Builder: OneSidedBuilder>(&'a mut Stack<T, Builder>);

    impl<'a, T, Builder: OneSidedBuilder> Iterator for Drain<'a, T, Builder> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.pop()
        }
    }
    
    impl<T, Builder: OneSidedBuilder> Stack<T, Builder> {
        pub fn drain<'a>(&'a mut self) -> Drain<'a, T, Builder> {Drain(self)}
    }

    impl<T, Builder: OneSidedBuilder> OneSidedCollection<T> for Stack<T, Builder> {
        fn new() -> Self {
            Self(Builder::Type::<T>::new())
        }

        fn top(&self) -> Option<&T> {
            self.0.top()
        }
        fn pop(&mut self) -> Option<T> {
            self.0.pop()
        }
        fn push(&mut self, val: T) {
            self.0.push(val)
        }

        fn len(&self) -> usize {
            self.0.len()
        }
    }
}

pub mod queue {
    use super::{VecDequeBuilder, TwoSidedBuilder, OneSidedCollection, TwoSidedCollection};

    #[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
    pub struct Queue<T, Builder: TwoSidedBuilder = VecDequeBuilder>(Builder::Type<T>);
    
    pub struct Drain<'a, T, Builder: TwoSidedBuilder>(&'a mut Queue<T, Builder>);

    impl<'a, T, Builder: TwoSidedBuilder> Iterator for Drain<'a, T, Builder> {
        type Item = T;
        fn next(&mut self) -> Option<Self::Item> {
            self.0.pop()
        }
    }

    impl<T, Builder: TwoSidedBuilder> Queue<T, Builder> {
        pub fn drain<'a>(&'a mut self) -> Drain<'a, T, Builder> {Drain(self)}
    }

    impl<T, Builder: TwoSidedBuilder> OneSidedCollection<T> for Queue<T, Builder> {
        fn new() -> Self {
            Self(Builder::Type::<T>::new())
        }

        fn top(&self) -> Option<&T> {
            self.0.back()
        }
        fn pop(&mut self) -> Option<T> {
            self.0.pop_back()
        }
        fn push(&mut self, val: T) {
            self.0.push_front(val)
        }
        
        fn len(&self) -> usize {
            self.0.len()
        }
    }
}