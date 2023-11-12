use std::{
    cell::{Cell, OnceCell},
    fmt::Debug,
};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OnceState {
    Uninitialized,
    Building,
    Initialized,
}

#[derive(Clone)]
pub struct OnceBuildable<T> {
    state: Cell<OnceState>,
    value: OnceCell<T>,
}

impl<T> OnceBuildable<T> {
    pub fn new() -> OnceBuildable<T> {
        OnceBuildable {
            state: Cell::new(OnceState::Uninitialized),
            value: OnceCell::new(),
        }
    }

    pub fn get(&self) -> Option<&T> {
        self.value.get()
    }

    pub fn set(&self, val: T) -> Result<(), T> {
        self.value.set(val)?;
        self.state.set(OnceState::Initialized);
        Ok(())
    }

    pub fn state(&self) -> OnceState {
        self.state.get()
    }

    pub fn begin_build(&self) {
        self.state.set(OnceState::Building);
    }

    pub fn is_init(&self) -> bool {
        self.state() == OnceState::Initialized
    }
    pub fn is_building(&self) -> bool {
        self.state() == OnceState::Building
    }
    pub fn is_uninit(&self) -> bool {
        self.state() == OnceState::Uninitialized
    }
}

impl<T> Debug for OnceBuildable<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.get())
    }
}
