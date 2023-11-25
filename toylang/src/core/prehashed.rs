use std::{hash::{Hash, Hasher}, collections::hash_map::DefaultHasher, cell::OnceCell, ops::Deref};

#[derive(Debug, Clone)]
pub struct Prehashed<T: Hash> {
    value: T,
    hash: OnceCell<u64>
}

impl<T: Hash> Deref for Prehashed<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T: Hash> Hash for Prehashed<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(
            *self.hash.get_or_init(|| {
                let mut hasher = DefaultHasher::new();
                self.value.hash(&mut hasher);
                hasher.finish()
            })
        );
    }
}