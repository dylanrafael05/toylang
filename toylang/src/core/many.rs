#[derive(Debug, Clone)]
pub enum Many<T> {
    None,
    One(T),
    Many(Vec<T>),
}

impl Many<()> {
    pub fn unit(&self) {}
}

impl<V> FromIterator<V> for Many<V> {
    fn from_iter<T: IntoIterator<Item = V>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        if let Some(a) = iter.next() {
            if let Some(b) = iter.next() {
                let mut vec = vec![a, b];
                vec.extend(iter);
                Self::Many(vec)
            } else {
                Self::One(a)
            }
        } else {
            Self::None
        }
    }
}

impl<T> Many<T> {
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a T> {
        let iter: Box<dyn Iterator<Item = &'a T>> = match self {
            Self::None => Box::new(std::iter::empty()),
            Self::One(x) => Box::new(std::iter::once(x)),
            Self::Many(x) => Box::new(x.iter()),
        };

        iter
    }

    pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut T> {
        let iter: Box<dyn Iterator<Item = &'a mut T>> = match self {
            Self::None => Box::from(std::iter::empty()),
            Self::One(x) => Box::from(std::iter::once(x)),
            Self::Many(x) => Box::from(x.iter_mut()),
        };

        iter
    }
}
