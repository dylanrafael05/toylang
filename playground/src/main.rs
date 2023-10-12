use std::{marker::PhantomData, hash::Hash, fmt::Debug, ops::{Add, Sub}};

pub struct ArenaID<T>(usize, PhantomData<T>);

impl<T> Debug for ArenaID<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T> Default for ArenaID<T>
{
    fn default() -> Self {
        Self(Default::default(), Default::default())
    }
}

impl<T> Clone for ArenaID<T>
{
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<T> Copy for ArenaID<T> {}

impl<T> PartialEq for ArenaID<T> 
{
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T> Eq for ArenaID<T> {}

impl<T> Hash for ArenaID<T> 
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.0)
    }
}

impl<T> From<usize> for ArenaID<T>
{
    fn from(x: usize) -> Self {Self(x, PhantomData)}
}

impl<T> From<ArenaID<T>> for usize
{
    fn from(x: ArenaID<T>) -> usize {x.0}
}

impl<T> Add<usize> for ArenaID<T>
{
    type Output = ArenaID<T>;
    fn add(self, rhs: usize) -> Self::Output {
        (usize::from(self) + rhs).into()
    }
}
impl<T> Add<ArenaID<T>> for usize
{
    type Output = ArenaID<T>;
    fn add(self, rhs: ArenaID<T>) -> Self::Output {
        (self + usize::from(rhs)).into()
    }
}

impl<T> Sub<usize> for ArenaID<T>
{
    type Output = ArenaID<T>;
    fn sub(self, rhs: usize) -> Self::Output {
        (usize::from(self) - rhs).into()
    }
}
impl<T> Sub<ArenaID<T>> for usize
{
    type Output = ArenaID<T>;
    fn sub(self, rhs: ArenaID<T>) -> Self::Output {
        (self - usize::from(rhs)).into()
    }
}

impl<T> ArenaID<T>
{
    pub fn get<'a>(&self, owner: &'a Arena<T>) -> Option<(Self, &'a T)>
    {
        owner.get(self)
    }
    pub fn get_mut<'a>(&self, owner: &'a mut Arena<T>) -> Option<(Self, &'a mut T)>
    {
        owner.get_mut(self)
    }
    pub fn take(&self, owner: &mut Arena<T>) -> Option<(Self, T)>
    {
        owner.take(self)
    }

    pub fn as_usize(&self) -> usize {self.0}
}

#[derive(Debug)]
pub struct Arena<T>
{
    next_id: ArenaID<T>,
    values: Vec<Option<T>>
}

impl<T> Arena<T>
{
    pub fn new() -> Arena<T>
    {
        Arena { next_id: 0.into(), values: Default::default() }
    }

    pub fn get(&self, id: &ArenaID<T>) -> Option<(ArenaID<T>, &T)> 
    {
        self.values.get(id.as_usize())
            .and_then(|x| x.as_ref())
            .map(|k| (id.clone(), k))
    }
    pub fn get_mut(&mut self, id: &ArenaID<T>) -> Option<(ArenaID<T>, &mut T)> 
    {
        self.values.get_mut(id.as_usize())
            .and_then(|x| x.as_mut())
            .map(|k| (id.clone(), k))
    }
    pub fn take(&mut self, id: &ArenaID<T>) -> Option<(ArenaID<T>, T)> 
    {
        self.values.get_mut(id.as_usize())
            .and_then(|x| x.take())
            .map(|k| (id.clone(), k))
    }

    pub fn add(&mut self, value: T) -> ArenaID<T>
    {
        let id = self.next_id;
        self.next_id = ArenaID::<T>::from(usize::from(self.next_id) + 1);

        self.values.push(Some(value));

        id
    }

    pub fn len(&self) -> usize {self.values.len()}
}

// TESTING //

#[derive(Debug, PartialEq, Eq)]
pub struct Data(pub i32);

fn main() {

    let mut x = Arena::new();

    x.add(Data(10));

    assert!(ArenaID::default().get(&x) == Some((ArenaID::default(), &Data(10))));

        let (_, val) = ArenaID::default().get_mut(&mut x).unwrap();

        *val = Data(40);

    assert!(ArenaID::default().get(&x) == Some((ArenaID::default(), &Data(40))));

    assert!(ArenaID::default().take(&mut x) == Some((ArenaID::default(), Data(40))));

    assert!(ArenaID::default().get(&x) == None);
}
