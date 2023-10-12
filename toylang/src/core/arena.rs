use std::{marker::PhantomData, hash::Hash, fmt::Debug, ops::{Add, Sub, AddAssign, DerefMut, Deref, SubAssign}};

/// An ID corresponding to some arena value.
pub struct ArenaID<T>(usize, PhantomData<T>);

impl<T> Debug for ArenaID<T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArenaID")
            .field(&self.0)
            .finish()
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

impl<T> AddAssign<usize> for ArenaID<T>
{
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
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

impl<T> SubAssign<usize> for ArenaID<T>
{
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

impl<T> ArenaID<T>
{
    pub fn get<'a>(self, owner: &'a Arena<T>) -> ArenaRef<'a, T>
    {
        owner.get(self)
    }
    pub fn get_mut<'a>(self, owner: &'a mut Arena<T>) -> ArenaMut<'a, T>
    {
        owner.get_mut(self)
    }
    pub fn take(self, owner: &mut Arena<T>) -> ArenaVal<T>
    {
        owner.take(self)
    }

    pub fn as_usize(&self) -> usize {self.0}
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaVal<T>
{
    id: ArenaID<T>,
    value: T
}

impl<T> ArenaVal<T>
{
    pub fn id(&self) -> ArenaID<T> {self.id}
    pub fn get(&self) -> &T {&self.value}
    
    fn from(id: ArenaID<T>, value: T) -> Self 
    {
        ArenaVal { id, value }
    }

    pub fn into_value(self) -> T {self.value}
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaRef<'a, T>
{
    id: ArenaID<T>,
    value: &'a T
}


impl<'a, T> Deref for ArenaRef<'a, T>
{
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<'a, T> ArenaRef<'a, T>
{
    pub fn id(this: &Self) -> ArenaID<T> {this.id}

    fn from(id: ArenaID<T>, value: &'a T) -> Self 
    {
        ArenaRef { id, value }
    }
    
    pub fn as_ref(this: &Self) -> &'a T
    {
        this.value
    }
    pub fn into_ref(this: Self) -> &'a T
    {
        this.value
    }
}

#[derive(Debug)]
pub struct ArenaMut<'a, T>
{
    id: ArenaID<T>,
    value: &'a mut T
}

impl<'a, T> ArenaMut<'a, T>
{
    pub fn id(this: &Self) -> ArenaID<T> {this.id}
    
    fn from(id: ArenaID<T>, value: &'a mut T) -> Self 
    {
        ArenaMut { id, value }
    }

    pub fn as_ref(this: &'a Self) -> &'a T
    {
        &this.value
    }
    pub fn as_mut(this: &'a mut Self) -> &'a mut T
    {
        this.value
    }

    pub fn into_ref(this: Self) -> &'a T
    {
        &*this.value
    }
    pub fn into_mut(this: Self) -> &'a mut T
    {
        this.value
    }
}

impl<'a, T> Deref for ArenaMut<'a, T>
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<'a, T> DerefMut for ArenaMut<'a, T>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
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

    pub fn next_id(&self) -> ArenaID<T>
    {
        self.next_id
    }

    pub fn get<'a>(&'a self, id: ArenaID<T>) -> ArenaRef<'a, T>
    {
        self.values.get(id.as_usize())
            .and_then(|x| x.as_ref())
            .map(|k| ArenaRef::from(id.clone(), k))
            .unwrap()
    }
    pub fn get_mut<'a>(&'a mut self, id: ArenaID<T>) -> ArenaMut<'a, T> 
    {
        self.values.get_mut(id.as_usize())
            .and_then(|x| x.as_mut())
            .map(|k| ArenaMut::from(id.clone(), k))
            .unwrap()
    }
    pub fn take(&mut self, id: ArenaID<T>) -> ArenaVal<T> 
    {
        self.values.get_mut(id.as_usize())
            .and_then(|x| x.take())
            .map(|k| ArenaVal::from(id.clone(), k))
            .unwrap()
    }

    pub fn add(&mut self, value: T) -> ArenaID<T>
    {
        let id = self.next_id;
        self.next_id += 1;

        self.values.push(Some(value));

        id
    }

    pub fn len(&self) -> usize {self.values.len()}
}