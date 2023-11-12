pub trait Wrapper
where
    Self: Sized,
{
    type Value;
    type Of<T>: Wrapper;

    fn map_ref<'a, T, F: FnOnce(&'a Self::Value) -> T>(&'a self, func: F) -> Self::Of<T>;
    fn map<T, F: FnOnce(Self::Value) -> T>(self, func: F) -> Self::Of<T>;

    fn map_ref_deep<T, F: FnOnce(&<Self::Value as Wrapper>::Value) -> T>(
        &self,
        func: F,
    ) -> Self::Of<<Self::Value as Wrapper>::Of<T>>
    where
        Self::Value: Wrapper,
    {
        self.map_ref(|x| x.map_ref(func))
    }
    fn map_deep<T, F: FnOnce(<Self::Value as Wrapper>::Value) -> T>(
        self,
        func: F,
    ) -> Self::Of<<Self::Value as Wrapper>::Of<T>>
    where
        Self::Value: Wrapper,
    {
        self.map(|x| x.map(func))
    }

    fn replace<T>(&self, value: T) -> Self::Of<T> {
        self.map_ref(|_| value)
    }
    fn replace_deep<T>(&self, value: T) -> Self::Of<<Self::Value as Wrapper>::Of<T>>
    where
        Self::Value: Wrapper,
    {
        self.map_ref_deep(|_| value)
    }

    fn as_ref<'a>(&'a self) -> Self::Of<&'a Self::Value> {
        self.map_ref(|x| x)
    }
}
