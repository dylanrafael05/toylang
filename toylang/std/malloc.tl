mod std

declare fn malloc(sz usize) *unit
declare fn free(ptr *unit) unit

fn malloc_many(type_size usize, amt i32) *unit: 
    malloc(type_size * amt as usize)
    
fn new[T]() *T: malloc(sizeof T) as *T
fn new_arr[T](sz i32) *T: malloc_many(sizeof T, sz) as *T

fn null[T]() *T: 0 as *T

fn copy[T](a *T, b *T, len usize): {
    let i = 0 as usize;
    while i < len {
        *(b + i) = *(a + i);
        i = i + 1 as usize;
    };
}