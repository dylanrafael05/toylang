import "std:malloc.tl"

mod std
mod vec

struct Vec[T] {
    ptr *T,
    len usize,
    cap usize
}

fn new[T]() Vec[T]: {
    let v = Vec::[T]();
    v.ptr = std::null::[T]();
    v.len = 0 as usize;
    v.cap = 0 as usize;
    v
}

fn push[T](_self &Vec[T], elem T): {
    let self = _self;
    if self.len >= self.cap {
        let newcap = self.cap * 2 as usize + 1 as usize;
        let new = std::new_arr::[T](newcap as i32);

        std::copy::[T](self.ptr, new, self.len);
        std::free(self.ptr as *unit);

        self.ptr = new;
        
        self.cap = newcap;
    };
    
    *(self.ptr + self.len) = elem;
    self.len = self.len + 1 as usize;
}

fn get[T](self &Vec[T], idx usize) &T: {
    &*(self.ptr + idx)
}

fn free[T](self &Vec[T]): {
    std::free(self.ptr as *unit);
}