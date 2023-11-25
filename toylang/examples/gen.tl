import "std:malloc.tl" 
import "std:io.tl" 

struct Box[T] {
    ptr *T
}

struct Point {
    x i32,
    y i32
}

fn get[T](b &Box[T]) &T: &*b.ptr

fn main(): {
    let m = Box::[Point]();

    m.ptr = std::new::[Point]();

    get::[Point](&m).x = 20;
    get::[Point](&m).y = 40;

    std::printf("(%d, %d)\n", get::[Point](&m).x, get::[Point](&m).y);
}