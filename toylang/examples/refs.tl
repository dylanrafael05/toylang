import "std:io.tl"

fn add_one(x &i32): {
    *x = *x + 1;
}

fn main(): {
    let x = 1;
    add_one(&x);

    std::printf("%d\n", x);
}