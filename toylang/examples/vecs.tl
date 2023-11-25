import "std:vec.tl"
import "std:io.tl"

fn main(): {
    let x = std::vec::new::[i32]();

    std::vec::push::[i32](&x, 10);
    std::vec::push::[i32](&x, 20);
    std::vec::push::[i32](&x, 30);
    std::vec::push::[i32](&x, 40);

    std::printf("%d\n", x.len);

    let i = 0 as usize;

    // NOTE: If semicolon after this loop is ommited, the
    // entire statement seemingly vanishes!

    while i < x.len {
        std::printf("%d\n", *std::vec::get::[i32](&x, i));
        i = i + 1 as usize;
    };
}