use std::{
    fs, env,
    path::PathBuf
};

const STD: &'static str = "std";

fn main() {
    let to = env::var("PROFILE").unwrap();
    let to = PathBuf::from(format!("target/{}", to));

    let from = STD;
    let to = to.join(STD);

    if to.exists() {
        fs::remove_dir_all(&to).unwrap();
    }
    fs::create_dir(&to).unwrap();

    for from in fs::read_dir(from).unwrap() {
        let from = from.unwrap().path();
        let to = to.join(from.file_name().unwrap());

        fs::copy(from, to).unwrap();
    }
}