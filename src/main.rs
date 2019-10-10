pub mod tensorflow {
    tonic::include_proto!("tensorflow");
    pub mod tensorflow_serving {
        tonic::include_proto!("tensorflow.serving");
    }
}

fn main() {
    println!("Hello, world!");
}
