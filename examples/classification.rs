use std::collections::HashMap;
use structopt::StructOpt;
use tensorflow_serving::TensorflowServing;

#[derive(StructOpt, Debug)]
struct Opts {
    #[structopt(short = "m", long = "model")]
    model: String,
    #[structopt(long = "hostname", default_value = "127.0.0.1")]
    hostname: String,
    #[structopt(long = "port", default_value = "9000")]
    port: u16,
}

#[tokio::main]
async fn main() {
    let opts = Opts::from_args();

    let mut serving = TensorflowServing::new()
        .await
        .hostname(opts.hostname)
        .port(opts.port)
        .build()
        .unwrap();

    let data = vec![1f32, 2f32, 3f32];

    let mut data_binding = HashMap::new();
    data_binding.insert("foobar", data);

    let result = serving
        .classify("resnet", data_binding)
        .await
        .expect("error classifying");
    println!("{:#?}", result);
}
