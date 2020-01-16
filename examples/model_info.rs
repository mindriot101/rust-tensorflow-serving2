use std::path::PathBuf;
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
        .hostname(opts.hostname)
        .port(opts.port)
        .build()
        .await
        .unwrap();

    println!("Tensorflow serving client created");

    println!("Sending request");
    let status = serving.model_status(&opts.model).await.expect("fetching model status");
    println!("Got result: {:#?}", status);
}
