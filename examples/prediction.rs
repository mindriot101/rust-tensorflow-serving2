use std::collections::HashMap;
use std::path::PathBuf;
use structopt::StructOpt;
use tensorflow_serving::TensorflowServing;

#[derive(StructOpt, Debug)]
struct Opts {
    #[structopt(parse(from_os_str))]
    image: PathBuf,
    #[structopt(short = "m", long = "model")]
    model: String,
    #[structopt(long = "version")]
    model_version: Option<i64>,
    #[structopt(long = "hostname", default_value = "127.0.0.1")]
    hostname: String,
    #[structopt(long = "port", default_value = "9000")]
    port: u16,
}

#[tokio::main]
async fn main() {
    let opts = Opts::from_args();

    let img = image::open(opts.image).expect("reading image");

    let mut serving = TensorflowServing::new()
        .await
        .hostname(opts.hostname)
        .port(opts.port)
        .build()
        .unwrap();

    let model_definition = tensorflow_serving::ModelDescription {
        name: opts.model,
        version: opts.model_version,
    };

    let result = serving
        .predict_with_preprocessing(img, model_definition, |value| value / 255.)
        .await
        .expect("error predicting");
    println!("{:#?}", result);
}
