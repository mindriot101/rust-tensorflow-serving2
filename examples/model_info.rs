use std::collections::HashMap;
use structopt::StructOpt;
use tensorflow_serving::{ModelConfig, TensorflowServing};

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

    println!("Getting model status");
    let status = serving
        .model_status(&opts.model)
        .await
        .expect("fetching model status");
    println!("Got result: {:#?}", status);

    println!("Fetching metadata");
    let metadata = serving
        .model_metadata(&opts.model)
        .await
        .expect("fetching model metadata");
    println!("Got result: {:#?}", metadata);

    // XXX after reloading the config, the model will be reloaded and for some reason become not
    // available afterwards. We therefore don't run this as part of the example.

    // Build up a model config
    // let config = vec![ModelConfig {
    //     base_path: "/".to_string(),
    //     logging_config: None,
    //     model_platform: "tensorflow".to_string(),
    //     name: opts.model.clone(),
    //     model_type: 0,
    //     model_version_policy: None,
    //     version_labels: HashMap::new(),
    // }];

    // println!("Reloading model");
    // let response = serving.reload(config).await.expect("reloading model");
    // println!("Got result: {:#?}", response);
}
