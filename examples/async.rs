use futures::future::join_all;
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
    env_logger::init();
    let opts = Opts::from_args();

    let serving = TensorflowServing::new()
        .hostname(opts.hostname)
        .port(opts.port)
        .build()
        .await
        .unwrap();

    println!("Tensorflow serving client created");

    let mut futs = Vec::new();

    let mut s = serving.clone();
    let m = opts.model.clone();
    futs.push(tokio::spawn(async move {
        let status = s.model_status(m).await.expect("fetching model status");
        println!("Got result: {:#?}", status);
    }));

    let mut s = serving.clone();
    let m = opts.model.clone();
    futs.push(tokio::spawn(async move {
        println!("Fetching metadata");
        let metadata = s.model_metadata(m).await.expect("fetching model metadata");
        println!("Got result: {:#?}", metadata);
    }));

    let results = join_all(&mut futs).await;
    for res in results {
        match res {
            Ok(_) => {}
            Err(e) => eprintln!("failure: {:?}", e),
        }
    }

    println!("Finished");
}
