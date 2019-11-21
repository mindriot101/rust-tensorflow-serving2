use std::path::PathBuf;

fn main() {
    let protos = &["protos/tensorflow_serving/apis/prediction_service.proto"]
        .iter()
        .map(PathBuf::from)
        .collect::<Vec<_>>();

    tonic_build::configure()
        .compile(&protos, &[PathBuf::from("protos")])
        .unwrap()
}
