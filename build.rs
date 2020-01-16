fn main() {
    let protos = &[
        "protos/tensorflow_serving/apis/prediction_service.proto",
        "protos/tensorflow_serving/apis/model_service.proto",
        "protos/tensorflow/core/lib/core/error_codes.proto",
    ];
    tonic_build::configure()
        .compile(protos, &["protos"])
        .unwrap()
}
