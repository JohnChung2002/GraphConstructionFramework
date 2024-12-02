from evaluator import evaluator
from datetime import datetime

data_loader = "cert_r42"
data_loader_args = {
    "logon" : { "path" : "/fred/oz382/packaged_dataset/CERT/r4.2/logon.feather", "type" : "feather" },
    "device" : { "path" : "/fred/oz382/packaged_dataset/CERT/r4.2/device.feather", "type" : "feather" },
    "http" : { "path" : "/fred/oz382/packaged_dataset/CERT/r4.2/http.feather", "type" : "feather" },
    "file" : { "path" : "/fred/oz382/packaged_dataset/CERT/r4.2/file.feather", "type" : "feather" },
    "answers" : [
        "/fred/oz382/dataset/CERT/answers/r4.2-1",
        "/fred/oz382/dataset/CERT/answers/r4.2-2",
        "/fred/oz382/dataset/CERT/answers/r4.2-3"
    ]
}
data_preprocessor = "log2vec"
edge_builder = "log2vec"
graph_builder = "log2vec"
model = "test"
results = "results"

evaluator(data_loader, data_loader_args, data_preprocessor, edge_builder, graph_builder, model, results)