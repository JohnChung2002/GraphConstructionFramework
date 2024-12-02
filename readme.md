# Graph Construction Evaluation Framework
This framework evaluates the efficacy of different edge relationship on the effectiveness of model

The basis of this code is performed on CERT CMU r4.2 in the field of Insider Threat Detection, and can be modified according to match your use case
- each folder consist a template which whereby you can recreate it based on your use case 
- load_data folder is where the data loading script should be placed in 
- preprocess_data is where the script for the preprocessing of data loaded should be placed in 
- rules is where the script for the edge construction should be placed in, there is a utils script which can assist the edge construction
- graph is where the script to combine the processed data (features) and edges into graphs
- model is where you should load your training model 

The flow of data should be as follows
load_data -> preprocess_data -> rules -> graph -> model -> results

You can run the script with 
`python main.py --data_loader='path' --data_preprocessor='path' --edge_builder='path' --graph_builder='path' --model='path' --results='path'`
- --data_loader and its subsequent path should be a file in the load_data folder
- --data_preprocessor and its subsequent path should be a file in the preprocess_data folder
- --edge_builder and its subsequent path should be a file in the rules folder
- --graph_builder and its subsequent path should be a file in the graph folder
- --model and its subsequent path should be a file in the model folder
- --results can be any folder be it relative or absolute

The sample script is a example execution with a drafted model, which is not sufficiently trained yet.