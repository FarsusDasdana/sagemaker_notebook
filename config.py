import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--main_path", type=str, default='s3://tutorial/simple_pipe/model')
parser.add_argument("--preped_path", type=str, default=None)
parser.add_argument("--training_data", type=str, default="diabetes")
parser.add_argument("--target_column", type=str, default=None)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--model_params_path", type=str, default='s3://tutorial/simple_pipe/model/model_params')
parser.add_argument("--save_the_model", type=bool, default=True)
parser.add_argument("--predictions_path", type=str, default="s3://tutorial/simple_pipe/model/predictions")
args, _ = parser.parse_known_args()

conn = {
    "aws_access_key_id": "###"
    "aws_secret_access_key": "###"
}

params = {
    'platform': 'aws', # options: 'local', 'aws'
    'main_path':  args.main_path,  
    'target_column': args.target_column, 
    'model_params_path': args.model_params_path 
    'save_the_model': args.save_the_model 
    'predictions_path': args.predictions_path 
}
data = {
    'data_type':'.csv',
    'training_data':args.training_data,
    'test_size':args.test_size,
    'pred':'location_product_predictions' 
}
