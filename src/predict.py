# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import joblib

from src.dataloader import DataReader
import config as conf


def Pred():
    
    os.environ['AWS_ACCESS_KEY_ID'] = conf.conn["data.aws.aws_access_key_id"]
    os.environ['AWS_SECRET_ACCESS_KEY'] = conf.conn["data.aws.aws_secret_access_key"]
    
    # Load Model for prediction
    ##------------------------------------------------------------------------------------------------------------------
    print("Prediction started!")

    model = joblib.load(conf.params['main_path'] +"/model_saving/"+'LightGBM_model.pkl')
    #X_test = joblib.load(conf.params['main_path'] +'X_test.pkl')
    y_test = joblib.load(conf.params['main_path'] +'y_test.pkl')

    predictions = model.predict(y_test)
    pd.Series(predictions)

    # Saving predictions
    ##------------------------------------------------------------------------------------------------------------------

    pd.DataFrame(predictions, columns=["Predictions"]).to_csv("predictions.csv",conf.params['predictions_path'],index=False)

    print("Model predicted and saved!")
    
