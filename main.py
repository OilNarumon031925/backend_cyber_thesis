"""
 The main.py is file of backend api used to communicating with Application
"""


from fastapi import FastAPI
import uvicorn
import joblib
from api.model.detect_model import DetectionModel
from src.helper.ml_helper import prepare
from api.model.response_model import ReponseModel


app=FastAPI()


@app.on_event("startup")
async def startupApp():
    print("====== startup ======")

    """
        Make the vectorizer is a global variable for all method call access it
    """
    global vectorizer

    """
        Make the model to a global variable for all method can access the model of machine learning
    """
    global model

    try:
        """
            The vectorizer and model path variable are used to set the paths that need to access the model or vactorizer file
        """
        vectorizer_path = "vectorizer\\vectorizer.pkl"
        model_path = "model\\finalized_model.pkl"

        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
    except Exception as e:
        """
            If the vectorizer and model cannot load the backend will error here
        """
        print(e)
    


@app.get("/")
def welcome():
    return {
        "message": "welcome"
    }


@app.post("/detect/bully")
async def detect(dect_model: DetectionModel):
    try:

        print(dect_model.text)
        df2 = prepare(dect_model.text, vectorizer=vectorizer)
        print(df2)
        y_pred = model.predict(df2)

        pred = "Bullying" if y_pred[0] == 1 else "Not Bullying"

        return {
            "message": "pred success",
            "data": {
                "pred": pred
            },
            "status": 200,
        }
    except Exception as e:
        print(e)

        return {
            "message": "Internale Server Erorr",
            "data": None,
            "status": 500,
        }

        # return ReponseModel(message="Internal Server Error", status=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
