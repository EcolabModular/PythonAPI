from flask import Flask,jsonify
app = Flask(__name__)


@app.route('/')
def hello_world():
    return jsonify({"Message":"This route has no data","data": []})
from alldata import ML

@app.route('/original/<string:data_set>')
def getInitialData(data_set):
    return jsonify({"name": "Original","data": ML.GetOriginalData(data_set)})

@app.route('/armodel/<string:data_set>')
def getARData(data_set):
    return jsonify({"data": ML.GetARPredictions(data_set)})

@app.route('/arima/<string:data_set>')
def getARIMAData(data_set):
    return jsonify({"data": ML.GetArimaPredictions(data_set)})

if __name__ == '__main__':
    app.run(debug=True,port = 5000)
