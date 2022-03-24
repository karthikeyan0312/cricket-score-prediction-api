import pickle as pk
import bz2
from flask import Flask, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS
from cachetools import cached, TTLCache
import numpy as np

app = Flask(__name__)
api = Api(app)

cache = TTLCache(maxsize=100, ttl=86400)

@cached(cache)
def load_file():
    sc=pk.load(open(r"/app/transform.pkl",'rb'))
    data = bz2.BZ2File(r"/app/randomforestmodelbz2.pbz2","rb")
    pkbz2 = pk.load(data)
    return pkbz2,sc
pkbz2,sc=load_file()
class status(Resource):    
    def get(self):
        try:
            return {'data': 'Api running'}
        except(error): 
            return {'data': error}

class Sum(Resource):
    def get(self, a, b):
        return jsonify({'data': a+b})

class model(Resource):
    def get(self,a,b,c,d,e):
        res=pkbz2.predict(sc.transform(np.array([[a,b,c,d,e]])))
        return jsonify({'Score': str(res)})
    
#100,0,13,50,50
api.add_resource(model, '/model/<int:a>,<int:b>,<int:c>,<int:d>,<int:e>')
api.add_resource(status,'/')
api.add_resource(Sum,'/add/<int:a>,<int:b>')

if __name__ == '__main__':
    app.run(debug=True)
