import numpy as np
from flask import Flask, request, jsonify, render_template,url_for
# from tagger import find_tag,disambiguer
from sentiment_analysis import build_models,predict_sentiment

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/button_clicked', methods=['POST'])
def handle_button_click():
    # Your Python function code here
    # Example:
    res = build_models()
    if res == "success":
        return 'MODELS BUILDING COMPLETED!' 
    else:
        return "MODELS BUILDING FAILED!!"

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [str(x) for x in request.form.values()]
    input = request.form.values()
    # for i in input:
    #     input_string = i
    # print(input_string)
    output_result = []
    # input_words = input_string.split()

    # for i in input_words:
    #     output_result.append([i,find_tag(i)])

    result = predict_sentiment(input)

    images = {
        "happy": "happy",
        "sad" : "sad"
    }

    # output = {
    #     "logistic_reg": "happy",
    #     "svc": "happy",
    #     "naive bayes": "sad",
    #     "decision tree": "happy",
    #     "random forest": "sad"
    # }
    
    
    # round(prediction[0], 2)

    prediction_with_emojis = {model: images[prediction] for model, prediction in result.items()}

    # output_disambi  =  disambiguer(output)

    return render_template('index.html',build_output='MODELS BUILDING COMPLETED!', prediction_text=prediction_with_emojis)


if __name__ == "__main__":
    app.run(debug=True)