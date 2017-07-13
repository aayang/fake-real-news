from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def my_index():
    return render_template("index.html")

@app.route('/model')
def my_model():
    return render_template("model.html")

#get url, scrape article text, feed into model, return prediction
@app.route('/', methods=['POST'])
def my_form_post():
    TXT = request.form['text'] #take url from user input

    #unpack and deploy trained count vectorizer
    count_vect = joblib.load('vectorizer_final.pkl')
    X_train_counts = count_vect.fit_transform([TXT])
    tf_transformer = TfidfTransformer()
    X_train_tfidf = tf_transformer.fit_transform(X_train_counts)

    #unpack and run trained model
    clf = joblib.load('mnnb_model_final.pkl')
    pred = clf.predict(X_train_tfidf)
    prob = clf.predict_proba(X_train_tfidf)
    pred_out = pred[0].decode('utf-8')
    if prob[0][0] >= .5:
        prob_out = str(round(prob[0][0]*100, 1))
    else:
        prob_out = str(round(prob[0][1]*100, 1))
    print(pred)
    return render_template("index.html")

if __name__ == '__main__':
    app.run()
