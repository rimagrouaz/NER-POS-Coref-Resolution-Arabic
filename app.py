from imports import *
import utils
app = Flask(__name__)

MODEL_DIR = os.environ.get('MODEL_DIR', 'model')
POS_MODEL_PATH = os.environ.get('POS_MODEL_PATH', os.path.join(MODEL_DIR, 'modele_BILSTM-CNN'))
NER_MODEL_PATH = os.environ.get('NER_MODEL_PATH', os.path.join(MODEL_DIR, 'modele_CRF_NER.pkl'))
COREF_MODEL_PATH = os.environ.get('COREF_MODEL_PATH', os.path.join(MODEL_DIR, 'Coref.h5'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_pos', methods=['POST'])
def predict_pos():
    pos_model = tf.saved_model.load(POS_MODEL_PATH)

    data = request.get_json()
    sentence = data.get('sentence', '')
    with open(os.path.join(MODEL_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label2idx.pkl'), 'rb') as f:
        tag2idx = pickle.load(f)
 
    pred_tags, words = utils.predict_pos_tags(sentence, pos_model, word2idx, tag2idx, utils.extract_features_for_sentence, utils.extract_features, 112)
    
    response = {'words': words, 'tags': pred_tags}
    return jsonify(response)

@app.route('/predict_ner', methods=['POST'])
def predict_ner():
    data = request.get_json()
    sentence = data.get('sentence', '')

    entities, tagged_sentence = utils.test_model_on_sentence(NER_MODEL_PATH, sentence)

    response = {
        'entities': entities,
        'tagged_sentence': [{'word': w, 'tag': t} for w, t in tagged_sentence]
    }
    return jsonify(response)

@app.route('/predict_coreference', methods=['POST'])
def predict_coreference():
    coref_model = tf.keras.models.load_model(COREF_MODEL_PATH)
    pos_model = tf.saved_model.load(POS_MODEL_PATH)   

    data = request.get_json()
    sentence = data.get('sentence', '')
        
    if not sentence.strip():
        return jsonify({'error': 'Sentence is required'}), 400
        
    with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'm1_type_encoder.pkl'), 'rb') as f:
        m1_type_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'm2_type_encoder.pkl'), 'rb') as f:
        m2_type_encoder = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'word2idx.pkl'), 'rb') as f:
        word2idx = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label2idx.pkl'), 'rb') as f:
        tag2idx = pickle.load(f)

    coref_pairs = utils.predict_coreference_with_pos_only(
            sentence=sentence,
            coref_model=coref_model,
            pos_model=pos_model,
            word2idx=word2idx,
            tag2idx=tag2idx,
            tokenizer=tokenizer,
            m1_type_encoder=m1_type_encoder,
            m2_type_encoder=m2_type_encoder,
            scaler=scaler,
            max_len=50,
            MAXLEN=112
        )

        # Préparer la réponse
    response = {
            'sentence': sentence,
            'coreference_pairs': coref_pairs,
            'total_pairs': len(coref_pairs)
        }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) 