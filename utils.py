from imports import *
def extract_features(word):
    features = []

    contains_digit = [1] if any(c.isdigit() for c in word) else [0]
    is_digit = [1] if word.isdigit() else [0]

    is_punct = [1] if all(not c.isalnum() for c in word) else [0]

    suffix = word[-3:].ljust(3)
    suffix_features = [ord(c) / 1200 for c in suffix]

    prefix = word[:3].ljust(3)
    prefix_features = [ord(c) / 1200 for c in prefix]

    length = [min(len(word) / 15, 1.0)]
    has_alef = [1] if 'ا' in word else [0]
    has_waw = [1] if 'و' in word else [0]
    has_yeh = [1] if 'ي' in word else [0]

    features = (contains_digit + is_digit + is_punct + suffix_features +
               prefix_features + length + has_alef + has_waw + has_yeh)

    return features

def extract_features_for_sentence(sentence):
    features = []
    for word in sentence:
        features.append(extract_features(word))
    return features

def predict_pos_tags(sentence, model, word2idx, tag2idx, extract_features_for_sentence, extract_features, MAXLEN):
    idx2tag = {v: k for k, v in tag2idx.items()}

    words = sentence.split()

    X = [word2idx.get(w, word2idx.get("UNK", 0)) for w in words]
    X_padded = pad_sequences([X], maxlen=MAXLEN, padding="post", value=word2idx.get("PAD", 0))

    sent_features = extract_features_for_sentence(words)

    while len(sent_features) < MAXLEN:
        zeros_vector = [0] * len(extract_features(""))
        sent_features.append(zeros_vector)

    X_features = np.array([sent_features])

    X_padded = X_padded.astype(np.float32)
    X_features = X_features.astype(np.float32)

    if hasattr(model, "signatures"):
        predict_fn = model.signatures["serving_default"]

        inputs = {
            "word_input": tf.constant(X_padded, dtype=tf.float32),
            "feature_input": tf.constant(X_features, dtype=tf.float32)
        }

        predictions = predict_fn(**inputs)
        print("Clés sorties du modèle:", predictions.keys())  
        
        predictions = predictions['time_distributed_6'].numpy()

    else:
        predictions = model.predict([X_padded, X_features])

    pred_tags = []
    for i in range(min(len(words), MAXLEN)):
        pred_idx = np.argmax(predictions[0, i])
        if pred_idx in idx2tag:
            pred_tags.append(idx2tag[pred_idx])
        else:
            # Utiliser un tag par défaut si l'indice n'existe pas
            pred_tags.append(tag2idx.get("X", "UNK"))

    return pred_tags, words

def normalize_arabic_text(text):

    if not isinstance(text, str):
        return text

    text = normalize_unicode(text)

    text = araby.strip_tashkeel(text)

    text = araby.normalize_alef(text)

    text = araby.normalize_hamza(text)

    # Normalisation des ya et alif maqsura
    text = araby.normalize_ligature(text)

    # Suppression des espaces multiples
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def get_word_features(word):
    return {
        'word.isdigit()': word.isdigit(),
        'word.is_year': bool(re.match(r'^[0-9]{4}$', word)),
        'word.length': len(word),
        'word.prefix1': word[:1] if len(word) > 0 else '',
        'word.prefix2': word[:2] if len(word) > 1 else '',
        'word.suffix1': word[-1:] if len(word) > 0 else '',
        'word.suffix2': word[-2:] if len(word) > 1 else '',
        'word.has_digit': any(c.isdigit() for c in word),
        'word.has_punctuation': any(not c.isalnum() for c in word),
        'word.has_alef': 'ا' in word,
        'word.has_waw': 'و' in word,
        'word.has_ya': 'ي' in word,
        'word.has_ta_marbuta': 'ة' in word,
        'word.has_alif_lam': 'ال' in word,
    }

def get_contextual_features(words, i):
    features = {}

    # Mot précédent
    if i > 0:
        prev_word = normalize_arabic_text(words[i-1])
        features['prev_word'] = prev_word
        if len(prev_word) >= 2:
            features['prev_word.suffix2'] = prev_word[-2:]

    # Mot suivant
    if i < len(words) - 1:
        next_word = normalize_arabic_text(words[i+1])
        features['next_word'] = next_word
        if len(next_word) >= 2:
            features['next_word.prefix2'] = next_word[:2]

    return features

def extract_feature(words, i):
    word = words[i]
    normalized_word = normalize_arabic_text(word)

    features = {}
    features.update(get_word_features(normalized_word))
    features.update(get_contextual_features(words, i))

    return features

def prepare_sentence_features(sentence):
    words = sentence.split()
    X_sent = []

    for i in range(len(words)):
        X_sent.append(extract_feature(words, i))

    return X_sent

def predict_entities(model_path, sentence):
    crf = joblib.load(model_path)

    words = sentence.split()

    X_sent = prepare_sentence_features(sentence)

    y_pred = crf.predict([X_sent])[0]

    tagged_sentence = list(zip(words, y_pred))

    return tagged_sentence

def extract_named_entities(tagged_sentence):
    entities = {}
    current_entity = []
    current_tag = None

    for word, tag in tagged_sentence:
        if tag.startswith('B-'):
            if current_entity:
                entity_text = ' '.join(current_entity)
                entity_type = current_tag[2:]

                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(entity_text)

                current_entity = []

            current_entity.append(word)
            current_tag = tag

        elif tag.startswith('I-') and current_entity and tag[2:] == current_tag[2:]:
            current_entity.append(word)

        else:  # Tag 'O' ou nouvelle entité sans avoir fermé la précédente
            if current_entity:
                entity_text = ' '.join(current_entity)
                entity_type = current_tag[2:]  # Enlever le préfixe "B-" ou "I-"

                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(entity_text)

                current_entity = []
                current_tag = None

            if tag.startswith('B-'):  # Nouvelle entité
                current_entity.append(word)
                current_tag = tag

    # Traiter la dernière entité si elle existe
    if current_entity:
        entity_text = ' '.join(current_entity)
        entity_type = current_tag[2:]  # Enlever le préfixe "B-" ou "I-"

        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(entity_text)

    return entities

def test_model_on_sentence(model_path, sentence):
    tagged_sentence = predict_entities(model_path, sentence)

    entities = extract_named_entities(tagged_sentence)

    return entities, tagged_sentence

def safe_transform(label, encoder):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    else:
        return encoder.transform([encoder.classes_[0]])[0]

def preprocess_text_for_arabert(text):
    if isinstance(text, str):
        text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def fix_scaler_warning(distance, scaler):
    import pandas as pd
    distance_df = pd.DataFrame([[distance]], columns=['token_distance'])
    distance_scaled = scaler.transform(distance_df)[0][0]
    return distance_scaled

def extract_mentions_with_patterns(sentence):
    import re
    
    mentions = []
    
    propn_patterns = [
        r'[\u0621-\u064A]{2,}(?=\s|$|\.)',  
        r'\b[A-Z][a-z]+\b' 
    ]
    
    pronouns = ['هو', 'هي', 'هم', 'هن', 'أنت', 'أنا', 'نحن', 'إياه', 'إياها']
    
    determinants = ['ال', 'هذا', 'هذه', 'ذلك', 'تلك', 'هؤلاء', 'أولئك']
    
    for pattern in propn_patterns:
        for match in re.finditer(pattern, sentence):
            word = match.group()
            if len(word) > 2 and not word in ['في', 'من', 'إلى', 'على', 'أن', 'كان', 'قال']:
                mentions.append({
                    'text': word,
                    'start': match.start(),
                    'end': match.end(),
                    'label': 'NOUN_PROP',
                    'score': 0.8
                })
    
    for pronoun in pronouns:
        start_pos = 0
        while True:
            pos = sentence.find(pronoun, start_pos)
            if pos == -1:
                break
            mentions.append({
                'text': pronoun,
                'start': pos,
                'end': pos + len(pronoun),
                'label': 'PRON',
                'score': 0.9
            })
            start_pos = pos + 1
    
    for det in determinants:
        pattern = det + r'[\u0621-\u064A]{2,}'
        for match in re.finditer(pattern, sentence):
            mentions.append({
                'text': match.group(),
                'start': match.start(),
                'end': match.end(),
                'label': 'NOUN',
                'score': 0.7
            })
    
    unique_mentions = []
    seen_positions = set()
    
    for mention in mentions:
        pos_key = (mention['start'], mention['end'])
        if pos_key not in seen_positions:
            unique_mentions.append(mention)
            seen_positions.add(pos_key)
    
    return sorted(unique_mentions, key=lambda x: x['start'])

def predict_pos_tags1(sentence, pos_model, word2idx, tag2idx, MAXLEN=112):
    idx2tag = {v: k for k, v in tag2idx.items()}
    
    words = sentence.split()
    
    X = [word2idx.get(w, word2idx.get("UNK", 0)) for w in words]
    X_padded = pad_sequences([X], maxlen=MAXLEN, padding="post", value=word2idx.get("PAD", 0))
    
    sent_features = extract_features_for_sentence(words)
    
    while len(sent_features) < MAXLEN:
        zeros_vector = [0] * len(extract_features(""))
        sent_features.append(zeros_vector)
    
    X_features = np.array([sent_features])
    
    X_padded = X_padded.astype(np.float32)
    X_features = X_features.astype(np.float32)
    
    if hasattr(pos_model, "signatures"):
        predict_fn = pos_model.signatures["serving_default"]
        
        inputs = {
            "word_input": tf.constant(X_padded, dtype=tf.float32),
            "feature_input": tf.constant(X_features, dtype=tf.float32)
        }
        
        predictions = predict_fn(**inputs)
        predictions = list(predictions.values())[0].numpy()
    else:
        predictions = pos_model.predict([X_padded, X_features])
    
    pred_tags = []
    for i in range(min(len(words), MAXLEN)):
        pred_idx = np.argmax(predictions[0, i])
        if pred_idx in idx2tag:
            pred_tags.append(idx2tag[pred_idx])
        else:
            pred_tags.append("UNK")
    
    return pred_tags, words

def extract_mentions_with_pos_integration(sentence, pos_model, word2idx, tag2idx, MAXLEN=112):
    
    pos_tags, words = predict_pos_tags1(sentence, pos_model, word2idx, tag2idx, MAXLEN)
    
    print(f"Analyse POS:")
    for word, tag in zip(words, pos_tags):
        print(f"  - '{word}' -> {tag}")
    
    mentions = []
    current_pos = 0
    
    for i, (word, pos_tag) in enumerate(zip(words, pos_tags)):
        word_start = sentence.find(word, current_pos)
        if word_start == -1:
            word_start = current_pos
        word_end = word_start + len(word)
        current_pos = word_end
        
        is_mention = False
        mention_type = None
        confidence = 0.0
        
        if pos_tag in ['NOUN', 'NN', 'NNS']: 
            is_mention = True
            mention_type = 'NOUN'
            confidence = 0.4
        elif pos_tag in ['PROPN', 'NNP', 'NNPS']: 
            is_mention = True
            mention_type = 'NOUN_PROP'
            confidence = 0.4
        elif pos_tag in ['PRON', 'PRP', 'PRP$']:  
            is_mention = True
            mention_type = 'PRON'
            confidence = 0.95
        elif pos_tag in ['DET']:  
            if i + 1 < len(pos_tags) and pos_tags[i + 1] in ['NOUN', 'NN', 'NNS', 'PROPN', 'NNP']:
                is_mention = True
                mention_type = 'DET_NOUN'
                confidence = 0.7
        
        if not is_mention:
            if len(word) > 3 and re.match(r'[\u0621-\u064A]+', word):
                is_mention = True
                mention_type = 'NOUN'
                confidence = 0.4
            
            elif re.match(r'[\u0621-\u064A]{2,}', word) and word[0].isupper():
                is_mention = True
                mention_type = 'NOUN_PROP'
                confidence = 0.4
        
        if is_mention:
            mentions.append({
                'text': word,
                'start': word_start,
                'end': word_end,
                'label': mention_type,
                'pos_tag': pos_tag,
                'score': confidence,
                'word_index': i
            })
    
    return mentions

def extract_mentions_hybrid_with_pos(sentence, pos_model, word2idx, tag2idx, MAXLEN=112):
    
    pos_mentions = extract_mentions_with_pos_integration(sentence, pos_model, word2idx, tag2idx, MAXLEN)
    
    pattern_mentions = extract_mentions_with_patterns(sentence)
    
    all_mentions = pos_mentions + pattern_mentions
    
    unique_mentions = {}
    
    for mention in all_mentions:
        key = (mention['start'], mention['end'])
        
        if key not in unique_mentions:
            unique_mentions[key] = mention
        else:
            existing = unique_mentions[key]
            if mention['score'] > existing['score']:
                mention.update({k: v for k, v in existing.items() if k not in mention})
                unique_mentions[key] = mention
            else:
                existing.update({k: v for k, v in mention.items() if k not in existing})
    
    final_mentions = sorted(unique_mentions.values(), key=lambda x: x['start'])
    
    return final_mentions

def predict_coreference_with_pos_only(sentence, coref_model, pos_model, word2idx, tag2idx, 
                                    tokenizer, scaler, m1_type_encoder, m2_type_encoder, 
                                    max_len=50, MAXLEN=112):
    
    mentions = extract_mentions_hybrid_with_pos(sentence, pos_model, word2idx, tag2idx, MAXLEN)
    
    print(f"\nMentions extraites (POS + Patterns): {len(mentions)}")
    for m in mentions:
        pos_info = f" [POS: {m.get('pos_tag', 'N/A')}]" if 'pos_tag' in m else ""
        pattern_info = f" [Pattern]" if 'pos_tag' not in m else ""
        print(f"- '{m['text']}' ({m['label']}) at {m['start']}-{m['end']} [score: {m['score']:.3f}]{pos_info}{pattern_info}")

    if len(mentions) < 2:
        print("Pas assez de mentions (< 2) pour la coréférence")
        return []

    mention_pairs = list(combinations(mentions, 2))
    print(f"\nNombre de paires à analyser: {len(mention_pairs)}")

    all_inputs = {
        'input_m1_text': [],
        'input_m2_text': [],
        'input_m1_type': [],
        'input_m2_type': [],
        'input_distances': []
    }

    for m1, m2 in mention_pairs:
        text1 = preprocess_text_for_arabert(m1['text'])
        text2 = preprocess_text_for_arabert(m2['text'])

        seq1 = tokenizer.texts_to_sequences([text1])
        seq2 = tokenizer.texts_to_sequences([text2])

        if not seq1 or not seq1[0]:
            seq1 = [[1]]
        if not seq2 or not seq2[0]:
            seq2 = [[1]]

        seq1_padded = pad_sequences(seq1, maxlen=max_len, padding='post')[0]
        seq2_padded = pad_sequences(seq2, maxlen=max_len, padding='post')[0]

        type1_label = m1.get('pos_tag', m1['label'])
        type2_label = m2.get('pos_tag', m2['label'])
        
        type1 = safe_transform(type1_label, m1_type_encoder)
        type2 = safe_transform(type2_label, m2_type_encoder)

        distance = abs(m2['start'] - m1['end'])
        distance_scaled = fix_scaler_warning(distance, scaler)

        all_inputs['input_m1_text'].append(seq1_padded)
        all_inputs['input_m2_text'].append(seq2_padded)
        all_inputs['input_m1_type'].append([type1])
        all_inputs['input_m2_type'].append([type2])
        all_inputs['input_distances'].append([distance_scaled])

    for key in all_inputs:
        all_inputs[key] = np.array(all_inputs[key])

    predictions = coref_model.predict(all_inputs, verbose=0)

    coref_pairs = []
    threshold = 0.5

    print(f"\nAnalyse des paires:")
    for i, (m1, m2) in enumerate(mention_pairs):
        score = predictions[i][0]
        pos1 = m1.get('pos_tag', 'N/A')
        pos2 = m2.get('pos_tag', 'N/A')
        print(f"'{m1['text']}' ({m1['label']}/{pos1}) ↔ '{m2['text']}' ({m2['label']}/{pos2}) -> Score: {score:.4f}")

        if score > threshold:
            coref_pairs.append({
                'mention1': m1,
                'mention2': m2,
                'score': float(score)
            })

    return coref_pairs