document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loader = document.getElementById('loader');
    const resultContainer = document.getElementById('resultContainer');

    // Enhanced loading state
    function showLoader() {
        loader.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        resultContainer.style.display = 'none';
    }

    function hideLoader() {
        loader.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze';
    }

    // Enhanced error handling
    function showError(message) {
        resultContainer.innerHTML = `
            <div class="error-container">
                <i class="fas fa-exclamation-triangle error-icon"></i>
                <h3>Analysis Error</h3>
                <p>${message}</p>
                <button onclick="location.reload()" class="retry-btn">
                    <i class="fas fa-redo"></i> Try Again
                </button>
            </div>
        `;
        resultContainer.style.display = 'block';
    }

    analyzeBtn.addEventListener('click', () => {
        const sentence = document.getElementById('textInput').value.trim();
        const selectedTask = document.querySelector('input[name="taskRadio"]:checked');

        if (!sentence) {
            showError("Please enter some text to analyze!");
            return;
        }

        if (!selectedTask) {
            showError("Please select an analysis type!");
            return;
        }

        showLoader();

        if (selectedTask.value === 'pos') {
            fetch('/predict_pos', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence: sentence })
            })
            .then(response => {
                if (!response.ok) throw new Error("Server error occurred");
                return response.json();
            })
            .then(data => {
                displayPOSResults(data);
            })
            .catch(err => {
                console.error("POS Error:", err);
                showError("An error occurred during Part-of-Speech analysis.");
            })
            .finally(() => {
                hideLoader();
            });
        }
        else if (selectedTask.value === 'ner') {
            fetch('/predict_ner', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence: sentence })
            })
            .then(response => {
                if (!response.ok) throw new Error("Server error occurred");
                return response.json();
            })
            .then(data => {
                displayNERResults(data);
            })
            .catch(err => {
                console.error("NER Error:", err);
                showError("An error occurred during Named Entity Recognition analysis.");
            })
            .finally(() => {
                hideLoader();
            });
        }
        else if (selectedTask.value === 'coref') {
            fetch('/predict_coreference', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    sentence: sentence,
                    threshold: 0.7  
                })
            })
            .then(response => {
                if (!response.ok) throw new Error("Server error occurred");
                return response.json();
            })
            .then(data => {
                displayCoreferenceResults(data);
            })
            .catch(err => {
                console.error("Coreference Error:", err);
                showError("An error occurred during Coreference Resolution analysis.");
            })
            .finally(() => {
                hideLoader();
            });
        }
    });

    // Enhanced POS Results Display
    function displayPOSResults(data) {
        const words = data.words;
        const tags = data.tags;

        let html = `
            <div class="results-header">
                <h3><i class="fas fa-sitemap"></i> Part-of-Speech Analysis Results</h3>
                <div class="results-stats">
                    <span class="stat-item">
                        <i class="fas fa-list-ol"></i> ${words.length} words analyzed
                    </span>
                </div>
            </div>
            <div class="pos-results-container">
        `;

        // Group by POS tags for better visualization
        const posGroups = {};
        for (let i = 0; i < words.length; i++) {
            if (!posGroups[tags[i]]) {
                posGroups[tags[i]] = [];
            }
            posGroups[tags[i]].push(words[i]);
        }

        // Display word-by-word analysis
        html += '<div class="pos-word-analysis">';
        for (let i = 0; i < words.length; i++) {
            const color = getPOSColor(tags[i]);
            html += `
                <div class="pos-word-item" style="border-left: 4px solid ${color};">
                    <div class="word-text">${words[i]}</div>
                    <div class="pos-tag" style="background-color: ${color};">${tags[i]}</div>
                    <div class="pos-description">${getPOSDescription(tags[i])}</div>
                </div>
            `;
        }
        html += '</div>';

        // Display POS tag summary
        html += '<div class="pos-summary"><h4><i class="fas fa-chart-pie"></i> POS Tag Summary</h4>';
        Object.entries(posGroups).forEach(([tag, wordList]) => {
            const color = getPOSColor(tag);
            html += `
                <div class="pos-group" style="border-left: 4px solid ${color};">
                    <div class="pos-group-header">
                        <span class="pos-tag-name" style="background-color: ${color};">${tag}</span>
                        <span class="pos-count">${wordList.length} word(s)</span>
                    </div>
                    <div class="pos-words">${wordList.join(', ')}</div>
                </div>
            `;
        });
        html += '</div></div>';

        document.getElementById('nerResult').style.display = 'none';
        document.getElementById('corefResult').style.display = 'none';
        document.getElementById('posResult').style.display = 'block';
        document.getElementById('posDisplay').innerHTML = html;
        document.getElementById('resultContainer').style.display = 'block';
        
    }

    // Enhanced NER Results Display with clearer descriptions
    function displayNERResults(data) {
        console.log("NER Data received:", data);
        const tagged_sentence = data.tagged_sentence;
        
        let html = `
            <div class="results-header">
                <h3><i class="fas fa-tags"></i> Named Entity Recognition Results</h3>
                <div class="results-stats">
                    <span class="stat-item">
                        <i class="fas fa-list-ol"></i> ${tagged_sentence.length} tokens analyzed
                    </span>
                </div>
            </div>
            <div class="ner-results-container">
        `;

        // Group entities by type with clearer categorization
        const entityGroups = {};
        const entities = tagged_sentence.filter(item => item.tag !== 'O');
        
        entities.forEach(item => {
            const entityType = getEntityType(item.tag);
            if (!entityGroups[entityType]) {
                entityGroups[entityType] = [];
            }
            entityGroups[entityType].push(item.word);
        });

        // Display word-by-word analysis
        html += '<div class="ner-word-analysis">';
        tagged_sentence.forEach(({word, tag}) => {
            const color = getNERColor(tag);
            const isEntity = tag !== 'O';
            const description = getNERDescription(tag);
            html += `
                <div class="ner-word-item ${isEntity ? 'entity' : 'non-entity'}" style="border-left: 4px solid ${color};">
                    <div class="word-text">${word}</div>
                    <div class="ner-tag" style="background-color: ${color};">${tag}</div>
                    <div class="ner-description">${description}</div>
                </div>
            `;
        });
        html += '</div>';

        // Display entities summary with clearer grouping
        if (Object.keys(entityGroups).length > 0) {
            html += '<div class="ner-summary"><h4><i class="fas fa-bookmark"></i> Identified Entities</h4>';
            Object.entries(entityGroups).forEach(([entityType, wordList]) => {
                const color = getEntityTypeColor(entityType);
                html += `
                    <div class="ner-group" style="border-left: 4px solid ${color};">
                        <div class="ner-group-header">
                            <span class="ner-tag-name" style="background-color: ${color};">${entityType}</span>
                            <span class="ner-count">${wordList.length} entity(ies)</span>
                        </div>
                        <div class="ner-entities">${[...new Set(wordList)].join(', ')}</div>
                    </div>
                `;
            });
            html += '</div>';
        } else {
            html += `
                <div class="no-entities">
                    <i class="fas fa-info-circle"></i>
                    <p>No named entities detected in the text.</p>
                </div>
            `;
        }

        html += '</div>';

        document.getElementById('posResult').style.display = 'none';
        document.getElementById('corefResult').style.display = 'none';
        document.getElementById('nerResult').style.display = 'block';
        document.getElementById('nerDisplay').innerHTML = html;
        document.getElementById('resultContainer').style.display = 'block';
    }

    // Enhanced Coreference Results Display
    function displayCoreferenceResults(data) {
        console.log("Coreference data received:", data);
        
        const corefPairs = data.coreference_pairs;
        const totalPairs = data.total_pairs;
        const threshold = data.threshold_used;

        let html = `
            <div class="results-header">
                <h3><i class="fas fa-link"></i> Coreference Resolution Results</h3>
                <div class="results-stats">
                    <span class="stat-item">
                        <i class="fas fa-percentage"></i> Threshold: ${threshold}
                    </span>
                    <span class="stat-item">
                        <i class="fas fa-link"></i> ${totalPairs} coreference(s) found
                    </span>
                </div>
            </div>
            <div class="coref-results-container">
        `;

        if (totalPairs === 0) {
            html += `
                <div class="no-coreferences">
                    <i class="fas fa-info-circle"></i>
                    <h4>No Coreferences Detected</h4>
                    <p>No coreferences were found above the threshold of ${threshold}.</p>
                    <small>Try lowering the threshold or using a different text.</small>
                </div>
            `;
        } else {
            corefPairs.forEach((pair, index) => {
                const mention1 = pair.mention1;
                const mention2 = pair.mention2;
                const score = (pair.score * 100).toFixed(2);
                
                const confidenceLevel = getConfidenceLevel(score);
                const scoreColor = getScoreColor(score);
                
                html += `
                    <div class="coref-pair-card">
                        <div class="coref-pair-header">
                            <div class="pair-number">
                                <i class="fas fa-link"></i> Pair ${index + 1}
                            </div>
                            <div class="confidence-badge ${confidenceLevel.class}" style="background-color: ${scoreColor};">
                                <i class="fas fa-chart-line"></i> ${score}% ${confidenceLevel.label}
                            </div>
                        </div>
                        
                        <div class="coref-pair-content">
                            <div class="mention-box mention-1">
                                <div class="mention-text">"${mention1.text}"</div>
                                <div class="mention-details">
                                    <span class="mention-type">${mention1.label}</span>
                                    <span class="mention-position">Position: ${mention1.start}-${mention1.end}</span>
                                </div>
                            </div>
                            
                            <div class="coref-connection">
                                <div class="connection-line"></div>
                                <div class="connection-icon">
                                    <i class="fas fa-arrows-alt-h"></i>
                                </div>
                                <div class="connection-label">${confidenceLevel.description}</div>
                            </div>
                            
                            <div class="mention-box mention-2">
                                <div class="mention-text">"${mention2.text}"</div>
                                <div class="mention-details">
                                    <span class="mention-type">${mention2.label}</span>
                                    <span class="mention-position">Position: ${mention2.start}-${mention2.end}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            });
        }

        html += '</div>';

        document.getElementById('posResult').style.display = 'none';
        document.getElementById('nerResult').style.display = 'none';
        document.getElementById('corefResult').style.display = 'block';
        document.getElementById('corefDisplay').innerHTML = html;
        document.getElementById('resultContainer').style.display = 'block';
        document.getElementById('backBtn').style.display = 'inline-block';
    }

    // Helper functions for colors and descriptions
    function getPOSColor(tag) {
        const colors = {
            'NOUN': '#3498db',
            'VERB': '#e74c3c',
            'ADJ': '#2ecc71',
            'ADV': '#f39c12',
            'PRON': '#9b59b6',
            'DET': '#1abc9c',
            'PREP': '#e67e22',
            'CONJ': '#34495e',
            'PUNCT': '#95a5a6'
        };
        return colors[tag] || '#7f8c8d';
    }

    function getPOSDescription(tag) {
        const descriptions = {
            'NOUN': 'Noun',
            'VERB': 'Verb',
            'ADJ': 'Adjective',
            'ADV': 'Adverb',
            'PRON': 'Pronoun',
            'DET': 'Determiner',
            'PREP': 'Preposition',
            'CONJ': 'Conjunction',
            'PUNCT': 'Punctuation'
        };
        return descriptions[tag] || 'Other';
    }

    // Enhanced NER color function with BIO tagging support
    function getNERColor(tag) {
        // Extract the entity type from BIO tags (B-PER, I-PER, etc.)
        const entityType = getEntityType(tag);
        const colors = {
            'PERSON': '#e74c3c',
            'LOCATION': '#2ecc71', 
            'ORGANIZATION': '#3498db',
            'MISCELLANEOUS': '#f39c12',
            'OUTSIDE': '#95a5a6'
        };
        return colors[entityType] || '#7f8c8d';
    }

    // Enhanced NER description function with clear explanations
    function getNERDescription(tag) {
        const descriptions = {
            'O': 'Outside (not an entity)',
            'B-PERS': 'Beginning - Person\'s name',
            'I-PERS': 'Inside - Person\'s name (continued)',
            'B-LOC': 'Beginning - Location name', 
            'I-LOC': 'Inside - Location name (continued)',
            'B-ORG': 'Beginning - Organization name',
            'I-ORG': 'Inside - Organization name (continued)',
            'B-MISC': 'Beginning - Miscellaneous entity',
            'I-MISC': 'Inside - Miscellaneous entity (continued)',
            'PERS': 'Person',
            'LOC': 'Location',
            'ORG': 'Organization', 
            'MISC': 'Miscellaneous'
        };
        return descriptions[tag] || 'Unknown entity type';
    }

    // Helper function to extract entity type from BIO tags
    function getEntityType(tag) {
        if (tag === 'O') return 'OUTSIDE';
        if (tag.startsWith('B-PERS') || tag.startsWith('I-PERS') || tag === 'PERS') return 'PERSON';
        if (tag.startsWith('B-LOC') || tag.startsWith('I-LOC') || tag === 'LOC') return 'LOCATION';
        if (tag.startsWith('B-ORG') || tag.startsWith('I-ORG') || tag === 'ORG') return 'ORGANIZATION';
        if (tag.startsWith('B-MISC') || tag.startsWith('I-MISC') || tag === 'MISC') return 'MISCELLANEOUS';
        return 'UNKNOWN';
    }

    // Color function for entity types in summary
    function getEntityTypeColor(entityType) {
        const colors = {
            'PERSON': '#e74c3c',
            'LOCATION': '#2ecc71',
            'ORGANIZATION': '#3498db', 
            'MISCELLANEOUS': '#f39c12',
            'OUTSIDE': '#95a5a6',
            'UNKNOWN': '#7f8c8d'
        };
        return colors[entityType] || '#7f8c8d';
    }

    function getConfidenceLevel(score) {
        if (score >= 80) return { class: 'high', label: 'High Confidence', description: 'Strong coreference' };
        if (score >= 60) return { class: 'medium', label: 'Medium Confidence', description: 'Likely coreference' };
        if (score >= 40) return { class: 'low', label: 'Low Confidence', description: 'Possible coreference' };
        return { class: 'very-low', label: 'Very Low', description: 'Weak coreference' };
    }

    function getScoreColor(score) {
        if (score >= 80) return '#27ae60';
        if (score >= 60) return '#2ecc71';
        if (score >= 40) return '#f39c12';
        return '#e74c3c';
    }
});