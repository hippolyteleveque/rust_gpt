use regex::Regex;
use std::collections::HashMap;

pub struct SimpleTokenizerV1 {
    tok_to_ix: HashMap<String, usize>,
    ix_to_tok: HashMap<usize, String>,
}

impl SimpleTokenizerV1 {
    pub fn new(vocab: HashMap<String, usize>) -> SimpleTokenizerV1 {
        let mut ix_to_tok = HashMap::new();
        for (token, ix) in &vocab {
            // first version we clone the strings
            ix_to_tok.insert(*ix, token.clone());
        }
        SimpleTokenizerV1 {
            ix_to_tok,
            tok_to_ix: vocab,
        }
    }

    pub fn tok_to_ix(&self) -> &HashMap<String, usize> {
        &self.tok_to_ix
    }

    pub fn ix_to_tok(&self) -> &HashMap<usize, String> {
        &self.ix_to_tok
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(r#"([,.?_!"()\']|--|\s)"#)
            .expect("This should be a valid regular expression.");

        let mut preprocessed = Vec::new();
        let mut left = 0;
        for m in re.find_iter(text) {
            let start = m.start();
            let end = m.end();
            if text[left..start].trim() != "" {
                preprocessed.push(text[left..start].trim());
            }
            if m.as_str().trim() != "" {
                preprocessed.push(m.as_str());
            }
            left = end;
        }
        let res = preprocessed
            .iter()
            .map(|token| {
                *self
                    .tok_to_ix
                    .get(*token)
                    .unwrap_or_else(|| panic!("Failed to find token: {:?}", token))
            })
            .collect();
        res
    }

    pub fn decode(&self, ixs: Vec<usize>) -> String {
        let tokens: Vec<&str> = ixs
            .iter()
            .map(|ix| &self.ix_to_tok.get(ix).unwrap()[..])
            .collect();
        let result = tokens.join(" ");
        result.to_string()
    }
}

pub struct SimpleTokenizerV2 {
    tok_to_ix: HashMap<String, usize>,
    ix_to_tok: HashMap<usize, String>,
}

impl SimpleTokenizerV2 {
    pub fn new(vocab: HashMap<String, usize>) -> SimpleTokenizerV2 {
        let mut ix_to_tok = HashMap::new();
        for (token, ix) in &vocab {
            // first version we clone the strings
            ix_to_tok.insert(*ix, token.clone());
        }
        SimpleTokenizerV2 {
            ix_to_tok,
            tok_to_ix: vocab,
        }
    }

    pub fn tok_to_ix(&self) -> &HashMap<String, usize> {
        &self.tok_to_ix
    }

    pub fn ix_to_tok(&self) -> &HashMap<usize, String> {
        &self.ix_to_tok
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let re = Regex::new(r#"([,.?_!"()\']|--|\s)"#)
            .expect("This should be a valid regular expression.");

        let mut preprocessed = Vec::new();
        let mut left = 0;
        for m in re.find_iter(text) {
            let start = m.start();
            let end = m.end();
            if text[left..start].trim() != "" {
                preprocessed.push(text[left..start].trim());
            }
            if m.as_str().trim() != "" {
                preprocessed.push(m.as_str());
            }
            left = end;
        }
        let res = preprocessed
            .iter()
            .map(|token| {
                *self
                    .tok_to_ix
                    .get(*token)
                    .unwrap_or(self.tok_to_ix.get("<|unk|>").unwrap())
            })
            .collect();
        res
    }

    pub fn decode(&self, ixs: Vec<usize>) -> String {
        let tokens: Vec<&str> = ixs
            .iter()
            .map(|ix| &self.ix_to_tok.get(ix).unwrap()[..])
            .collect();
        let result = tokens.join(" ");
        result.to_string()
    }
}
