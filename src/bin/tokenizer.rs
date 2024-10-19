use regex::Regex;
use std::collections::{HashSet, HashMap};
use std::fs;
use rust_gpt::tokenizer::SimpleTokenizerV1;

fn main() {
    let file_content =
        fs::read_to_string("./data/the-verdict.txt").expect("The file should be present");
    println!("The len of the file is {}", file_content.len());
    println!("The first lines of the fiels are {:?}", &file_content[..99]);
    let text = "I HAD always thought Jack Gisburn rather";
    let re =
        Regex::new(r#"([,.?_!"()\']|--|\s)"#).expect("This should be a valid regular expression.");
    // println!("The splitted text is : {words:?}");

    let mut words = Vec::new();
    let mut left = 0;
    for m in re.find_iter(&text) {
        let start = m.start();
        let end = m.end();
        if text[left..start].trim() != "" {
            words.push(text[left..start].trim());
        }
        if m.as_str().trim() != "" {
            words.push(m.as_str());
        }
        left = end;
    }
    println!("The splitted text is : {words:?}");
    let mut left = 0;
    let mut preprocessed = Vec::new();
    for m in re.find_iter(&file_content) {
        let start = m.start();
        let end = m.end();
        if file_content[left..start].trim() != "" {
            preprocessed.push(file_content[left..start].trim());
        }
        if m.as_str().trim() != "" {
            preprocessed.push(m.as_str());
        }
        left = end;
    }

    println!("Num tokens find in the text: {}", preprocessed.len());
    println!("The 30 first tokens are: {:?}", &preprocessed[..30]);

    let set: HashSet<&str> = preprocessed.drain(..).collect();
    let mut preprocessed: Vec<&str> = set.into_iter().collect();
    preprocessed.sort();

    println!("Num unique tokens find in the text: {}", preprocessed.len());

    let mut vocab = HashMap::new();

    for (ix, el) in preprocessed.iter().enumerate() {
        vocab.insert(el.to_string(), ix);
    }
    let mut i = 0;

    // for (key, value) in &vocab {
    //     println!("{key}: {value}");
    //     i += 1;
    //     if i > 50 {
    //        break;
    //     }
    // } 


    let  tokenizer = SimpleTokenizerV1::new(vocab);
    let text = "It's the last he painted, you know,\" Mrs. Gisburn said with pard";
    // let text = "I HAD always thought Jack Gisburn rather";
    let ids = tokenizer.encode(text);
    println!("{ids:?}");
    let decoded = tokenizer.decode(ids);
    println!("{decoded}");


}
