use regex::Regex;
use rust_gpt::dataset::create_dataloader_v1;
use rust_gpt::tokenizer::SimpleTokenizerV2;
use std::collections::{HashMap, HashSet};
use std::fs;
use tiktoken_rs::r50k_base;
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
    for m in re.find_iter(text) {
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
    // Add special tokens
    preprocessed.push("<|endoftext|>");
    preprocessed.push("<|unk|>");

    println!("Num unique tokens find in the text: {}", preprocessed.len());

    let mut vocab = HashMap::new();

    for (ix, el) in preprocessed.iter().enumerate() {
        vocab.insert(el.to_string(), ix);
    }
    let i = 0;

    // for (key, value) in &vocab {
    //     println!("{key}: {value}");
    //     i += 1;
    //     if i > 50 {
    //        break;
    //     }
    // }

    // let  tokenizer = SimpleTokenizerV1::new(vocab);
    // let text = "It's the last he painted, you know,\" Mrs. Gisburn said with pard";
    // // let text = "I HAD always thought Jack Gisburn rather";
    // let ids = tokenizer.encode(text);
    // println!("{ids:?}");
    // let decoded = tokenizer.decode(ids);
    // println!("{decoded}");

    let text1 = "Hello, do you like tea?";
    let text2 = "In the sunlit terraces of the palace.";
    let text = [text1, text2].join(" <|endoftext|> ");
    println!("{text}");

    let tokenizer = SimpleTokenizerV2::new(vocab);
    let encoded = tokenizer.encode(&text);
    println!("{encoded:?}");

    // Sanity check
    let decoded = tokenizer.decode(encoded);
    println!("{decoded:?}");

    let text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of som";
    let bpe = r50k_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(text);
    println!("{tokens:?}");

    let strings = bpe
        .decode(tokens)
        .expect("tiktoken is supposed to be robust");
    println!("{strings:?}");

    let file_content =
        fs::read_to_string("./data/the-verdict.txt").expect("the file should be present");

    let encoded_file = bpe.encode_with_special_tokens(&file_content[..]);
    println!("Number of tokens: {}", encoded_file.len());
    let sample = &encoded_file[..50];
    let context_size = 4;
    let x = &sample[..context_size];
    let y = &sample[1..context_size + 1];
    println!("x={x:?}\ny={y:?}");

    // for i in 1..context_size+1 {
    //     let context = &sample[..i];
    //     let desired = sample[i];

    //     println!("{:?} -----> {:?}", bpe.decode(context.to_vec()).unwrap(), bpe.decode(vec![desired]).unwrap());
    // }
    let file_content =
        fs::read_to_string("./data/the-verdict.txt").expect("the file should be present");

    let mut dataloader = create_dataloader_v1(&file_content, Some(1), Some(4), Some(1), None, None);
    // for res in dataloader {
    //     match res {
    //         Ok((inputs, targets)) => {
    //             println!("{inputs:?}, {targets:?}");
    //         }
    //         _ => {
    //             println!("Something went wrong")
    //         }
    //     }
    //     // println!("Input: {input}\nTarget: {target}");
    // }

    if let Some(Ok((inputs, targets))) = dataloader.next() {
        // println!("{inputs:?}, {targets:?}");
        println!("inputs: {}\n outputs: {}", inputs, targets);
    }

}
