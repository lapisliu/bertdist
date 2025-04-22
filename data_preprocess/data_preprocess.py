import argparse
import spacy
import glob
import os
import spacy.cli

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")
# Download the spaCy English model
spacy.cli.download("en_core_web_sm")

def split_sentences_with_spacy(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        text = infile.read()

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for sentence in sentences:
            outfile.write(sentence + '\n')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split text files into sentences using spaCy and write to output directory."
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing raw text files"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory to write sentence-separated text files"
    )
    return parser.parse_args()

def main():
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    print("Model download complete.")
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in glob.glob(os.path.join(args.input_dir, "*.txt")):
        filename = os.path.basename(file_path)
        output_path = os.path.join(args.output_dir, filename)
        split_sentences_with_spacy(file_path, output_path)

if __name__ == "__main__":
    main()
