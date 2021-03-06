#
# Pyserini: Reproducible IR research with sparse and dense representations
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import argparse
import string
from nltk import bigrams, word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert KILT Knowledge Source into a Document-level JSONL that can be processed by Pyserini')
    parser.add_argument('--input', required=True, help='Path to the kilt_knowledgesource.json file')
    parser.add_argument('--output', required=True, help='Path to the output directory and file name')
    parser.add_argument('--bigrams', action='store_true', help='Enable bigrams')
    parser.add_argument('--stem', action='store_true', help='Enable stemming on bigrams')
    parser.add_argument('--flen', default=5903530, type=int, help='Number of lines in the file')

    args = parser.parse_args()

    FILE_LENGTH = args.flen
    STOPWORDS = set(stopwords.words('english') + list(string.punctuation))
    stemmer = SnowballStemmer("english")

    with open(args.input, 'r') as f, open(f'{args.output}', 'w') as outp:
        for line in tqdm(f, total=FILE_LENGTH, mininterval=10.0, maxinterval=20.0):
            raw = json.loads(line)
            doc = {}
            doc["id"] = raw["_id"]
            doc["contents"] = "".join(raw["text"])
            if args.bigrams:
                tokens = filter(lambda word: word.lower() not in STOPWORDS, word_tokenize(doc["contents"]))
                if args.stem:
                    tokens = map(stemmer.stem, tokens)
                bigram_doc = bigrams(tokens)
                bigram_doc = " ".join(["".join(bigram) for bigram in bigram_doc])
                doc["contents"] += " " + bigram_doc
            doc["wikipedia_id"] = raw["wikipedia_id"]
            doc["wikipedia_title"] = raw["wikipedia_title"]
            doc["categories"] = raw["categories"]
            _ = outp.write(json.dumps(doc))
            _ = outp.write('\n')

