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

import argparse
import pandas as pd
from tqdm import tqdm
import sys

# We're going to explicitly use a local installation of Pyserini (as opposed to a pip-installed one).
# Comment these lines out to use a pip-installed one instead.
sys.path.insert(0, './')
sys.path.insert(0, '../pyserini/')

from pyserini.dsearch import AnceQueryEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    parser.add_argument('--input', type=str, help='query file to be encoded.', required=True)
    parser.add_argument('--output', type=str, help='path to store query embeddings', required=True)
    parser.add_argument('--device', type=str,
                        help='device cpu or cuda [cuda:0, cuda:1...]', default='cpu', required=False)
    args = parser.parse_args()

    encoder = AnceQueryEncoder(args.encoder, device=args.device)
    embeddings = {'id': [], 'text': [], 'embedding': []}
    for line in tqdm(open(args.input, 'r').readlines()):
        qid, text = line.rstrip().split('\t')
        qid = qid.strip()
        text = text.strip()
        embeddings['id'].append(qid)
        embeddings['text'].append(text)
        embeddings['embedding'].append(encoder.encode(text))
    embeddings = pd.DataFrame(embeddings)
    embeddings.to_pickle(args.output)
