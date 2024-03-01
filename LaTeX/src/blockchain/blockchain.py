# Create Blockchain

import datetime
import hashlib
import json
from flask import Flask, jsonify


# Part 1 - Building a Blockchain


class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(proof=1, previous_hash='0')

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'previous_hash': previous_hash
                 }
        self.chain.append(block)
        return block

    def get_previous_block(self):  # actually the last block
        return self.chain[-1]

    def proof_of_work(self, previous_proof):
        new_proof = 1
        check_proof = False
        while check_proof is False:
            # to make it simple, have far less leading 0's ie 4
            hash_operation = hashlib.sha256(str(new_proof ** 2 - previous_proof ** 2).encode()).hexdigest()
            # needs to be non-symmetrical, order independent, square to make it challenging
            if hash_operation[:4] == '0000':
                check_proof = True
            else:
                new_proof += 1
            return new_proof

    def hash(self, block):
        encoded_block = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded_block).hexdigest()

    def is_chain_valid(self, chain):
        block_index = 1
        previous_block = chain[0]
        while block_index < len(chain):
            block = chain[block_index]
            if block['previous_hash'] != self.hash(previous_block):
                return False
            previous_proof = previous_block['proof']
            proof = block['proof']
            hash_operation = hashlib.sha256(str(proof ** 2 - previous_proof ** 2).encode()).hexdigest()
            if hash_operation[:4] != '0000':
                return False
            block_index += 1
            previous_block = block
        return True


# Part 2 - Mining the Blockchain

# Create flask Web app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Create a blockchain
blockcahin = Blockchain()


# Mining new block
@app.route('/mine_block', methods=['GET'])
def mine_block():
    previous_block = blockcahin.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockcahin.proof_of_work(previous_proof)
    previous_hash = blockcahin.hash(previous_block)
    block = blockcahin.create_block(proof, previous_hash)
    response = {'message': 'Congrats, you mined a block',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash']}
    return jsonify(response), 200


@app.route('/get_chain', methods=['GET'])
def get_chain():
    response = {'chain': blockcahin.chain,
                'length': len(blockcahin.chain)}
    return jsonify(response), 200


@app.route('/is_valid', methods=['GET'])
def is_valid():
    is_valid_boolean = blockcahin.is_chain_valid(blockcahin.chain)
    if is_valid_boolean:
        response = {'message': 'All good, blockchain is valid'}
    else:
        response = {'message': 'Huston, we have a problem'}
    return jsonify(response), 200


app.run(host='0.0.0.0', port=5000)
