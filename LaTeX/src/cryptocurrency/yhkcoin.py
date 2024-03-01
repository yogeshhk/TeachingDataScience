# Create Blockchain

import datetime
import hashlib
import json
from flask import Flask, jsonify, request
import requests
from uuid import uuid4
from urllib.parse import urlparse


# Part 1 - Building a Blockchain
class Blockchain:
    def __init__(self):
        self.chain = []
        self.transactions = []
        self.create_block(proof=1, previous_hash='0')
        self.nodes = set()  # no particular order

    def create_block(self, proof, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'timestamp': str(datetime.datetime.now()),
                 'proof': proof,
                 'transactions': self.transactions,
                 'previous_hash': previous_hash
                 }
        self.transactions = []  # empty this temp storage
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

    def add_transaction(self, sender, receiver, amount):
        self.transactions.append({'sender': sender,
                                  'receiver': receiver,
                                  'amount': amount})
        previous_block = self.get_previous_block()  # last block
        return previous_block['index'] + 1  # block index to add these transactions to

    def add_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def replace_chain(self):
        network = self.nodes
        longest_chain = None
        max_length = len(self.chain)
        for node in network:
            response = requests.get('http://' + node + '/get_chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.is_chain_valid(chain):
                    max_length = length
                    longest_chain = chain
        if longest_chain:
            self.chain = longest_chain
            return True
        return False


# Part 2 - Mining the Blockchain

# Create flask Web app
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Creating nodes/computers
node_address = str(uuid4()).replace("-", "")

# Create a blockchain
blockchain = Blockchain()


# Mining new block
@app.route('/mine_block', methods=['GET'])
def mine_block():
    previous_block = blockchain.get_previous_block()
    previous_proof = previous_block['proof']
    proof = blockchain.proof_of_work(previous_proof)
    previous_hash = blockchain.hash(previous_block)
    blockchain.add_transaction(sender=node_address, receiver='Yogesh', amount=1)
    block = blockchain.create_block(proof, previous_hash)
    response = {'message': 'Congrats, you mined a block',
                'index': block['index'],
                'timestamp': block['timestamp'],
                'proof': block['proof'],
                'transactions': block['transactions'],
                'previous_hash': block['previous_hash']}
    return jsonify(response), 200


@app.route('/get_chain', methods=['GET'])
def get_chain():
    response = {'chain': blockchain.chain,
                'length': len(blockchain.chain)}
    return jsonify(response), 200


@app.route('/is_valid', methods=['GET'])
def is_valid():
    is_valid_boolean = blockchain.is_chain_valid(blockchain.chain)
    if is_valid_boolean:
        response = {'message': 'All good, blockchain is valid'}
    else:
        response = {'message': 'Huston, we have a problem'}
    return jsonify(response), 200


# Adding a new transaction to the blockchain
@app.route('/add_transaction', methods=['POST'])
def add_transaction():
    json_response = request.get_json()
    transaction_keys = ['sender', 'receiver', 'amount']
    if not all(key in json_response for key in transaction_keys):
        return 'Some fields of transactions are missing', 400
    index = blockchain.add_transaction(json_response['sender'],
                                       json_response['receiver'],
                                       json_response['amount'])
    response = {'message': f"This transaction will be added to Block{index}"}
    return jsonify(response), 201


# Decentralizing Blockchain
# Connecting new nodes
@app.route('/connect_node', methods=['POST'])
def connect_node():
    json_response = request.get_json()
    nodes = json_response.get('nodes')
    if nodes is None:
        return "No nodes", 400
    for node in nodes:
        blockchain.add_node(node)
    response = {'message': "Added node to network",
                'total_nodes': list(blockchain.nodes)}
    return jsonify(response), 201


@app.route('/replace_chain', methods=['GET'])
def replace_chain():
    is_chain_replaced = blockchain.replace_chain()
    if is_chain_replaced:
        response = {'message': 'Chain is replaced by longest one',
                    'new_chain': blockchain.chain}
    else:
        response = {'message': 'All good, the chain is largest',
                    'new_chain': blockchain.chain}
    return jsonify(response), 200


app.run(host='0.0.0.0', port=5000)
