# Notes on Quantum Machine Learning

## IKIGAI, Specific Knowledge
- Unique, different, rare field
- Good research funding, big companies scope, entry barrier to others
- Specific Knowledge if apprenticeship is done with some group
- Spread knowledge by talks, conferences, meet-ups
- Contribute to tf quantum, IBM? open source

## ToDos
- IISc courses, AI, ML on audit, QML with 60% paid discount
- Udemy Course
- Audit Coursera course
- Reference papers, Letcure notes and books
- TensorFlow Quantum site tutorials.
- Fermatik category theory playlist and others

## References

### QnA with Himanshu Vaidya
Q - Why do a ML Engineering has to bother about underlying hardware, it can be CPU, GPU, TPU or QPU?
A - QPU is not an improved and similar replaceable processor. Quantum computing paradigm itself is very different than normal computing. The classical computing is based on bits (0s, 1s) where is Quantum is on Qubits. This data type definitions, the data storage methodologies are very different. The abstractions are not yet developed to make QPU replaceable.

Q - How practical is it to use Quantum Computing for ML?
A - Not all ML problems can be solved using Quantum. Similar to parallel processing paradigm, where problems themselves are split to be parallel, some problems themselves have to be quantumiz-able. Even with 70 qubits, with overlapping/superimpose storage, huge simulations can be solved.

### My experience with TensorFlow Quantum,  Owen Lockwood, Rensselaer Polytechnic Institute
- QML vs. traditional neural network/deep learning:
- Similarities:
	- using “stacked layers” of transformations that make up a larger model
	- data is used to inform updates to model parameters, typically to minimize some loss function
- Differences:
	- QML models have access to the power of quantum mechanics and deep neural networks do not.
- Variational quantum circuits (QVC) is the QNN, has encoder circuit, the variational circuit and the measurement operators. 
	- The encoder circuit either takes naturally quantum data (i.e. a nonparametrized quantum circuit) or converts classical data into quantum data. 
	- Variational circuit is defined by its learnable parameters. The parametrized part of the circuit is the part that is updated during the learning process. 
	- Measurement operators extract information from the QVC some sort of quantum measurement (such as a Pauli X, Y, or Z basis measurements), a loss function (and gradients) is calculated on a classical computer and the parameters can be updated.
	- QVC’s can also be combined with traditional neural networks (as is shown in the diagram) as the quantum circuit is differentiable and thus gradients can be backpropagated through.
	
	![QVC](images/qvc.png)
	
- QML can harness quantum phenomena, such as superposition and entanglement. 
- Superposition stems from the wave function being a linear combination of multiple states and enables a qubit to represent two different states simultaneously (in a probabilistic manner). The ability to operate on these superpositions, i.e. operate on multiple states simultaneously, is integral to the power of quantum computing.
- Entanglement is a complex phenomenon that is induced via multi-qubit gates.
- Near term and current quantum devices have 10s-100s of quantum bits (qubits) like the Google sycamore processor. Because of their size and the noise, this hardware is often called Noisy Intermediate Scale Quantum (NISQ) technology.
- For classical ML researchers with experience in TensorFlow, TFQ makes it easy to transition and experiment with QML at small or large scales. The API of TFQ and the modules it provides (i.e. Keras-esque layers and differentiators) share design principles with TF and their similarities make for an easier programming transition.

### TensorFlow Quantum: beauty and the beast  - Shubham Goyal
- While a normal Turing machine can only perform one calculation at a time, a quantum Turing machine can perform many calculations at once.
- TFQ adds the ability to process quantum data, consisting of both quantum circuits and quantum operators. 
	![TFQ](images/tfq.png)
- TFQ steps to train and build QML models:
	- Prepare a quantum dataset: Quantum data is loaded as tensors, specified as a quantum circuit written in Cirq. The tensor is executed by TensorFlow on the quantum computer to generate a quantum dataset.
	- Evaluate a quantum neural network model: prototype a quantum neural network using Cirq that will be embedded inside of a TensorFlow compute graph.
	- Sample or Average: This step leverages methods for averaging over several runs involving steps (1) and (2).
	- Evaluate a classical neural networks model: This step uses classical deep neural networks to distill such correlations between the measures extracted in the previous steps.
	- Evaluate Cost Function: Similar to traditional machine learning models, TFQ uses this step to evaluate a cost function.
	- Evaluate Gradients & Update Parameters.
	
	
### Quantum AI Google
- What lies beyond classical
Classical computers and quantum computers are used to solve different types of problems. For classical computing, we represent what we think about the world as bits of information in sequences of zeros and ones, and use logic to make decisions. Is there someone at the door? Yes (1) or no (0)? If yes (1); then open the door.
- NISQ computers can perform tasks with imperfect reliability, but beyond the capability of classical computers. Even with imperfect reliability, we can advance our knowledge of science in the NISQ era.
- Chemical reactions are most accurately described by quantum mechanics. We've developed computational techniques to describe some chemical reactions, but some important reactions remain out of reach for classical computers.
- Once we have high-quality physical qubits organized into a logical qubit, the next challenge is scale. With 1000 qubits, it should be possible to store quantum data for nearly a year. Scaling up requires making many of our components smaller: including our wiring, amplifiers, filters, and electronics.
- Once we have successfully created a very long-lived, error-corrected logical qubit and the cryostat to hold it, the next task will be to stitch several logical qubits together so that they can all work together on increasingly complex algorithms. For this, we’ll need coherence between chips so that the processor can communicate across qubits. This requires significant work on our fabrication technology, control software, and more
- To solve quantum-friendly problems, we built computers that use qubits instead of bits. Qubits represent the world in terms of uncertainty, and we use quantum mechanics to make decisions based on the probability of our qubits being in one state (1) or another (0). These computers enable us to simulate how the world actually works, instead of how we think it works.

### The Emerging Paths Of Quantum Computing - Chuck Brooks
Qubit zoo: Quantum vocabulary and terminology

- Qubits not bits. Quantum computers do calculations with quantum bits, or qubits, rather than the digital bits in traditional computers. Qubits allow quantum computers to consider previously unimaginable amounts of information.

- Superposition. Quantum objects can be in more than one state at the same time, a situation depicted by Schrödinger’s cat, a fictional feline that is simultaneously alive and dead. For example, a qubit can represent the values 0 and 1 simultaneously, whereas classical bits can only be either a 0 or a 1.

- Entanglement. When qubits are entangled, they form a connection to each other that survives no matter the distance between them. A change to one qubit will alter its entangled twin, a finding that baffled even Einstein, who called entanglement “spooky action at a distance.”

- Types of qubits. At the core of the quantum computer is the qubit, a quantum bit of information typically made from a particle so small that it exhibits quantum properties rather than obeying the classical laws of physics that govern our everyday lives. Several types of qubits are in development:

	- Superconducting qubits, or transmons. Already in use in prototype computers made by Google, IBM and others, these qubits are made from superconducting electrical circuits.
	- Trapped atoms. Atoms trapped in place by lasers can behave as qubits. Trapped ions (charged atoms) can also act as qubits.
	- Silicon spin qubits. An up-and-coming technology involves trapping electrons in silicon chambers to manipulate a quantum property known as spin.
	- Topological qubits. Still quite early in development, quasi-particles called Majorana fermions, which exist in certain materials, have the potential for use as qubits. Quantum computing: Opening new realms of possibilities (princeton.edu)
	
### QML Learning with TensorFlow Quantum - Nicholas Teague
- In quantum neural networks we’ll be feeding our input data embedded into quantum bits, aka qubits, and so any operations applied onto that form will need to be conducted by aggregations of quantum gates applied sequentially to and between those qubits through the progression of time steps. With each of these gate operations, we’ll be rotating the superposition state around an axis of those qubits’ “Bloch spheres”.
- Quantum algorithms are means to craft by such gate rotations a shaped superposition of collective qubit states such that the probabilistic measurement operations on the returned qubit states are more likely to collapse to classical states corresponding to some desired output.
- Each layer of gates only interacts with adjacent qubits instead of all to all. This reduced information flow is compensated by the depth of information capacity of multi-qubit superpositions.
- The realized gated network can then be given a form of supervised training to fit the parameterized gate sets to result in a returned superposition more aligned with our target state as a function of inputs. 
- The chain rule of backpropagation doesn’t directly translate to quantum primitives. However, there is a very clean solution realized by the simple modularity inherent in the networks. In other words, if in a forward pass the returned measurements from a quantum network are fed as input to a classical network, then in the backward pass we can simply apply the returned gradients from the classical network as input to the gradient calculations of the quantum network.
