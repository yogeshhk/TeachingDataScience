# Notes

- [The Coin Bureau Podcast: Crypto Without the Hype](https://podnews.net/podcast/i9xk4)
	Wonderful podcasts about Blockchain, Bitcoint, Ethereum and all about crypto
	
- [Decentriq - Introduction to zk-SNARKs (Part 1)](https://blog.decentriq.com/zk-snarks-primer-part-one/)
	- ZKPs can be used as the building blocks for verifiable computation: a method of offloading computation from a weak client to a more computationally powerful worker, which enables the client to cryptographically verify the correctness of the computation that was carried out by the worker with a much smaller computational effort compared to executing the original program.
	-  In machine learning, the technology can be used to prove possession of specific data without the need of full disclosure as well as to enable 3rd parties to verfiy that specific data has been used in the process of model training or prediction
	- Given problem is transformed by: code flattening, conversion to a rank-1 constraint system (R1CS) and finally formulation of the QAP.
	- Original problem: Peggie knows roots for the following equation
		```
		def f(x):
			y = x ** 2 - 4
		```
	- The corresponding flattened code is:
		```
		def f(x):
				out_1 = x * x
				y = out_1 - 4	
		```
	- A R1CS is a list of triplets of vectors and its solution is a vector such that:
		$\langle \vec{a_i}, \vec{s} \rangle \ast \langle \vec{b_i}, \vec{s} \rangle - \langle \vec{c_i}, \vec{s} \rangle = 0 \quad \textrm{for} \quad \forall i$ where $\vec{s} = \begin{pmatrix} 1 \\ x \\ out\_1 \\ y \end{pmatrix} $
	- For example, to encode the first line in the function body we set: $a = [0 1 0 0]; b = [0 1 0 0]; c=[0 0 1 0]$
	- To encode the second line we use: $a = [-4 0 1 0]; b = [1 0 0 0]; c=[0 0 0 1]$
	- Line 1: $\begin{equation} 
\langle \vec{a_1}, \vec{s} \rangle \ast \langle \vec{b_1}, \vec{s} \rangle - \langle \vec{c_1}, \vec{s} \rangle = \sum_{i=1}^{n} a_{1,i} s_i \ast \sum_{i=1}^{n} b_{1,i} s_i - \sum_{i=1}^{n} c_{1,i} s_i = x \ast x - out\_1 = 0 
\end{equation}$
	- Line 2: $\begin{equation} 
\langle \vec{a_2}, \vec{s} \rangle \ast \langle \vec{b_2}, \vec{s} \rangle - \langle \vec{c_2}, \vec{s} \rangle = \sum_{i=1}^{n} a_{2,i} s_i \ast \sum_{i=1}^{n} b_{2,i} s_i - \sum_{i=1}^{n} c_{2,i} s_i = out\_1 - 4 - y = 0 
\end{equation}$
	- To complete the transformation to a QAP we need to switch from vectors to polynomials.
	- Doing a Lagrange interpolation, we arrive at the polynomials $A(x) \ast B(x) - C(x) = H(x) \ast Z(x)$
	- One nifty trick that we can do with polynomials is to evaluate them at a point that is not known to us, a.k.a. blind evaluation of polynomials.
	- First, Victor chooses a point x0 at which he wants the polynomial to be evaluated. He then feeds it to f (another linear function) to get a new point, which we denote z, and sends z to Peggy. Note that because of the 1st property of f, there is no way for Peggy to find x0 from z.
	- Peggie can now compute P(z), which is same as f(P(x0)). Victor can now look at P(z) computed by Peggy and comparei t with f(P(x0)) locally, since he knows f, P and x0. If the two results match, he can be certain that she evaluated P
 at x0.
 
 - [Zero-Knowledge Proofs - Bharat Kumar Ramesh and Swapnika Nag](https://trilemma.substack.com/p/zero-knowledge-proofs?sd=pf)
	- Rollup, combines multiple transactions off-chain and computes verified total output.
	- Rollups look to construct a zero-knowledge proof such that any other nodes can verify it, and be certain that the transactions in the rollup are all valid, without requiring any more information about the transactions
	- How does a ZK-SNARK work?
		- The prover does the following:
			- Takes a list of transactions and adds to the rollup - this is the private input priv_t
			- Computes the new state of the chain from these transactions - this is a public
			- value pub_x
			- Takes the prover_key
			- Runs the prover function in the program with these parameters (priv_t, pub_x, prover_key) and generates a proof
			- Broadcasts the proof and the new state of the chain pub_x to all verifiers
		- Each verifier can now do the following:
			- Takes the public output or the new state of the chain i.e. pub_x
			- Receives the proof (from the prover)
			- Takes the verification_key
			- Runs the verification function in the program with these parameters (pub_x, proof, verification_key)
			- The program returns true or false			
	- The program uses cryptographic methods such that it is possible for the prover to compute a true proof if they have a valid priv_t. But computationally infeasible to do so, if they do not have a valid priv_t
	- And correspondingly, it is easy for a verifier to verify the same. Therefore, if the proof is computed to true, the verifier can be confident that the private input is valid, without knowing anything more about the input
	- Therefore, by storing just the proof, along with the new state of the chain on the base layer of Ethereum, one can abstract away a lot of the transactions away from L1, and store them off-chain. 
			
- [Zero Knowledge Proof: A Introductory Guide 101 Blockchains](https://101blockchains.com/zero-knowledge-proof/)	
	- A zero-knowledge proof needs to have three different properties to be fully described. They are:
		- Completeness: If the statement is really true and both users follow the rules properly, then the verifier would be convinced without any artificial help.
		- Soundness: In case of the statement being false, the verifier would not be convinced in any scenario. (The method is probabilistically checked to ensure that the probability of falsehood is equal to zero)
		- Zero-knowledge: The verifier in every case would not know any more information.
- [But how does bitcoin actually work?](https://www.youtube.com/watch?v=bBC-nXj3Ng4)
- [A General Introduction to Modern Cryptography - Josh Benaloh, Senior Cryptographer, Microsoft](https://www.youtube.com/watch?v=_Rf15nDic24)
	- Kerckhoffs's Principle (1883): The security of crypto-system should depend on the key. Meaning, even attacher knows all about the system, except the key.
- [Fundamentals Webinar 2020 - Blockchain at Berkeley](https://www.youtube.com/playlist?list=PLSONl1AVlZNWzsyZfhd9yDJRuGv3WPBck)
- [Blockchain Fundamentals Fall 2021 - Blockchain at Berkeley](https://www.youtube.com/playlist?list=PLSONl1AVlZNWoeYjazuvIVTeX7rxBtDNh)
- [MIT 15.S12 Blockchain and Money, Fall 2018 - MIT OpenCourseWare](https://www.youtube.com/playlist?list=PLUl4u3cNGP63UUkfL0onkxF6MYgVa04Fn)
- [MIT MAS.S62 Cryptocurrency Engineering and Design, Spring 2018 - MIT OpenCourseWare](https://www.youtube.com/playlist?list=PLUl4u3cNGP61KHzhg3JIJdK08JLSlcLId)
- [How to become a Web3 Developer](https://www.youtube.com/watch?v=q54j35z3fPQ) (End Goal - Get a Job or start a Web3 Startup) Harpalsinh Jadeja (@harpaljadeja11 on twitter)
	- Skills from Web2 that will help.
		- Typescript, NextJS, Chakra UI, Material UI, Testing.
		- Core computer concepts (operating systems, networking) and problem solving skills are fundamental and required no matter web2 or web3.
	- Differentiating Skills
		- Knowledge of Cryptography (ECDSA - Elliptic curve cryptography, Asymmetric key cryptography).
		- Remote Procedure calls is the base of blockchain (very helpful if you already know it), WebSockets.

	- [Metamask](https://metamask.io/) (wallet) (use web3 to learn how it works)
		- this is a wallet used to interact with blockchain.
		- this is a very basic skill expected from a crypto user, however developer should know more than average user like how to programmatically request network change and detect current accounts and network.
		- have a look at its docs.

	- Solidity (new language) (EVM based) (For solana & near) (learn Rust)
		- Learn basics first the language is under very quick development new versions roll out pretty quick so don’t get overwhelmed.
		- Best place to start [CryptoZombies](https://cryptozombies.io/).
		- Reading habit is recommended as most of the good tutorials are in text not in video format.
		- Other Resources:
				- Beginner Level
						- [Eattheblocks Youtube](https://www.youtube.com/watch?v=p3C7jljTXaA&list=PLbbtODcOYIoE0D6fschNU4rqtGFRpk3ea)
						- [Smart Contract Programmer Youtube](https://www.youtube.com/watch?v=hMwdd664_iw&list=PLO5VPQH6OWdULDcret0S0EYQ7YcKzrigz)
						- [DApp University Youtube](https://www.youtube.com/c/DappUniversity/playlists)
						- [Solidity by Example (text based tutorial)](https://solidity-by-example.org/)
				- Intermediate Level
						- [Solidity Docs](https://docs.soliditylang.org/en/v0.8.10/)
						- Read other people’s code.
				- Expert Level
						- [Ethereum Docs](https://ethereum.org/en/developers/docs/)
						- [Secureum](https://www.youtube.com/c/SecureumVideos)
						- Mastering Ethereum - Gavin Wood

	- [Etherscan](https://etherscan.io/) (Blockchain Explorer)
		- This is the blockchain explorer. You might think this is basic but there is a lot to learn here. A lot of people don't know how to read the transactions.
		- Learn all the terms mentioned in a transaction google them.
		- Learn by reading code of other popular smart contracts most of them are Open Source.
		- Learn how to verify a contract from explorer, learn how to verify using remix and then programmatically from hardhat.
	- Remix (online IDE for solidity) (beginner level) Learn how to deploy contracts.

	- [Hardhat](https://hardhat.org/) (framework for development) (intermediate user) (widely used) (highly recommended) (Javascript)

	- Openzeppelin Contracts (EVM based only)

	- Anchor Protocol (Solana)

	- [Brownie](https://eth-brownie.readthedocs.io/en/stable/) (intermediate user) (development Framework) (python)

	- Start building!

	- Project based learning Videos
		- [1](https://www.youtube.com/watch?v=GKJBEEXUha0)
		- [2](https://www.youtube.com/watch?v=M576WGiDBdQ)
		- Moralis Youtube Channel


	- Decide DeFi, NFT or DAOs (multiple is fine)
		- DeFi - [Uniswap](https://app.uniswap.org/#/swap) (decentralized Dex), [Aave](https://aave.com/), [Compound](https://compound.finance/). [Openzeppelin](https://openzeppelin.com/contracts/)
		- NFT - [Openzeppelin](https://openzeppelin.com/contracts/) (ERC721, ERC20, ERC1155)
		- DAOs - [Compound Governance](https://compound.finance/governance), [Snapshot](https://snapshot.org/#/)
		- Metaverse - Decentraland, Sandbox.

	- Good to learn
		- Chainlink Oracles ⭐
		- Chainlink External Adapters
		- IPFS ⭐
		- Ceramic 
		- Moralis
	- Participate in Hackathons
		- [Devpost](https://devpost.com/hackathons?themes[]=Blockchain)
		- [Gitcoin](https://gitcoin.co/)
		- [Hackerlink](https://hackerlink.io/)
		- [ETHGlobal](https://ethglobal.com/)
		- Announcements on Medium.com 

	- Contribute to Open Source
		- Gitcoin Bounties
		- Follow projects on Github many require help for documentation.

	- Ethereum Security (optional) (very high demand very low supply)
		- [Ethernaut](https://ethernaut.openzeppelin.com/)
		- [Damn Vulnerable Defi](https://www.damnvulnerabledefi.xyz/)
		- Capture the Ether
		- Trail of Bits blogs
		- Paradigm Blogs

	- Communities to join (highly recommended)
		Get discord if you don’t have web3 people use discord.
		- [Figment.io](https://www.figment.io/)
		- [Buildspace](https://buildspace.so/)
		- [Questbook (India Specific)](https://www.questbook.app/)
		- [SuperTeamDAO](https://superteam.fun/)
		- Developer DAO (not free technically)

	- Get a Job
		- [Cryptocurrency Jobs](https://cryptocurrencyjobs.co/)
		- AngelList
		- LinkedIn
		- EthGlobal Discord
		- Buildspace Discord
		- Chainlink Discord
		- Developer DAO Discord

	- Freelance Gigs on Gitcoin Bounties and various other discord groups. If you go the extra mile and learn Blockchain Security you will get paid very well!
 
	- Start a Web3 Startup: Participate in hackathons. Win the hackathons. The VCs are usually judges interact with them during the hackathon build something they are expecting someone to build. There are team formation session and brainstorming sessions.

		- [Tachyon Accelerator](https://mesh.xyz/tachyon/)
		- [Celo Camp 5 (Forms open)](https://www.celocamp.com/)
		- [TechStars](https://www.techstars.com/accelerators/launchpool-web3)
		- [Outlier Ventures](https://outlierventures.io/)
		- [DeFi Alliance](https://www.defialliance.co/)
		- [Web3 Foundation](https://web3.foundation/)
		- [Kernel Foundation (Gitcoin)](https://kernel.community/en/)
		- [Encode Club](https://www.encode.club/)

	- Hot Topics
		- MEV
		- DAOs
		- Metaverse
		- Zero-Knowledge Proofs

- Zero Knowledge Proofs - Aryan Shah (LinkedIn)
	Zero knowledge proofs are one of the most significant crypto technologies being developed
	Here are five ways zero knowledge proofs are making dapps more #efficient, #private & #decentralised

	1. MultiPlayer games on chain - #darkforest

	DarkForest is a #multiplayer game completely on-chain where players compete in rounds. Players start on a planet & can conquer other unconquered planets or planets of other players
	The goal is to get as many points as possible

	How ZK helps?

	Both the location of the player's planet and the moves that they make are hidden from other players
	Players generate ZK proofs on their system & publish them on the blockchain where the #contract verifies that the moves they are making are valid.

	2. Private Transactions - #TornadoCash

	Tornado #cash enables users to transfer their #ETH to a different account in a private way. Users deposit ETH into a pool & can withdraw those ETH from the pool using a diff account.

	ZK helps break the link between the #deposit and #withdrawal .

	How ZK helps ?

	While depositing users submit the hash of a secret.
	While withdrawing users submit a ZK Proof that they know the secret without revealing which #secret they know.
	This enables them to #withdraw their ETH without revealing which #deposit was theirs

	3. Proof of Storage - #filecoin

	Filecoin enables everyone to store their data in a #decentralised manner.
	Every storage provider must submit two proofs
	1. Proof of replication(PoRep)- data was downloaded by provider
	2. Proof of space time(PoSt)- data is still being stored.

	How ZK helps?

	Using ZK, filecoin compresses both PoRep & PoSt from 100s of KBs to 192Bs, a reduction of 10000!. Each storage provider generates 10 ZKPs each for PoRep & PoSt respectively.
	These ZKPs can be very cheaply verified on #chain, saving both bandwidth & mining costs

	4. Private Identity & Credentials - #identhree

	Iden3 enables solutions like #polygon
	 where users share information about their data while keeping it private
	Eg. proving you are a DAO member without disclosing your #identity. Or that you are above 18 yrs w/o sharing your age

	How ZK helps?

	Users can submit proofs to the #DAO that they are one of the member addresses without revealing which #address is them

	Users can also submit proofs that they hold a valid document (>18 age) without #revealing the actual details of that document.

	5. Proof of Alpha #minaprotocol

	 helps traders & funds verify that they actually made a #profit in their trades without revealing the real #trades.
	LPs or followers can verify trader's #profitability without knowing which trades or #investments were executed.

	How ZK helps?

	Traders can link their exchange accounts to an application which fetches all their trading history.
	It generates a ZKP of the profit without revealing the actual trades.

- PRIVACY-PRESERVING DATA SCIENCE, EXPLAINED, May 19th, 2020, Private ML
	- What is Secure Multi-Party Computation?: Secure multi-party computation allows multiple parties to collectively perform
some computation and receive the resulting output without ever exposing any party’s sensitive input.
	-  A Zero Knowledge Proof (ZKP) is a mathematical method to prove that one party possesses something without actually revealing the information.


- LinkedIn post by Suraj R Mulla

	Zero Knowledge Proof Decentralized Digital Identity - Bringing Democratic Regulatory Governance!

	Web3 industry is being built on the principles of self-sovereignty, decentralization, transparency, censorship resistance and democratic governance. 

	Governments are gradually building regulatory frameworks for consideration of this emerging industry and embrace it with its utmost potential. 

	At the same time, government agencies are reluctant towards anonymous governance of users performing illicit, fraudulent activities which can be harmful towards state nations. 

	So, we need a novel solution which bridges this gap and allows Web3 users to continue enjoying the Web3 perks, and Government agencies to bring this industry in a holistic regulatory framework. 

	The Zero Knowledge Proof Decentralized Digital Identity(ZK-DID) is a perfect solution for bridging this gap without losing dynamics of either parties. 

	The ZK-DID can help build a regulatory framework by verifying identities of each individual participating in the Web3 without giving away any sensitive information of that individual and keeping the anonymity in a decentralized network. 

	Each user possessing ZK-DID will have authenticity, self-sovereignty, anonymity, and a democratic governance of its operations across different Web3 applications. 

	At the same time, the governments can imply frameworks for the Web3 applications to have a trustless verification method without having to monitor individual activities and breach digital rights. 

	The Polygon ID is explicitly working towards allowing Web3 users to have a self-sovereign governance of their identity without having to share any personal sensitive information to prove the user authenticity. 

	Polygon ID is catalyzing the authentic Web3 user adoption while preventing Web3 applications from regulatory scrutiny. 

	Can Polygon ID act as a novice solution for the regulatory approved, self-sovereign, authentic, democratic and decentralized mass adoption of the Web 3.0 industry?

- Introduction to Modern Cryptography - Navin Kabra https://www.youtube.com/watch?v=trgBxahTVdU
	- How to send a secure message over public channel?
	- How to sign a word document? because it can be signature
	- Symmetric Encryption: encode pdf with password and share the password with those who want to see it, they use the same password. Not suitable for public communication as password needs to be shared
	- Asymmetric Encryption: Encrypt with one password, and decrypt with another password.
	- Modular Arithmetic, '%' or 'mod' is remainder operator: 
		14 % 10 = 4
		5 % 2 = 1
		a = b (mod n) means a and b have same remainder when divided by n
		14 = 4 (mod 10)
		If I am interested in last digit, its 'mod 10'
		If I am interested in hours, its 'mod 12'
	- Rules:
		a mod n + b mod n = a+b mod n
		You get smaller numbers, even after operations
		a mod n x b mod n = ab mod n
		What is -1 mod arithmetic? -1 mod 5 ==? its 4... its same as adding -1 and 5
		Division?? Opposite of Multiplication. 12/3 =?, ie 3 into what gives 12?
		for mod 15 arithmetic, 3 x 4 = 12; 3 x 9 = 27 mod 15 = 12, so 12/3 is it 4 as normal arithmetic or 9 as in mod arithmetic. Undefined.
		Such things occur when '3' has common factor with '15'
		Division is Multiply by Inverse giving answer 1. a x b = 1 so a = b^-1 or b = a^-1
		In mod 15 arithmetic, 4 x 4 = 16 mod 15 = 1; 7 x 13 mod 15 = 1; so 7 and 13 are inverses of each other wrt mod 15
		but for 5 x anything with mod 15, the answer is never 1, because 5 has a common factor with 15; same with 3, 6, 9, 10, etc
		a^n = a x a x ...n times, In exponential table, row^2, columns wrt say mod 21 repeat after 6...thats magic number Lambda 'L'
		In 'mod n' arithmetic: m^{L+1} = m (mod n) for all m, basically you get back original m.
		How L is calculated?  if p and q are prime numbers, and n = p x q, then L is lcm of p-1 and q-1; 21 is 3 x 7, and lcm of 3-1=2 and 7-1=6 is 6 thus things were repeating after every 6 columns.
	- send a secure message over public channel? Alice to Bob where Carol is watching
		- Alice to find two numbers e and d, if e x d = L +1, then ed mod L = 1, 
		- Use property(m^e)^d = m^{ed} in m^{L+1} = m (mod n)  becomes (m^e)^d = m^{L+1} = m mod n
		Let C = m^e mod n, then C^d = m mod n

		In public key cryptography, there are two keys. Suppose Alice wishes to receive encrypted messages; she publishes one of the keys, the public key, and anyone, say Bob, can use it to encrypt a message and send it to her. When Alice gets the encrypted message, she uses the private key to decrypt it and read the original message. If Alice needs to reply to Bob, Bob will publish his own public key, and Alice can use it to encrypt her reply.
		
		This is RSA, after Ron Rivest, Adi Shamir and Leonard Adleman.
		
		Alice does: 
			- pick p and q as large prime numbers, multiply to get n, is of the required bit length, e.g. 1024 bits. 
			- then get L, L is lcm of p-1 and q-1
			- pick any random small prime number e, compute d as e inverse mod L. d = e^-1 mod L; 1 = e x d mod L; what number d times e when divided by L leave s remainder 1? 
			- Save d carefully as it is a secrete, its a private key. On public channel she send n and e. 
			- n is known as the modulus. 
			- e is known as the public exponent or encryption exponent or just the exponent.
			- d is known as the secret exponent or decryption exponent.
		
		The RSA algorithm works because, when n is sufficiently large, deriving d from a known e and n will be an impractically long calculation — unless we know p, in which case we can use the shortcut. This is why p and q must remain secret.
		
		Encryption: 
			- Bob wants to send one message m = 101 
			- Uses e declared as public key by Alice
			- He encrypts it like, C = m^e mod n = 101^11 mod 2173 = 1305, 
			- So Bob sends encrypted message '1305' to Alice.
		Decryption: 
			- Alice receives Bob’s message '1305'
			- She decrypts it with the saved private key, d, say 331, so (n = 2173, d = 331). 
			- So Alice decrypts the message like: 	plaintext = cyphertext^d mod n = 1305^331 mod 2173 = '101'
