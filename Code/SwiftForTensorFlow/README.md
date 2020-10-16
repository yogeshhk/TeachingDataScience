# Swift for TensorFlow

## How to Use
- Google Colab: Open  https://colab.research.google.com/notebook\#create=true\&language=swift; Save in your GDrive with appropriate name. If needed move to appropriate folder.
- Write file as xxx.swift. Compile using
```
SDKROOT=C:/Library/Developer/Platforms/Windows.platform/Developer/SDKs/Windows.sdk
swiftc -sdk %SDKROOT% -I %SDKROOT%/usr/lib/swift -L %SDKROOT%/usr/lib/swift/windows -emit-executable -o xxx.exe xxx.swift
xxx.exe
```
- Local Jupyter notebook is not working. Error llvm module not found
- Local Interpreter not working. Error.

# ToDos
- Prep 1/1.5 hrs Intro seminar on "Swift for TensorFlow" (Intro to Swift, Intro to TensorFlow, MNIST usecase)
- Go through official S4TF tutorials
- Understand contribute pythonTF tutorials to siwthTF by translation
- Contribute to Swift-APIs github repo

## Observations
- https://www.tensorflow.org/swift/tutorials/model_training_walkthrough
	- Import data with the Epochs API.?? or Data APIs? 
	- Data import seems clumsy. Why read from np, twice, why not pandas. Canned functions are neeed.
	- Filed as https://github.com/tensorflow/swift/issues/547
	
	