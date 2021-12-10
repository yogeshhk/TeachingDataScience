# 3D Machine/Deep Learning for Augmented/Virtual Reality

*Activity Recognition & Reconstruction* (ARR) : 
- Recognition: 3D Skeleton extraction from images/videos ie 3D pose estimation
- Reconstruction: Taking 3D skeletons, building next action predictions, and constructing corresponding 3D mesh models.

NO Scene generation but "stick" to Skeleton generation as its close to Midcurve, within AR/VR domain.


## ToDos (2022)
- TFUG: Prep tutorial on Activity Recognition from images/videos using Pytorch
- Implement 3DPoselite or likes, Yoga poses validation app, etc and demonstrate, CVPR Paper?

## Rational
- Big demand: AR/VR in many domains like manufacturing, health, architecture, media & entertainment.
- Reality: Computer vision (2D) domain is well established but not 3D (many representations like, point cloud, mesh, implicit surface equations, voxels, primitives, etc)
- Potential: Big push by Meta, Nvidia, Google, etc on Machine/Deep Learning on 3D data. Large number of research/commercial/individual opportunities. 
- [Sample job profile at Facebook - Research Scientist, 3D Machine Learning (PhD)](https://www.linkedin.com/jobs/view/2817385473)
- [Sample PhD position](https://polytechnicpositions.com/phd-positions-in-3d-machine-learning-3d-vision,i7150.html)

## General sub-topics under AR/VR
- Differential rendering
- 3D shape completion
- 3D segmentation
- Estimating depth from 2D
- Eye-tracking ground-truth models.
- 3D Computer Vision and Machine Learning.
- 3D Semantic Scene Understanding
- *Pose Estimation/Activity Recognition*, Skeletonization | stick figures, image-to-skeleton, skeleton-to-3dMesh
- People/Object classification/detection/re-identification
- Single/multi-camera subject/object tracking and association
- Human attributes (gender, age, etc.) pose estimation and gesture recognition
- Activity classification/detection/recognition
- Liveness detection/anti-spoofing
= Feature extraction and matching; face detection/verification/recognition
- Image and video synthesis, 3D modeling and reconstruction,
- Image denoising and quality improvements
- Head, Hand and Body Pose Tracking, 
- SLAM (Simultaneous localization and mapping)

Target Conferences such as NeurIPS, ICML, ICPR, CVPR, and journals such as JMLR, IEEE Trans., Neural Computation.

Libraries:
- [Pytorch3D by Facebook](https://github.com/facebookresearch/pytorch3d)
- [Kaolin by NVidia](https://github.com/NVIDIAGameWorks/kaolin)
- [MeshCNN by Rana Hanocka](https://github.com/ranahanocka/MeshCNN)
- [Open Pose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) Python code, C++ implementation and Unity Plugin. 

Courses:
- [Lecture: 3D Scanning and Motion Capture](https://justusthies.github.io/posts/3D-Scanning-and-Motion-Capture-SS18/)
- [CS231A: Computer Vision, From 3D Reconstruction to Recognition](https://web.stanford.edu/class/cs231a/)

Researchers:
- [Angela Dai](https://www.professoren.tum.de/en/dai-angela)
- [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/)
- [Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)
- [Andreas Geiger](http://www.cvlibs.net/)[Talks](http://www.cvlibs.net/talks.php)

Projects:
- TCS Research
	- [3DPoseLite](https://openaccess.thecvf.com/content/WACV2021/html/Dani_3DPoseLite_A_Compact_3D_Pose_Estimation_Using_Node_Embeddings_WACV_2021_paper.html) [video](https://www.youtube.com/watch?v=aPlHyxF7I1k)
- IIITH
	- [Quo Vadis, Skeleton Action Recognition ?](http://cvit.iiit.ac.in/research/projects/cvit-projects/quo-vadis-skeleton-action-recognition) 
	- [Zero Shot](http://cvit.iiit.ac.in/research/projects/cvit-projects/syntactically-guided-generative-embeddings-for-zero-shot-skeleton-action-recognition)
- CVPR
	- [SkeletonOnCVPR19](http://ubee.enseeiht.fr/skelneton/)
<!-- 
## Why me?
- Over 2 decades of professional experience in 3D modeling software development along with Masters and Doctoral degrees.
- Over half-a-decade of professional/teaching experience in Machine/Deep Learning.

## IKIGAI 
- World needs: huge demand, unique combination of 3D+ML needed forever!!
- Good at: both domains, got PhD, taught these subjects, vast experience
- Love doing: Wow projects in visualization, 3D and Pytorch
- Paid for: FAANG, $$$, Own, global impact, conferences, global reach

## Specific Knowledge 
- rare, un-trainable, 
- only through apprenticeship/experience, 
- unique ability of domain expertise plus teaching/counseling, 
- be a reliable brand!! 

 -->
## Resources
- Fundamental concepts [Artificial Intelligence for Geometry Processing (Rana Hanocka, Tel Aviv University)](https://www.youtube.com/watch?v=h8VRNYDrIAM)
- Introductory [Learning 3D Reconstruction in Function Space](https://www.youtube.com/watch?v=kxKI8_Si2a0)
- Implementation framework [Pytorch3D](https://github.com/facebookresearch/pytorch3d)
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin): A Pytorch Library for Accelerating 3D Deep Learning Research
- Amazing Scene generation ["Towards AI for 3D Content Creation" - Prof. Sanja Fidler, University of Toronto and NVIDIA](https://www.youtube.com/watch?v=lkkFcg9k9ho
- [2D to 3D surface reconstruction from images and point clouds](https://www.youtube.com/playlist?list=PL3OV2Akk7XpDjlhJBDGav08bef_DvIdH2)
- Detailed research series [CVSS 2019 - Computational Vision Summer School](https://www.youtube.com/playlist?list=PLeCNfJWZKqxsvidOlVLtWq9s7sIsX1QTC)
- Attention leverage [TransformerFusion: Monocular RGB SceneReconstruction using Transformers](https://www.youtube.com/watch?v=LIpTKYfKSqw)