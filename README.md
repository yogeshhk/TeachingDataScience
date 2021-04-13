# Teaching Data Science

Repository for LaTeX course notes for Python, Machine Learning, Deep Learning, Natural Language Processing, etc. Core content is in the form for Beamer slides, which in turn can get compiled into presentation mode as well as two-column course notes mode.

All tex sources and images have been open sourced as I have taken from others, learnt from others, although I have added some of mine, I need to give it back!! LinkedIn post: https://www.linkedin.com/feed/update/urn:li:activity:6523000857385103360

Copyright (C) 2019 Yogesh H Kulkarni

<!--
## Rational
- Global Recognition and getting opportunities and respect (Part II, 40+, chalk in hand)
- For Giving back from classes to masses, dyan-eshwar. 
- If any $: Donate to CoEP Alumni, Jagriti, Aai for causes (Experfy/INE already)
- For better own understanding
-->
<!-- 
Avenues: @TFUG Pune, Colleges/Confs, GDE speakers opportunities

## Playbook (conduct lectures under TFUG Pune)
Logistics:
- Day/time: Saturdays Mornings, 9 to 10 am, else NO
- Pre-Talk:
    - Prep materials _seminar.tex and ipynb for 40 minutes talk and then QnA
    - Create Google calendar event with Google Meet link
    - Create Banner png from admin/template.ppt
		- Create TFUG Pune Meetup event with Google Meet link and Banner png, good intro text
    - Advertise the Meetup link: my LinkedIn, Company Teams, GDE WhatsApp groups
    - Copy material pdf/ipynbs under github Talks/presentations folder
		- No recording or YouTube streaming, content is open-sourced anyway
- During Talk:
<!-- 		- Simplify with Doodles (?, tablet)

    - Take pictures or screenshots
    - At the end of the talk, share links to my github page, yati.io, linkedin profile
- Post-Talk:
    - Fill GDE Advocu entry with picture taken
    - Fill GDE New event survey link
    - Share beamer pdf of the talk on meetup, post pictures there

## Video Recordings
- Using OBS: Open-source Free-ware
	- Sources: Windows Capture
	- Settings:
		- Hot Keys: Alt + 10 to start recording, Alt + F11 to stop
		- Output format : mkv, easier to mix, can be converted to mp4 by File->Remux
		- Output gets stored in Videos folder
- OpenShot to edit video, save to join
- Handbrake to compress video
 -->
 
## Trainings/Talks

### Seminar Topics (1-2 hours long)
- Introduction to Artificial Intelligence (Non-technical)
- Introduction to Artificial Intelligence and Machine Learning (Technical)

- Introduction to Maths 4 AI: Calculus
- Introduction to Maths 4 AI: Linear Algebra
- Introduction to Maths 4 AI: Statistics

- Introduction to Python (Technical)

- Introduction to Data (Data Engineering, Tensorflow)
- Introduction to Data Analytics (Data Processing with Pandas)

- Introduction to Machine Learning (Technical)
- Introduction to ANN (Artificial Neural Networks, Tensorflow)
- Introduction to CNN (Convolutional Neural Networks, Tensorflow)
- Introduction to RNN (Convolutional Neural Networks, Tensorflow)

- Introduction to NLP (Natural Language Processing)
- Introduction to Word Embedding (Word 2 Vec, Tensorflow)
- Introduction to Deep NLP (Natural Language Processing with Neural Networks)
- Introduction to Conversational AI (Chatbot)
- Introduction to Text Mining (Topic Modeling, Custom NER)

- Introduction to Explainable AI (Non-technical)
- Decoding Gartner Hype Cycles for Emerging Technologies (Non-technical)
- Deep Learning for Geometric Algorithms (Research)

### Workshop Topics (2-5 days long if full-time)
- Introduction to Python (Technical)
- Introduction to Data Analytics (Data Processing with Pandas)

- Introduction to Maths 4 AI

- Introduction to Machine Learning (Technical)
- Introduction to Deep Learning (Technical)

- Introduction to NLP (Natural Language Processing)
- Introduction to Deep NLP (Natural Language Processing with Neural Networks)

- Introduction to Conversational AI (Chatbot)

### Talks (An hour long)
- Choose To Thinq: Mid-Career Transitions into ML-AI, with Yogesh Kulkarni https://www.youtube.com/watch?v=IQzWosVzkM4
- ODSC 2019: Encoder-Decoder Neural Network for Computing Midcurve of Thin Polygon by Yogesh Kulkarni https://www.youtube.com/watch?v=ZY0nuykqgoE
- AI Pillipinas: Introduction to RNN with TensorFlow https://www.youtube.com/watch?v=qFrdm_9fjJY

### Online Courses (Paid)
- Experfy: Introduction to Artificial Intelligence, Learn what exactly AI is, what are its different facets.
https://www.experfy.com/training/courses/introduction-to-artificial-intelligence

- Experfy: Unsupervised Learning: Dimensionality Reduction and Representation, Overcoming the Curse of Dimensionality
https://www.experfy.com/training/courses/unsupervised-learning-dimensionality-reduction-and-representation

- Experfy: Hands-on Project - Data Preparation, Modeling & Visualization
https://www.experfy.com/training/courses/hands-on-project-data-preparation-modeling-visualization

- INE: Introduction to Machine Learning 
https://my.ine.com/DataScience/courses/2eaf6b98/introduction-to-machine-learning

## Steps for LaTex source files to pdfs

### Code Arrangement
*	LaTeX directory 
	* Has tex sources along with necessary images
	*	Naming: subject_maintopic_subtopic.tex eg maths_linearalgebra_matrices.tex
	*	Main_Workshop/Seminar_Presentation/CourseMaterial.tex are the driver files
	*	They intern contain common content files, which have included actual source files
	*	Make bat files compile the appropriate sources
*	Code directory 
	*	Has running python or ipython notebook files with necessary images and data
	*	Naming should be such that corresponding latex file can be associated
	*	Library based tex file, say, sklearn_decisiontree.tex will have just template code and short fully working examples from Mastering Machine Learning whereas the sklearn_decisiontree.ipynb will have running workflows. No need to sync both. You may want to keep ipynb’s pdf in LaTeX/images directory for reference
*	References directory (not uploaded, as it is mostly from others github repos, nothing much of mine)
	*	Has papers, code, ppts as base material to be used for content preparation

### Requirements
* LaTeX (tested with MikTex 2.9 on Windows 7, 64bit)
* Need to install LaTeX packages, as and when you get such warning/suggestions.
* Using TexWorks as IDE


<!-- ## Data Science Course Series

<img src="LaTeX/images/teaching_data_science_series.png"/> -->

### How to Run LaTeX:
* Driver files for the courses are named with "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex"
* Both the Cheatsheet (meaning course notes in two column format) and Presentation.tex refer to same core content file, which in turn contains are the topic files.
* Run make bat for the course you need. Inside, its just a texify command, so you can modify it as per your OS.
* You can compile individual "Main_Workshop/Seminar_<course>_CheatSheet/Presentation.tex" also using your LaTeX system.
* Instead of these given driver files, you can have your own main files and include just the *content.tex files.

<!-- ## Notes

<!-- ## Good resources for learning
*	Machine Learning
    * ML Victor Levrenko https://www.youtube.com/user/victorlavrenko/playlists
    * Statistics ML https://www.youtube.com/user/BCFoltz/playlists 
*	Deep Learning
    * Deep Learning by Google https://in.udacity.com/course/deep-learning--ud730
    * Deep Learning Book lectures https://www.youtube.com/channel/UCF9O8Vj-FEbRDA5DcDGz-Pg/playlists

*	General
    * Open Data Science Masters http://datasciencemasters.org/
    * GeekForGeeks https://www.youtube.com/watch?v=v4cd1O4zkGw
 -->

## Disclaimer:
* Author (yogeshkulkarni@yahoo.com) gives no guarantee of the correctness of the content. Notes have been built using lots of publicly available material. 
* Although care has been taken to cite the original sources as much as possible, but there could be some missing ones. Do point them and I will update wherever possible. 
* Lots of improvements are still to be made. So, don’t depend on it at all, fully. 
* Do send in your suggestions/comments/corrections/Pull-requests.